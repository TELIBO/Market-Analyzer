"""Token-conscious LangGraph workflow for stock research reports.

Configure GROQ_MODEL, GROQ_CONTEXT_WINDOW, GROQ_INPUT_TOKEN_BUDGET,
GROQ_MAX_TOKENS, GROQ_MAX_RETRIES, NEWS_ARTICLE_LIMIT, NEWS_EXCERPT_CHARS,
NEWS_SEARCH_TIMEOUT, and NEWS_OFFICIAL_FETCH_LIMIT through environment variables.
"""

from __future__ import annotations

import json
import html
import math
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any, TypedDict
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import yfinance as yf

from stock_research.financials import (
    FINANCIAL_FIELDS,
    _annual_income_metrics,
    _date_string,
    _finite_number,
    _select_financials,
    _statement_columns,
    _statement_frame,
    _statement_metrics,
    _statement_value,
)

load_dotenv()

PERIOD_ALIASES = {
    "1d": "1d", "1 day": "1d", "5d": "5d", "5 days": "5d",
    "1mo": "1mo", "1 month": "1mo", "30d": "1mo",
    "3mo": "3mo", "3 months": "3mo", "90d": "3mo",
    "6mo": "6mo", "6 months": "6mo", "180d": "6mo",
    "1y": "1y", "1 year": "1y", "12mo": "1y", "1yr": "1y",
    "2y": "2y", "2 years": "2y", "5y": "5y", "5 years": "5y",
    "10y": "10y", "10 years": "10y", "ytd": "ytd", "max": "max",
}
PERIOD_DAYS = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "10y": 3650}
SYMBOL_STOP_WORDS = {
    "A", "AN", "AND", "ARE", "AT", "BUY", "FOR", "FROM", "HOW", "IN", "IS",
    "LATEST", "ME", "MY", "OF", "ON", "OR", "OUTLOOK", "PRICE", "RESEARCH",
    "SHOULD", "STOCK", "THE", "TO", "UPDATE", "WHAT", "WHEN", "WITH",
}
REQUIRED_SECTIONS = (
    "EXECUTIVE SUMMARY",
    "MARKET SNAPSHOT",
    "HISTORICAL PERFORMANCE",
    "COMPANY OVERVIEW",
    "RECENT NEWS & EVENTS",
    "BULLISH FACTORS",
    "RISKS & BEARISH FACTORS",
    "PRICE & VOLUME ANALYSIS",
    "NEWS SENTIMENT",
    "FINAL ASSESSMENT",
    "DATA QUALITY & LIMITATIONS",
)
LLM_REPORT_SECTIONS = (
    "EXECUTIVE SUMMARY",
    "BULLISH FACTORS",
    "RISKS & BEARISH FACTORS",
)


class ResearchPlan(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    companies: list[str] = Field(default_factory=list)
    intent: str = "general stock research"


class ResearchState(TypedDict, total=False):
    started_at: float
    execution_time_seconds: float
    user_question: str
    company_name: str
    ticker: str
    requested_ticker: str
    period: str
    requested_period: str
    tickers: list[str]
    companies: list[str]
    intent: str
    input_status: str
    market_summary: dict[str, Any]
    financial_metrics: dict[str, dict[str, Any]]
    news_evidence: list[dict[str, Any]]
    issuer_reported_fcf: dict[str, dict[str, Any]]
    market_errors: list[str]
    news_errors: list[str]
    warnings: list[str]
    report: str
    report_error: str
    validation_status: str
    validation_warnings: list[str]
    correction_attempts: int
    llm_usage: dict[str, int]
    execution_errors: list[str]


class PromptTooLargeError(ValueError):
    pass


class GroqRequestError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, int]) -> None:
        super().__init__(message)
        self.usage = usage


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def llm(max_tokens: int | None = None) -> ChatGroq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        max_tokens=max_tokens or _env_int("GROQ_MAX_TOKENS", 1800),
        max_retries=0,
    )


def estimate_tokens(text: str) -> int:
    """Conservative rough estimate; provider-reported usage is preferred."""
    return math.ceil(len(text) / 3.5)


def _status_code(error: Exception) -> int | None:
    response = getattr(error, "response", None)
    return getattr(error, "status_code", None) or getattr(response, "status_code", None)


def _retry_after(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _retryable(error: Exception) -> bool:
    code = _status_code(error)
    if code in {400, 413, 422}:
        return False
    message = str(error).lower()
    if any(token in message for token in ("context length", "maximum context", "request too large", "token limit")):
        return False
    return code == 429 or (code is not None and 500 <= code < 600)


def _usage_from_response(response: Any, estimated_input: int) -> dict[str, int]:
    metadata = getattr(response, "usage_metadata", None) or {}
    details = getattr(response, "response_metadata", None) or {}
    usage = details.get("token_usage") or details.get("usage") or {}
    input_tokens = metadata.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = metadata.get("output_tokens", usage.get("completion_tokens", 0))
    try:
        input_count, output_count = int(input_tokens or 0), int(output_tokens or 0)
    except (TypeError, ValueError):
        input_count, output_count = 0, 0
    return {
        "calls": 1,
        "retry_attempts": 0,
        "estimated_input_tokens": estimated_input,
        "actual_input_tokens": input_count,
        "actual_output_tokens": output_count,
    }


def _add_usage(current: dict[str, int] | None, addition: dict[str, int]) -> dict[str, int]:
    result = dict(current or {})
    for key in ("calls", "retry_attempts", "estimated_input_tokens", "actual_input_tokens", "actual_output_tokens"):
        result[key] = result.get(key, 0) + addition.get(key, 0)
    return result


def _invoke_groq(
    prompt: str,
    output_budget: int | None = None,
) -> tuple[Any, dict[str, int]]:
    context_limit = _env_int("GROQ_CONTEXT_WINDOW", 8192)
    max_output = output_budget or _env_int("GROQ_MAX_TOKENS", 1800)
    input_limit = min(_env_int("GROQ_INPUT_TOKEN_BUDGET", 5000), context_limit - max_output)
    estimated = estimate_tokens(prompt)
    if input_limit < 1 or estimated > input_limit:
        raise PromptTooLargeError(
            f"Estimated prompt size {estimated} tokens exceeds the configured input budget {input_limit}; request was not sent."
        )

    client = llm(max_output)
    retries = _env_int("GROQ_MAX_RETRIES", 2, minimum=0)
    for attempt in range(retries + 1):
        try:
            response = client.invoke(prompt)
            usage = _usage_from_response(response, estimated)
            usage["calls"] = attempt + 1
            usage["retry_attempts"] = attempt
            usage["estimated_input_tokens"] *= attempt + 1
            return response, usage
        except Exception as exc:
            if attempt >= retries or not _retryable(exc):
                raise GroqRequestError(
                    str(exc),
                    {
                        "calls": attempt + 1,
                        "retry_attempts": attempt,
                        "estimated_input_tokens": estimated * (attempt + 1),
                        "actual_input_tokens": 0,
                        "actual_output_tokens": 0,
                    },
                ) from exc
            delay = _retry_after(exc)
            time.sleep(delay if delay is not None else min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError("Unreachable Groq retry state")


def normalize_period(value: str | None) -> str:
    candidate = (value or "2y").strip().lower()
    normalized = PERIOD_ALIASES.get(candidate)
    if normalized is None:
        raise ValueError(f"Unsupported period {value!r}. Supported periods: {', '.join(sorted(set(PERIOD_ALIASES.values())))}")
    return normalized


def _extract_period(text: str) -> str | None:
    match = re.search(r"\b(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)\b", text, flags=re.IGNORECASE)
    if match:
        return normalize_period(match.group(1))
    patterns = (
        (r"\b(one|1)\s+day\b", "1d"),
        (r"\b(five|5)\s+days?\b", "5d"),
        (r"\b(one|1)\s+month\b", "1mo"),
        (r"\b(three|3)\s+months?\b", "3mo"),
        (r"\b(six|6)\s+months?\b", "6mo"),
        (r"\b(one|1|twelve|12)\s+years?\b", "1y"),
        (r"\b(two|2)\s+years?\b", "2y"),
    )
    for pattern, period in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return period
    return None


def _extract_tickers(text: str) -> list[str]:
    explicit = re.findall(r"\$([A-Za-z][A-Za-z0-9.-]{0,9})\b", text)
    candidates = explicit or re.findall(r"\b[A-Z][A-Z0-9]{0,4}(?:[.-][A-Z0-9]+)*\b", text)
    return list(dict.fromkeys(symbol.upper() for symbol in candidates if symbol.upper() not in SYMBOL_STOP_WORDS))[:5]


def _normalize_tickers(tickers: list[str]) -> list[str]:
    normalized: list[str] = []
    for ticker in tickers:
        symbol = ticker.strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9]{0,4}(?:[.-][A-Z0-9]+)*", symbol):
            raise ValueError(f"Invalid ticker symbol: {ticker!r}")
        if symbol not in normalized:
            normalized.append(symbol)
    return normalized[:5]


def _parse_research_plan(response: Any) -> ResearchPlan:
    if isinstance(response, ResearchPlan):
        return response

    content = getattr(response, "content", response)
    if not isinstance(content, str):
        raise ValueError("The planner returned an unsupported response format.")

    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    try:
        return ResearchPlan.model_validate_json(stripped)
    except ValueError:
        pass

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            data, _end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return ResearchPlan.model_validate(data)

    labels: dict[str, str] = {}
    for label in ("tickers", "companies", "intent"):
        match = re.search(
            rf"(?im)^\s*(?:\*\*)?{label}(?:\*\*)?\s*:\s*(?:\*\*)?\s*(.+?)\s*(?:\n|$)",
            content,
        )
        if match:
            labels[label] = match.group(1).strip().strip("`")
    if "tickers" in labels:
        def parse_list(value: str) -> list[str]:
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
            return re.findall(r"\b[A-Za-z][A-Za-z0-9.-]{0,9}\b", value)

        return ResearchPlan(
            tickers=parse_list(labels["tickers"]),
            companies=parse_list(labels.get("companies", "[]")),
            intent=labels.get("intent", "general stock research"),
        )
    raise ValueError("The planner response did not contain a valid research plan.")


def _usage_result(prompt: str) -> tuple[ResearchPlan | None, dict[str, int], str | None]:
    usage: dict[str, int] = {}
    try:
        estimate = estimate_tokens(prompt)
        context_limit = _env_int("GROQ_CONTEXT_WINDOW", 8192)
        input_limit = min(_env_int("GROQ_INPUT_TOKEN_BUDGET", 5000), context_limit - _env_int("GROQ_MAX_TOKENS", 1800))
        if estimate > input_limit:
            raise PromptTooLargeError(f"Estimated planning prompt size {estimate} tokens exceeds budget {input_limit}; request was not sent.")
        response, usage = _invoke_groq(
            prompt,
            output_budget=min(500, _env_int("GROQ_MAX_TOKENS", 1800)),
        )
        return _parse_research_plan(response), usage, None
    except Exception as exc:
        return None, getattr(exc, "usage", usage), str(exc)


def validate_input(state: ResearchState) -> dict[str, Any]:
    question = (state.get("user_question") or "").strip()
    input_ticker = state.get("requested_ticker") or state.get("ticker")
    if not question and not input_ticker:
        return {"input_status": "invalid", "execution_errors": ["Provide a stock ticker or a research question."]}

    try:
        period = normalize_period(state.get("requested_period") or state.get("period") or _extract_period(question) or "2y")
        tickers = _normalize_tickers(
            [input_ticker] if input_ticker else _extract_tickers(question)
        )
        usage: dict[str, int] = {}
        company_name = (state.get("company_name") or "").strip()
        companies: list[str] = [company_name] if company_name else []
        intent = question or "general stock research"
        if not tickers:
            planning_prompt = (
                "Identify public-company ticker symbols explicitly relevant to this request. Do not guess. "
                "Return exactly one JSON object and no markdown, with keys tickers (array of strings), "
                "companies (array of strings), and intent (string). Use an empty tickers array if uncertain.\n"
                + question
            )
            plan, usage, error = _usage_result(planning_prompt)
            if error:
                return {
                    "input_status": "invalid",
                    "llm_usage": usage,
                    "execution_errors": [f"Ticker planning failed: {error}"],
                }
            tickers = _normalize_tickers(plan.tickers)
            companies, intent = plan.companies, plan.intent
        if not tickers:
            return {
                "input_status": "invalid",
                "llm_usage": usage,
                "execution_errors": ["No valid ticker could be identified; no market or news requests were made."],
            }
        return {
            "input_status": "valid",
            "tickers": tickers,
            "companies": companies,
            "intent": intent,
            "period": period,
            "llm_usage": usage,
        }
    except ValueError as exc:
        return {"input_status": "invalid", "execution_errors": [str(exc)]}




from stock_research import sec as _sec
from stock_research.sec import (
    _effective_financial_value,
    _financial_source_citation,
    _is_stale_metric,
    _latest_dated_metric,
    _latest_fresh_dated_metric,
    _latest_sec_fact_by_end,
    _merge_statement_sources,
    _metric_period_date,
    _read_sec_json,
    _sec_company_ciks,
    _sec_company_facts,
    _sec_fact_rows,
    _sec_filing_details,
    _sec_standalone_quarters,
    _sec_statement_metrics,
)

def _retrieve_sec_statement_metrics(
    ticker: str,
    known_cik: Any = None,
) -> tuple[dict[str, Any], str | None]:
    previous_urlopen = _sec.urlopen
    _sec.urlopen = urlopen
    try:
        return _sec._retrieve_sec_statement_metrics(ticker, known_cik)
    finally:
        _sec.urlopen = previous_urlopen


def _return_metric(close: Any, target: date, end_price: float, end_date: date, prior_close: bool = False) -> dict[str, Any]:
    eligible = close[close.index.date < end_date] if prior_close else close[close.index.date <= target]
    if eligible.empty:
        return {"return_percent": None, "start_price": None, "end_price": round(end_price, 2), "start_date": None, "end_date": str(end_date)}
    start_index = eligible.index[-1]
    start_price = _finite_number(eligible.iloc[-1])
    if start_price is None or start_price <= 0:
        return {"return_percent": None, "start_price": None, "end_price": round(end_price, 2), "start_date": None, "end_date": str(end_date)}
    return {
        "return_percent": round((end_price / start_price - 1) * 100, 2),
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "start_date": str(start_index.date()),
        "end_date": str(end_date),
    }


def _calculate_market_data(symbol: str, period: str) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    ticker = yf.Ticker(symbol)
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        info = ticker.info or {}
    except Exception as exc:
        info = {}
        errors.append(f"{symbol} fundamentals unavailable: {exc}")

    financial_metrics = _select_financials(info)
    sec_metrics, sec_error = _retrieve_sec_statement_metrics(
        symbol, financial_metrics.get("cik")
    )
    required_sec_metrics = (
        "quarterly_revenue_growth_yoy",
        "quarterly_operating_income_growth_yoy",
        "quarterly_profit_margin",
        "trailing_12_month_free_cash_flow",
    )
    sec_covers_screen = all(
        isinstance(sec_metrics.get(key), dict)
        and isinstance(sec_metrics[key].get("value"), (int, float))
        and _metric_period_date(sec_metrics[key]) is not None
        for key in required_sec_metrics
    )
    statements: dict[str, Any] = {}
    comparisons: list[str] = []
    if sec_covers_screen:
        statement_metrics = dict(sec_metrics)
        financial_metrics["metadata"]["yahoo_statement_fallback_skipped"] = (
            "The latest SEC filing supplied dated values for the core score; "
            "older Yahoo statement requests were skipped."
        )
    else:
        try:
            annual_income = _statement_frame(
                ticker,
                "get_income_stmt",
                "income_stmt",
                freq="yearly",
            )
        except Exception as exc:
            annual_income = None
            financial_metrics["metadata"]["annual_statement_warning"] = (
                f"{symbol} annual income statement unavailable: {exc}"
            )
        for key, method, property_name in (
            ("quarterly_income_statement", "get_income_stmt", "quarterly_income_stmt"),
            ("quarterly_cash_flow_statement", "get_cash_flow", "quarterly_cashflow"),
            ("quarterly_balance_sheet", "get_balance_sheet", "quarterly_balance_sheet"),
        ):
            try:
                statements[key] = _statement_frame(ticker, method, property_name)
            except Exception as exc:
                errors.append(f"{symbol} {key.replace('_', ' ')} unavailable: {exc}")
                statements[key] = None
        statement_metrics = _statement_metrics(
            statements["quarterly_income_statement"],
            statements["quarterly_cash_flow_statement"],
            statements["quarterly_balance_sheet"],
        )
        statement_metrics.update(_annual_income_metrics(annual_income))
    if sec_metrics:
        if not sec_covers_screen:
            statement_metrics, comparisons = _merge_statement_sources(
                statement_metrics, sec_metrics
            )
    financial_metrics["statement_metrics"] = statement_metrics
    if sec_metrics:
        financial_metrics["metadata"]["official_filing_source"] = "SEC EDGAR company facts"
        financial_metrics["metadata"]["official_filing_retrieved_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")
        if comparisons:
            financial_metrics["metadata"]["financial_source_comparisons"] = comparisons
        if (
            statement_metrics.get("trailing_12_month_free_cash_flow", {}).get("source")
            == "SEC EDGAR company facts"
        ):
            financial_metrics["metadata"]["free_cash_flow_definition_note"] = (
                "The SEC-derived free-cash-flow figure uses operating cash flow less reported "
                "payments to acquire productive assets. An issuer's own free-cash-flow figure "
                "may differ if it nets proceeds from asset sales, incentives, or other adjustments; "
                "check the cited earnings release for that company-specific definition."
            )
        dated_liquidity = statement_metrics.get(
            "reported_total_cash_and_short_term_investments", {}
        )
        profile_cash = financial_metrics.get("total_cash")
        if isinstance(dated_liquidity.get("value"), (int, float)) and isinstance(
            profile_cash, (int, float)
        ):
            financial_metrics["metadata"]["cash_profile_period_note"] = (
                f"Yahoo Finance's total_cash profile value "
                f"({_format_metric('market_cap', profile_cash)}) has no reporting date. "
                f"The dated SEC figure for cash, cash equivalents and current marketable "
                f"securities is {_format_metric('market_cap', dated_liquidity['value'])} "
                f"as of {dated_liquidity.get('period_end')}; the undated profile value is "
                "not used as a current balance."
            )
    quarterly_fcf = statement_metrics.get("quarterly_free_cash_flow", {})
    trailing_fcf = statement_metrics.get("trailing_12_month_free_cash_flow", {})
    quarterly_end = _metric_period_date(quarterly_fcf)
    trailing_end = _metric_period_date(trailing_fcf)
    if quarterly_end and trailing_end and quarterly_end > trailing_end:
        financial_metrics["metadata"].setdefault(
            "financial_source_comparisons", []
        ).append(
            f"Latest quarterly free cash flow is for the period ended {quarterly_end.isoformat()}, "
            f"but the newest available TTM free cash flow ends {trailing_end.isoformat()}; "
            "the score uses the dated TTM figure, which could not be refreshed without four "
            "consecutive quarterly cash-flow and capital-expenditure periods."
        )
    if sec_error:
        financial_metrics["metadata"]["official_filing_warning"] = sec_error
        if os.getenv("SEC_USER_AGENT", "").strip():
            errors.append(sec_error)
    revenue_growth_selection = _latest_dated_metric([
        ("quarterly_revenue_growth_yoy", statement_metrics.get("quarterly_revenue_growth_yoy", {})),
        ("annual_revenue_growth_yoy", statement_metrics.get("annual_revenue_growth_yoy", {})),
    ])
    revenue_growth_entry = revenue_growth_selection[1] if revenue_growth_selection else {}
    revenue_growth = revenue_growth_entry.get("value")
    if revenue_growth is not None:
        financial_metrics["revenue_growth"] = revenue_growth / 100
        financial_metrics["metadata"]["profile_metric_periods"]["revenue_growth"] = (
            revenue_growth_entry.get("basis")
            or "Calculated from a comparable dated income statement"
        )
    quarterly_margin = None
    sec_quarterly_margin = statement_metrics.get("quarterly_profit_margin")
    if isinstance(sec_quarterly_margin, dict) and isinstance(sec_quarterly_margin.get("value"), (int, float)):
        quarterly_margin = float(sec_quarterly_margin["value"])
        financial_metrics["quarterly_profit_margin"] = quarterly_margin
        financial_metrics["metadata"]["quarterly_profit_margin_period_end"] = (
            sec_quarterly_margin.get("period_end")
        )
        financial_metrics["metadata"]["quarterly_profit_margin_source"] = (
            sec_quarterly_margin.get("source")
        )
    quarterly_revenue = statement_metrics.get("quarterly_revenue", {}).get("value")
    quarterly_net_income = statement_metrics.get("quarterly_net_income", {}).get("value")
    if quarterly_margin is None and quarterly_revenue and quarterly_net_income is not None:
        quarterly_margin = quarterly_net_income / quarterly_revenue
        financial_metrics["quarterly_profit_margin"] = quarterly_margin
        financial_metrics["metadata"]["quarterly_profit_margin_period_end"] = (
            statement_metrics.get("quarterly_net_income", {}).get("period_end")
        )
        financial_metrics["metadata"]["quarterly_profit_margin_source"] = (
            "Yahoo Finance via yfinance"
        )
        financial_metrics["metadata"]["quarterly_profit_margin_basis"] = (
            "quarterly income statement"
        )
    if quarterly_margin is None:
        annual_margin = statement_metrics.get("annual_profit_margin", {})
        if isinstance(annual_margin.get("value"), (int, float)):
            quarterly_margin = float(annual_margin["value"])
            financial_metrics["quarterly_profit_margin"] = quarterly_margin
            financial_metrics["metadata"]["quarterly_profit_margin_period_end"] = (
                annual_margin.get("period_end")
            )
            financial_metrics["metadata"]["quarterly_profit_margin_source"] = (
                annual_margin.get("source", "Yahoo Finance via yfinance")
            )
            financial_metrics["metadata"]["quarterly_profit_margin_basis"] = (
                "annual income statement fallback"
            )

    try:
        history = ticker.history(period=period, interval="1d", auto_adjust=False)
        if history is not None and not history.empty:
            history = history.sort_index()
    except Exception as exc:
        history = None
        errors.append(f"{symbol} historical prices unavailable: {exc}")
    if history is None or history.empty or "Close" not in history:
        errors.append(f"{symbol} returned no usable historical prices.")
        return {}, _select_financials(info), errors

    close = history["Close"].dropna()
    close = close[close.map(lambda value: (_finite_number(value) or 0) > 0)]
    if close.empty:
        errors.append(f"{symbol} returned no finite positive closing prices.")
        return {}, financial_metrics, errors

    end_index = close.index[-1]
    end_date = end_index.date()
    end_price = float(close.iloc[-1])
    prior = float(close.iloc[-2]) if len(close) > 1 else None
    daily_return = round((end_price / prior - 1) * 100, 2) if prior and prior > 0 else None

    latest = history.iloc[-1]
    adjusted_close = history["Adj Close"].dropna() if "Adj Close" in history else close
    adjusted_close = adjusted_close[adjusted_close.map(lambda value: (_finite_number(value) or 0) > 0)]
    if adjusted_close.empty:
        adjusted_close = close
    adjusted_end_price = float(adjusted_close.iloc[-1])
    volume_series = history["Volume"].dropna() if "Volume" in history else None
    current_volume = _finite_number(latest.get("Volume")) if "Volume" in history else None
    if current_volume is not None and current_volume < 0:
        current_volume = None
    average_volume = _finite_number(volume_series.tail(20).mean()) if volume_series is not None and not volume_series.empty else None
    volume_delta = round((current_volume / average_volume - 1) * 100, 2) if current_volume is not None and average_volume and average_volume > 0 else None

    returns = {
        "1-day": _return_metric(adjusted_close, end_date, adjusted_end_price, end_date, prior_close=True),
        "1-month": _return_metric(adjusted_close, end_date - timedelta(days=30), adjusted_end_price, end_date),
        "3-month": _return_metric(adjusted_close, end_date - timedelta(days=90), adjusted_end_price, end_date),
        "6-month": _return_metric(adjusted_close, end_date - timedelta(days=180), adjusted_end_price, end_date),
        "1-year": _return_metric(adjusted_close, end_date - timedelta(days=365), adjusted_end_price, end_date),
    }
    previous_year = adjusted_close[(adjusted_close.index.date < end_date) & (adjusted_close.index.year < end_date.year)]
    if not previous_year.empty and float(previous_year.iloc[-1]) > 0:
        returns["YTD"] = _return_metric(adjusted_close, previous_year.index[-1].date(), adjusted_end_price, end_date)
    else:
        returns["YTD"] = {"return_percent": None, "start_price": None, "end_price": round(end_price, 2), "start_date": None, "end_date": str(end_date)}
    requested_days = PERIOD_DAYS.get(period)
    if requested_days is not None and period not in {"1d", "1y", "2y", "5y", "10y"}:
        returns[f"requested period ({period})"] = _return_metric(adjusted_close, end_date - timedelta(days=requested_days), adjusted_end_price, end_date)

    market = {
        "ticker": symbol,
        "company_name": info.get("longName") or info.get("shortName") or symbol,
        "date": str(end_date),
        "close": round(end_price, 2),
        "daily_change_percent": daily_return,
        "open": _finite_number(latest.get("Open")),
        "high": _finite_number(latest.get("High")),
        "low": _finite_number(latest.get("Low")),
        "volume": int(current_volume) if current_volume is not None else None,
        "average_volume_20d": round(average_volume) if average_volume is not None else None,
        "volume_vs_average_percent": volume_delta,
        "returns": returns,
        "retrieved_at_utc": retrieved_at,
        "requested_history_period": period,
        "price_basis": "Unadjusted Close for quoted OHLC; Adj Close for historical returns when available",
    }
    financial_metrics["metadata"]["retrieved_at_utc"] = retrieved_at
    financial_metrics["metadata"]["price_history_end_date"] = str(end_date)
    return market, financial_metrics, errors


def retrieve_market(state: ResearchState) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    financials: dict[str, Any] = {}
    errors: list[str] = []
    for symbol in state["tickers"]:
        try:
            market, metrics, symbol_errors = _calculate_market_data(symbol, state["period"])
            if market:
                summary[symbol] = market
            if metrics:
                financials[symbol] = metrics
            errors.extend(symbol_errors)
        except Exception as exc:
            errors.append(f"{symbol} market retrieval failed: {exc}")
    return {"market_summary": summary, "financial_metrics": financials, "market_errors": errors}


def _canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def _published_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _company_tokens(company: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[a-z0-9]+", company, flags=re.IGNORECASE)
        if len(token) >= 3
    ]


def _is_official_company_news_host(host: str, company: str) -> bool:
    normalized_host = host.lower().removeprefix("www.")
    tokens = _company_tokens(company)
    if normalized_host.startswith(("ir.", "investor.", "investors.")):
        return any(token in normalized_host for token in tokens)
    return any(
        normalized_host in {f"{token}.com", f"about{token}.com"}
        or normalized_host.endswith((f".{token}.com", f".about{token}.com"))
        for token in tokens
    )


def _article_score(article: dict[str, Any], tickers: list[str], companies: list[str]) -> tuple[int, float]:
    text = f"{article.get('title', '')} {article.get('snippet', '')}".lower()
    relevance = sum(
        3
        for ticker in tickers
        if re.search(rf"(?<![a-z0-9]){re.escape(ticker.lower())}(?![a-z0-9])", text)
    )
    relevance += sum(2 for company in companies if company.lower() in text)
    domain = urlsplit(article.get("url", "")).netloc.lower()
    trusted_sources = (
        "sec.gov", "reuters.com", "apnews.com", "bloomberg.com",
        "wsj.com", "cnbc.com", "finance.yahoo.com",
    )
    authority = 5 if domain.endswith((".gov", ".edu")) or any(
        domain == source or domain.endswith("." + source) for source in trusted_sources
    ) or any(_is_official_company_news_host(domain, company) for company in companies) else 0
    published = _published_datetime(article.get("published", ""))
    recency = published.timestamp() if published else 0.0
    return relevance + authority, recency


def _is_low_information_article(article: dict[str, Any]) -> bool:
    title = _title_key(str(article.get("title") or ""))
    excerpt = _title_key(str(article.get("snippet") or ""))
    path = urlsplit(str(article.get("url") or "")).path.lower()
    if path.rstrip("/").endswith("/overview") or re.search(
        r"\boverview\b$", title
    ):
        return True
    financial_directory = bool(re.search(
        r"/(?:financials?|cash-flow|income-statement|balance-sheet)(?:/|$)",
        path,
    ))
    generic_financial_page = any(phrase in excerpt for phrase in (
        "cash flow statement",
        "latest financials balance sheet income statement",
        "annual and quarterly data",
    ))
    if financial_directory and (
        generic_financial_page or "financials" in title or "cash flow" in title
    ):
        return True
    generic_title = any(phrase in title for phrase in (
        "latest news",
        "latest updates analysis",
        "stock price latest news",
        "latest stock news",
        "latest news and real time alerts",
    ))
    generic_excerpt = any(phrase in excerpt for phrase in (
        "get the latest news and real time alerts",
        "tracks the full story in real time",
        "latest news and updates",
        "in depth reporting",
    ))
    return generic_title and (generic_excerpt or len(excerpt) < 100)


def _clean_source(source: str, url: str) -> str:
    candidate = str(source or "").strip()
    domain = urlsplit(url).netloc.lower().removeprefix("www.")
    if not candidate or candidate.lower().removeprefix("www.") == domain:
        known_sources = {
            "reuters.com": "Reuters",
            "bloomberg.com": "Bloomberg",
            "cnbc.com": "CNBC",
            "wsj.com": "The Wall Street Journal",
            "ft.com": "Financial Times",
            "seekingalpha.com": "Seeking Alpha",
            "marketbeat.com": "MarketBeat",
            "sec.gov": "U.S. SEC",
            "nvidianews.nvidia.com": "NVIDIA Newsroom",
            "ir.aboutamazon.com": "Amazon Investor Relations",
            "aboutamazon.com": "Amazon Investor Relations",
        }
        for known_domain, name in known_sources.items():
            if domain == known_domain or domain.endswith("." + known_domain):
                return name
        return domain or "Source unavailable"
    candidate = re.sub(r"\s+", " ", candidate)
    if len(candidate) > 80:
        return domain or candidate[:80]
    return candidate


def _clean_publication_date(value: str) -> str:
    text = str(value or "").strip()
    parsed = _published_datetime(text)
    if parsed is None:
        for date_format in ("%B %d, %Y", "%b %d, %Y"):
            try:
                parsed = datetime.strptime(text, date_format).replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
    if parsed is None:
        relative = re.fullmatch(
            r"(?i)(\d+)\s+(hour|day|week|month)s?\s+ago",
            text,
        )
        if relative:
            amount = int(relative.group(1))
            unit = relative.group(2).lower()
            delta = {
                "hour": timedelta(hours=amount),
                "day": timedelta(days=amount),
                "week": timedelta(weeks=amount),
                "month": timedelta(days=30 * amount),
            }[unit]
            parsed = datetime.now(timezone.utc) - delta
    return parsed.date().isoformat() if parsed else ""


def _article_publication_date(item: dict[str, Any]) -> str:
    for key in ("date", "published", "publishedAt", "published_date", "datetime", "timestamp"):
        value = item.get(key)
        if value:
            published = _clean_publication_date(str(value))
            if published:
                return published
    body = str(item.get("body") or item.get("snippet") or "")
    relative = re.match(
        r"(?i)^\s*(\d+\s+(?:hour|day|week|month)s?\s+ago)\b",
        body,
    )
    if relative:
        return _clean_publication_date(relative.group(1))
    url = str(item.get("url") or item.get("href") or "")
    date_match = re.search(
        r"(?<!\d)(20\d{2})[-/](\d{1,2})[-/](\d{1,2})(?!\d)",
        url,
    )
    if date_match:
        try:
            return date(
                int(date_match.group(1)),
                int(date_match.group(2)),
                int(date_match.group(3)),
            ).isoformat()
        except ValueError:
            pass
    return ""


def _official_article_metadata_and_excerpt(
    url: str,
    company: str,
) -> tuple[str, str]:
    parsed_url = urlsplit(url)
    host = parsed_url.netloc.lower().removeprefix("www.")
    if (
        parsed_url.scheme != "https"
        or not _is_official_company_news_host(host, company)
    ):
        return "", ""
    request = Request(
        url,
        headers={
            "User-Agent": "StockResearchWorkflow/1.0",
            "Accept": "text/html",
        },
    )
    with urlopen(request, timeout=_env_int("NEWS_OFFICIAL_TIMEOUT", 4)) as response:
        page = response.read(2_000_001)
    if len(page) > 2_000_000:
        raise ValueError("official news page exceeded the 2 MB enrichment limit")
    document = page.decode("utf-8", errors="replace")
    published = ""
    meta_tags = re.findall(r"(?is)<meta\b[^>]*>", document)
    for tag in meta_tags:
        attrs = {}
        for attribute in re.finditer(
            r"""([a-zA-Z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""",
            tag,
        ):
            attrs[attribute.group(1).lower()] = html.unescape(
                attribute.group(2) or attribute.group(3) or ""
            )
        meta_key = (attrs.get("property") or attrs.get("name") or "").lower()
        if meta_key in {
            "article:published_time",
            "datepublished",
            "pubdate",
            "publishdate",
        }:
            published = _clean_publication_date(attrs.get("content", ""))
            if published:
                break
    if not published:
        date_match = re.search(
            r'"datePublished"\s*:\s*"([^"]+)"',
            document,
            flags=re.IGNORECASE,
        )
        if date_match:
            published = _clean_publication_date(date_match.group(1))
    if not published:
        date_match = re.search(
            r"""(?is)<(?:span|time)\b[^>]*(?:module_date-text|datetime)[^>]*>(.*?)</(?:span|time)\s*>""",
            document,
        )
        if date_match:
            published_text = html.unescape(re.sub(r"(?is)<[^>]+>", " ", date_match.group(1)))
            published = _clean_publication_date(" ".join(published_text.split()))

    body = re.sub(r"(?is)<(script|style|noscript)\b[^>]*>.*?</\1\s*>", " ", document)
    paragraphs = re.findall(
        r"(?is)<(?:p|li)\b[^>]*>(.*?)</(?:p|li)\s*>",
        body,
    )
    for paragraph in paragraphs:
        text = html.unescape(re.sub(r"(?is)<[^>]+>", " ", paragraph))
        text = re.sub(r"\s+", " ", text).strip()
        if "free cash flow" not in text.lower():
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        relevant = next(
            (
                sentence.strip()
                for sentence in sentences
                if "free cash flow" in sentence.lower()
                and re.search(r"\b(?:trailing\s+(?:twelve|12)\s+months|ttm)\b", sentence, re.I)
            ),
            "",
        )
        if relevant:
            return published, _clean_excerpt(relevant, _env_int("NEWS_EXCERPT_CHARS", 500))
    return published, ""


def _clean_excerpt(value: str, limit: int) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    source_is_truncated = bool(re.search(r"(?:…|\.{3,})\s*$", text))
    if source_is_truncated:
        text = re.sub(r"(?:…|\.{3,})\s*$", "", text).rstrip()
    complete_sentences = [
        match for match in re.finditer(r".+?[.!?](?:[\"')\]]*)?(?=\s|$)", text)
        if len(text[:match.end()].strip()) <= limit
    ]
    if not complete_sentences:
        return ""
    excerpt = text[:complete_sentences[-1].end()].strip()
    if not source_is_truncated and len(text) <= limit and excerpt == text:
        return text
    return excerpt


def _title_key(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", title).lower()))


def _earnings_event(article: dict[str, Any]) -> tuple[str, str] | None:
    text = " ".join(str(article.get(key) or "") for key in ("title", "url")).lower()
    year_match = re.search(r"\b(20\d{2})\b", text)
    quarter_match = re.search(r"\bq([1-4])\b", text)
    if not quarter_match:
        ordinal = re.search(r"\b(first|second|third|fourth|[1-4](?:st|nd|rd|th))\s+quarter\b", text)
        if ordinal:
            quarter = ordinal.group(1)
            quarter_number = {
                "first": "1", "second": "2", "third": "3", "fourth": "4",
                "1st": "1", "2nd": "2", "3rd": "3", "4th": "4",
            }[quarter]
        else:
            return None
    else:
        quarter_number = quarter_match.group(1)
    if not year_match or not re.search(r"\b(earnings|results|release)\b", text):
        return None
    return year_match.group(1), quarter_number


def _same_story(first: dict[str, Any], second: dict[str, Any]) -> bool:
    first_tickers = set(first.get("related_tickers") or [])
    second_tickers = set(second.get("related_tickers") or [])
    if first_tickers and second_tickers and first_tickers.isdisjoint(second_tickers):
        return False
    first_url, second_url = first.get("url", ""), second.get("url", "")
    if first_url and second_url and _canonical_url(first_url) == _canonical_url(second_url):
        return True
    if first.get("news_scope") != second.get("news_scope"):
        return False
    first_title, second_title = _title_key(first.get("title", "")), _title_key(second.get("title", ""))
    if not first_title or not second_title:
        return False
    if first_title == second_title:
        return True
    if len(first_title) < 24 or len(second_title) < 24:
        return False
    first_domain = urlsplit(first_url).netloc.lower().removeprefix("www.")
    second_domain = urlsplit(second_url).netloc.lower().removeprefix("www.")
    first_root = ".".join(first_domain.split(".")[-2:])
    second_root = ".".join(second_domain.split(".")[-2:])
    first_event = _earnings_event(first)
    second_event = _earnings_event(second)
    if first_root == second_root and first_event and first_event == second_event:
        return True
    similarity = SequenceMatcher(None, first_title, second_title).ratio()
    return similarity >= 0.94 or (first_domain == second_domain and similarity >= 0.88)


def _merge_duplicate_article(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    tickers: list[str],
    companies: list[str],
) -> dict[str, Any]:
    merged = dict(existing)
    if _article_score(candidate, tickers, companies) > _article_score(existing, tickers, companies):
        merged.update(candidate)
    for key in ("published", "excerpt", "snippet", "url", "source"):
        if not merged.get(key) and candidate.get(key):
            merged[key] = candidate[key]
    if len(str(candidate.get("excerpt") or "")) > len(str(merged.get("excerpt") or "")):
        merged["excerpt"] = candidate["excerpt"]
    merged["related_tickers"] = list(dict.fromkeys([
        *existing.get("related_tickers", []),
        *candidate.get("related_tickers", []),
    ]))
    return merged


def _classify_article(article: dict[str, Any]) -> list[str]:
    text = f"{article.get('title', '')}. {article.get('snippet', '')}".lower()
    categories: list[str] = []
    if re.search(r"\b(quarter|q[1-4]|earnings|results|profit|eps|net income)\b", text):
        categories.append("earnings_results")
    if re.search(r"\b(revenue|net sales|sales)\b.{0,70}\b(grow\w*|increase\w*|rise\w*|up|higher|record\w*|billion|million)\b|\b(grow\w*|increase\w*|rise\w*|up|higher)\b.{0,70}\b(revenue|net sales|sales)\b", text):
        categories.append("revenue_increase")
    if re.search(r"\b(revenue|net sales|sales)\b.{0,70}\b(declin\w*|fall|fell|drop|down|lower|miss|shortfall)\b|\b(declin\w*|fell|drop|miss|shortfall)\b.{0,70}\b(revenue|net sales|sales)\b", text):
        categories.append("revenue_decline_or_miss")
    if re.search(r"\b(operating income|operating profit)\b.{0,70}\b(grow\w*|increase\w*|rise\w*|up|higher|record)\b|\b(grow\w*|increase\w*|rise\w*|up|higher)\b.{0,70}\b(operating income|operating profit)\b", text):
        categories.append("operating_income_increase")
    if re.search(r"\b(operating income|operating profit)\b.{0,70}\b(declin\w*|fall|fell|drop|down|lower|miss)\b|\b(declin\w*|fell|drop|miss)\b.{0,70}\b(operating income|operating profit)\b", text):
        categories.append("operating_income_decline")
    if re.search(r"\b(aws|cloud)\b.{0,80}\b(grow|growth|increase|accelerat|record|highest|strong)\b|\b(grow|growth|increase|accelerat|record|highest|strong)\b.{0,80}\b(aws|cloud)\b", text):
        categories.append("cloud_growth")
    if re.search(r"\b(free cash flow|cash flow|cash burn|liquidity)\b", text):
        categories.append("cash_flow")
    if re.search(r"\b(free cash flow|cash flow|cash burn)\b.{0,70}\b(negative|declin|fall|drop|wors|balloon|weak|turn)\b|\b(negative|declin|fall|drop|wors|balloon|weak)\b.{0,70}\b(free cash flow|cash flow|cash burn)\b", text):
        categories.append("cash_flow_pressure")
    if re.search(r"\b(capex|capital expenditure|capital spending|investment spending)\b", text):
        categories.append("capital_spending")
    if re.search(r"\b(capex|capital expenditure|capital spending|investment spending)\b.{0,70}\b(increas|higher|rise|grow|expand)\b|\b(increas|higher|rise|grow|expand)\b.{0,70}\b(capex|capital expenditure|capital spending)\b", text):
        categories.append("capital_spending_increase")
    if re.search(r"\b(earnings|profit|eps|results)\b.{0,70}\b(miss|below|fell short|disappoint)\b|\b(miss|below|fell short|disappoint)\b.{0,70}\b(earnings|profit|eps|expectation)\b", text):
        categories.append("earnings_miss")
    if re.search(r"\b(earnings|profit|eps|results)\b.{0,70}\b(beat|above|exceed)\b|\b(beat|above|exceed)\b.{0,70}\b(earnings|profit|eps|expectation)\b", text):
        categories.append("earnings_beat")
    if re.search(r"\b(guidance|forecast|outlook|expects?|projected|estimates?)\b", text):
        categories.append("guidance")
    if re.search(r"\b(regulator|antitrust|lawsuit|court|investigation|fine|settlement|compliance)\b", text):
        categories.append("regulatory_legal")
    if re.search(r"\b(shares?|stock|trading)\b.{0,60}\b(fell|drop|rose|gain|declin|surge|sell.off|higher|lower)\b|\b(fell|drop|rose|gain|declin|surge|sell.off)\b.{0,60}\b(shares?|stock)\b", text):
        categories.append("market_move")
    if re.search(r"\b(warehouse|delivery|workforce|layoff|labor|supply chain|launch|service outage|product)\b", text):
        categories.append("operations")
    categories = list(dict.fromkeys(categories))
    return categories or ["company_update"]


def prepare_news_evidence(
    results: list[dict[str, Any]],
    tickers: list[str],
    companies: list[str],
    limit: int,
    excerpt_length: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for item in results:
        url = str(item.get("url") or "").strip()
        title = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(item.get("title") or ""))).strip()
        snippet = _clean_excerpt(str(item.get("snippet") or ""), max(excerpt_length * 2, excerpt_length))
        candidate = {
            **item,
            "title": title,
            "url": url,
            "snippet": snippet,
            "related_tickers": list(item.get("related_tickers") or tickers),
        }
        if not title and not snippet:
            continue
        if _is_low_information_article(candidate):
            continue
        duplicate_index = next(
            (index for index, existing in enumerate(selected) if _same_story(existing, candidate)),
            None,
        )
        if duplicate_index is None:
            selected.append(candidate)
        else:
            selected[duplicate_index] = _merge_duplicate_article(
                selected[duplicate_index], candidate, tickers, companies
            )

    ranked = sorted(
        selected,
        key=lambda item: _article_score(item, tickers, companies),
        reverse=True,
    )
    evidence: list[dict[str, Any]] = []
    for item in ranked[:limit]:
        url = str(item.get("url") or "")
        source = _clean_source(str(item.get("source") or ""), url)[:80]
        published = _clean_publication_date(str(item.get("published") or ""))
        excerpt = _clean_excerpt(str(item.get("snippet") or ""), excerpt_length)
        categories = _classify_article({**item, "snippet": excerpt})
        evidence.append({
            "id": f"news_{len(evidence) + 1}",
            "title": str(item.get("title") or "")[:240],
            "source": source,
            "published": published,
            "url": url[:800],
            "excerpt": excerpt,
            "categories": ", ".join(categories),
            "related_tickers": list(item.get("related_tickers") or tickers),
            "news_scope": item.get("news_scope", "stock"),
        })
    return evidence


def _official_issuer_fcf_from_news(
    articles: list[dict[str, Any]],
    ticker: str,
    company: str,
) -> dict[str, Any] | None:
    for article in articles:
        if ticker not in (article.get("related_tickers") or [ticker]):
            continue
        url = str(article.get("url") or "")
        host = urlsplit(url).netloc.lower().removeprefix("www.")
        is_sec = host == "sec.gov" or host.endswith(".sec.gov")
        is_investor_site = _is_official_company_news_host(host, company)
        if not (is_sec or is_investor_site):
            continue
        published = str(article.get("published") or "")
        if not published:
            continue
        content = " ".join(
            str(article.get(key) or "")
            for key in ("title", "excerpt", "snippet")
        )
        match = re.search(
            r"free cash flow.{0,220}?"
            r"(outflow|inflow)\s+(?:of\s+)?\$?\s*"
            r"([\d,]+(?:\.\d+)?)\s*(billion|bn|million|m)\b"
            r".{0,120}?(?:trailing\s+(?:twelve|12)\s+months|ttm)",
            content,
            flags=re.IGNORECASE,
        )
        if not match:
            match = re.search(
                r"free cash flow.{0,160}?"
                r"(-?)\$\s*([\d,]+(?:\.\d+)?)\s*(billion|bn|million|m)\b"
                r".{0,120}?(?:trailing\s+(?:twelve|12)\s+months|ttm)",
                content,
                flags=re.IGNORECASE,
            )
        if not match:
            continue
        direction = match.group(1).lower() if match.lastindex == 3 else ""
        amount_text = match.group(2) if match.lastindex == 3 else match.group(2)
        unit = match.group(3).lower()
        amount = float(amount_text.replace(",", ""))
        amount *= 1_000_000_000 if unit in {"billion", "bn"} else 1_000_000
        if direction == "outflow" or (match.lastindex == 3 and direction == "-"):
            amount = -amount
        return {
            "value": amount,
            "published": published,
            "source": article.get("source") or host,
            "url": url,
            "news_id": article.get("id"),
            "evidence_text": content,
        }
    return None


def _reconcile_issuer_fcf(
    financial_metrics: dict[str, Any],
    issuer_evidence: dict[str, Any],
) -> bool:
    statements = financial_metrics.get("statement_metrics", {})
    calculated = statements.get("trailing_12_month_free_cash_flow", {})
    filed = str(calculated.get("filed_date") or "")
    published = str(issuer_evidence.get("published") or "")
    if not filed or not published:
        return False
    try:
        if abs((date.fromisoformat(filed) - date.fromisoformat(published)).days) > 14:
            return False
    except ValueError:
        return False

    issuer_entry = dict(issuer_evidence)
    issuer_entry.update({
        "period_start": calculated.get("period_start"),
        "period_end": calculated.get("period_end"),
        "source": issuer_evidence.get("source") or "Company investor-relations release",
        "basis": (
            f"Issuer-reported trailing-12-month free cash flow, published "
            f"{published}; aligned to SEC filing period ended {calculated.get('period_end')}; "
            f"[News {str(issuer_evidence.get('news_id') or '').removeprefix('news_')}]"
        ),
    })
    statements["issuer_reported_trailing_12_month_free_cash_flow"] = issuer_entry
    financial_metrics["statement_metrics"] = statements
    if isinstance(calculated.get("value"), (int, float)):
        delta = abs(float(calculated["value"]) - float(issuer_entry["value"]))
        if not math.isclose(delta, 0, abs_tol=0.01):
            delta_text = _format_metric("free_cash_flow", delta)
            standard_text = _format_metric("free_cash_flow", calculated["value"])
            issuer_text = _format_metric("free_cash_flow", issuer_entry["value"])
            evidence_text = str(issuer_entry.get("evidence_text") or "").lower()
            if "net of proceeds" in evidence_text and "incentiv" in evidence_text:
                definition = (
                    "the issuer release defines its figure using property/equipment purchases "
                    "net of proceeds from sales and incentives"
                )
            else:
                definition = (
                    "the issuer release uses its own free-cash-flow definition; the exact "
                    "reconciling adjustments were not fully identified in the retrieved excerpt"
                )
            financial_metrics.setdefault("metadata", {})["free_cash_flow_definition_note"] = (
                f"For the TTM period ended {calculated.get('period_end')}, SEC cash flow less "
                f"reported productive-asset purchases is {standard_text}; the issuer-reported "
                f"figure is {issuer_text}. The {delta_text} difference reflects that {definition}. "
                "The difference is not a separately verified proceeds amount."
            )
    return True


def _business_search_terms(metrics: dict[str, Any]) -> list[str]:
    text = " ".join(
        str(metrics.get(key) or "")
        for key in ("sector", "industry", "business_summary")
    ).lower()
    topic_patterns = (
        (r"\baws\b|amazon web services", "cloud computing"),
        (r"\bcloud\b|compute|data cent(?:er|re)", "cloud infrastructure"),
        (r"artificial intelligence|\bai\b|machine learning", "artificial intelligence"),
        (r"\bgpu|graphics process", "GPUs"),
        (r"semiconductor|semiconductors|chipmaker|\bchips?\b", "semiconductors"),
        (r"\bretail\b|e-commerce|ecommerce|online stor", "online retail"),
        (r"\badvertis", "digital advertising"),
        (r"subscription|streaming|media content", "streaming and subscriptions"),
        (r"automotive|electric vehicle|\bev\b|self.driving", "automotive technology"),
        (r"cybersecurity|security software", "cybersecurity"),
        (r"wireless|telecommunications|mobile network", "telecommunications"),
        (r"pharmaceutical|drug development|biotechnology", "biopharmaceuticals"),
        (r"banking|financial services|payments", "financial services"),
        (r"renewable|solar|wind power|electric utility", "energy"),
    )
    selected = [
        topic
        for pattern, topic in topic_patterns
        if re.search(pattern, text)
    ]
    return list(dict.fromkeys(selected))[:4]


def retrieve_news(state: ResearchState) -> dict[str, Any]:
    companies = state.get("companies") or []
    company_by_ticker = {
        ticker: companies[index] if index < len(companies) else ""
        for index, ticker in enumerate(state["tickers"])
    }
    financial_metrics = state.get("financial_metrics", {})
    raw: list[dict[str, Any]] = []
    errors: list[str] = []
    limit = _env_int("NEWS_SEARCH_LIMIT", 5)
    timeout = _env_int("NEWS_SEARCH_TIMEOUT", 4)
    tasks: list[tuple[str, str, str, str]] = []
    for ticker in state["tickers"]:
        company = company_by_ticker[ticker]
        metrics = financial_metrics.get(ticker, {})
        company_query = " ".join(
            part for part in (ticker, company, "earnings results stock latest news")
            if part
        )
        business_topics = _business_search_terms(metrics)
        business_context = " ".join(
            part for part in (
                metrics.get("industry"),
                metrics.get("sector"),
                *business_topics,
            ) if part
        )
        business_query = " ".join(
            part for part in (business_context, "industry business news latest")
            if part
        )
        tasks.extend((
            (ticker, company, "stock", company_query),
            (ticker, company, "business", business_query),
        ))

    def search_news(task: tuple[str, str, str, str]) -> tuple[tuple[str, str, str, str], list[dict[str, Any]], str | None]:
        ticker, _company, scope, query = task
        try:
            rows = list(
                DDGS(timeout=timeout).text(
                    query, timelimit="m", max_results=limit
                )
            )
            normalized = [{
                "title": item.get("title"),
                "url": item.get("href"),
                "source": urlsplit(item.get("href") or "").netloc,
                "published": _article_publication_date({
                    **item,
                    "url": item.get("href"),
                }),
                "snippet": re.sub(
                    r"(?i)^\s*\d+\s+(?:hour|day|week|month)s?\s+ago\s*[|\-–—�]\s*",
                    "",
                    str(item.get("body") or ""),
                ),
                "related_tickers": [ticker],
                "news_scope": scope,
            } for item in rows]
            return task, normalized, None
        except Exception as exc:
            return task, [], f"{ticker} {scope} news retrieval failed: {exc}"

    if tasks:
        with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as executor:
            futures = [executor.submit(search_news, task) for task in tasks]
            for future in as_completed(futures):
                task, articles, error = future.result()
                raw.extend(articles)
                if error:
                    errors.append(error)

    official_fetch_limit = _env_int("NEWS_OFFICIAL_FETCH_LIMIT", 1, minimum=0)
    for ticker, company in company_by_ticker.items():
        official_articles = sorted(
            (
                article for article in raw
                if ticker in article.get("related_tickers", [])
                and article.get("news_scope") == "stock"
                and _is_official_company_news_host(
                    urlsplit(str(article.get("url") or "")).netloc, company
                )
                and not _is_low_information_article(article)
            ),
            key=lambda article: _article_score(article, [ticker], [company]),
            reverse=True,
        )
        for article in official_articles[:official_fetch_limit]:
            url = str(article.get("url") or "")
            try:
                published, excerpt = _official_article_metadata_and_excerpt(url, company)
                if published:
                    article["published"] = published
                if excerpt:
                    article["snippet"] = excerpt
            except Exception as exc:
                errors.append(
                    f"Official news enrichment failed for {ticker} "
                    f"({urlsplit(url).netloc}): {exc}"
                )

    per_ticker_limit = _env_int("NEWS_ARTICLE_LIMIT", 5)
    stock_limit = min(3, per_ticker_limit)
    business_limit = max(0, per_ticker_limit - stock_limit)
    recent_days = _env_int("NEWS_MAX_AGE_DAYS", 45)
    excerpt_length = _env_int("NEWS_EXCERPT_CHARS", 500)
    evidence: list[dict[str, Any]] = []
    for ticker in state["tickers"]:
        ticker_rows = [
            item for item in raw
            if ticker in item.get("related_tickers", [])
        ]
        cutoff = date.today() - timedelta(days=recent_days)
        recent_rows = [
            item for item in ticker_rows
            if not item.get("published")
            or not _published_datetime(str(item["published"]))
            or _published_datetime(str(item["published"])).date() >= cutoff
        ]
        stock_rows = [item for item in recent_rows if item.get("news_scope") == "stock"]
        business_rows = [item for item in recent_rows if item.get("news_scope") == "business"]
        stock_evidence = prepare_news_evidence(
            stock_rows,
            [ticker],
            [company_by_ticker[ticker]] if company_by_ticker[ticker] else [],
            stock_limit,
            excerpt_length,
        )
        business_evidence = prepare_news_evidence(
            business_rows,
            [ticker],
            [company_by_ticker[ticker]] if company_by_ticker[ticker] else [],
            business_limit,
            excerpt_length,
        )
        selected = [*stock_evidence, *business_evidence]
        if len(selected) < per_ticker_limit:
            selected_urls = {_canonical_url(str(item.get("url") or "")) for item in selected}
            remainder = prepare_news_evidence(
                recent_rows,
                [ticker],
                [company_by_ticker[ticker]] if company_by_ticker[ticker] else [],
                per_ticker_limit,
                excerpt_length,
            )
            selected.extend(
                item for item in remainder
                if _canonical_url(str(item.get("url") or "")) not in selected_urls
            )
        evidence.extend(selected[:per_ticker_limit])

    deduplicated: list[dict[str, Any]] = []
    for article in evidence:
        duplicate = next(
            (existing for existing in deduplicated if _same_story(existing, article)),
            None,
        )
        if duplicate is None:
            deduplicated.append(article)
        else:
            merged = _merge_duplicate_article(
                duplicate, article, state["tickers"], companies
            )
            duplicate.update(merged)
    for index, article in enumerate(deduplicated, start=1):
        article["id"] = f"news_{index}"
    issuer_reported_fcf = {}
    for ticker, company in company_by_ticker.items():
        evidence = _official_issuer_fcf_from_news(deduplicated, ticker, company)
        if evidence:
            issuer_reported_fcf[ticker] = evidence
    return {
        "news_evidence": deduplicated,
        "issuer_reported_fcf": issuer_reported_fcf,
        "news_errors": errors,
    }


def prepare_report_input(state: ResearchState) -> dict[str, Any]:
    warnings: list[str] = []
    if not state.get("market_summary"):
        warnings.append("Market prices are unavailable.")
    if not state.get("news_evidence"):
        warnings.append("No news evidence was retrieved.")
    warnings.extend(state.get("market_errors", []))
    warnings.extend(state.get("news_errors", []))
    financial_metrics = {
        ticker: dict(metrics)
        for ticker, metrics in state.get("financial_metrics", {}).items()
    }
    for ticker, evidence in state.get("issuer_reported_fcf", {}).items():
        metrics = financial_metrics.get(ticker)
        if metrics and _reconcile_issuer_fcf(metrics, evidence):
            continue
        if metrics:
            metrics.setdefault("metadata", {})["issuer_fcf_warning"] = (
                "An issuer-reported TTM free-cash-flow value was found but could not be safely "
                "matched to the latest SEC filing period; it was not used for scoring."
            )
    for ticker, metrics in financial_metrics.items():
        statements = metrics.get("statement_metrics", {})
        if (
            ticker not in state.get("issuer_reported_fcf", {})
            and statements.get("trailing_12_month_free_cash_flow", {}).get("source")
            == "SEC EDGAR company facts"
        ):
            metrics.setdefault("metadata", {})["issuer_fcf_warning"] = (
                "The latest SEC filing supports a calculated TTM free-cash-flow figure, but "
                "no period-matched issuer-reported figure was verified in the retrieved news; "
                "the score uses operating cash flow less reported productive-asset purchases."
            )
    return {"warnings": warnings, "financial_metrics": financial_metrics}


def _market_section(state: ResearchState) -> str:
    def format_price(value: Any) -> str:
        return "N/A" if value is None else f"{value:,.2f}"

    lines: list[str] = []
    for symbol in state["tickers"]:
        market = state.get("market_summary", {}).get(symbol)
        if not market:
            lines.append(f"{symbol}: market data unavailable.")
            continue
        daily = market.get("daily_change_percent")
        change = f"{daily:+.2f}%" if isinstance(daily, (int, float)) else "N/A"
        lines.append(
            f"{symbol} ({market.get('company_name', symbol)}): close ${market['close']:.2f} "
            f"on {market['date']}; daily change {change}."
        )
        volume = f"{market['volume']:,}" if market.get("volume") is not None else "N/A"
        lines.append(
            f"Open/high/low: ${format_price(market.get('open'))} / "
            f"${format_price(market.get('high'))} / "
            f"${format_price(market.get('low'))}; volume: {volume}"
        )
        lines[-1] += (
            f"; 20-day average volume: {market['average_volume_20d']:,}."
            if market.get("average_volume_20d") is not None
            else "; 20-day average volume: N/A."
        )
    return "\n".join(lines) or "Market data unavailable."


def _history_section(state: ResearchState) -> str:
    lines: list[str] = [
        "Returns use adjusted closing prices when Yahoo Finance supplies them; the market snapshot uses the unadjusted closing quote."
    ]
    periods = ("1-day", "1-month", "3-month", "6-month", "YTD", "1-year")
    for symbol in state["tickers"]:
        market = state.get("market_summary", {}).get(symbol)
        lines.append(f"{symbol}:")
        if not market:
            lines.append("Historical performance unavailable.")
            continue
        for label in periods:
            metric = market.get("returns", {}).get(label)
            if not metric or metric.get("return_percent") is None:
                lines.append(f"- {label}: N/A (insufficient data for the requested period).")
            else:
                lines.append(
                    f"- {label}: {metric['return_percent']:+.2f}% "
                    f"(from ${metric['start_price']:.2f} on {metric['start_date']} "
                    f"to ${metric['end_price']:.2f} on {metric['end_date']})."
                )
    return "\n".join(lines)


def _price_volume_section(state: ResearchState) -> str:
    lines: list[str] = []
    for symbol in state["tickers"]:
        market = state.get("market_summary", {}).get(symbol)
        if not market:
            lines.append(f"- {symbol}: Price and volume data unavailable.")
            continue
        daily = market.get("daily_change_percent")
        daily_text = f"{daily:+.2f}%" if isinstance(daily, (int, float)) else "unavailable"
        lines.append(
            f"- {symbol}: daily change {daily_text}; "
            f"close ${market['close']:.2f} on {market['date']}."
        )
        current = market.get("volume")
        average = market.get("average_volume_20d")
        if current is None or average is None or average <= 0:
            lines.append("- Volume comparison unavailable.")
        else:
            multiple = current / average
            lines.append(
                f"- Volume: {current:,} shares versus the 20-day average of "
                f"{average:,} ({multiple:.2f}x average). This describes relative volume only "
                "and does not establish a cause for the price move."
            )
    return "\n".join(lines) or "Price and volume data unavailable."


def _normalize_heading(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    normalized = re.sub(r"^#{1,6}\s*", "", normalized)
    normalized = re.sub(r"^(?:\*\*|__)\s*|\s*(?:\*\*|__)$", "", normalized)
    normalized = normalized.strip().rstrip(":").strip()
    normalized = re.sub(r"\s*\([^()\n]*\)\s*$", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.upper()


def _heading_line_pattern(title: str) -> re.Pattern[str]:
    escaped = re.escape(title)
    return re.compile(
        rf"(?im)^[ \t]*(?:#{{1,6}}[ \t]*)?(?:\*\*|__)?[ \t]*{escaped}"
        rf"(?:[ \t]*\([^()\n]*\))?[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*$"
    )


def _news_section(state: ResearchState) -> str:
    articles = state.get("news_evidence", [])
    if not articles:
        return "No recent news evidence was available."
    lines: list[str] = []
    for article in articles:
        citation = f"[News {article['id'].removeprefix('news_')}]"
        published = article.get("published") or "publication date unavailable"
        lines.append(
            f"- {citation} {article.get('title') or 'Untitled article'} "
            f"({article.get('source') or 'source unavailable'}; {published}; "
            f"{'business-area news' if article.get('news_scope') == 'business' else 'company news'})."
        )
        related_tickers = article.get("related_tickers") or []
        if related_tickers:
            lines.append(f"  Retrieved for: {', '.join(related_tickers)}.")
        categories = article.get("categories")
        if categories:
            lines.append(f"  Evidence topics: {categories}.")
        if article.get("excerpt"):
            lines.append(f"  Evidence excerpt: {article['excerpt']}")
    return "\n".join(lines)


def _compact_business_summary(value: Any, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    sentences = [
        match.group().strip()
        for match in re.finditer(r".+?[.!?](?=\s|$)", text)
    ]
    compact = " ".join(sentences[:2])
    if not compact and text:
        compact = text[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    if len(compact) > limit:
        first_sentence = sentences[0] if sentences else ""
        compact = first_sentence if len(first_sentence) <= limit else ""
    return compact


def _normalize_news_citations(content: str, state: ResearchState) -> str:
    available = {
        article["id"].removeprefix("news_")
        for article in state.get("news_evidence", [])
    }

    def replace(match: re.Match[str]) -> str:
        number = match.group(1) or match.group(2)
        if number in available:
            return f"[News {number}]"
        return f"[Unsupported News {number}]"

    return re.sub(
        r"【\s*news[_\s]?(\d+)\s*】|\[\s*news[_\s]?(\d+)\s*\]",
        replace,
        content,
        flags=re.IGNORECASE,
    )


def _news_factors(state: ResearchState, positive: bool) -> list[str]:
    categories = (
        {"revenue_increase", "cloud_growth", "earnings_beat", "operating_income_increase"}
        if positive
        else {"revenue_decline_or_miss", "cash_flow_pressure", "capital_spending_increase", "earnings_miss", "operating_income_decline"}
    )
    factors: list[str] = []
    for article in state.get("news_evidence", []):
        related_tickers = article.get("related_tickers") or state.get("tickers", [])
        if len(state.get("tickers", [])) > 1 and not set(related_tickers).intersection(state["tickers"]):
            continue
        excerpt = " ".join((article.get("excerpt") or "").split())
        article_categories = set((article.get("categories") or "").split(", "))
        matching = article_categories & categories
        if not excerpt or not matching:
            continue
        number = article["id"].removeprefix("news_")
        title = article.get("title") or "News report"
        topics = ", ".join(sorted(matching)).replace("_", " ")
        factors.append(f"- Evidence on {topics}: {excerpt[:260]} [News {number}]")
        if len(factors) == 3:
            break
    return factors


def _format_metric(key: str, value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    if key == "market_cap":
        absolute = abs(value)
        sign = "-" if value < 0 else ""
        if absolute >= 1_000_000_000_000:
            return f"{sign}${absolute / 1_000_000_000_000:.2f}T"
        if absolute >= 1_000_000_000:
            return f"{sign}${absolute / 1_000_000_000:.2f}B"
        if absolute >= 1_000_000:
            return f"{sign}${absolute / 1_000_000:.2f}M"
        return f"{sign}${absolute:,.2f}"
    if key in {"revenue_growth", "profit_margin", "return_on_equity", "dividend_yield"}:
        return f"{value * 100:.2f}%"
    if key == "debt_to_equity":
        return f"{value:.2f}%"
    if key in {"pe", "forward_pe", "enterprise_to_ebitda"}:
        return f"{value:.2f}x"
    if key in {"total_cash", "free_cash_flow"}:
        return _format_metric("market_cap", value)
    return f"{value:,.2f}"


def _ticker_state(state: ResearchState, ticker: str) -> ResearchState:
    return {
        **state,
        "tickers": [ticker],
        "market_summary": {
            ticker: state.get("market_summary", {}).get(ticker, {})
        },
        "financial_metrics": {
            ticker: state.get("financial_metrics", {}).get(ticker, {})
        },
        "news_evidence": [
            article for article in state.get("news_evidence", [])
            if ticker in (article.get("related_tickers") or state.get("tickers", []))
        ],
    }


def _screen_evidence(state: ResearchState, ticker: str) -> dict[str, Any]:
    metrics = state.get("financial_metrics", {}).get(ticker, {})
    market = state.get("market_summary", {}).get(ticker, {})
    statements = metrics.get("statement_metrics", {})
    factors: list[dict[str, Any]] = []

    growth, growth_basis = _effective_financial_value(
        metrics, "revenue_growth", allow_profile=False
    )
    if growth is not None:
        factors.append({
            "name": "Revenue growth",
            "value": growth,
            "display": _format_metric("revenue_growth", growth),
            "period": growth_basis,
            "points": 1 if growth > 0 else -1 if growth < 0 else 0,
            "weight": 1,
            "dated": bool(re.search(r"\d{4}-\d{2}-\d{2}", growth_basis)),
        })

    operating_growth_selection = _latest_fresh_dated_metric(metrics, [
        ("quarterly_operating_income_growth_yoy", statements.get("quarterly_operating_income_growth_yoy", {})),
        ("annual_operating_income_growth_yoy", statements.get("annual_operating_income_growth_yoy", {})),
    ])
    operating_growth = operating_growth_selection[1] if operating_growth_selection else {}
    if isinstance(operating_growth.get("value"), (int, float)):
        value = float(operating_growth["value"])
        source = operating_growth.get("source") or "Yahoo Finance via yfinance"
        factors.append({
            "name": "Operating-income growth",
            "value": value,
            "display": f"{value:+.2f}%",
            "period": (
                f"{operating_growth.get('basis') or 'year-over-year operating-income comparison'}; "
                f"{operating_growth.get('period_end') or 'date unavailable'} versus "
                f"{operating_growth.get('comparison_period_end') or 'prior fiscal period'}; source: {source}"
            ),
            "points": 2 if value >= 15 else 1 if value > 0 else -2 if value <= -15 else -1 if value < 0 else 0,
            "weight": 2,
            "dated": True,
        })

    margin, margin_basis = _effective_financial_value(
        metrics, "profit_margin", allow_profile=False
    )
    if margin is not None:
        factors.append({
            "name": "Profit margin",
            "value": margin,
            "display": _format_metric("profit_margin", margin),
            "period": margin_basis,
            "points": 1 if margin > 0 else -1 if margin < 0 else 0,
            "weight": 1,
            "dated": bool(re.search(r"\d{4}-\d{2}-\d{2}", margin_basis)),
        })

    cash_flow, cash_basis = _effective_financial_value(metrics, "free_cash_flow", allow_profile=False)
    if cash_flow is not None:
        uses_ttm = "trailing 12 months" in cash_basis
        weight = 2 if uses_ttm else 1
        factors.append({
            "name": "Free cash flow",
            "value": cash_flow,
            "display": _format_metric("free_cash_flow", cash_flow),
            "period": cash_basis,
            "points": weight if cash_flow > 0 else -weight if cash_flow < 0 else 0,
            "weight": weight,
            "dated": True,
        })

    pe = metrics.get("pe")
    valuation = (
        {"value": float(pe), "display": f"{pe:.2f}x", "period": "Yahoo Finance trailing P/E; quote can update independently"}
        if isinstance(pe, (int, float)) and pe > 0
        else None
    )
    close = market.get("close")
    factor_by_name = {factor["name"]: factor for factor in factors}
    cash_flow_weight = (
        2 if "trailing 12 months" in cash_basis
        else 1 if cash_flow is not None
        else 2
    )
    factor_breakdown = []
    for name, potential_weight in (
        ("Revenue growth", 1),
        ("Operating-income growth", 2),
        ("Profit margin", 1),
        ("Free cash flow", cash_flow_weight),
    ):
        factor = factor_by_name.get(name)
        factor_breakdown.append({
            "name": name,
            "weight": factor["weight"] if factor else potential_weight,
            "available": factor is not None,
            "evidence": factor["display"] if factor else "Not available",
            "period": factor["period"] if factor else "No comparable dated statement value",
            "points": factor["points"] if factor else 0,
            "dated": factor.get("dated", False) if factor else False,
        })
    maximum_score = 4 + cash_flow_weight
    return {
        "factors": factors,
        "factor_breakdown": factor_breakdown,
        "score": sum(factor["points"] for factor in factors),
        "available_factor_count": len(factors),
        "dated_factor_count": sum(bool(factor.get("dated")) for factor in factors),
        "max_score": maximum_score,
        "available_max_score": sum(factor["weight"] for factor in factors),
        "minimum_score": -maximum_score,
        "maximum_score": maximum_score,
        "cash_flow_score_basis": (
            "trailing-12-month free cash flow" if cash_flow_weight == 2
            else "quarterly free cash flow fallback"
        ),
        "valuation": valuation,
        "has_price": isinstance(close, (int, float)) and close > 0,
    }


def _research_stance(
    state: ResearchState,
    ticker: str | None = None,
) -> tuple[str, list[str], list[str], str]:
    tickers = [ticker] if ticker else state.get("tickers", [])
    if not ticker and len(tickers) > 1:
        per_ticker = [
            (symbol, _research_stance(_ticker_state(state, symbol), symbol))
            for symbol in tickers
        ]
        positive = [fact for _symbol, result in per_ticker for fact in result[1]]
        risks = [fact for _symbol, result in per_ticker for fact in result[2]]
        rationale = " ".join(
            f"{symbol}: {result[0]}. {result[3]}"
            for symbol, result in per_ticker
        )
        return "PER-TICKER SCREENS (see each ticker separately)", positive[:8], risks[:8], rationale

    positive_facts: list[str] = []
    risk_facts: list[str] = []
    if not tickers:
        return "INSUFFICIENT EVIDENCE", [], [], "No ticker was available to screen."
    symbol = tickers[0]
    evidence = _screen_evidence(state, symbol)
    factors = evidence["factors"]
    score = evidence["score"]
    available_max = evidence["available_max_score"]
    if not evidence["has_price"]:
        stance = "INSUFFICIENT DATA — no current market price"
    elif not factors:
        stance = "INSUFFICIENT DATA — no usable fundamental measures were retrieved"
    elif len(factors) < 3 or evidence["dated_factor_count"] < 2:
        stance = (
            f"LIMITED DATA COVERAGE — {len(factors)} of 4 measures available, "
            f"{evidence['dated_factor_count']} with dated support; score is not a directional conclusion"
        )
    elif score >= max(2, (available_max + 1) // 2):
        stance = "CONSTRUCTIVE FUNDAMENTALS — valuation and suitability still need review"
    elif score <= -max(2, (available_max + 1) // 2):
        stance = "ELEVATED FUNDAMENTAL RISK — review the drivers before deciding"
    else:
        stance = "MIXED FUNDAMENTALS — evidence points in different directions"

    rationale_parts = [
        f"Deterministic fundamental score: {score:+d} points from {len(factors)} of 4 available measures "
        f"({evidence['dated_factor_count']} dated); revenue growth ±1; operating-income growth "
        "(stronger change ≥15% or ≤−15% scores ±2, smaller change scores ±1; "
        "profit margin ±1; "
        "trailing-12-month free cash flow ±2, or quarterly free cash flow ±1 when TTM is unavailable; "
        "missing measures contribute zero)."
    ]
    for factor in factors:
        description = (
            "supports the operating case"
            if factor["points"] > 0
            else "weighs against the operating case"
            if factor["points"] < 0
            else "is neutral"
        )
        rationale_parts.append(
            f"{factor['name']}: {factor['display']} ({factor['period']}; "
            f"{factor['points']:+d} points), which {description} "
            f"{_financial_source_citation(factor['period'])}."
        )
        citation = _financial_source_citation(factor["period"])
        fact = (
            f"{symbol} {factor['name'].lower()} was {factor['display']} "
            f"({factor['period']}) {citation}."
        )
        (positive_facts if factor["points"] > 0 else risk_facts if factor["points"] < 0 else positive_facts).append(fact)

    if evidence["valuation"]:
        rationale_parts.append(
            f"Valuation: trailing P/E {evidence['valuation']['display']} "
            f"({evidence['valuation']['period']}); it is not scored because no peer or historical benchmark "
            "is calculated."
        )
    else:
        rationale_parts.append(
            "Valuation is unassessed: no reliable trailing P/E was retrieved, and no peer or historical benchmark is calculated."
        )
    rationale_parts.append(
        "Valuation is disclosed but not scored because this workflow does not benchmark P/E against peers or history. "
        "The score is a transparent fundamental screen, not a forecast or personalized investment recommendation."
    )
    if any(
        isinstance(state.get("financial_metrics", {}).get(symbol, {}).get(metric), (int, float))
        and not state.get("financial_metrics", {}).get(symbol, {}).get(
            "statement_metrics", {}
        ).get(statement_key)
        for metric, statement_key in (
            ("revenue_growth", "quarterly_revenue_growth_yoy"),
            ("profit_margin", "quarterly_profit_margin"),
            ("free_cash_flow", "trailing_12_month_free_cash_flow"),
        )
    ):
        rationale_parts.append(
            "Undated Yahoo profile growth, margin, and cash-flow fields are excluded from scoring."
        )
    return stance, positive_facts[:6], risk_facts[:6], " ".join(rationale_parts)


def _analyst_summary_for_ticker(state: ResearchState, ticker: str) -> str:
    ticker_state = _ticker_state(state, ticker)
    stance, _positives, _risks, rationale = _research_stance(ticker_state, ticker)
    market = state.get("market_summary", {}).get(ticker, {})
    financials = state.get("financial_metrics", {}).get(ticker, {})
    company = (
        financials.get("company_name")
        or market.get("company_name")
        or ticker
    )
    sentences = [f"**Bottom line:** {company} screens as **{stance.lower()}**."]

    returns = market.get("returns", {})
    one_month = returns.get("1-month", {}).get("return_percent")
    one_year = returns.get("1-year", {}).get("return_percent")
    if one_month is not None and one_year is not None:
        if one_month < 0 < one_year:
            trend = "recent weakness sits against a positive one-year trend"
        elif one_month > 0 > one_year:
            trend = "the recent recovery has not yet erased a negative one-year trend"
        elif one_month < 0 and one_year < 0:
            trend = "both the recent and one-year price trends are negative"
        elif one_month > 0 and one_year > 0:
            trend = "both the recent and one-year price trends are positive"
        else:
            trend = "the recent and one-year price signals are mixed"
        sentences.append(
            f"The share price is {one_month:+.2f}% over one month and {one_year:+.2f}% over one year, so {trend} [YFINANCE]."
        )

    growth, growth_basis = _effective_financial_value(financials, "revenue_growth", allow_profile=False)
    margin, margin_basis = _effective_financial_value(financials, "profit_margin", allow_profile=False)
    cash_flow, cash_basis = _effective_financial_value(financials, "free_cash_flow", allow_profile=False)
    operating_points: list[str] = []
    statement_metrics = financials.get("statement_metrics", {})
    operating_growth_selection = _latest_fresh_dated_metric(financials, [
        ("quarterly_operating_income_growth_yoy", statement_metrics.get("quarterly_operating_income_growth_yoy", {})),
        ("annual_operating_income_growth_yoy", statement_metrics.get("annual_operating_income_growth_yoy", {})),
    ])
    operating_growth = operating_growth_selection[1] if operating_growth_selection else {}
    if isinstance(operating_growth.get("value"), (int, float)):
        source_citation = (
            "[SEC EDGAR]"
            if operating_growth.get("source") == "SEC EDGAR company facts"
            else "[YFINANCE]"
        )
        operating_points.append(
            "operating income, a measure of profit from core operations before interest and taxes, "
            f"changed {operating_growth['value']:+.2f}% year over year for the period ended "
            f"{operating_growth.get('period_end') or 'date unavailable'} "
            f"(comparison period ended {operating_growth.get('comparison_period_end') or 'date unavailable'}) "
            f"{source_citation}"
        )
    if growth is not None:
        growth_read = (
            "indicates sales expansion"
            if growth > 0
            else "indicates sales contraction"
            if growth < 0
            else "indicates flat sales"
        )
        operating_points.append(
            f"revenue growth is {_format_metric('revenue_growth', growth)} ({growth_basis}), which {growth_read} "
            f"{_financial_source_citation(growth_basis)}"
        )
    if margin is not None:
        operating_points.append(
            f"profit margin is {_format_metric('profit_margin', margin)} ({margin_basis}), the share of sales retained as profit"
            f" {_financial_source_citation(margin_basis)}"
        )
    if cash_flow is not None:
        if cash_flow < 0:
            cash_read = "a cash outflow after operating and investment activity, a key counterweight to growth"
        elif cash_flow > 0:
            cash_read = "cash left after operating and investment activity"
        else:
            cash_read = "roughly break-even cash generation after operating and investment activity"
        operating_points.append(
            f"free cash flow is {_format_metric('free_cash_flow', cash_flow)} ({cash_basis}), {cash_read} "
            f"{_financial_source_citation(cash_basis)}"
        )
    if operating_points:
        sentences.append("On the business side, " + "; ".join(operating_points) + ".")
    elif not (one_month is not None and one_year is not None):
        sentences.append(
            "Comparable dated operating figures or a complete price trend were not available, so the evidence does not establish a clear direction."
        )

    pe = financials.get("pe")
    if isinstance(pe, (int, float)) and pe > 0:
        sentences.append(
            f"The trailing P/E is {pe:.2f}x; without a peer or historical comparison, that multiple alone cannot show whether the shares are cheap or expensive [YFINANCE]."
        )
    sentences.append(rationale)
    sentences.append(
        "This is a screening view, not a personalized buy or sell recommendation."
    )
    return " ".join(sentences)


def _final_assessment_for_ticker(state: ResearchState, ticker: str) -> str:
    ticker_state = _ticker_state(state, ticker)
    stance, _positives, _risks, rationale = _research_stance(ticker_state, ticker)
    financials = state.get("financial_metrics", {}).get(ticker, {})
    checks: list[str] = []
    for key, label in (
        ("revenue_growth", "revenue growth"),
        ("profit_margin", "profit margin"),
        ("free_cash_flow", "free cash flow"),
    ):
        value, _basis = _effective_financial_value(
            financials, key, allow_profile=False
        )
        if value is None:
            checks.append(f"confirm the latest comparable {label}")
    if not isinstance(financials.get("pe"), (int, float)) or financials["pe"] <= 0:
        checks.append("review valuation using a reliable trailing P/E and peer/history context")
    elif not checks:
        checks.append("compare valuation and operating trends against peers and prior periods")
    next_check = "; ".join(checks)
    return (
        f"**Evidence-based screen: {stance}.** {rationale} "
        f"**What to check next:** {next_check}. "
        "This is a general research screen, not personalized investment advice."
    )


def _fallback_section(title: str, state: ResearchState) -> str:
    tickers = state.get("tickers", [])
    market = state.get("market_summary", {})
    financials = state.get("financial_metrics", {})
    news = state.get("news_evidence", [])
    positive = _news_factors(state, positive=True)
    negative = _news_factors(state, positive=False)

    if title == "EXECUTIVE SUMMARY":
        if tickers:
            return "\n\n".join(
                _analyst_summary_for_ticker(state, symbol)
                for symbol in tickers
            )
        return "The retrieved evidence is too limited to form an analyst view."

    if title == "COMPANY OVERVIEW":
        lines: list[str] = []
        for symbol in tickers:
            name = market.get(symbol, {}).get("company_name", symbol)
            metrics = financials.get(symbol, {})
            details = [metrics[key] for key in ("sector", "industry") if metrics.get(key)]
            summary = _compact_business_summary(metrics.get("business_summary"))
            description = f"; {' / '.join(str(item) for item in details)}" if details else ""
            lines.append(f"- {name}{description} [YFINANCE].")
            if summary:
                lines.append(f"  Business focus: {summary} [YFINANCE]")
        if not lines:
            return "Company profile information was not available from the retrieved market data."
        return "\n".join(lines)

    if title == "BULLISH FACTORS":
        if len(tickers) > 1:
            return "\n\n".join(
                f"**{symbol}**\n"
                + (_fallback_section(title, _ticker_state(state, symbol)))
                for symbol in tickers
            )
        factors: list[str] = []
        for symbol in tickers:
            metrics = financials.get(symbol, {})
            growth, growth_basis = _effective_financial_value(
                metrics, "revenue_growth", allow_profile=False
            )
            margin, margin_basis = _effective_financial_value(
                metrics, "profit_margin", allow_profile=False
            )
            cash_flow, cash_basis = _effective_financial_value(
                metrics, "free_cash_flow", allow_profile=False
            )
            if growth is not None and growth > 0:
                factors.append(
                    f"- {symbol} revenue grew {_format_metric('revenue_growth', growth)} "
                    f"({growth_basis}), supporting a sales-expansion case; growth alone does not establish "
                    f"that sales are profitable. {_financial_source_citation(growth_basis)}"
                )
            if margin is not None and margin > 0:
                factors.append(
                    f"- {symbol} retained {_format_metric('profit_margin', margin)} of revenue as profit "
                    f"({margin_basis}), evidence that sales are translating into earnings. "
                    f"{_financial_source_citation(margin_basis)}"
                )
            if cash_flow is not None and cash_flow > 0:
                factors.append(
                    f"- {symbol} generated {_format_metric('free_cash_flow', cash_flow)} in free cash flow "
                    f"({cash_basis}), leaving cash after operating and investment activity. "
                    f"{_financial_source_citation(cash_basis)}"
                )
        factors.extend(
            f"{item} This is a reported operating signal, not proof it will continue."
            for item in positive
        )
        return "\n".join(factors[:4]) or "The available evidence does not establish a clear fundamental tailwind."

    if title == "RISKS & BEARISH FACTORS":
        if len(tickers) > 1:
            return "\n\n".join(
                f"**{symbol}**\n"
                + (_fallback_section(title, _ticker_state(state, symbol)))
                for symbol in tickers
            )
        _stance, _financial_positives, financial_risks, _rationale = _research_stance(state)
        factors = [*negative, *financial_risks]
        for symbol in tickers:
            row = market.get(symbol)
            if row and row.get("daily_change_percent") is not None and row["daily_change_percent"] < 0:
                factors.append(
                    f"- {symbol} declined {row['daily_change_percent']:.2f}% in the latest session [YFINANCE]; "
                    "the price move alone does not establish its cause."
                )
        return "\n".join(factors[:4]) or "No specific bearish factors were established by the available evidence; missing information remains a risk."

    if title == "NEWS SENTIMENT" and len(tickers) > 1:
        return "\n\n".join(
            f"{symbol}: {_fallback_section(title, _ticker_state(state, symbol))}"
            for symbol in tickers
        )

    if title == "NEWS SENTIMENT":
        if positive and negative:
            tone = "Mixed: the selected excerpts include both positive and negative signals."
        elif positive:
            tone = "Positive in the selected excerpts, subject to limited coverage and publication-date uncertainty."
        elif negative:
            tone = "Negative in the selected excerpts, subject to limited coverage and publication-date uncertainty."
        else:
            tone = "Indeterminate: the available articles do not provide enough substantive evidence to assess sentiment."
        citations = sorted({
            f"[News {article['id'].removeprefix('news_')}]"
            for article in news
            if article.get("excerpt")
        })
        cite_text = " Evidence: " + ", ".join(citations) + "." if citations else ""
        return f"- {tone}{cite_text} This summarizes retrieved headlines and excerpts; it does not establish a cause for stock-price changes."

    if title == "FINAL ASSESSMENT":
        if len(tickers) > 1:
            assessments = []
            for symbol in tickers:
                one_ticker_state = _ticker_state(state, symbol)
                assessment = _final_assessment_for_ticker(one_ticker_state, symbol)
                assessments.append(f"**{symbol}.** {assessment}")
            return "\n\n".join(assessments)
        return _final_assessment_for_ticker(state, tickers[0]) if tickers else (
            "The available evidence is insufficient to provide a stock-specific screen."
        )

    if title == "DATA QUALITY & LIMITATIONS":
        limitations = []
        if not market:
            limitations.append("Market prices were unavailable.")
        if not financials or not any(financials.values()):
            limitations.append("No financial metrics were retrieved.")
        for metrics in financials.values():
            metadata = metrics.get("metadata", {})
            warning = metadata.get("official_filing_warning")
            if warning:
                limitations.append(warning)
            annual_warning = metadata.get("annual_statement_warning")
            if annual_warning:
                limitations.append(annual_warning)
            issuer_fcf_warning = metadata.get("issuer_fcf_warning")
            if issuer_fcf_warning:
                limitations.append(issuer_fcf_warning)
            limitations.extend(
                metadata.get("financial_source_comparisons", [])
            )
            fcf_definition_note = metadata.get("free_cash_flow_definition_note")
            if fcf_definition_note:
                limitations.append(fcf_definition_note)
            cash_profile_note = metadata.get("cash_profile_period_note")
            if cash_profile_note:
                limitations.append(cash_profile_note)
        if not news:
            limitations.append("No news evidence was retrieved.")
        elif any(not article.get("published") for article in news):
            limitations.append("One or more news publication dates are unavailable.")
        if state.get("warnings"):
            limitations.extend(state["warnings"][:4])
        if not limitations:
            limitations.append("Market returns are calculated from retrieved historical prices; company and news facts may be incomplete or delayed.")
        return "\n".join(f"- {item}" for item in dict.fromkeys(limitations))

    return "Insufficient supported evidence to complete this section."


def _company_overview_section(state: ResearchState) -> str:
    lines: list[str] = []
    for symbol in state.get("tickers", []):
        market_row = state.get("market_summary", {}).get(symbol, {})
        metrics = state.get("financial_metrics", {}).get(symbol, {})
        name = metrics.get("company_name") or market_row.get("company_name") or symbol
        sector = metrics.get("sector")
        industry = metrics.get("industry")
        classification = " / ".join(str(item) for item in (sector, industry) if item)
        details = f" ({classification})" if classification else ""
        lines.append(f"- {name}{details} [YFINANCE].")
        summary = _compact_business_summary(metrics.get("business_summary"))
        if summary:
            lines.append(f"  Business focus: {summary} [YFINANCE]")
        else:
            lines.append("  A detailed company description was not available in the retrieved company profile.")
        labels = (
            ("market_cap", "Market capitalization"),
            ("revenue_growth", "Revenue growth"),
            ("profit_margin", "Profit margin"),
            ("return_on_equity", "Return on equity"),
            ("pe", "Trailing P/E"),
            ("forward_pe", "Forward P/E"),
            ("total_cash", "Cash"),
            ("free_cash_flow", "Free cash flow"),
        )
        statement_metrics = metrics.get("statement_metrics", {})
        statement_equivalents = {
            "revenue_growth": "quarterly_revenue_growth_yoy",
            "profit_margin": "quarterly_profit_margin",
            "free_cash_flow": "trailing_12_month_free_cash_flow",
            "total_cash": "reported_total_cash_and_short_term_investments",
        }
        selected = []
        omitted_profile_values = False
        for key, label in labels:
            if metrics.get(key) is None:
                continue
            if key in {
                "revenue_growth", "profit_margin", "return_on_equity",
                "total_cash", "free_cash_flow",
            }:
                omitted_profile_values = True
                continue
            statement_key = statement_equivalents.get(key)
            statement_value = (
                metrics.get("quarterly_profit_margin")
                if statement_key == "quarterly_profit_margin"
                else statement_metrics.get(statement_key, {}).get("value")
                if statement_key
                else None
            )
            if key == "free_cash_flow":
                statement_value = any(
                    isinstance(statement_metrics.get(candidate), dict)
                    and isinstance(statement_metrics[candidate].get("value"), (int, float))
                    for candidate in (
                        "issuer_reported_trailing_12_month_free_cash_flow",
                        "trailing_12_month_free_cash_flow",
                        "quarterly_free_cash_flow",
                    )
                )
            if statement_value is not None:
                omitted_profile_values = True
                continue
            period = metrics.get("metadata", {}).get("profile_metric_periods", {}).get(
                key, "Yahoo Finance profile field; period unavailable"
            )
            selected.append(f"{label}: {_format_metric(key, metrics[key])} ({period})")
        if selected:
            lines.append(f"  Selected reported metrics: {'; '.join(selected)} [YFINANCE].")
        statements = metrics.get("statement_metrics", {})
        statement_fields = (
            ("quarterly_revenue", "Quarterly revenue", "amount"),
            ("quarterly_operating_income", "Quarterly operating income", "amount"),
            ("quarterly_operating_income_growth_yoy", "Quarterly operating-income growth year over year", "percent"),
            ("quarterly_net_income", "Quarterly net income", "amount"),
            ("quarterly_revenue_growth_yoy", "Quarterly revenue growth year over year", "percent"),
            ("annual_revenue", "Annual revenue", "amount"),
            ("annual_operating_income", "Annual operating income", "amount"),
            ("annual_operating_income_growth_yoy", "Annual operating-income growth year over year", "percent"),
            ("annual_net_income", "Annual net income", "amount"),
            ("annual_revenue_growth_yoy", "Annual revenue growth year over year", "percent"),
            ("quarterly_free_cash_flow", "Quarterly free cash flow", "amount"),
            ("trailing_12_month_free_cash_flow", "Calculated TTM free cash flow (operating cash flow less productive-asset purchases)", "amount"),
            ("issuer_reported_trailing_12_month_free_cash_flow", "Issuer-reported TTM free cash flow", "amount"),
            ("reported_cash_and_equivalents", "Reported cash and equivalents", "amount"),
            ("reported_short_term_investments", "Reported short-term investments", "amount"),
            ("reported_total_cash_and_short_term_investments", "Cash, cash equivalents and short-term investments", "amount"),
            ("reported_total_debt", "Reported total debt", "amount"),
            ("reported_stockholders_equity", "Reported stockholders' equity", "amount"),
            ("reported_total_liabilities", "Reported total liabilities (distinct from total debt)", "amount"),
            ("calculated_debt_to_equity", "Calculated debt-to-equity", "ratio"),
        )
        reported = []
        latest_quarter_end = metrics.get("metadata", {}).get("most_recent_quarter_end")
        has_quarterly_results = any(
            isinstance(statements.get(key), dict)
            and statements[key].get("value") is not None
            for key in ("quarterly_revenue", "quarterly_operating_income", "quarterly_net_income")
        )
        for key, label, value_type in statement_fields:
            statement = statements.get(key)
            if not isinstance(statement, dict) or statement.get("value") is None:
                continue
            if key.startswith("annual_") and has_quarterly_results:
                continue
            if _is_stale_metric(metrics, statement):
                omitted_profile_values = True
                continue
            if (
                key == "reported_total_liabilities"
                and latest_quarter_end
                and _metric_period_date(statement)
                and _metric_period_date({
                    "period_end": latest_quarter_end,
                }) > _metric_period_date(statement)
            ):
                continue
            value = statement["value"]
            if value_type == "percent":
                formatted = f"{value:+.2f}%"
            elif value_type == "ratio":
                formatted = (
                    f"{value:.2f}% (total debt {_format_metric('market_cap', statement.get('numerator'))} / "
                    f"stockholders' equity {_format_metric('market_cap', statement.get('denominator'))}; "
                    "total liabilities excluded)"
                )
            else:
                formatted = _format_metric("free_cash_flow", value)
            source_citation = _financial_source_citation(
                f"{statement.get('source') or ''} {statement.get('basis') or ''}"
            )
            period_description = (
                f"period {statement['period_start']} to {statement['period_end']}"
                if statement.get("period_start") and statement.get("period_end")
                else f"as of {statement.get('period_end') or 'date unavailable'}"
                if key.startswith("reported_") or key == "calculated_debt_to_equity"
                else f"period ended {statement.get('period_end') or 'date unavailable'}"
            )
            reported.append(
                f"{label}: {formatted} "
                f"({period_description}; "
                f"{statement.get('source') or statement.get('basis') or 'source unavailable'}"
                + (
                    f"; filed {statement.get('filed_date')} ({statement.get('filing_form')})"
                    if statement.get("filed_date")
                    else ""
                )
                + f") {source_citation}"
            )
        margin_value, margin_basis = _effective_financial_value(
            metrics, "profit_margin", allow_profile=False
        )
        if margin_value is not None:
            period_end = metrics.get("metadata", {}).get("quarterly_profit_margin_period_end") or "date unavailable"
            profit_statement = statements.get("quarterly_profit_margin", {})
            profit_source = (
                "SEC EDGAR company facts"
                if profit_statement.get("source") == "SEC EDGAR company facts"
                else "Yahoo Finance via yfinance"
            )
            profit_citation = "[SEC EDGAR]" if profit_source == "SEC EDGAR company facts" else "[YFINANCE]"
            reported.append(
                f"Profit margin: {_format_metric('profit_margin', margin_value)} "
                f"({margin_basis}; "
                f"period ended {period_end}; {profit_source}) {profit_citation}"
            )
        if reported:
            lines.append("  Dated statement figures: " + "; ".join(reported))
        if omitted_profile_values:
            lines.append(
                "  Dated statement figures above take precedence over Yahoo profile values with a different or unclear reporting date."
            )
    return "\n".join(lines) or "Company profile information was not available."


def _replace_section(report: str, title: str, replacement: str) -> str:
    heading = _heading_line_pattern(title)
    next_heading = re.compile(
        r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
        r"[A-Z][A-Z0-9 &/'-]*(?:[ \t]*\([^()\n]*\))?"
        r"[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*$"
    )
    matches = list(heading.finditer(report))
    section = f"{title}\n\n{replacement.strip()}\n\n"
    if matches:
        first = matches[0]
        end = next((match.start() for match in next_heading.finditer(report, first.end()) if match.start() > first.start()), len(report))
        normalized = report[:first.start()] + section + report[end:]

        duplicates = list(_heading_line_pattern(title).finditer(normalized))
        for duplicate in reversed(duplicates[1:]):
            duplicate_end = next(
                (match.start() for match in next_heading.finditer(normalized, duplicate.end()) if match.start() > duplicate.start()),
                len(normalized),
            )
            normalized = normalized[:duplicate.start()] + normalized[duplicate_end:]
        return normalized
    return section + report.lstrip()


def _extract_report_sections(report: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in report.splitlines():
        normalized = _normalize_heading(line)
        if normalized == "SOURCES":
            current = None
            continue
        if normalized in REQUIRED_SECTIONS:
            sections.setdefault(normalized, [])
            current = normalized
            continue
        if current:
            sections[current].append(line)
    return {
        title: "\n".join(lines).strip()
        for title, lines in sections.items()
        if "\n".join(lines).strip()
    }


def _section_implication(title: str, state: ResearchState) -> str:
    tickers = state.get("tickers", [])
    if len(tickers) > 1 and title in {
        "EXECUTIVE SUMMARY",
        "BULLISH FACTORS",
        "RISKS & BEARISH FACTORS",
        "NEWS SENTIMENT",
        "FINAL ASSESSMENT",
    }:
        return "\n".join(
            f"{ticker}: {_section_implication(title, _ticker_state(state, ticker))}"
            for ticker in tickers
        )
    markets = state.get("market_summary", {})
    financials = state.get("financial_metrics", {})
    articles = state.get("news_evidence", [])
    symbol = ", ".join(tickers) or "The company"

    if title == "EXECUTIVE SUMMARY":
        stance, _positive, _risks, rationale = _research_stance(state)
        observations = []
        for ticker in tickers:
            row = markets.get(ticker, {})
            daily = row.get("daily_change_percent")
            yearly = row.get("returns", {}).get("1-year", {}).get("return_percent")
            if daily is not None and yearly is not None:
                observations.append(
                    f"{ticker}'s latest session was {daily:+.2f}% while its one-year return was {yearly:+.2f}%"
                )
            elif daily is not None:
                observations.append(f"{ticker}'s latest session was {daily:+.2f}%")
        context = "; ".join(observations)
        prefix = f"{context}. " if context else ""
        return f"{prefix}For this stock, the evidence points to **{stance}**. {rationale}"

    if title == "MARKET SNAPSHOT":
        interpretations = []
        for ticker in tickers:
            row = markets.get(ticker, {})
            daily = row.get("daily_change_percent")
            volume = row.get("volume")
            average = row.get("average_volume_20d")
            if daily is None:
                continue
            direction = "rose" if daily > 0 else "fell" if daily < 0 else "was unchanged"
            note = f"{ticker} {direction} {abs(daily):.2f}% in the latest session"
            if volume is not None and average:
                ratio = volume / average
                activity = "above" if ratio > 1.05 else "below" if ratio < 0.95 else "near"
                note += f", with trading volume {activity} its 20-day average ({ratio:.2f}x)"
            interpretations.append(note)
        if interpretations:
            return "; ".join(interpretations) + ". This describes the latest trading session, not a confirmed longer-term trend or the reason for the move."
        return f"No current price movement is available to interpret for {symbol}."

    if title == "HISTORICAL PERFORMANCE":
        interpretations = []
        for ticker in tickers:
            row = markets.get(ticker, {})
            returns = row.get("returns", {})
            short = returns.get("1-month", {}).get("return_percent")
            medium = returns.get("3-month", {}).get("return_percent")
            long = returns.get("1-year", {}).get("return_percent")
            if short is not None and long is not None:
                if short < 0 < long:
                    trend = "recent weakness contrasts with a positive one-year return"
                elif short > 0 > long:
                    trend = "a recent recovery contrasts with a negative one-year return"
                elif short < 0 and long < 0:
                    trend = "both recent and one-year returns are negative"
                elif short > 0 and long > 0:
                    trend = "both recent and one-year returns are positive"
                else:
                    trend = "the recent and one-year returns are mixed"
                interpretations.append(f"{ticker}: {trend} ({short:+.2f}% over one month; {long:+.2f}% over one year)")
            elif medium is not None:
                interpretations.append(f"{ticker}: the available three-month return is {medium:+.2f}%")
        if interpretations:
            return "; ".join(interpretations) + ". These are historical price movements, not forecasts; return calculations use adjusted closes when available."
        return f"There is not enough historical pricing to interpret a trend for {symbol}."

    if title == "COMPANY OVERVIEW":
        interpretations = []
        for ticker in tickers:
            metrics = financials.get(ticker, {})
            growth, growth_basis = _effective_financial_value(
                metrics, "revenue_growth", allow_profile=False
            )
            margin, margin_basis = _effective_financial_value(
                metrics, "profit_margin", allow_profile=False
            )
            cash_flow, cash_flow_basis = _effective_financial_value(metrics, "free_cash_flow")
            points = []
            if isinstance(growth, (int, float)):
                points.append(
                    f"reported revenue growth of {_format_metric('revenue_growth', growth)} "
                    + ("supports a growing-sales picture" if growth > 0 else "shows sales contraction" if growth < 0 else "shows flat sales")
                    + f" ({growth_basis})"
                )
            if isinstance(margin, (int, float)):
                points.append(
                    f"reported profit margin of {_format_metric('profit_margin', margin)} "
                    f"indicates the share of revenue retained as profit ({margin_basis})"
                )
            if isinstance(cash_flow, (int, float)):
                if cash_flow > 0:
                    cash_text = f"positive free cash flow ({_format_metric('free_cash_flow', cash_flow)}) indicates cash remained after operating and investment spending ({cash_flow_basis})"
                elif cash_flow < 0:
                    cash_text = f"negative free cash flow ({_format_metric('free_cash_flow', cash_flow)}) means spending exceeded cash generated over the reported period ({cash_flow_basis})"
                else:
                    cash_text = "reported free cash flow was approximately zero"
                points.append(cash_text)
            if points:
                interpretations.append(f"{ticker}: " + "; ".join(points))
        if interpretations:
            return " ".join(interpretations) + ". The period labels above matter: quarterly statement figures are preferred, while profile figures may reflect different trailing or reporting periods. These measures do not alone establish future performance."
        return f"The retrieved company profile for {symbol} lacks enough financial metrics to infer its current operating strength."

    if title == "RECENT NEWS & EVENTS":
        substantive = [
            article for article in articles
            if set((article.get("categories") or "").split(", ")) != {"company_update"}
        ]
        if substantive:
            labels = []
            for article in substantive[:3]:
                number = article["id"].removeprefix("news_")
                categories = (article.get("categories") or "").replace("_", " ")
                labels.append(f"{categories} [News {number}]")
            return (
                f"For {symbol}, the retrieved coverage includes: " + "; ".join(labels)
                + ". These items identify topics to investigate; they do not prove financial impact or explain a share-price move."
            )
        return f"The retrieved results provide little specific, decision-relevant information about {symbol}; avoid drawing a news-based conclusion from these headlines alone."

    if title == "BULLISH FACTORS":
        stance, positives, _risks, _rationale = _research_stance(state)
        if positives:
            return (
                f"For {symbol}, the cited positive indicators may support the investment case, but they must be weighed "
                f"against the risks below. The overall screen remains: {stance}."
            )
        return f"The listed positive evidence is limited for {symbol}; it is not strong enough on its own to support a buy conclusion ({stance})."

    if title == "RISKS & BEARISH FACTORS":
        stance, _positives, risks, _rationale = _research_stance(state)
        if risks:
            return (
                f"For {symbol}, these indicators could weaken the investment case if they persist. "
                f"The current overall screen is: {stance}; verify the latest filings before acting."
            )
        return f"No severe negative financial indicator was established from the retrieved data for {symbol}; missing data still limit confidence ({stance})."

    if title == "PRICE & VOLUME ANALYSIS":
        interpretations = []
        for ticker in tickers:
            row = markets.get(ticker, {})
            daily = row.get("daily_change_percent")
            volume = row.get("volume")
            average = row.get("average_volume_20d")
            if daily is None or volume is None or not average:
                continue
            ratio = volume / average
            if daily < 0 and ratio > 1:
                meaning = "a down day with above-average trading activity"
            elif daily > 0 and ratio > 1:
                meaning = "an up day with above-average trading activity"
            elif daily < 0:
                meaning = "a down day with below-average trading activity"
            elif daily > 0:
                meaning = "an up day with below-average trading activity"
            else:
                meaning = "a nearly unchanged day"
            interpretations.append(f"{ticker} had {meaning} ({daily:+.2f}%, {ratio:.2f}x average volume)")
        if interpretations:
            return "; ".join(interpretations) + ". This describes trading activity only and does not identify its cause."
        return f"Price or volume comparisons are incomplete for {symbol}, so this section provides no reliable activity signal."

    if title == "NEWS SENTIMENT":
        positive_categories = {"revenue_increase", "cloud_growth", "earnings_beat"}
        risk_categories = {"revenue_decline_or_miss", "cash_flow_pressure", "capital_spending_increase", "earnings_miss"}
        positive_count = sum(
            bool(positive_categories & set((article.get("categories") or "").split(", ")))
            for article in articles
        )
        risk_count = sum(
            bool(risk_categories & set((article.get("categories") or "").split(", ")))
            for article in articles
        )
        if positive_count and risk_count:
            return f"The selected coverage of {symbol} includes both growth-related and risk-related topics; this is mixed evidence, not a reliable forecast."
        if positive_count:
            return f"The selected coverage highlights growth-related topics for {symbol}, but does not establish that growth will continue or raise the share price."
        if risk_count:
            return f"The selected coverage highlights financial or operating risks for {symbol}; confirm their scale and duration in company filings."
        return f"The retrieved headlines about {symbol} lack enough substantive financial detail to infer a meaningful news trend."

    if title == "FINAL ASSESSMENT":
        stance, _positives, _risks, rationale = _research_stance(state)
        return f"For {symbol}, the facts in this report support **{stance}**. {rationale}"

    if title == "DATA QUALITY & LIMITATIONS":
        missing = []
        for ticker in tickers:
            metrics = financials.get(ticker, {})
            metadata = metrics.get("metadata", {})
            statement_metrics = metrics.get("statement_metrics", {})
            absent = []
            quarterly_growth = statement_metrics.get("quarterly_revenue_growth_yoy", {})
            annual_growth = statement_metrics.get("annual_revenue_growth_yoy", {})
            if quarterly_growth.get("value") is None and annual_growth.get("value") is None:
                absent.append("comparable quarterly or annual revenue growth")
            elif quarterly_growth.get("value") is None:
                missing.append(f"{ticker} revenue growth uses annual statements because comparable quarterly data were unavailable")
            if metrics.get("quarterly_profit_margin") is None:
                if statement_metrics.get("annual_profit_margin", {}).get("value") is None:
                    absent.append("comparable quarterly or annual profit margin")
            elif metadata.get("quarterly_profit_margin_basis") == "annual income statement fallback":
                missing.append(f"{ticker} profit margin uses the annual income statement fallback")
            if (
                statement_metrics.get("trailing_12_month_free_cash_flow", {}).get("value") is None
                and statement_metrics.get("quarterly_free_cash_flow", {}).get("value") is None
            ):
                absent.append("free cash flow")
            if metrics.get("pe") is None:
                absent.append("trailing P/E")
            if absent:
                missing.append(f"{ticker} is missing {', '.join(absent)}")
            if metadata.get("retrieved_at_utc"):
                missing.append(
                    f"{ticker} yfinance data snapshot retrieved at {metadata['retrieved_at_utc']}"
                )
            market_row = markets.get(ticker, {})
            if market_row.get("price_basis"):
                missing.append(f"{ticker} price basis: {market_row['price_basis']}")
            latest_statement_dates = [
                value.get("period_end")
                for value in statement_metrics.values()
                if isinstance(value, dict) and value.get("period_end")
            ]
            if latest_statement_dates:
                missing.append(
                    f"{ticker} latest statement periods end on {', '.join(sorted(set(latest_statement_dates)))}"
                )
            missing.extend(
                f"{ticker}: {comparison}"
                for comparison in metadata.get("financial_source_comparisons", [])
            )
            filing_warning = metadata.get("official_filing_warning")
            if filing_warning:
                missing.append(f"{ticker}: {filing_warning}")
            screen = _screen_evidence(_ticker_state(state, ticker), ticker)
            undated = [
                factor["name"] for factor in screen["factors"]
                if not factor.get("dated")
            ]
            if undated:
                missing.append(
                    f"{ticker} scoring uses undated Yahoo profile values for {', '.join(undated)}"
                )
        if any(not article.get("published") for article in articles):
            missing.append("some news dates are unavailable, so recency cannot be checked")
        if missing:
            return "For this stock: " + "; ".join(missing) + ". Yahoo profile metrics may have different or unspecified reporting periods; do not compare them as though they were one dated statement. These gaps reduce confidence in the assessment."
        return f"The key retrieved fields for {symbol} are present, but provider data can still be delayed or incomplete; confirm important figures against current company filings."

    return f"The available information does not support a stock-specific interpretation for {symbol} in this section."


def _report_prompt(state: ResearchState) -> str:
    financial_metrics = json.loads(json.dumps(
        state.get("financial_metrics", {}), separators=(",", ":"), default=str
    ))
    for metrics in financial_metrics.values():
        if metrics.get("business_summary"):
            metrics["business_summary"] = _compact_business_summary(
                metrics["business_summary"]
            )
        metrics.pop("free_cash_flow", None)
        metrics.pop("debt_to_equity", None)
        statements = metrics.get("statement_metrics", {})
        if statements.get("quarterly_revenue_growth_yoy", {}).get("value") is not None:
            metrics.pop("revenue_growth", None)
        if metrics.get("quarterly_profit_margin") is not None:
            metrics.pop("profit_margin", None)
    evidence = {
        "tickers": state["tickers"],
        "requested_period": state["period"],
        "market": state.get("market_summary", {}),
        "financial_metrics": financial_metrics,
        "news": state.get("news_evidence", []),
        "known_data_warnings": state.get("warnings", []),
    }
    system = (
        "Write only these three analytical report sections, with exactly these headings: "
        + "; ".join(LLM_REPORT_SECTIONS)
        + ". Keep the complete response under 360 words. Write for an intelligent reader who is new to investing: "
        "use short, natural sentences and explain any necessary financial term in context. "
        "Make the executive summary a compact analyst thesis, not a list of figures: explain the strongest "
        "business signal, the most important counterpoint, and why that combination matters to an investor. "
        "For bullish and risk sections, use 2-4 concise markdown bullet points per section. Each bullet should connect "
        "each cited fact to its business implication and name an important caveat; do not merely restate numbers or headlines. "
        "Do not repeat the deterministic market snapshot, historical performance, price/volume analysis, "
        "recent news list, company description, or final screening; those are added from retrieved evidence. "
        "State what is supported, what remains uncertain, and which missing evidence limits the conclusion. "
        "Use evidence-topic categories to describe what the article reports, not to infer overall news sentiment. "
        "Use only supplied [News N] citations. "
        "Treat every ticker independently: label each ticker-specific claim with its ticker and cite news only when that "
        "source's related_tickers includes the same ticker. Never transfer evidence or conclusions between companies. "
        "Cite company financial and market facts with [SEC EDGAR] when the supplied metric source is SEC EDGAR, "
        "otherwise [YFINANCE]. Prefer dated statement metrics over profile metrics "
        "and keep their reporting periods explicit. Never invent financial data, dates, sources, technical "
        "indicators, analyst expectations, valuation comparisons, or causal claims. A market move coinciding "
        "with a news report does not prove that the report caused the move. Avoid canned phrases such as 'what this "
        "means for the stock'; deliver the interpretation directly. Do not present a buy/sell conclusion as certain."
    )
    payload = json.dumps(evidence, ensure_ascii=True, separators=(",", ":"), default=str)
    prompt = f"{system}\nResearch request: {state.get('user_question') or 'General stock research'}\nEvidence JSON:\n{payload}"
    return _compact_prompt(prompt, state)


def _compact_prompt(prompt: str, state: ResearchState) -> str:
    context_limit = _env_int("GROQ_CONTEXT_WINDOW", 8192)
    output_budget = _env_int("GROQ_MAX_TOKENS", 1800)
    budget = min(_env_int("GROQ_INPUT_TOKEN_BUDGET", 5000), context_limit - output_budget)
    if estimate_tokens(prompt) <= budget:
        return prompt

    payload_start = prompt.find("Evidence JSON:\n") + len("Evidence JSON:\n")
    prefix, raw_payload = prompt[:payload_start], prompt[payload_start:]
    evidence = json.loads(raw_payload)
    evidence["known_data_warnings"] = [
        str(warning)[:240] for warning in evidence.get("known_data_warnings", [])[:20]
    ]
    financial_metrics = evidence.get("financial_metrics", {})
    for metrics in financial_metrics.values():
        if metrics.get("business_summary"):
            metrics["business_summary"] = metrics["business_summary"][:400]
    news = evidence.get("news", [])
    for article in news:
        article["excerpt"] = article.get("excerpt", "")[:240]
    while estimate_tokens(prefix + json.dumps(evidence, ensure_ascii=True, separators=(",", ":"), default=str)) > budget and news:
        ticker_counts = {
            ticker: sum(ticker in article.get("related_tickers", []) for article in news)
            for ticker in evidence.get("tickers", [])
        }
        removable = next(
            (
                index for index in range(len(news) - 1, -1, -1)
                if all(
                    ticker_counts.get(ticker, 0) > 1
                    for ticker in news[index].get("related_tickers", [])
                )
            ),
            None,
        )
        news.pop(removable if removable is not None else -1)
    compact = prefix + json.dumps(evidence, ensure_ascii=True, separators=(",", ":"), default=str)
    if estimate_tokens(compact) > budget:
        raise PromptTooLargeError(
            f"Required market input exceeds configured token budget {budget}; request was not sent."
        )
    return compact


def _as_bullet_list(content: str) -> str:
    content = re.sub(r"\\([*_\-+#])", r"\1", content)
    bullet_lines = []
    for paragraph in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if all(re.match(r"^(?:[-*+]|\d+[.)])\s+", line) for line in lines):
            bullet_lines.extend(
                re.sub(r"^(?:[*+]|\d+[.)])\s+", "- ", line)
                for line in lines
            )
            continue
        text = " ".join(
            re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line)
            for line in lines
        )
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
        bullet_lines.extend(f"- {sentence.strip()}" for sentence in sentences if sentence.strip())
    return "\n".join(bullet_lines)


def _insert_deterministic_sections(report: str, state: ResearchState) -> str:
    extracted = _extract_report_sections(report)
    deterministic = {
        "EXECUTIVE SUMMARY": _fallback_section("EXECUTIVE SUMMARY", state),
        "MARKET SNAPSHOT": _market_section(state),
        "HISTORICAL PERFORMANCE": _history_section(state),
        "COMPANY OVERVIEW": _company_overview_section(state),
        "RECENT NEWS & EVENTS": _news_section(state),
        "PRICE & VOLUME ANALYSIS": _price_volume_section(state),
        "BULLISH FACTORS": _fallback_section("BULLISH FACTORS", state),
        "RISKS & BEARISH FACTORS": _fallback_section("RISKS & BEARISH FACTORS", state),
        "NEWS SENTIMENT": _fallback_section("NEWS SENTIMENT", state),
        "FINAL ASSESSMENT": _fallback_section("FINAL ASSESSMENT", state),
        "DATA QUALITY & LIMITATIONS": _fallback_section("DATA QUALITY & LIMITATIONS", state),
    }
    sections: list[str] = []
    for title in REQUIRED_SECTIONS:
        content = deterministic.get(title) or extracted.get(title) or _fallback_section(title, state)
        content = _normalize_news_citations(content, state)
        if title in {"BULLISH FACTORS", "RISKS & BEARISH FACTORS"}:
            content = _as_bullet_list(content)
        if title == "NEWS SENTIMENT" and state.get("news_evidence") and not re.search(
            r"\[News\s+\d+\]", content, flags=re.IGNORECASE
        ):
            content = _fallback_section(title, state)
        sections.append(f"{title}\n\n{content.strip()}")
    report = "\n\n".join(sections)
    sources = ["PRICE_DATA / yfinance (retrieved market prices and calculated returns)."]
    sources.extend(
        f"[News {article['id'].removeprefix('news_')}] {article['source']} - {article['title']} ({article['published'] or 'date unavailable'})\n{article['url']}"
        for article in state.get("news_evidence", [])
    )
    sec_sources: dict[str, dict[str, str]] = {}
    for metrics in state.get("financial_metrics", {}).values():
        for entry in metrics.get("statement_metrics", {}).values():
            if (
                isinstance(entry, dict)
                and entry.get("source") == "SEC EDGAR company facts"
                and entry.get("url")
            ):
                sec_sources[entry["url"]] = entry
    sources.extend(
        f"[SEC EDGAR] {entry.get('filing_form', 'Company filing')} filed "
        f"{entry.get('filed_date', 'date unavailable')}; period ended "
        f"{entry.get('period_end', 'date unavailable')}\n{url}"
        for url, entry in sec_sources.items()
    )
    source_section = (
        "\n\nSOURCES\n\n"
        + ("\n\n".join(sources) if sources else "No news sources retrieved.")
    )
    return report.rstrip() + source_section + "\n"


def generate_report(state: ResearchState) -> dict[str, Any]:
    try:
        prompt = _report_prompt(state)
        selected_news = json.loads(prompt.split("Evidence JSON:\n", 1)[1]).get("news", [])
        report_state = {**state, "news_evidence": selected_news}
        response, usage = _invoke_groq(prompt)
        report = _insert_deterministic_sections(str(response.content), report_state)
        return {
            "report": report,
            "news_evidence": selected_news,
            "llm_usage": _add_usage(state.get("llm_usage"), usage),
            "report_error": "",
        }
    except Exception as exc:
        return {
            "report": "",
            "report_error": str(exc),
            "execution_errors": [*state.get("execution_errors", []), f"Report generation failed: {exc}"],
            "llm_usage": _add_usage(state.get("llm_usage"), getattr(exc, "usage", {})),
        }


def _validate_report(report: str, state: ResearchState) -> list[str]:
    warnings: list[str] = []
    headings: set[str] = set()
    for line in report.splitlines():
        if len(line.strip()) > 100:
            continue
        candidate = _normalize_heading(line)
        if candidate in REQUIRED_SECTIONS:
            headings.add(candidate)
    for section in REQUIRED_SECTIONS:
        if section not in headings:
            warnings.append(f"Required report section is missing: {section}.")

    valid_citations = {
        f"[News {article['id'].removeprefix('news_')}]"
        for article in state.get("news_evidence", [])
    }
    cited = set(re.findall(r"\[News\s+\d+\]", report, flags=re.IGNORECASE))
    invalid = {citation for citation in cited if citation.lower() not in {valid.lower() for valid in valid_citations}}
    if invalid:
        warnings.append("Report contains unsupported news citations: " + ", ".join(sorted(invalid)) + ".")
    unsupported = set(re.findall(r"\[Unsupported News\s+\d+\]", report, flags=re.IGNORECASE))
    if unsupported:
        warnings.append(
            "Report contains citations that do not match retrieved evidence: "
            + ", ".join(sorted(unsupported))
            + "."
        )

    body = re.sub(r"(?is)\nSOURCES\s*\n.*$", "", report)
    recent_news_heading = _heading_line_pattern("RECENT NEWS & EVENTS")
    recent_news_match = recent_news_heading.search(body)
    if recent_news_match:
        following_heading = re.compile(
            r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
            r"[A-Z][A-Z0-9 &/'-]*[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*$"
        ).search(body, recent_news_match.end())
        body = body[:recent_news_match.start()] + body[following_heading.start():] if following_heading else body[:recent_news_match.start()]
    if state.get("news_evidence") and not re.search(r"\[News\s+\d+\]", body, flags=re.IGNORECASE):
        warnings.append("News-derived factual claims outside the recent-news evidence section need [News N] citations.")
    elif state.get("news_evidence"):
        uncited_news_facts = []
        news_fact_terms = re.compile(
            r"\b(revenue|net sales|earnings|guidance|outlook|AWS|cloud growth|"
            r"capital spending|capital expenditure|lawsuit|regulatory|acquisition|"
            r"product launch|layoffs|cash burn)\b",
            flags=re.IGNORECASE,
        )
        factual_claim_terms = re.compile(
            r"\b(increased|grew|declined|fell|rose|announced|expects?|"
            r"forecast|plans?|launched|settled|filed|faces|missed|beat|raised|cut)\b"
            r"|\b\d+(?:\.\d+)?%|\$\d",
            flags=re.IGNORECASE,
        )
        sections = _extract_report_sections(report)
        for title in LLM_REPORT_SECTIONS:
            section = sections.get(title, "")
            for line in section.splitlines():
                if (
                    news_fact_terms.search(line)
                    and factual_claim_terms.search(line)
                    and "Deterministic fundamental score:" not in line
                    and not re.search(
                            r"\[(?:News\s+\d+|YFINANCE|SEC\s+EDGAR)\]", line, flags=re.IGNORECASE
                    )
                ):
                    uncited_news_facts.append(line.strip())
        if uncited_news_facts:
            warnings.append(
                "Potential news-derived factual claims lack a [News N] or [YFINANCE] citation."
            )
    if len(state.get("tickers", [])) > 1:
        articles_by_id = {
            article["id"].removeprefix("news_"): article
            for article in state.get("news_evidence", [])
        }
        mismatched_citations = False
        unlabeled_citations = False
        for line in body.splitlines():
            citation_ids = re.findall(r"\[News\s+(\d+)\]", line, flags=re.IGNORECASE)
            if not citation_ids:
                continue
            mentioned = {
                ticker for ticker in state["tickers"]
                if re.search(rf"\b{re.escape(ticker)}\b", line, flags=re.IGNORECASE)
            }
            if not mentioned:
                unlabeled_citations = True
                continue
            for citation_id in citation_ids:
                article = articles_by_id.get(citation_id)
                if article and not mentioned.issubset(
                    article.get("related_tickers") or state["tickers"]
                ):
                    mismatched_citations = True
        if unlabeled_citations:
            warnings.append("News-cited claims in a multi-ticker report must identify the ticker they concern.")
        if mismatched_citations:
            warnings.append("A news citation is attributed to a ticker not associated with that evidence.")

    snapshot_heading = _heading_line_pattern("MARKET SNAPSHOT")
    snapshot_match = snapshot_heading.search(report)
    if snapshot_match:
        next_heading = re.compile(
            r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?[ \t]*"
            r"[A-Z][A-Z0-9 &/'-]*[ \t]*(?:\*\*|__)?[ \t]*:?[ \t]*$"
        ).search(report, snapshot_match.end())
        snapshot_text = report[snapshot_match.end():next_heading.start() if next_heading else len(report)]
    else:
        snapshot_text = ""
    for symbol, market in state.get("market_summary", {}).items():
        expected = f"${market['close']:.2f}"
        if expected not in snapshot_text or str(market["date"]) not in snapshot_text:
            warnings.append(f"Market snapshot does not match verified close/date for {symbol}.")
    return warnings


def _missing_report_sections(report: str) -> list[str]:
    present = {
        normalized
        for line in report.splitlines()
        if len(line.strip()) <= 140
        for normalized in (_normalize_heading(line),)
        if normalized in REQUIRED_SECTIONS
    }
    return [section for section in REQUIRED_SECTIONS if section not in present]


def validate_report(state: ResearchState) -> dict[str, Any]:
    if state.get("report_error"):
        return {
            "validation_status": "failed",
            "validation_warnings": [state["report_error"]],
        }
    warnings = _validate_report(state.get("report", ""), state)
    if not warnings:
        return {"validation_status": "passed", "validation_warnings": []}
    if state.get("correction_attempts", 0) == 0:
        return {"validation_status": "correction_needed", "validation_warnings": warnings}
    return {"validation_status": "failed", "validation_warnings": warnings}


def correct_report(state: ResearchState) -> dict[str, Any]:
    missing = _missing_report_sections(state.get("report", ""))
    if missing:
        evidence = {
            "market": state.get("market_summary", {}),
            "financial_metrics": state.get("financial_metrics", {}),
            "news": state.get("news_evidence", []),
            "warnings": state.get("warnings", []),
        }
        prompt = (
            "Write only the missing report sections listed below. Use each section title exactly as given, "
            "with at most two concise bullets. Use only this JSON evidence; if evidence is insufficient, say so. "
            "Cite news facts with the exact available [News N] IDs. Cite financial metrics as [YFINANCE]. "
            "Do not infer causation, technical indicators, analyst expectations, or missing figures.\n"
            f"Missing sections:\n" + "\n".join(f"- {section}" for section in missing)
            + "\nEvidence JSON:\n"
            + json.dumps(evidence, ensure_ascii=True, separators=(",", ":"), default=str)
        )
    else:
        issue_text = "\n".join(f"- {warning}" for warning in state.get("validation_warnings", []))
        prompt = (
            "Return a corrected version of this report, fixing only the listed validation issues. "
            "Preserve all section headings and verified values. Do not add unsupported facts or citations.\n"
            f"Issues:\n{issue_text}\n\nReport:\n{state.get('report', '')}"
        )
    try:
        response, usage = _invoke_groq(_compact_correction_prompt(prompt))
        if missing:
            updated = state.get("report", "").rstrip() + "\n\n" + str(response.content).strip() + "\n"
        else:
            updated = str(response.content)
        updated = _insert_deterministic_sections(updated, state)
        return {
            "report": updated,
            "correction_attempts": state.get("correction_attempts", 0) + 1,
            "llm_usage": _add_usage(state.get("llm_usage"), usage),
        }
    except Exception as exc:
        return {
            "correction_attempts": state.get("correction_attempts", 0) + 1,
            "execution_errors": [*state.get("execution_errors", []), f"Targeted report correction failed: {exc}"],
            "validation_status": "failed",
            "validation_warnings": [*state.get("validation_warnings", []), f"Correction failed: {exc}"],
            "llm_usage": _add_usage(state.get("llm_usage"), getattr(exc, "usage", {})),
        }


def _compact_correction_prompt(prompt: str) -> str:
    context_limit = _env_int("GROQ_CONTEXT_WINDOW", 8192)
    budget = min(_env_int("GROQ_INPUT_TOKEN_BUDGET", 5000), context_limit - _env_int("GROQ_MAX_TOKENS", 1800))
    if estimate_tokens(prompt) <= budget:
        return prompt
    report_start = prompt.find("Report:\n")
    if report_start < 0:
        raise PromptTooLargeError(f"Correction input exceeds configured token budget {budget}; request was not sent.")
    prefix = prompt[:report_start]
    report = prompt[report_start + len("Report:\n"):].split("\nSOURCES\n", 1)[0]
    sections = re.split(r"(?m)^(?=#{0,3}\s*[A-Z][A-Z &/]+\s*$)", report)
    compact_sections = [section[:500] for section in sections if section.strip()]
    compact = prefix + "Report:\n" + "\n\n".join(compact_sections)
    if estimate_tokens(compact) > budget:
        raise PromptTooLargeError(f"Correction input exceeds configured token budget {budget}; request was not sent.")
    return compact


def route_validated(state: ResearchState) -> str:
    return "retrieve" if state.get("input_status") == "valid" else "invalid"


def begin_retrieval(_state: ResearchState) -> dict[str, Any]:
    return {}


def start_execution(_state: ResearchState) -> dict[str, float]:
    return {"started_at": time.perf_counter()}


def finalize_execution(state: ResearchState) -> dict[str, float]:
    started_at = state.get("started_at")
    return {"execution_time_seconds": max(0.0, time.perf_counter() - started_at)} if started_at else {}


def route_validation_result(state: ResearchState) -> str:
    return "correct" if state.get("validation_status") == "correction_needed" and state.get("report") else "finish"


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("start_execution", start_execution)
    graph.add_node("validate_input", validate_input)
    graph.add_node("begin_retrieval", begin_retrieval)
    graph.add_node("retrieve_market", retrieve_market)
    graph.add_node("retrieve_news", retrieve_news)
    graph.add_node("prepare_report_input", prepare_report_input)
    graph.add_node("generate_report", generate_report)
    graph.add_node("validate_report", validate_report)
    graph.add_node("correct_report", correct_report)
    graph.add_node("finalize_execution", finalize_execution)

    graph.add_edge(START, "start_execution")
    graph.add_edge("start_execution", "validate_input")
    graph.add_conditional_edges("validate_input", route_validated, {
        "retrieve": "begin_retrieval",
        "invalid": "finalize_execution",
    })
    graph.add_conditional_edges(
        "begin_retrieval",
        lambda _state: ["retrieve_market", "retrieve_news"],
        {"retrieve_market": "retrieve_market", "retrieve_news": "retrieve_news"},
    )
    graph.add_edge(["retrieve_market", "retrieve_news"], "prepare_report_input")
    graph.add_edge("prepare_report_input", "generate_report")
    graph.add_edge("generate_report", "validate_report")
    graph.add_conditional_edges("validate_report", route_validation_result, {
        "correct": "correct_report",
        "finish": "finalize_execution",
    })
    graph.add_edge("correct_report", "validate_report")
    graph.add_edge("finalize_execution", END)
    return graph.compile()


__all__ = [
    "ResearchState",
    "build_graph",
    "estimate_tokens",
    "normalize_period",
    "prepare_news_evidence",
]
