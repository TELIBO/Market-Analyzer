from __future__ import annotations

import re
from typing import Any

import pandas as pd
import streamlit as st

import stock_research_workflow as workflow


PROGRESS_LABELS = {
    "validate_input": "Validating ticker and period",
    "retrieve_market": "Retrieving market prices and financial statements",
    "retrieve_news": "Searching news for this ticker",
    "prepare_report_input": "Preparing compact evidence",
    "generate_report": "Generating research report",
    "validate_report": "Validating report and citations",
    "correct_report": "Applying one targeted correction",
    "finalize_execution": "Finalizing report",
}


def search_companies(query: str) -> list[dict[str, str]]:
    company_name = query.strip()
    if not company_name:
        raise ValueError("Enter a company name to search.")
    response = workflow.yf.Search(
        company_name,
        max_results=10,
        news_count=0,
        lists_count=0,
        raise_errors=True,
    )
    candidates: list[dict[str, str]] = []
    seen_symbols: set[str] = set()
    for quote in getattr(response, "quotes", []) or []:
        symbol = str(quote.get("symbol") or "").strip().upper()
        name = str(quote.get("longname") or quote.get("shortname") or "").strip()
        if not symbol or not name or quote.get("quoteType") not in {"EQUITY", None}:
            continue
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        candidates.append({
            "symbol": symbol,
            "name": name,
            "exchange": str(quote.get("exchDisp") or quote.get("exchange") or "").strip(),
        })
    if not candidates:
        raise ValueError(
            f"No publicly traded company matched {company_name!r}. Check the spelling and try again."
        )
    return candidates


def run_ticker(
    company_name: str,
    ticker: str,
    period: str,
    status: Any,
) -> dict[str, Any]:
    graph = workflow.build_graph()
    result: dict[str, Any] = {}
    for event in graph.stream(
        {
            "user_question": f"Research {company_name} stock",
            "company_name": company_name,
            "requested_ticker": ticker,
            "period": period,
        },
        stream_mode="updates",
    ):
        for node, update in event.items():
            if update is not None:
                result.update(update)
            status.write(PROGRESS_LABELS.get(node, f"Completed {node}"))
    return result


def _format_money(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "Unavailable"
    return workflow._format_metric("market_cap", value)


def _money_with_currency(value: Any, currency: Any = None) -> str:
    if not isinstance(value, (int, float)):
        return "Unavailable"
    absolute = abs(value)
    sign = "-" if value < 0 else ""
    prefix = workflow._currency_prefix(currency)
    if absolute >= 1_000_000_000_000:
        return f"{sign}{prefix}{absolute / 1_000_000_000_000:.2f}T"
    if absolute >= 1_000_000_000:
        return f"{sign}{prefix}{absolute / 1_000_000_000:.2f}B"
    if absolute >= 1_000_000:
        return f"{sign}{prefix}{absolute / 1_000_000:.2f}M"
    return f"{sign}{prefix}{absolute:,.2f}"


def _escape_currency_markdown(text: str) -> str:
    return re.sub(r"(?<!\\)\$(?=[+-]?\d)", "&#36;", text)


def _metric_value(metrics: dict[str, Any], key: str, currency: Any = None) -> tuple[str, str]:
    value, basis = workflow._effective_financial_value(metrics, key, allow_profile=False)
    if value is None:
        value, basis = workflow._effective_financial_value(metrics, key)
    if value is None:
        return "Unavailable", "No comparable value was retrieved"
    if key in {"revenue_growth", "profit_margin"}:
        return f"{value * 100:.2f}%", basis
    if key == "free_cash_flow":
        return _money_with_currency(value, currency), basis
    return str(value), basis


def render_dashboard(ticker: str, result: dict[str, Any]) -> None:
    markets = result.get("market_summary", {})
    market = markets.get(ticker, {})
    financials = result.get("financial_metrics", {}).get(ticker, {})
    currency = (
        financials.get("metadata", {}).get("financial_currency")
        or market.get("currency")
    )
    sections = workflow._extract_report_sections(result.get("report", ""))
    warnings = result.get("validation_warnings", [])
    validation = result.get("validation_status", "not_run")

    st.subheader(f"{financials.get('company_name') or ticker} ({ticker})")
    if validation == "passed":
        st.success("Report checks passed")
    elif validation in {"correction_needed", "failed"}:
        st.warning(f"Report checks: {validation.replace('_', ' ')}")
    if result.get("report_error"):
        st.error(f"Report generation failed: {result['report_error']}")
    if result.get("execution_errors"):
        for error in result["execution_errors"]:
            st.error(error)
    for warning in warnings:
        st.warning(warning)

    summary, performance, financial_tab, news_tab, report_tab = st.tabs(
        ["Summary", "Performance", "Financials", "News & evidence", "Full report"]
    )
    with summary:
        close = market.get("close")
        daily = market.get("daily_change_percent")
        year_return = market.get("returns", {}).get("1-year", {}).get("return_percent")
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest close", _money_with_currency(close, currency), f"{daily:+.2f}%" if daily is not None else None)
        col2.metric("1-year return", f"{year_return:+.2f}%" if year_return is not None else "Unavailable")
        col3.metric("Market cap", _money_with_currency(financials.get("market_cap"), currency))
        st.caption(
            f"Quote date: {market.get('date', 'unavailable')} · "
            f"Price basis: {market.get('price_basis', 'unavailable')}"
        )
        st.markdown("#### The analyst's read")
        st.markdown(_escape_currency_markdown(
            sections.get("EXECUTIVE SUMMARY", "No analyst summary was produced.")
        ))
        screen = workflow._screen_evidence({"tickers": [ticker], **result}, ticker)
        st.metric(
            "Fundamental screen score",
            f"{screen['score']:+d} points",
            help=(
                "The score adds signed points for revenue growth (±1), operating-income growth (±1 for a modest "
                "change or ±2 at 15% growth/decline), profit margin (±1), and free cash flow (±2 for trailing "
                "12 months or ±1 for a quarter). Positive values add points; negative values subtract them. "
                "Missing values add zero. Undated Yahoo profile metrics are shown but reduce coverage confidence. "
                "P/E is not scored."
            ),
        )
        statements = financials.get("statement_metrics", {})
        cash_flow_basis = screen.get("cash_flow_score_basis")
        if not cash_flow_basis:
            cash_flow_basis = (
                "trailing-12-month free cash flow"
                if isinstance(statements.get("trailing_12_month_free_cash_flow", {}).get("value"), (int, float))
                else "quarterly free cash flow fallback"
                if isinstance(statements.get("quarterly_free_cash_flow", {}).get("value"), (int, float))
                else "default trailing-12-month free cash flow weighting"
            )
        available_max = screen.get("available_max_score", screen.get("max_score", 0))
        st.caption(
            f"Possible range for this scoring basis ({cash_flow_basis}): "
            f"{screen.get('minimum_score', -6):+d} to {screen.get('maximum_score', 6):+d} points. "
            f"Range from measures available this run: {-available_max:+d} to {available_max:+d}. "
            f"{screen.get('dated_factor_count', 0)} of {screen.get('available_factor_count', 0)} "
            "available measures have dated statement support; missing measures contribute 0 and valuation is excluded."
        )
        with st.expander("How the score was calculated"):
            factor_explanations = []
            for factor in screen["factor_breakdown"]:
                point_text = (
                    f"{factor.get('points', 0):+d} point{'s' if abs(factor.get('points', 0)) != 1 else ''}"
                    if factor.get("available")
                    else "0 points (missing comparable data)"
                )
                factor_explanations.append(
                    f"- **{factor['name']}** (potential impact ±{factor.get('weight', 0)}): "
                    f"{factor.get('evidence', 'Not available')}; "
                    f"{factor.get('period', 'No comparable dated statement value')}; {point_text}."
                )
            st.markdown(_escape_currency_markdown("\n".join(factor_explanations)))
            if screen["valuation"]:
                st.caption(
                    f"Valuation context only: P/E {screen['valuation']['display']} "
                    f"({screen['valuation']['period']}); it does not affect the score."
                )
        bullish, risks = st.columns(2)
        with bullish:
            st.markdown("##### What supports the case")
            st.markdown(_escape_currency_markdown(
                sections.get("BULLISH FACTORS", "No supported positives were identified.")
            ))
        with risks:
            st.markdown("##### What could challenge it")
            st.markdown(_escape_currency_markdown(
                sections.get("RISKS & BEARISH FACTORS", "No specific risks were established.")
            ))
        with st.expander("Screen result and what to verify"):
            st.markdown(_escape_currency_markdown(
                sections.get("FINAL ASSESSMENT", "No final assessment was produced.")
            ))
        with st.expander("Company overview"):
            st.markdown(_escape_currency_markdown(
                sections.get("COMPANY OVERVIEW", "Company profile was unavailable.")
            ))

    with performance:
        returns = market.get("returns", {})
        values = {
            label: entry.get("return_percent")
            for label, entry in returns.items()
            if isinstance(entry, dict) and isinstance(entry.get("return_percent"), (int, float))
        }
        if values:
            st.bar_chart(pd.DataFrame.from_dict(values, orient="index", columns=["Return (%)"]))
        else:
            st.info("Historical returns were unavailable.")
        global_indices = result.get("global_indices", {})
        if global_indices:
            st.markdown("#### Global index snapshot")
            index_rows = [
                {
                    "Index": row.get("name", symbol),
                    "Symbol": symbol,
                    "Close": _money_with_currency(row.get("close"), row.get("currency")),
                    "Daily change (%)": row.get("daily_change_percent"),
                    "Date": row.get("date", "unavailable"),
                }
                for symbol, row in global_indices.items()
            ]
            st.dataframe(pd.DataFrame(index_rows), use_container_width=True, hide_index=True)
            index_moves = {
                row.get("name", symbol): row.get("daily_change_percent")
                for symbol, row in global_indices.items()
                if isinstance(row.get("daily_change_percent"), (int, float))
            }
            if index_moves:
                st.bar_chart(pd.DataFrame.from_dict(index_moves, orient="index", columns=["Daily change (%)"]))
        st.markdown(_escape_currency_markdown(
            sections.get("HISTORICAL PERFORMANCE", "Historical performance was unavailable.")
        ))
        st.markdown(_escape_currency_markdown(
            sections.get("PRICE & VOLUME ANALYSIS", "Price and volume details were unavailable.")
        ))

    with financial_tab:
        st.caption("Dated quarterly statement values are preferred for screening. Profile fields may use a different or unavailable period.")
        metric_columns = st.columns(3)
        for column, key, label in zip(
            metric_columns,
            ("revenue_growth", "profit_margin", "free_cash_flow"),
            ("Revenue growth", "Profit margin", "Free cash flow"),
        ):
            value, basis = _metric_value(financials, key, currency=currency)
            display_label = (
                f"Quarterly {label.lower()}"
                if basis.startswith("quarterly")
                else f"Profile {label.lower()}"
                if "profile" in basis.lower()
                else label
            )
            column.metric(display_label, value)
            column.caption(basis)
        valuation_columns = st.columns(2)
        valuation_columns[0].metric(
            "Trailing P/E",
            f"{financials['pe']:.2f}x" if isinstance(financials.get("pe"), (int, float)) else "Unavailable",
        )
        valuation_columns[1].metric(
            "Forward P/E",
            f"{financials['forward_pe']:.2f}x" if isinstance(financials.get("forward_pe"), (int, float)) else "Unavailable",
        )
        statement_metrics = financials.get("statement_metrics", {})
        statement_rows = []
        for key, entry in statement_metrics.items():
            if not isinstance(entry, dict) or entry.get("value") is None:
                continue
            if entry.get("period_start") and entry.get("period_end"):
                period_text = f"{entry['period_start']} to {entry['period_end']}"
            else:
                period_text = entry.get("period_end", "unavailable")
            value = entry["value"]
            if key in {
                "quarterly_revenue_growth_yoy",
                "quarterly_operating_income_growth_yoy",
                "annual_revenue_growth_yoy",
                "annual_operating_income_growth_yoy",
            }:
                value = f"{value:+.2f}%"
            elif key in {
                "quarterly_revenue", "quarterly_operating_income", "quarterly_net_income",
                "quarterly_free_cash_flow", "trailing_12_month_free_cash_flow",
                "reported_cash_and_equivalents", "reported_total_debt",
                "reported_stockholders_equity", "reported_total_liabilities",
                "annual_revenue", "annual_operating_income", "annual_net_income",
            }:
                value = _money_with_currency(value, currency)
            elif key == "calculated_debt_to_equity":
                value = (
                    f"{value:.2f}% = {_money_with_currency(entry['numerator'], currency)} total debt / "
                    f"{_money_with_currency(entry['denominator'], currency)} stockholders' equity"
                )
            statement_rows.append({
                "Metric": key.replace("_", " ").title(),
                "Value": value,
                "Period": period_text,
                "Source / calculation": entry.get("source") or entry.get("basis", "unavailable"),
                "Filed": entry.get("filed_date", ""),
            })
        if statement_rows:
            st.dataframe(pd.DataFrame(statement_rows), use_container_width=True, hide_index=True)
        st.markdown(_escape_currency_markdown(
            sections.get("DATA QUALITY & LIMITATIONS", "No data-quality notes were reported.")
        ))

    with news_tab:
        articles = [
            article for article in result.get("news_evidence", [])
            if ticker in (article.get("related_tickers") or [ticker])
        ]
        if not articles:
            st.info("No ticker-specific news evidence was retrieved.")
        for article in articles:
            citation = f"[News {article['id'].removeprefix('news_')}]"
            st.markdown(_escape_currency_markdown(
                f"**{citation} {article.get('title', 'Untitled article')}**"
            ))
            st.caption(
                f"{article.get('source') or 'Source unavailable'} · "
                f"{article.get('published') or 'Publication date unavailable'} · "
                f"{article.get('categories') or 'Topic not categorized'}"
            )
            st.write(article.get("excerpt") or "No excerpt was available.")
            if article.get("url"):
                st.link_button("Open source", article["url"])
        st.markdown(_escape_currency_markdown(
            sections.get("NEWS SENTIMENT", "News sentiment could not be assessed.")
        ))

    with report_tab:
        st.markdown(_escape_currency_markdown(result.get("report") or "No report was generated."))
        usage = result.get("llm_usage", {})
        st.caption(
            "LLM calls: "
            f"{usage.get('calls', 0)} · estimated input tokens: "
            f"{usage.get('estimated_input_tokens', 0)} · provider input/output tokens: "
            f"{usage.get('actual_input_tokens', 0)}/{usage.get('actual_output_tokens', 0)}"
        )


def main() -> None:
    st.set_page_config(page_title="Stock Research Dashboard", page_icon="📊", layout="wide")
    st.title("Stock Research Dashboard")
    st.write("Enter a company name to explore an evidence-linked stock research report.")
    with st.sidebar:
        st.header("Research settings")
        company_input = st.text_input(
            "Company name",
            value="Amazon",
            placeholder="e.g. Amazon, NVIDIA, Microsoft",
        )
        period = st.selectbox(
            "Historical period",
            options=["1y", "2y", "5y", "10y", "max"],
            index=1,
        )
        find_company = st.button("Find company", use_container_width=True)

    if find_company:
        try:
            st.session_state["company_candidates"] = search_companies(company_input)
            st.session_state["company_search_query"] = company_input.strip()
            st.session_state.pop("stock_research_results", None)
            st.session_state.pop("stock_research_error", None)
        except Exception as exc:
            st.session_state["stock_research_error"] = (
                f"Company search failed ({type(exc).__name__}): {exc}"
            )
            st.session_state.pop("company_candidates", None)

    candidates = st.session_state.get("company_candidates", [])
    if candidates:
        options = {
            f"{item['name']}"
            + (f" · {item['exchange']}" if item["exchange"] else "")
            + f" · {item['symbol']}": item
            for item in candidates
        }
        selected_label = st.selectbox("Select the matching company", list(options))
        selected_company = options[selected_label]
        if st.button("Run research", type="primary", use_container_width=True):
            st.session_state.pop("stock_research_results", None)
            st.session_state.pop("stock_research_error", None)
            results: dict[str, dict[str, Any]] = {}
            research_exception: Exception | None = None
            validation_errors: list[str] = []
            with st.status(f"Researching {selected_company['name']}", expanded=True) as status:
                try:
                    result = run_ticker(
                        selected_company["name"],
                        selected_company["symbol"],
                        period,
                        status,
                    )
                    if result.get("input_status") != "valid":
                        validation_errors = result.get("execution_errors") or [
                            "The workflow could not validate the company request."
                        ]
                        for message in validation_errors:
                            status.write(message)
                        status.update(
                            label="Research could not start",
                            state="error",
                            expanded=True,
                        )
                    else:
                        results[selected_company["symbol"]] = result
                        status.update(label=f"Finished {selected_company['name']}", state="complete", expanded=False)
                except Exception as exc:
                    research_exception = exc
                    status.write(f"{type(exc).__name__}: {exc}")
                    status.update(
                        label=f"Research failed for {selected_company['name']}",
                        state="error",
                        expanded=True,
                    )
            if research_exception is not None:
                st.session_state["stock_research_error"] = (
                    f"Research failed ({type(research_exception).__name__}): {research_exception}"
                )
                with st.expander("Technical details", expanded=True):
                    st.exception(research_exception)
            elif validation_errors:
                st.session_state["stock_research_error"] = "; ".join(validation_errors)
                for message in validation_errors:
                    st.error(message)
            elif results:
                st.session_state.pop("stock_research_error", None)
            if results:
                st.session_state["stock_research_results"] = results

    results = st.session_state.get("stock_research_results", {})
    if st.session_state.get("stock_research_error"):
        st.error(st.session_state["stock_research_error"])
    if not results:
        if not st.session_state.get("stock_research_error"):
            st.info("Search for a company name, choose the correct listing, then run research.")
        return
    selected = st.selectbox("Ticker dashboard", options=list(results))
    render_dashboard(selected, results[selected])


if __name__ == "__main__":
    main()
