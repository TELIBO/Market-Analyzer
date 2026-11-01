"""Statement-based financial metrics and Yahoo Finance profile normalization."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

FINANCIAL_FIELDS = {
    "company_name": "longName",
    "sector": "sector",
    "industry": "industry",
    "business_summary": "longBusinessSummary",
    "market_cap": "marketCap",
    "pe": "trailingPE",
    "forward_pe": "forwardPE",
    "dividend_yield": "dividendYield",
    "average_volume": "averageVolume",
    "52_week_high": "fiftyTwoWeekHigh",
    "52_week_low": "fiftyTwoWeekLow",
    "revenue_growth": "revenueGrowth",
    "profit_margin": "profitMargins",
    "return_on_equity": "returnOnEquity",
    "debt_to_equity": "debtToEquity",
    "total_cash": "totalCash",
    "free_cash_flow": "freeCashflow",
    "enterprise_to_ebitda": "enterpriseToEbitda",
    "exchange": "exchange",
    "cik": "cik",
}

def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _select_financials(info: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for output, field in FINANCIAL_FIELDS.items():
        value = info.get(field)
        if output == "business_summary" and value:
            selected[output] = " ".join(str(value).split())
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            selected[output] = value
        elif isinstance(value, str) and value.strip():
            selected[output] = value.strip()[:160]
    selected["metadata"] = {
        "provider": "Yahoo Finance via yfinance",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "most_recent_quarter_end": _date_string(info.get("mostRecentQuarter")),
        "last_fiscal_year_end": _date_string(info.get("lastFiscalYearEnd")),
        "financial_currency": str(info.get("financialCurrency") or info.get("currency") or ""),
        "profile_metric_periods": {
            "revenue_growth": "Yahoo Finance reported growth field; exact comparison period not provided",
            "profit_margin": "Yahoo Finance reported margin field; exact statement period not provided",
            "free_cash_flow": "Yahoo Finance reported cash-flow field; exact statement period not provided",
            "total_cash": "Yahoo Finance reported balance-sheet field; exact statement date not provided",
            "market_cap": "Market value at retrieval; price and share count may update independently",
            "pe": "Yahoo Finance trailingPE; market quote can update independently",
            "forward_pe": "Yahoo Finance forwardPE; estimate-based, not a reported historical result",
        },
    }
    return selected


def _date_string(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc).date().isoformat()
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _statement_frame(
    ticker: Any,
    method_name: str,
    property_name: str,
    freq: str = "quarterly",
) -> Any:
    method = getattr(ticker, method_name, None)
    if callable(method):
        return method(freq=freq)
    return getattr(ticker, property_name, None)


def _statement_columns(frame: Any) -> list[Any]:
    if frame is None or getattr(frame, "empty", True):
        return []
    try:
        return sorted(list(frame.columns), key=lambda column: pd.Timestamp(column), reverse=True)
    except (TypeError, ValueError):
        return list(frame.columns)


def _statement_value(frame: Any, labels: tuple[str, ...], column: Any) -> float | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    normalized_labels = {
        re.sub(r"[^a-z0-9]", "", str(label).lower())
        for label in labels
    }
    for row in frame.index:
        normalized_row = re.sub(r"[^a-z0-9]", "", str(row).lower())
        if normalized_row not in normalized_labels:
            continue
        try:
            return _finite_number(frame.loc[row, column])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _statement_metrics(income: Any, cash_flow: Any, balance: Any) -> dict[str, Any]:
    income_columns = _statement_columns(income)
    cash_columns = _statement_columns(cash_flow)
    balance_columns = _statement_columns(balance)
    metrics: dict[str, Any] = {}
    if income_columns:
        latest_column = income_columns[0]
        period_end = _date_string(latest_column)
        revenue = _statement_value(income, ("Total Revenue", "Operating Revenue", "Revenue"), latest_column)
        net_income = _statement_value(income, ("Net Income", "Net Income Common Stockholders"), latest_column)
        operating_income = _statement_value(
            income, ("Operating Income", "Operating Income Loss"), latest_column
        )
        metrics["quarterly_revenue"] = {
            "value": revenue,
            "period_end": period_end,
            "source": "Yahoo Finance via yfinance",
            "basis": "latest available quarterly income statement",
        }
        metrics["quarterly_operating_income"] = {
            "value": operating_income,
            "period_end": period_end,
            "source": "Yahoo Finance via yfinance",
            "basis": "latest available quarterly income statement",
        }
        metrics["quarterly_net_income"] = {
            "value": net_income,
            "period_end": period_end,
            "source": "Yahoo Finance via yfinance",
            "basis": "latest available quarterly income statement",
        }
        if len(income_columns) >= 2:
            latest_timestamp = pd.to_datetime(latest_column, errors="coerce")
            if not pd.isna(latest_timestamp):
                target_year_ago = latest_timestamp - pd.DateOffset(years=1)
                comparable_columns = [
                    (column, pd.to_datetime(column, errors="coerce"))
                    for column in income_columns[1:]
                ]
                comparable_columns = [
                    (column, timestamp)
                    for column, timestamp in comparable_columns
                    if not pd.isna(timestamp)
                ]
                if comparable_columns:
                    prior_column, prior_timestamp = min(
                        comparable_columns,
                        key=lambda item: abs((item[1] - target_year_ago).days),
                    )
                else:
                    prior_column, prior_timestamp = None, None
                if prior_column is not None and abs((prior_timestamp - target_year_ago).days) <= 45:
                    prior_revenue = _statement_value(
                        income, ("Total Revenue", "Operating Revenue", "Revenue"), prior_column
                    )
                    growth = (
                        round((revenue / prior_revenue - 1) * 100, 2)
                        if revenue is not None and prior_revenue is not None and prior_revenue > 0
                        else None
                    )
                    metrics["quarterly_revenue_growth_yoy"] = {
                        "value": growth,
                        "period_end": period_end,
                        "comparison_period_end": _date_string(prior_column),
                        "source": "Yahoo Finance via yfinance",
                        "basis": "calculated from latest quarterly revenue versus the comparable quarter one year earlier",
                    }
                    prior_operating_income = _statement_value(
                        income,
                        ("Operating Income", "Operating Income Loss"),
                        prior_column,
                    )
                    operating_income_growth = (
                        round((operating_income / prior_operating_income - 1) * 100, 2)
                        if operating_income is not None
                        and prior_operating_income is not None
                        and prior_operating_income > 0
                        else None
                    )
                    metrics["quarterly_operating_income_growth_yoy"] = {
                        "value": operating_income_growth,
                        "period_end": period_end,
                        "comparison_period_end": _date_string(prior_column),
                        "source": "Yahoo Finance via yfinance",
                        "basis": "calculated from latest quarterly operating income versus the comparable quarter one year earlier",
                    }

    if cash_columns:
        latest_cash_column = cash_columns[0]
        operating_cash = _statement_value(
            cash_flow,
            ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"),
            latest_cash_column,
        )
        capex = _statement_value(
            cash_flow,
            ("Capital Expenditure", "Capital Expenditures", "Capital Expenditure Reported"),
            latest_cash_column,
        )
        if operating_cash is not None and capex is not None:
            free_cash_flow = operating_cash + capex if capex <= 0 else operating_cash - capex
            metrics["quarterly_free_cash_flow"] = {
                "value": round(free_cash_flow, 2),
                "period_end": _date_string(latest_cash_column),
                "basis": "calculated as operating cash flow minus capital expenditure for the latest available quarterly cash-flow statement",
                "operating_cash_flow": operating_cash,
                "capital_expenditure_reported": capex,
                "source": "Yahoo Finance via yfinance",
            }
        else:
            reported_free_cash_flow = _statement_value(
                cash_flow,
                ("Free Cash Flow", "Free Cash Flow From Continuing Operations"),
                latest_cash_column,
            )
            if operating_cash is not None:
                metrics["quarterly_operating_cash_flow"] = {
                    "value": operating_cash,
                    "period_end": _date_string(latest_cash_column),
                    "basis": "latest available quarterly cash-flow statement; capex was unavailable so free cash flow was not calculated",
                }
            if reported_free_cash_flow is not None:
                metrics["quarterly_free_cash_flow"] = {
                    "value": reported_free_cash_flow,
                    "period_end": _date_string(latest_cash_column),
                    "basis": "reported free cash flow from the latest available quarterly cash-flow statement",
                }
        quarter_cash_flows: list[tuple[date, float]] = []
        for column in cash_columns:
            period_end = pd.to_datetime(column, errors="coerce")
            if pd.isna(period_end):
                continue
            quarter_operating_cash = _statement_value(
                cash_flow,
                ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities", "Total Cash From Operating Activities"),
                column,
            )
            quarter_capex = _statement_value(
                cash_flow,
                ("Capital Expenditure", "Capital Expenditures", "Capital Expenditure Reported"),
                column,
            )
            if quarter_operating_cash is None or quarter_capex is None:
                continue
            quarter_fcf = (
                quarter_operating_cash + quarter_capex
                if quarter_capex <= 0
                else quarter_operating_cash - quarter_capex
            )
            quarter_start = (
                period_end - pd.DateOffset(months=3) + pd.Timedelta(days=1)
            ).date()
            quarter_cash_flows.append((period_end.date(), quarter_start, quarter_fcf))
        if len(quarter_cash_flows) >= 4:
            latest_end = quarter_cash_flows[0][0]
            trailing_four = [
                (period_end, quarter_start, value)
                for period_end, quarter_start, value in quarter_cash_flows
                if 0 <= (latest_end - period_end).days <= 400
            ][:4]
            quarter_gaps = [
                (trailing_four[index][0] - trailing_four[index + 1][0]).days
                for index in range(len(trailing_four) - 1)
            ]
            if len(trailing_four) == 4 and all(70 <= gap <= 110 for gap in quarter_gaps):
                metrics["trailing_12_month_free_cash_flow"] = {
                    "value": round(sum(value for _period_end, _quarter_start, value in trailing_four), 2),
                    "period_start": trailing_four[-1][1].isoformat(),
                    "period_end": latest_end.isoformat(),
                    "basis": "calculated from four consecutive quarterly operating cash flow less capital expenditure figures",
                    "quarter_count": 4,
                    "source": "Yahoo Finance via yfinance",
                }

    if balance_columns:
        latest_balance_column = balance_columns[0]
        for output, labels in (
            ("reported_cash_and_equivalents", ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents And Short Term Investments")),
            ("reported_total_debt", ("Total Debt", "Long Term Debt And Capital Lease Obligation", "Total Debt And Capital Lease Obligation")),
            ("reported_stockholders_equity", ("Stockholders Equity", "Total Stockholders Equity", "Common Stock Equity")),
            ("reported_total_liabilities", ("Total Liabilities Net Minority Interest", "Total Liabilities")),
        ):
            value = _statement_value(balance, labels, latest_balance_column)
            if value is not None:
                metrics[output] = {
                    "value": value,
                    "period_end": _date_string(latest_balance_column),
                    "basis": "latest available quarterly balance sheet",
                }
        debt = metrics.get("reported_total_debt", {}).get("value")
        equity = metrics.get("reported_stockholders_equity", {}).get("value")
        if debt is not None and equity is not None and equity > 0:
            metrics["calculated_debt_to_equity"] = {
                "value": round(debt / equity * 100, 2),
                "period_end": _date_string(latest_balance_column),
                "basis": "calculated as reported total debt divided by reported stockholders' equity, multiplied by 100",
                "numerator": debt,
                "denominator": equity,
                "unit": "percent",
            }
    for metric in metrics.values():
        if isinstance(metric, dict):
            metric.setdefault("source", "Yahoo Finance via yfinance")
    return metrics


def _annual_income_metrics(income: Any) -> dict[str, Any]:
    columns = _statement_columns(income)
    if not columns:
        return {}
    latest_column = columns[0]
    latest_end = _date_string(latest_column)
    revenue = _statement_value(
        income, ("Total Revenue", "Operating Revenue", "Revenue"), latest_column
    )
    operating_income = _statement_value(
        income, ("Operating Income", "Operating Income Loss"), latest_column
    )
    net_income = _statement_value(
        income, ("Net Income", "Net Income Common Stockholders"), latest_column
    )
    result: dict[str, Any] = {}
    if latest_end:
        result["annual_revenue"] = {
            "value": revenue,
            "period_end": latest_end,
            "source": "Yahoo Finance via yfinance",
            "basis": "latest available annual income statement",
        }
        result["annual_operating_income"] = {
            "value": operating_income,
            "period_end": latest_end,
            "source": "Yahoo Finance via yfinance",
            "basis": "latest available annual income statement",
        }
        result["annual_net_income"] = {
            "value": net_income,
            "period_end": latest_end,
            "source": "Yahoo Finance via yfinance",
            "basis": "latest available annual income statement",
        }
        if revenue is not None and revenue > 0 and net_income is not None:
            result["annual_profit_margin"] = {
                "value": net_income / revenue,
                "period_end": latest_end,
                "source": "Yahoo Finance via yfinance",
                "basis": "calculated from latest annual net income divided by annual revenue",
            }
    if len(columns) < 2:
        return result
    latest_timestamp = pd.to_datetime(latest_column, errors="coerce")
    if pd.isna(latest_timestamp):
        return result
    target_year_ago = latest_timestamp - pd.DateOffset(years=1)
    comparable = [
        (column, pd.to_datetime(column, errors="coerce"))
        for column in columns[1:]
    ]
    comparable = [
        (column, timestamp)
        for column, timestamp in comparable
        if not pd.isna(timestamp)
        and abs((timestamp - target_year_ago).days) <= 120
    ]
    if not comparable:
        return result
    prior_column, _prior_timestamp = min(
        comparable,
        key=lambda item: abs((item[1] - target_year_ago).days),
    )
    prior_end = _date_string(prior_column)
    prior_revenue = _statement_value(
        income, ("Total Revenue", "Operating Revenue", "Revenue"), prior_column
    )
    if revenue is not None and prior_revenue is not None and prior_revenue > 0:
        result["annual_revenue_growth_yoy"] = {
            "value": round((revenue / prior_revenue - 1) * 100, 2),
            "period_end": latest_end,
            "comparison_period_end": prior_end,
            "source": "Yahoo Finance via yfinance",
            "basis": "calculated from latest annual revenue versus prior fiscal year",
        }
    prior_operating_income = _statement_value(
        income, ("Operating Income", "Operating Income Loss"), prior_column
    )
    if (
        operating_income is not None
        and prior_operating_income is not None
        and prior_operating_income > 0
    ):
        result["annual_operating_income_growth_yoy"] = {
            "value": round((operating_income / prior_operating_income - 1) * 100, 2),
            "period_end": latest_end,
            "comparison_period_end": prior_end,
            "source": "Yahoo Finance via yfinance",
            "basis": "calculated from latest annual operating income versus prior fiscal year",
        }
    return result

__all__ = [
    "FINANCIAL_FIELDS",
    "_annual_income_metrics",
    "_date_string",
    "_finite_number",
    "_select_financials",
    "_statement_columns",
    "_statement_frame",
    "_statement_metrics",
    "_statement_value",
]
