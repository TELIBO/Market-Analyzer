"""SEC EDGAR retrieval and dated-source reconciliation for financial metrics."""

from __future__ import annotations

import gzip
import json
import math
import os
import re
import zlib
from datetime import date, timedelta
from functools import lru_cache
from typing import Any
from urllib.request import Request, urlopen

def _read_sec_json(response: Any) -> dict[str, Any]:
    body = response.read()
    headers = getattr(response, "headers", {})
    encoding = str(headers.get("Content-Encoding", "")).lower()
    if encoding == "gzip" or body.startswith(b"\x1f\x8b"):
        body = gzip.decompress(body)
    elif encoding == "deflate":
        body = zlib.decompress(body)
    return json.loads(body)


@lru_cache(maxsize=4)
def _sec_company_ciks(user_agent: str) -> dict[str, str]:
    request = Request(
        "https://www.sec.gov/files/company_tickers.json",
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
    )
    with urlopen(request, timeout=12) as response:
        rows = _read_sec_json(response)
    return {
        str(item["ticker"]).upper(): str(item["cik_str"]).zfill(10)
        for item in rows.values()
        if item.get("ticker") and item.get("cik_str") is not None
    }


@lru_cache(maxsize=32)
def _sec_company_facts(cik: str, user_agent: str) -> dict[str, Any]:
    request = Request(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
        headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"},
    )
    with urlopen(request, timeout=12) as response:
        return _read_sec_json(response)


def _sec_fact_rows(
    facts: dict[str, Any],
    tags: tuple[str, ...],
    *,
    duration_days: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    candidates: list[dict[str, Any]] = []
    for tag_priority, tag in enumerate(tags):
        units = us_gaap.get(tag, {}).get("units", {})
        rows = units.get("USD", [])
        if not rows:
            continue
        for row in rows:
            if row.get("form") not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
                continue
            if not row.get("end") or not row.get("filed"):
                continue
            if duration_days:
                if not row.get("start"):
                    continue
                span = (date.fromisoformat(row["end"]) - date.fromisoformat(row["start"])).days
                if not duration_days[0] <= span <= duration_days[1]:
                    continue
            elif row.get("start"):
                continue
            candidates.append({**row, "_tag_priority": tag_priority})
    return candidates


def _latest_sec_fact_by_end(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            item["end"],
            item["filed"],
            -int(item.get("_tag_priority", 0)),
        ),
    ):
        latest[row["end"]] = {
            key: value for key, value in row.items() if key != "_tag_priority"
        }
    return latest


def _sec_standalone_quarters(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_period: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        start, end = row.get("start"), row.get("end")
        if not start or not end:
            continue
        period_key = (str(start), str(end))
        existing = by_period.get(period_key)
        if existing is None or (
            row.get("filed", ""),
            -int(row.get("_tag_priority", 0)),
        ) > (
            existing.get("filed", ""),
            -int(existing.get("_tag_priority", 0)),
        ):
            by_period[period_key] = row

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in by_period.values():
        grouped.setdefault(str(row["start"]), []).append(row)

    quarters: dict[str, dict[str, Any]] = {}
    direct_quarters: dict[str, dict[str, Any]] = {}
    for fiscal_start, fiscal_rows in grouped.items():
        ordered = sorted(fiscal_rows, key=lambda row: row["end"])
        cumulative: list[dict[str, Any]] = []
        for row in ordered:
            span = (date.fromisoformat(row["end"]) - date.fromisoformat(fiscal_start)).days + 1
            if 60 <= span <= 120:
                direct_quarters[row["end"]] = {
                    **{key: value for key, value in row.items() if key != "_tag_priority"},
                    "start": row["start"],
                    "period_basis": "reported standalone quarterly filing fact",
                }
                cumulative.append(row)
                continue
            if span > 120:
                previous = next(
                    (
                        candidate for candidate in reversed(cumulative)
                        if 70 <= (
                            date.fromisoformat(row["end"])
                            - date.fromisoformat(candidate["end"])
                        ).days <= 110
                    ),
                    None,
                )
                if previous:
                    quarter_start = (
                        date.fromisoformat(previous["end"]) + timedelta(days=1)
                    ).isoformat()
                    quarters[row["end"]] = {
                        **{key: value for key, value in row.items() if key != "_tag_priority"},
                        "start": quarter_start,
                        "val": row["val"] - previous["val"],
                        "period_basis": "derived as current year-to-date filing value less the prior cumulative filing value",
                    }
                cumulative.append(row)
    return {**quarters, **direct_quarters}


def _sec_filing_details(cik: str, fact: dict[str, Any]) -> dict[str, str]:
    accession = str(fact.get("accn") or "")
    accession_path = accession.replace("-", "")
    return {
        "source": "SEC EDGAR company facts",
        "basis": f"SEC EDGAR company facts reported on {fact.get('filed') or 'filing date unavailable'}",
        "filing_form": str(fact.get("form") or "SEC filing"),
        "filed_date": str(fact.get("filed") or ""),
        "period_end": str(fact.get("end") or ""),
        "url": (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{accession_path}/{accession}-index.htm"
            if accession
            else f"https://www.sec.gov/edgar/browse/?CIK={int(cik)}&owner=exclude"
        ),
    }


def _sec_statement_metrics(facts: dict[str, Any], cik: str) -> dict[str, Any]:
    revenue_rows = _latest_sec_fact_by_end(_sec_fact_rows(
        facts,
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
        ),
        duration_days=(60, 120),
    ))
    operating_rows = _latest_sec_fact_by_end(_sec_fact_rows(
        facts, ("OperatingIncomeLoss",), duration_days=(60, 120)
    ))
    income_rows = _latest_sec_fact_by_end(_sec_fact_rows(
        facts, ("NetIncomeLoss", "ProfitLoss"), duration_days=(60, 120)
    ))
    operating_cash_periods = _sec_standalone_quarters(_sec_fact_rows(
        facts,
        (
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
        duration_days=(60, 400),
    ))
    capex_periods = _sec_standalone_quarters(_sec_fact_rows(
        facts,
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ),
        duration_days=(60, 400),
    ))
    balance_tags = {
        "reported_cash_and_equivalents": ("CashAndCashEquivalentsAtCarryingValue",),
        "reported_short_term_investments": ("MarketableSecuritiesCurrent",),
        "reported_stockholders_equity": ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        "reported_total_liabilities": ("Liabilities",),
        "reported_total_debt": (
            "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent",
            "LongTermDebtCurrentAndNoncurrent",
        ),
    }

    metrics: dict[str, Any] = {}

    def add_latest(name: str, rows_by_end: dict[str, dict[str, Any]]) -> None:
        if not rows_by_end:
            return
        fact = rows_by_end[max(rows_by_end)]
        metrics[name] = {
            "value": fact["val"],
            "period_start": fact.get("start"),
            **_sec_filing_details(cik, fact),
        }

    add_latest("quarterly_revenue", revenue_rows)
    add_latest("quarterly_operating_income", operating_rows)
    add_latest("quarterly_net_income", income_rows)
    if revenue_rows and income_rows:
        latest_end = max(revenue_rows)
        if latest_end in income_rows and revenue_rows[latest_end]["val"]:
            revenue_fact = revenue_rows[latest_end]
            income_fact = income_rows[latest_end]
            metrics["quarterly_profit_margin"] = {
                "value": income_fact["val"] / revenue_fact["val"],
                "basis": f"SEC EDGAR income statement for period ended {latest_end}",
                **_sec_filing_details(cik, income_fact),
            }
    balance_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for metric_name, tags in balance_tags.items():
        rows = _latest_sec_fact_by_end(_sec_fact_rows(facts, tags))
        balance_rows[metric_name] = rows
        if rows:
            fact = rows[max(rows)]
            metrics[metric_name] = {
                "value": fact["val"],
                "basis": "SEC EDGAR reported balance-sheet fact",
                **_sec_filing_details(cik, fact),
            }
    cash_rows = balance_rows.get("reported_cash_and_equivalents", {})
    investment_rows = balance_rows.get("reported_short_term_investments", {})
    shared_liquidity_dates = cash_rows.keys() & investment_rows.keys()
    if shared_liquidity_dates:
        liquidity_end = max(shared_liquidity_dates)
        cash_fact = cash_rows[liquidity_end]
        investment_fact = investment_rows[liquidity_end]
        metrics["reported_total_cash_and_short_term_investments"] = {
            "value": cash_fact["val"] + investment_fact["val"],
            "period_end": liquidity_end,
            "basis": (
                "calculated as cash and cash equivalents plus current marketable securities "
                "reported on the same balance-sheet date"
            ),
            "cash_component": cash_fact["val"],
            "short_term_investments_component": investment_fact["val"],
            **_sec_filing_details(cik, cash_fact),
        }
    if not balance_rows.get("reported_total_debt"):
        current_debt = _latest_sec_fact_by_end(
            _sec_fact_rows(facts, ("LongTermDebtCurrent",))
        )
        noncurrent_debt = _latest_sec_fact_by_end(
            _sec_fact_rows(facts, ("LongTermDebtNoncurrent",))
        )
        short_term_debt = _latest_sec_fact_by_end(
            _sec_fact_rows(facts, ("ShortTermBorrowings",))
        )
        combined_debt: dict[str, dict[str, Any]] = {}
        for period_end in current_debt.keys() & noncurrent_debt.keys():
            current, noncurrent = current_debt[period_end], noncurrent_debt[period_end]
            short_term = short_term_debt.get(period_end)
            debt_fact = {
                **noncurrent,
                "val": (
                    current["val"]
                    + noncurrent["val"]
                    + (short_term["val"] if short_term else 0)
                ),
                "basis": (
                    "SEC EDGAR total debt calculated from current and non-current long-term debt"
                    + (" and short-term borrowings" if short_term else "")
                ),
            }
            combined_debt[period_end] = debt_fact
        balance_rows["reported_total_debt"] = combined_debt
        if combined_debt:
            latest_end = max(combined_debt)
            fact = combined_debt[latest_end]
            metrics["reported_total_debt"] = {
                **_sec_filing_details(cik, fact),
                "value": fact["val"],
                "basis": fact["basis"],
            }

    if revenue_rows:
        latest_end = max(revenue_rows)
        comparable_end = next(
            (period_end for period_end in revenue_rows
             if abs((date.fromisoformat(latest_end) - date.fromisoformat(period_end)).days - 365) <= 45),
            None,
        )
        if comparable_end:
            current, prior = revenue_rows[latest_end], revenue_rows[comparable_end]
            if prior["val"]:
                metrics["quarterly_revenue_growth_yoy"] = {
                    "value": round((current["val"] / prior["val"] - 1) * 100, 2),
                    "comparison_period_end": comparable_end,
                    **_sec_filing_details(cik, current),
                }

    if operating_rows:
        latest_end = max(operating_rows)
        comparable_end = next(
            (period_end for period_end in operating_rows
             if abs((date.fromisoformat(latest_end) - date.fromisoformat(period_end)).days - 365) <= 45),
            None,
        )
        if comparable_end and operating_rows[comparable_end]["val"] > 0:
            current, prior = operating_rows[latest_end], operating_rows[comparable_end]
            metrics["quarterly_operating_income_growth_yoy"] = {
                "value": round((current["val"] / prior["val"] - 1) * 100, 2),
                "comparison_period_end": comparable_end,
                **_sec_filing_details(cik, current),
            }

    cashflow_by_end: dict[str, dict[str, Any]] = {}
    for period_end in operating_cash_periods.keys() & capex_periods.keys():
        operating = operating_cash_periods[period_end]
        capex = capex_periods[period_end]
        cashflow_by_end[period_end] = {
            **_sec_filing_details(cik, operating),
            "value": operating["val"] - abs(capex["val"]),
            "operating_cash_flow": operating["val"],
            "capital_expenditure": capex["val"],
            "period_start": operating.get("start"),
            "operating_cash_flow_period_basis": operating.get("period_basis"),
            "capital_expenditure_period_basis": capex.get("period_basis"),
            "capital_expenditure_filed_date": capex.get("filed"),
            "basis": (
                "calculated as operating cash flow less payments to acquire productive assets"
            ),
        }
    if cashflow_by_end:
        latest_end = max(cashflow_by_end)
        latest = cashflow_by_end[latest_end]
        metrics["quarterly_free_cash_flow"] = {
            **latest,
            "basis": (
                "SEC EDGAR: calculated as standalone quarterly operating cash flow less "
                "payments to acquire productive assets"
            ),
        }
        trailing_periods = [
            cashflow_by_end[period_end]
            for period_end in sorted(cashflow_by_end, reverse=True)
            if 0 <= (date.fromisoformat(latest_end) - date.fromisoformat(period_end)).days <= 400
        ][:4]
        quarter_gaps = [
            (
                date.fromisoformat(trailing_periods[index]["period_end"])
                - date.fromisoformat(trailing_periods[index + 1]["period_end"])
            ).days
            for index in range(len(trailing_periods) - 1)
        ]
        if len(trailing_periods) == 4 and all(70 <= gap <= 110 for gap in quarter_gaps):
            metrics["trailing_12_month_free_cash_flow"] = {
                "value": round(sum(item["value"] for item in trailing_periods), 2),
                "period_start": min(
                    item.get("period_start") or (
                        date.fromisoformat(item["period_end"]) - timedelta(days=90)
                    ).isoformat()
                    for item in trailing_periods
                ),
                "period_end": latest_end,
                "basis": (
                    "SEC EDGAR: calculated from four consecutive quarterly operating cash flow "
                    "less payments to acquire productive assets; year-to-date filing values were "
                    "converted to standalone quarters when needed. Issuer-defined free cash flow "
                    "may differ when it nets asset-sale proceeds or incentives."
                ),
                "quarter_count": 4,
                "source": "SEC EDGAR company facts",
                "url": latest["url"],
                "filed_date": latest["filed_date"],
                "filing_form": latest["filing_form"],
            }

    debt_rows = balance_rows.get("reported_total_debt", {})
    equity_rows = balance_rows.get("reported_stockholders_equity", {})
    common_balance_periods = debt_rows.keys() & equity_rows.keys()
    if common_balance_periods:
        common_end = max(common_balance_periods)
        debt_fact = debt_rows[common_end]
        equity_fact = equity_rows[common_end]
        if debt_fact["val"] is not None and equity_fact["val"] > 0:
            debt_details = _sec_filing_details(cik, debt_fact)
            debt_details["period_end"] = common_end
            metrics["calculated_debt_to_equity"] = {
                **debt_details,
                "value": round(debt_fact["val"] / equity_fact["val"] * 100, 2),
                "basis": (
                    "calculated from total debt and stockholders' equity reported for the same "
                    "balance-sheet date; total liabilities are not used"
                ),
                "numerator": debt_fact["val"],
                "denominator": equity_fact["val"],
                "unit": "percent",
                "equity_filed_date": str(equity_fact.get("filed") or ""),
            }

    return metrics


def _retrieve_sec_statement_metrics(
    ticker: str,
    known_cik: Any = None,
) -> tuple[dict[str, Any], str | None]:
    user_agent = os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent:
        return {}, (
            "Official SEC filings were not checked because SEC_USER_AGENT is not configured; "
            "configure an app/organization name and contact email to verify U.S. SEC filers. "
            "Non-U.S. companies require their local regulator or investor-relations source."
        )
    try:
        cik = str(int(known_cik)).zfill(10) if known_cik else _sec_company_ciks(user_agent)[ticker.upper()]
        facts = _sec_company_facts(cik, user_agent)
        return _sec_statement_metrics(facts, cik), None
    except (KeyError, ValueError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        return {}, f"SEC filing enrichment unavailable for {ticker}: {exc}"


def _metric_period_date(metric: dict[str, Any]) -> date | None:
    period_end = metric.get("period_end")
    try:
        return date.fromisoformat(str(period_end))
    except (TypeError, ValueError):
        return None


def _merge_statement_sources(
    yahoo_metrics: dict[str, Any],
    sec_metrics: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    merged = dict(yahoo_metrics)
    comparisons: list[str] = []
    for key, official in sec_metrics.items():
        yahoo = merged.get(key)
        if not isinstance(yahoo, dict):
            merged[key] = official
            continue
        if not isinstance(official, dict):
            continue

        yahoo_end = _metric_period_date(yahoo)
        sec_end = _metric_period_date(official)
        if yahoo_end and sec_end and yahoo_end != sec_end:
            selected = "SEC EDGAR" if sec_end > yahoo_end else "Yahoo Finance"
            comparisons.append(
                f"{key.replace('_', ' ')}: SEC EDGAR period ended {sec_end.isoformat()}, "
                f"Yahoo Finance period ended {yahoo_end.isoformat()}; using the newer "
                f"{selected} period"
            )
        elif (
            yahoo_end
            and sec_end
            and yahoo_end == sec_end
            and isinstance(yahoo.get("value"), (int, float))
            and isinstance(official.get("value"), (int, float))
            and not math.isclose(
                float(yahoo["value"]),
                float(official["value"]),
                rel_tol=0.005,
                abs_tol=0.01,
            )
        ):
            comparisons.append(
                f"{key.replace('_', ' ')} differs between Yahoo Finance and SEC EDGAR "
                f"for the period ended {sec_end.isoformat()}; using the SEC-reported value"
            )

        # Official values win when periods match; otherwise use whichever source is newer.
        if sec_end and (not yahoo_end or sec_end >= yahoo_end):
            merged[key] = official
    return merged, comparisons


def _latest_dated_metric(
    entries: list[tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]] | None:
    dated_entries = [
        (key, entry, _metric_period_date(entry), priority)
        for priority, (key, entry) in enumerate(entries)
        if isinstance(entry, dict)
        and isinstance(entry.get("value"), (int, float))
        and _metric_period_date(entry) is not None
    ]
    if not dated_entries:
        return None
    key, entry, _period, _priority = max(
        dated_entries,
        key=lambda item: (item[2], -item[3]),
    )
    return key, entry


def _is_stale_metric(metrics: dict[str, Any], entry: dict[str, Any]) -> bool:
    period_end = _metric_period_date(entry)
    latest_quarter = _metric_period_date({
        "period_end": metrics.get("metadata", {}).get("most_recent_quarter_end")
    })
    return bool(
        period_end
        and latest_quarter
        and latest_quarter - period_end > timedelta(days=180)
    )


def _latest_fresh_dated_metric(
    metrics: dict[str, Any],
    entries: list[tuple[str, dict[str, Any]]],
) -> tuple[str, dict[str, Any]] | None:
    return _latest_dated_metric([
        (key, entry)
        for key, entry in entries
        if isinstance(entry, dict) and not _is_stale_metric(metrics, entry)
    ])


def _financial_source_citation(basis: str) -> str:
    news_citation = re.search(r"\[(News\s+\d+)\]", basis, flags=re.IGNORECASE)
    if news_citation:
        return f"[{news_citation.group(1)}]"
    return "[SEC EDGAR]" if "SEC EDGAR" in basis else "[YFINANCE]"


def _effective_financial_value(
    metrics: dict[str, Any],
    key: str,
    allow_profile: bool = True,
) -> tuple[float | None, str]:
    statement_metrics = metrics.get("statement_metrics", {})
    if key == "free_cash_flow":
        issuer_reported = statement_metrics.get(
            "issuer_reported_trailing_12_month_free_cash_flow", {}
        )
        if (
            isinstance(issuer_reported.get("value"), (int, float))
            and not _is_stale_metric(metrics, issuer_reported)
        ):
            return (
                float(issuer_reported["value"]),
                f"issuer-reported trailing 12 months "
                f"({issuer_reported.get('period_start') or 'start unavailable'} to "
                f"{issuer_reported.get('period_end') or 'date unavailable'}); "
                f"{issuer_reported.get('basis')}; source: {issuer_reported.get('source')}",
            )
        trailing = statement_metrics.get("trailing_12_month_free_cash_flow", {})
        if (
            isinstance(trailing.get("value"), (int, float))
            and not _is_stale_metric(metrics, trailing)
        ):
            selected_key, entry = "trailing_12_month_free_cash_flow", trailing
        else:
            quarterly = statement_metrics.get("quarterly_free_cash_flow", {})
            selected_key, entry = (
                ("quarterly_free_cash_flow", quarterly)
                if isinstance(quarterly.get("value"), (int, float))
                and not _is_stale_metric(metrics, quarterly)
                else (None, {})
            )
        if selected_key:
            source = entry.get("source") or "reported cash-flow statements"
            period = (
                f"{entry.get('period_start') or 'start unavailable'} to {entry.get('period_end') or 'date unavailable'}"
                if selected_key == "trailing_12_month_free_cash_flow"
                else f"period ended {entry.get('period_end') or 'date unavailable'}"
            )
            description = (
                "trailing 12 months" if selected_key == "trailing_12_month_free_cash_flow"
                else "quarterly"
            )
            return (
                float(entry["value"]),
                f"{description} ({period}); {entry.get('basis') or 'reported cash-flow statements'}; "
                f"source: {source}",
            )
    statement_keys = {
        "revenue_growth": (
            "quarterly_revenue_growth_yoy",
            "annual_revenue_growth_yoy",
        ),
        "profit_margin": ("quarterly_profit_margin", "annual_profit_margin"),
        "free_cash_flow": ("quarterly_free_cash_flow",),
    }.get(key, ())
    if key == "profit_margin":
        metadata = metrics.get("metadata", {})
        profile_period_end = metadata.get("quarterly_profit_margin_period_end")
        dated_quarterly_margin = (
            {
                "value": metrics["quarterly_profit_margin"],
                "period_end": profile_period_end,
                "source": metadata.get(
                    "quarterly_profit_margin_source", "Yahoo Finance via yfinance"
                ),
                "basis": metadata.get(
                    "quarterly_profit_margin_basis", "quarterly income statement"
                ),
            }
            if isinstance(metrics.get("quarterly_profit_margin"), (int, float))
            and profile_period_end
            else {}
        )
        candidates = [
            ("quarterly_profit_margin", statement_metrics.get("quarterly_profit_margin", {})),
            ("dated_quarterly_profit_margin", dated_quarterly_margin),
            ("annual_profit_margin", statement_metrics.get("annual_profit_margin", {})),
        ]
        selected = _latest_fresh_dated_metric(metrics, candidates)
        if selected:
            _selected_key, entry = selected
            source = entry.get("source") or "Yahoo Finance via yfinance"
            basis = entry.get("basis") or "income statement"
            return (
                float(entry["value"]),
                f"{source}, {basis}, period ended {entry.get('period_end')}",
            )
        value = metrics.get("quarterly_profit_margin")
        if allow_profile and isinstance(value, (int, float)):
            period = metadata.get("quarterly_profit_margin_period_end") or "date unavailable"
            source = metadata.get("quarterly_profit_margin_source", "Yahoo Finance via yfinance")
            basis = metadata.get("quarterly_profit_margin_basis", "income statement")
            return float(value), f"{source}, {basis}, period ended {period}"
    if key == "revenue_growth":
        selected = _latest_fresh_dated_metric(metrics, [
            (statement_key, statement_metrics.get(statement_key, {}))
            for statement_key in statement_keys
        ])
        if selected:
            _selected_key, entry = selected
            source = entry.get("source") or "Yahoo Finance via yfinance"
            basis = entry.get("basis") or "year-over-year revenue growth"
            return (
                float(entry["value"]) / 100,
                f"{basis} (period ended {entry.get('period_end') or 'date unavailable'}; "
                f"source: {source})",
            )
    for statement_key in statement_keys:
        entry = (
            statement_metrics.get(statement_key)
            if statement_key != "quarterly_profit_margin"
            else statement_metrics.get(statement_key)
        )
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if isinstance(value, (int, float)) and not _is_stale_metric(metrics, entry):
            source = entry.get("source") or "Yahoo Finance via yfinance"
            basis = entry.get("basis") or statement_key.replace("_", " ")
            value = float(value) / 100 if key == "revenue_growth" else float(value)
            return (
                value,
                f"{basis} (period ended {entry.get('period_end') or 'date unavailable'}; source: {source})",
            )
    if not allow_profile:
        return None, "not available from comparable statement data"
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        metadata = metrics.get("metadata", {})
        basis = metadata.get("profile_metric_periods", {}).get(key, "Yahoo Finance profile value; exact period unavailable")
        return float(value), basis
    return None, "not available"

__all__ = [
    "_effective_financial_value",
    "_financial_source_citation",
    "_is_stale_metric",
    "_latest_dated_metric",
    "_latest_fresh_dated_metric",
    "_latest_sec_fact_by_end",
    "_merge_statement_sources",
    "_metric_period_date",
    "_read_sec_json",
    "_retrieve_sec_statement_metrics",
    "_sec_company_ciks",
    "_sec_company_facts",
    "_sec_fact_rows",
    "_sec_filing_details",
    "_sec_standalone_quarters",
    "_sec_statement_metrics",
]
