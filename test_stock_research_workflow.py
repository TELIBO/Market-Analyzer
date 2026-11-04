from __future__ import annotations

import gzip
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

import stock_research_workflow as workflow
import streamlit_app as dashboard


def test_sec_json_reader_decompresses_gzip_response() -> None:
    payload = b'{"facts": {"ok": true}}'
    response = SimpleNamespace(
        read=lambda: gzip.compress(payload),
        headers={"Content-Encoding": "gzip"},
    )

    assert workflow._read_sec_json(response) == {"facts": {"ok": True}}


def test_normalizes_ticker_and_period() -> None:
    assert workflow._normalize_tickers([" amzn "]) == ["AMZN"]
    assert workflow.normalize_period("1 year") == "1y"
    assert workflow._extract_tickers("Research $amzn for 6 months") == ["AMZN"]
    assert workflow._extract_period("Research AMZN for 6 months") == "6mo"
    with pytest.raises(ValueError):
        workflow.normalize_period("13 months")


def test_calculates_market_summary_from_history(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    dates = pd.date_range(datetime.now().date() - timedelta(days=40), periods=41, freq="D")
    history = pd.DataFrame(
        {
            "Close": [100.0 + index for index in range(len(dates))],
            "Open": [99.0 + index for index in range(len(dates))],
            "High": [101.0 + index for index in range(len(dates))],
            "Low": [98.0 + index for index in range(len(dates))],
            "Volume": [1000 + index for index in range(len(dates))],
        },
        index=dates,
    )

    class FakeTicker:
        info = {"longName": "Amazon", "marketCap": 10_000}

        def history(self, **_kwargs):
            return history

    monkeypatch.setattr(workflow.yf, "Ticker", lambda _symbol: FakeTicker())
    market, financials, errors = workflow._calculate_market_data("AMZN", "2y")

    assert not errors
    assert market["close"] == 140.0
    assert market["daily_change_percent"] == pytest.approx((140 / 139 - 1) * 100, abs=0.01)
    assert market["returns"]["1-day"]["return_percent"] == market["daily_change_percent"]
    assert market["returns"]["1-month"]["return_percent"] is not None
    assert financials["company_name"] == "Amazon"


def test_market_cap_falls_back_to_yahoo_fast_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    history = pd.DataFrame(
        {
            "Close": [275.0, 278.22],
            "Adj Close": [275.0, 278.22],
            "Open": [274.0, 276.0],
            "High": [279.0, 280.0],
            "Low": [273.0, 275.0],
            "Volume": [1_000, 1_100],
        },
        index=pd.to_datetime(["2026-10-29", "2026-10-30"]),
    )

    class FakeTicker:
        info = {}
        fast_info = {"market_cap": 3_400_000_000_000}

        def history(self, **_kwargs):
            return history

    monkeypatch.setattr(workflow.yf, "Ticker", lambda _symbol: FakeTicker())
    _market, financials, errors = workflow._calculate_market_data("GOOG", "2y")

    assert not errors
    assert financials["market_cap"] == 3_400_000_000_000
    assert "fast_info" in financials["metadata"]["profile_metric_periods"]["market_cap"]


def test_retrieve_market_includes_global_indices_and_currency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    dates = pd.to_datetime(["2026-10-29", "2026-10-30"])

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            if symbol == "RELIANCE.NS":
                self.info = {
                    "longName": "Reliance Industries",
                    "marketCap": 20_000_000_000_000,
                    "currency": "INR",
                    "exchange": "NSI",
                }
            else:
                self.info = {"currency": "INR" if symbol in {"^NSEI", "^BSESN"} else "USD"}

        def history(self, **_kwargs):
            if self.symbol == "RELIANCE.NS":
                return pd.DataFrame(
                    {
                        "Close": [100.0, 101.0],
                        "Open": [99.0, 100.0],
                        "High": [102.0, 103.0],
                        "Low": [98.0, 99.0],
                        "Volume": [1_000, 1_100],
                    },
                    index=dates,
                )
            return pd.DataFrame({"Close": [10_000.0, 10_100.0]}, index=dates)

    monkeypatch.setattr(workflow.yf, "Ticker", lambda symbol: FakeTicker(symbol))
    result = workflow.retrieve_market({"tickers": ["RELIANCE.NS"], "period": "1y"})

    assert result["market_summary"]["RELIANCE.NS"]["currency"] == "INR"
    assert {"^NSEI", "^BSESN"} <= set(result["global_indices"])
    assert result["global_indices"]["^NSEI"]["name"] == "NIFTY 50 (NSE)"


def test_calculates_comparable_quarterly_financial_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    latest = pd.Timestamp("2026-09-30")
    prior = pd.Timestamp("2025-09-30")
    history = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "Adj Close": [99.0, 100.0],
            "Open": [99.0, 100.0],
            "High": [102.0, 102.0],
            "Low": [98.0, 99.0],
            "Volume": [1_000, 1_100],
        },
        index=pd.to_datetime(["2026-09-29", "2026-09-30"]),
    )
    income = pd.DataFrame(
        {
            latest: [120.0, 12.0],
            prior: [100.0, 8.0],
        },
        index=["Total Revenue", "Net Income"],
    )
    annual_income = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [500.0, 50.0, 40.0],
            pd.Timestamp("2024-12-31"): [400.0, 40.0, 20.0],
        },
        index=["Total Revenue", "Net Income", "Operating Income"],
    )
    cashflow = pd.DataFrame(
        {latest: [30.0, -10.0]},
        index=["Operating Cash Flow", "Capital Expenditure"],
    )
    balance = pd.DataFrame(
        {latest: [50.0, 20.0, 80.0, 140.0]},
        index=[
            "Cash And Cash Equivalents",
            "Total Debt",
            "Stockholders Equity",
            "Total Liabilities Net Minority Interest",
        ],
    )

    class FakeTicker:
        info = {
            "longName": "Example Corp",
            "revenueGrowth": 0.9,
            "profitMargins": 0.5,
            "freeCashflow": 999.0,
            "trailingPE": 25.0,
        }

        def get_income_stmt(self, freq="yearly"):
            return income if freq == "quarterly" else annual_income

        def get_cash_flow(self, freq="yearly"):
            assert freq == "quarterly"
            return cashflow

        def get_balance_sheet(self, freq="yearly"):
            assert freq == "quarterly"
            return balance

        def history(self, **kwargs):
            assert kwargs["auto_adjust"] is False
            return history

    monkeypatch.setattr(workflow.yf, "Ticker", lambda _symbol: FakeTicker())
    market, financials, errors = workflow._calculate_market_data("EXM", "2y")

    assert not errors
    assert market["close"] == 101.0
    assert market["returns"]["1-day"]["return_percent"] == pytest.approx(1.01, abs=0.01)
    assert market["price_basis"].startswith("Unadjusted Close")
    assert financials["revenue_growth"] == pytest.approx(0.2)
    assert financials["quarterly_profit_margin"] == pytest.approx(0.1)
    assert financials["statement_metrics"]["quarterly_free_cash_flow"]["value"] == 20.0
    assert financials["statement_metrics"]["quarterly_revenue_growth_yoy"]["comparison_period_end"] == "2025-09-30"
    assert financials["statement_metrics"]["annual_revenue_growth_yoy"]["value"] == pytest.approx(25.0)
    assert financials["statement_metrics"]["annual_operating_income_growth_yoy"]["value"] == pytest.approx(100.0)
    assert financials["statement_metrics"]["reported_cash_and_equivalents"]["period_end"] == "2026-09-30"
    debt_to_equity = financials["statement_metrics"]["calculated_debt_to_equity"]
    assert debt_to_equity["value"] == pytest.approx(25.0)
    assert debt_to_equity["numerator"] == 20.0
    assert debt_to_equity["denominator"] == 80.0
    assert debt_to_equity["unit"] == "percent"
    assert financials["statement_metrics"]["reported_total_liabilities"]["value"] == 140.0
    assert "retrieved_at_utc" in financials["metadata"]
    assert "period ended 2026-09-30" in workflow._effective_financial_value(financials, "free_cash_flow")[1]


def test_annual_income_metrics_provide_fallback_growth_margin_and_screen_score() -> None:
    annual = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [500.0, 50.0, 40.0],
            pd.Timestamp("2024-12-31"): [400.0, 40.0, 20.0],
        },
        index=["Total Revenue", "Net Income", "Operating Income"],
    )
    annual_metrics = workflow._annual_income_metrics(annual)
    financial_metrics = {"statement_metrics": annual_metrics}
    screen = workflow._screen_evidence({
        "tickers": ["EXM"],
        "market_summary": {"EXM": {"close": 100.0}},
        "financial_metrics": {"EXM": financial_metrics},
    }, "EXM")

    assert annual_metrics["annual_revenue_growth_yoy"]["value"] == 25.0
    assert annual_metrics["annual_operating_income_growth_yoy"]["value"] == 100.0
    assert annual_metrics["annual_profit_margin"]["value"] == pytest.approx(0.1)
    assert screen["score"] == 4
    assert screen["dated_factor_count"] == 3


def test_annual_operating_income_growth_omitted_when_comparison_period_is_a_loss() -> None:
    annual = pd.DataFrame(
        {
            pd.Timestamp("2025-12-31"): [500.0, 50.0, 40.0],
            pd.Timestamp("2024-12-31"): [400.0, 20.0, -10.0],
        },
        index=["Total Revenue", "Net Income", "Operating Income"],
    )

    metrics = workflow._annual_income_metrics(annual)

    assert "annual_operating_income_growth_yoy" not in metrics


def test_official_statement_merge_uses_newest_period_and_official_value_on_ties() -> None:
    yahoo = {
        "trailing_12_month_free_cash_flow": {
            "value": -2.0,
            "period_end": "2026-03-31",
            "source": "Yahoo Finance via yfinance",
        },
        "quarterly_free_cash_flow": {
            "value": -10.0,
            "period_end": "2026-06-30",
            "source": "Yahoo Finance via yfinance",
        },
    }
    sec = {
        "trailing_12_month_free_cash_flow": {
            "value": -42.4,
            "period_end": "2026-03-31",
            "source": "SEC EDGAR company facts",
        },
        "quarterly_free_cash_flow": {
            "value": -12.0,
            "period_end": "2026-03-31",
            "source": "SEC EDGAR company facts",
        },
    }

    merged, comparisons = workflow._merge_statement_sources(yahoo, sec)

    assert merged["trailing_12_month_free_cash_flow"]["value"] == -42.4
    assert merged["trailing_12_month_free_cash_flow"]["source"] == "SEC EDGAR company facts"
    assert merged["quarterly_free_cash_flow"]["value"] == -10.0
    assert any("using the newer Yahoo Finance period" in note for note in comparisons)


def test_financial_screen_prefers_ttm_over_a_single_newer_quarter() -> None:
    financials = {
        "statement_metrics": {
            "quarterly_revenue_growth_yoy": {
                "value": 4.0,
                "period_end": "2026-03-31",
            },
            "annual_revenue_growth_yoy": {
                "value": 18.0,
                "period_end": "2025-12-31",
            },
            "trailing_12_month_free_cash_flow": {
                "value": -2_000_000_000,
                "period_end": "2026-03-31",
            },
            "quarterly_free_cash_flow": {
                "value": -18_000_000_000,
                "period_end": "2026-06-30",
            },
        },
    }

    growth, _growth_basis = workflow._effective_financial_value(financials, "revenue_growth")
    free_cash_flow, free_cash_flow_basis = workflow._effective_financial_value(
        financials, "free_cash_flow", allow_profile=False
    )

    assert growth == pytest.approx(0.04)
    assert free_cash_flow == -2_000_000_000
    assert "trailing 12 months" in free_cash_flow_basis


def test_fcf_prefers_recalculated_ttm_over_newer_single_quarter_for_scoring() -> None:
    financials = {
        "statement_metrics": {
            "trailing_12_month_free_cash_flow": {
                "value": -42_400_000_000,
                "period_start": "2025-10-01",
                "period_end": "2026-09-30",
                "source": "SEC EDGAR company facts",
            },
            "quarterly_free_cash_flow": {
                "value": -18_000_000_000,
                "period_end": "2026-09-30",
                "source": "SEC EDGAR company facts",
            },
        },
    }

    value, basis = workflow._effective_financial_value(
        financials, "free_cash_flow", allow_profile=False
    )
    screen = workflow._screen_evidence({
        "tickers": ["AMZN"],
        "market_summary": {"AMZN": {"close": 100.0}},
        "financial_metrics": {"AMZN": financials},
    }, "AMZN")

    cash_factor = next(factor for factor in screen["factors"] if factor["name"] == "Free cash flow")
    assert value == -42_400_000_000
    assert "SEC EDGAR company facts" in basis
    assert cash_factor["points"] == -2
    assert cash_factor["weight"] == 2
    assert "2026-09-30" in cash_factor["period"]


def test_sec_cashflow_recalculates_ttm_through_latest_quarter() -> None:
    ends = ["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"]

    def rows(values: list[float]) -> list[dict[str, object]]:
        return [
            {
                "start": (pd.Timestamp(end) - pd.DateOffset(months=3) + pd.Timedelta(days=1)).date().isoformat(),
                "end": end,
                "filed": end,
                "form": "10-Q",
                "val": value,
            }
            for end, value in zip(ends, values)
        ]

    facts = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": rows([10, 20, 30, 40, 50])}},
                "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": rows([-1, -2, -3, -4, -5])}},
            },
        },
    }

    metrics = workflow._sec_statement_metrics(facts, "0000000001")

    assert metrics["quarterly_free_cash_flow"]["value"] == 45
    assert metrics["quarterly_free_cash_flow"]["period_end"] == "2026-09-30"
    assert metrics["trailing_12_month_free_cash_flow"]["value"] == 126
    assert metrics["trailing_12_month_free_cash_flow"]["period_end"] == "2026-09-30"


def test_sec_cashflow_combines_tag_fallbacks_and_derives_missing_q4() -> None:
    def make_rows(entries: list[tuple[str, str, float]]) -> list[dict[str, object]]:
        return [
            {
                "start": start,
                "end": end,
                "filed": end,
                "form": "10-K" if end == "2025-12-31" else "10-Q",
                "val": value,
            }
            for start, end, value in entries
        ]

    operating_rows = [
        ("2025-01-01", "2025-09-30", 100.0),
        ("2025-01-01", "2025-12-31", 140.0),
        ("2026-01-01", "2026-03-31", 30.0),
        ("2026-04-01", "2026-06-30", 40.0),
        ("2026-07-01", "2026-09-30", 45.0),
    ]
    capex_rows = [
        ("2025-01-01", "2025-09-30", 80.0),
        ("2025-01-01", "2025-12-31", 120.0),
        ("2026-01-01", "2026-03-31", 40.0),
        ("2026-04-01", "2026-06-30", 50.0),
        ("2026-07-01", "2026-09-30", 65.0),
    ]
    facts = {
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": {
                    "units": {"USD": make_rows(operating_rows)},
                },
                "PaymentsToAcquirePropertyPlantAndEquipment": {
                    "units": {"USD": make_rows([("2016-01-01", "2016-03-31", 5.0)])},
                },
                "PaymentsToAcquireProductiveAssets": {
                    "units": {"USD": make_rows(capex_rows)},
                },
            },
        },
    }

    metrics = workflow._sec_statement_metrics(facts, "0000000001")

    assert metrics["quarterly_free_cash_flow"]["value"] == -20.0
    assert metrics["quarterly_free_cash_flow"]["period_end"] == "2026-09-30"
    assert metrics["trailing_12_month_free_cash_flow"]["value"] == -40.0
    assert metrics["trailing_12_month_free_cash_flow"]["period_end"] == "2026-09-30"


def test_official_issuer_fcf_reconciles_with_sec_formula_and_updates_score() -> None:
    state = {
        "tickers": ["AMZN"],
        "companies": ["Amazon"],
        "news_evidence": [{
            "id": "news_1",
            "related_tickers": ["AMZN"],
            "title": "Amazon Q3 results",
            "source": "Amazon Investor Relations",
            "url": "https://ir.aboutamazon.com/news-release/q3-results",
            "published": "2026-10-29",
            "excerpt": (
                "Free cash flow decreased to an outflow of $42.4 billion for the trailing "
                "twelve months, driven primarily by purchases of property and equipment, "
                "net of proceeds from sales and incentives."
            ),
        }],
        "financial_metrics": {
            "AMZN": {
                "company_name": "Amazon",
                "statement_metrics": {
                    "quarterly_free_cash_flow": {
                        "value": -34_837_000_000,
                        "period_start": "2026-07-01",
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                    },
                    "trailing_12_month_free_cash_flow": {
                        "value": -46_892_000_000,
                        "period_start": "2025-10-01",
                        "period_end": "2026-09-30",
                        "filed_date": "2026-10-30",
                        "source": "SEC EDGAR company facts",
                    },
                },
            },
        },
    }

    issuer_value = workflow._official_issuer_fcf_from_news(
        state["news_evidence"], "AMZN", "Amazon"
    )
    assert issuer_value is not None
    prepared = workflow.prepare_report_input({**state, "issuer_reported_fcf": {"AMZN": issuer_value}})
    updated_metrics = prepared["financial_metrics"]["AMZN"]

    value, basis = workflow._effective_financial_value(
        updated_metrics, "free_cash_flow", allow_profile=False
    )
    screen = workflow._screen_evidence({
        **state,
        "financial_metrics": {"AMZN": updated_metrics},
    }, "AMZN")

    assert value == -42_400_000_000
    assert "[News 1]" in basis
    assert screen["score"] == -2
    assert screen["cash_flow_score_basis"] == "trailing-12-month free cash flow"
    note = updated_metrics["metadata"]["free_cash_flow_definition_note"]
    assert "$4.49B" in note
    assert "net of proceeds from sales and incentives" in note


def test_unmatched_or_undated_issuer_fcf_is_not_used_for_scoring() -> None:
    article = {
        "id": "news_1",
        "related_tickers": ["AMZN"],
        "title": "Amazon earnings",
        "source": "Amazon Investor Relations",
        "url": "https://ir.aboutamazon.com/earnings",
        "published": "",
        "excerpt": "Free cash flow is an outflow of $42.4 billion for the trailing twelve months.",
    }
    assert workflow._official_issuer_fcf_from_news([article], "AMZN", "Amazon") is None


def test_sec_debt_to_equity_uses_a_shared_balance_sheet_date() -> None:
    def instant_rows(values: list[tuple[str, float]]) -> list[dict[str, object]]:
        return [
            {"end": end, "filed": end, "form": "10-Q", "val": value}
            for end, value in values
        ]

    facts = {
        "facts": {
            "us-gaap": {
                "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent": {
                    "units": {"USD": instant_rows([
                        ("2025-12-31", 100.0),
                        ("2026-06-30", 120.0),
                    ])},
                },
                "StockholdersEquity": {
                    "units": {"USD": instant_rows([
                        ("2025-12-31", 50.0),
                        ("2026-09-30", 60.0),
                    ])},
                },
            },
        },
    }

    metrics = workflow._sec_statement_metrics(facts, "0000000001")
    ratio = metrics["calculated_debt_to_equity"]

    assert ratio["value"] == 200.0
    assert ratio["numerator"] == 100.0
    assert ratio["denominator"] == 50.0
    assert ratio["period_end"] == "2025-12-31"


def test_sec_debt_fallback_combines_components_for_latest_period() -> None:
    def instant_rows(value: float) -> list[dict[str, object]]:
        return [
            {"end": "2026-09-30", "filed": "2026-10-30", "form": "10-Q", "val": value},
        ]

    facts = {
        "facts": {
            "us-gaap": {
                "LongTermDebtCurrent": {"units": {"USD": instant_rows(10.0)}},
                "LongTermDebtNoncurrent": {"units": {"USD": instant_rows(90.0)}},
                "ShortTermBorrowings": {"units": {"USD": instant_rows(5.0)}},
                "StockholdersEquity": {"units": {"USD": instant_rows(50.0)}},
            },
        },
    }

    metrics = workflow._sec_statement_metrics(facts, "0000000001")

    assert metrics["reported_total_debt"]["value"] == 105.0
    assert metrics["reported_total_debt"]["period_end"] == "2026-09-30"
    assert metrics["calculated_debt_to_equity"]["value"] == 210.0
    assert metrics["calculated_debt_to_equity"]["period_end"] == "2026-09-30"
    assert "same balance-sheet date" in metrics["calculated_debt_to_equity"]["basis"]


def test_bullish_factors_use_financial_metric_source_citations() -> None:
    state = {
        "tickers": ["EXM"],
        "market_summary": {},
        "financial_metrics": {
            "EXM": {
                "statement_metrics": {
                    "quarterly_revenue_growth_yoy": {
                        "value": 20.0,
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                        "basis": "SEC EDGAR quarterly filing",
                    },
                    "quarterly_profit_margin": {
                        "value": 0.2,
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                        "basis": "SEC EDGAR filing",
                    },
                    "trailing_12_month_free_cash_flow": {
                        "value": 1_000_000_000,
                        "period_start": "2025-10-01",
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                        "basis": "SEC EDGAR calculated from quarterly filings",
                    },
                },
                "revenue_growth": 0.2,
                "quarterly_profit_margin": 0.2,
                "metadata": {
                    "quarterly_profit_margin_period_end": "2026-09-30",
                    "quarterly_profit_margin_source": "SEC EDGAR company facts",
                    "quarterly_profit_margin_basis": "SEC EDGAR filing",
                },
            },
        },
        "news_evidence": [],
    }

    result = workflow._fallback_section("BULLISH FACTORS", state)

    assert "[SEC EDGAR]" in result
    assert "[YFINANCE]" not in result


def test_report_data_quality_surfaces_source_comparisons_and_official_source_status() -> None:
    state = {
        "tickers": ["EXM"],
        "financial_metrics": {
            "EXM": {
                "metadata": {
                    "official_filing_warning": "Official SEC filings were not checked.",
                    "financial_source_comparisons": [
                        "free cash flow periods differ; using the newer Yahoo Finance period"
                    ],
                    "free_cash_flow_definition_note": (
                        "Issuer-reported free cash flow may use net capex adjustments."
                    ),
                },
                "statement_metrics": {},
            },
        },
        "market_summary": {},
        "news_evidence": [],
    }

    section = workflow._fallback_section("DATA QUALITY & LIMITATIONS", state)

    assert "Official SEC filings were not checked." in section
    assert "using the newer Yahoo Finance period" in section
    assert "net capex adjustments" in section


def test_company_overview_uses_dated_liquidity_and_omits_undated_cash_profile() -> None:
    state = {
        "tickers": ["AMZN"],
        "market_summary": {"AMZN": {"company_name": "Amazon"}},
        "financial_metrics": {
            "AMZN": {
                "company_name": "Amazon",
                "total_cash": 114_230_000_000,
                "metadata": {
                    "profile_metric_periods": {
                        "total_cash": "Yahoo Finance total cash profile; exact date unavailable",
                    },
                },
                "statement_metrics": {
                    "reported_total_cash_and_short_term_investments": {
                        "value": 114_226_000_000,
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                    },
                },
            },
        },
        "news_evidence": [],
    }

    overview = workflow._company_overview_section(state)

    assert "Cash, cash equivalents and short-term investments: $114.23B" in overview
    assert "as of 2026-09-30" in overview
    assert "Selected reported metrics: Cash:" not in overview


def test_news_date_uses_search_relative_date_and_excerpt_ends_cleanly() -> None:
    assert workflow._clean_publication_date("October 29, 2026") == "2026-10-29"
    assert workflow._article_publication_date({
        "date": None,
        "body": "3 days ago � Amazon published quarterly results.",
    }) == (datetime.now().date() - timedelta(days=3)).isoformat()
    assert workflow._article_publication_date({
        "href": "https://www.cnbc.com/2026/10/29/amazon-q3-earnings.html",
    }) == "2026-10-29"
    assert workflow._article_publication_date({
        "href": "https://www.marketbeat.com/earnings/2026-10-29-amazon/",
    }) == "2026-10-29"
    excerpt = workflow._clean_excerpt(
        "A complete first sentence. A second complete sentence with more detail. "
        "An incomplete third sentence fragment",
        60,
    )
    assert excerpt == "A complete first sentence."
    assert not excerpt.endswith(("…", "..."))
    assert workflow._clean_excerpt(
        "The full sentence is usable. A search result ended abruptly without punctuation",
        500,
    ) == "The full sentence is usable."


def test_official_news_enrichment_recovers_publication_date_and_fcf_excerpt(monkeypatch) -> None:
    page = b"""
    <html><head></head>
    <body><ul><li><span>Free cash flow</span> decreased to an outflow of $42.4 billion
    for the trailing twelve months, driven by purchases net of proceeds and incentives.</li>
    <span id=\"_ctrl0_ctl54_lblDate\" class=\"module_date-text\">October 29, 2026</span>
    </ul></body></html>
    """

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, _limit):
            return page

    monkeypatch.setattr(workflow, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    published, excerpt = workflow._official_article_metadata_and_excerpt(
        "https://www.aboutamazon.com/news/q3-results",
        "Amazon",
    )

    assert published == "2026-10-29"
    assert "outflow of $42.4 billion" in excerpt
    assert excerpt.endswith("incentives.")


def test_business_topic_query_terms_follow_company_profile() -> None:
    amazon_topics = workflow._business_search_terms({
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
        "business_summary": (
            "Amazon provides online retail, advertising, and cloud compute through AWS."
        ),
    })
    nvidia_topics = workflow._business_search_terms({
        "sector": "Technology",
        "industry": "Semiconductors",
        "business_summary": (
            "The company makes GPUs for data centers and artificial intelligence."
        ),
    })

    assert {"cloud computing", "online retail", "digital advertising"} <= set(amazon_topics)
    assert {"semiconductors", "GPUs", "cloud infrastructure", "artificial intelligence"} <= set(nvidia_topics)


def test_news_selection_prefers_three_company_and_two_business_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSearch:
        def text(self, query, **_kwargs):
            scope = "business" if "industry business news" in query else "stock"
            topics = (
                "cloud expansion", "retail demand", "regulatory changes",
                "AI infrastructure", "market competition",
            )
            return [{
                "title": f"Amazon {scope} report covering {topics[index - 1]}",
                "href": f"https://news{index}.example/{scope}/{index}",
                "date": "2026-10-29",
                "body": f"Amazon {scope} reported a notable development {index}.",
            } for index in range(1, 6)]

    monkeypatch.setattr(workflow, "DDGS", lambda **_kwargs: FakeSearch())
    result = workflow.retrieve_news({
        "tickers": ["AMZN"],
        "companies": ["Amazon"],
        "financial_metrics": {
            "AMZN": {
                "sector": "Technology",
                "industry": "Cloud Computing",
                "business_summary": "Cloud services and artificial intelligence.",
            },
        },
    })

    assert len(result["news_evidence"]) == 5
    assert sum(article["news_scope"] == "stock" for article in result["news_evidence"]) == 3
    assert sum(article["news_scope"] == "business" for article in result["news_evidence"]) == 2


def test_official_earnings_release_mirrors_are_deduplicated() -> None:
    investor_relations = {
        "title": "Amazon.com Announces Third Quarter Results",
        "url": "https://ir.aboutamazon.com/news-release/2026/third-quarter-results",
    }
    newsroom = {
        "title": "Amazon Q3 2026 earnings report: Read the release",
        "url": "https://www.aboutamazon.com/news/company-news/amazon-earnings-q3-2026-report",
    }

    assert workflow._earnings_event(investor_relations) == ("2026", "3")
    assert workflow._earnings_event(newsroom) == ("2026", "3")
    assert workflow._same_story(investor_relations, newsroom)


def test_generic_financial_statement_pages_are_not_selected_as_news() -> None:
    assert workflow._is_low_information_article({
        "title": "AMZN Cash Flow 2026 | Free Cash Flow and Capex",
        "url": "https://example.com/stock/AMZN/financials/cash-flow",
        "snippet": (
            "Amazon cash flow statement: $-8.82B free cash flow. "
            "Capex, dividends, and annual and quarterly data."
        ),
    })
    assert not workflow._is_low_information_article({
        "title": "Amazon Q3 results: cash flow and capex outlook",
        "url": "https://example.com/news/amazon-q3-results",
        "snippet": "Amazon reported results and discussed investment spending.",
    })


def test_duplicate_news_merges_publication_date_and_excerpt_metadata() -> None:
    first = {
        "title": "Amazon reports Q3 results and cash flow",
        "url": "https://ir.aboutamazon.com/q3",
        "source": "Amazon IR",
        "published": "",
        "excerpt": "Amazon reported quarterly results.",
        "related_tickers": ["AMZN"],
    }
    second = {
        **first,
        "url": "https://ir.aboutamazon.com/q3#details",
        "published": "2026-10-29",
        "excerpt": "Amazon reported quarterly results and provided cash-flow detail.",
    }

    assert workflow._same_story(first, second)
    merged = workflow._merge_duplicate_article(first, second, ["AMZN"], ["Amazon"])

    assert merged["published"] == "2026-10-29"
    assert "cash-flow detail" in merged["excerpt"]


def test_undated_yahoo_profile_metrics_are_excluded_from_scoring() -> None:
    state = {
        "tickers": ["AMZN"],
        "market_summary": {"AMZN": {"close": 100.0}},
        "financial_metrics": {
            "AMZN": {
                "revenue_growth": 0.12,
                "profit_margin": 0.15,
                "metadata": {
                    "profile_metric_periods": {
                        "revenue_growth": "Yahoo Finance reported growth field; exact period not provided",
                        "profit_margin": "Yahoo Finance reported margin field; exact statement period not provided",
                    },
                },
                "statement_metrics": {
                    "trailing_12_month_free_cash_flow": {
                        "value": -20_000_000,
                        "period_start": "2025-10-01",
                        "period_end": "2026-09-30",
                    },
                },
            },
        },
    }

    evidence = workflow._screen_evidence(state, "AMZN")
    stance = workflow._research_stance(state)

    assert evidence["available_factor_count"] == 1
    assert evidence["dated_factor_count"] == 1
    assert stance[0].startswith("LIMITED DATA COVERAGE")
    assert "Undated Yahoo profile" in stance[3]


def test_ttm_free_cash_flow_sums_four_quarters_and_requires_complete_periods() -> None:
    dates = pd.to_datetime(["2026-09-30", "2026-06-30", "2026-03-31", "2025-12-31"])
    complete = pd.DataFrame(
        {
            dates[0]: [10.0, -1.0],
            dates[1]: [20.0, -2.0],
            dates[2]: [30.0, -3.0],
            dates[3]: [40.0, -4.0],
        },
        index=["Operating Cash Flow", "Capital Expenditure"],
    )
    incomplete = complete.loc[:, dates[:3]]
    missing_quarter = pd.DataFrame(
        {
            pd.Timestamp("2026-09-30"): [10.0, -1.0],
            pd.Timestamp("2026-06-30"): [20.0, -2.0],
            pd.Timestamp("2026-03-31"): [30.0, -3.0],
            pd.Timestamp("2025-09-30"): [40.0, -4.0],
        },
        index=["Operating Cash Flow", "Capital Expenditure"],
    )

    metrics = workflow._statement_metrics(None, complete, None)
    incomplete_metrics = workflow._statement_metrics(None, incomplete, None)
    missing_quarter_metrics = workflow._statement_metrics(None, missing_quarter, None)

    assert metrics["trailing_12_month_free_cash_flow"]["value"] == pytest.approx(90.0)
    assert metrics["trailing_12_month_free_cash_flow"]["quarter_count"] == 4
    assert metrics["trailing_12_month_free_cash_flow"]["period_start"] == "2025-10-01"
    assert "trailing_12_month_free_cash_flow" not in incomplete_metrics
    assert "trailing_12_month_free_cash_flow" not in missing_quarter_metrics


def test_missing_market_history_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    class EmptyTicker:
        info = {}

        def history(self, **_kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(workflow.yf, "Ticker", lambda _symbol: EmptyTicker())
    market, _, errors = workflow._calculate_market_data("AMZN", "1y")
    assert market == {}
    assert any("no usable historical prices" in error for error in errors)


def test_external_retrieval_failures_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenTicker:
        def __init__(self, _symbol):
            raise RuntimeError("market provider unavailable")

    class BrokenSearch:
        def text(self, *_args, **_kwargs):
            raise RuntimeError("news provider unavailable")

    monkeypatch.setattr(workflow.yf, "Ticker", BrokenTicker)
    monkeypatch.setattr(workflow, "DDGS", lambda **_kwargs: BrokenSearch())
    state = {"tickers": ["AMZN"], "companies": ["Amazon"], "period": "1y"}

    market = workflow.retrieve_market(state)
    news = workflow.retrieve_news(state)

    assert any("market provider unavailable" in error for error in market["market_errors"])
    assert news["news_evidence"] == []
    assert any("news provider unavailable" in error for error in news["news_errors"])


def test_company_name_falls_back_to_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        workflow,
        "_invoke_groq",
        lambda *_args, **_kwargs: (
            workflow.ResearchPlan(tickers=["AMZN"], companies=["Amazon"], intent="company research"),
            {"calls": 1, "retry_attempts": 0, "estimated_input_tokens": 25, "actual_input_tokens": 20, "actual_output_tokens": 10},
        ),
    )
    result = workflow.validate_input({"user_question": "Research Amazon's stock"})
    assert result["input_status"] == "valid"
    assert result["tickers"] == ["AMZN"]
    assert result["companies"] == ["Amazon"]
    assert result["llm_usage"]["calls"] == 1


def test_selected_company_listing_skips_planner_and_preserves_company_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        workflow,
        "_invoke_groq",
        lambda *_args, **_kwargs: pytest.fail("A selected listing should not need ticker planning"),
    )

    result = workflow.validate_input({
        "user_question": "Research Amazon.com, Inc. stock",
        "company_name": "Amazon.com, Inc.",
        "requested_ticker": "AMZN",
        "period": "2y",
    })

    assert result["input_status"] == "valid"
    assert result["tickers"] == ["AMZN"]
    assert result["companies"] == ["Amazon.com, Inc."]
    assert result.get("llm_usage", {}).get("calls", 0) == 0


def test_planner_parses_plain_text_response_without_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    response_text = (
        '**Tickers:** `["AMZN"]`  \n'
        '**Companies:** `["Amazon.com, Inc."]`  \n'
        '**Intent:** General stock research on Amazon.'
    )
    monkeypatch.setattr(
        workflow,
        "_invoke_groq",
        lambda *_args, **_kwargs: (
            SimpleNamespace(content=response_text),
            {"calls": 1, "retry_attempts": 0, "estimated_input_tokens": 25, "actual_input_tokens": 20, "actual_output_tokens": 10},
        ),
    )
    result = workflow.validate_input({"user_question": "amazon"})
    assert result["input_status"] == "valid"
    assert result["tickers"] == ["AMZN"]
    assert result["companies"] == ["Amazon.com, Inc."]
    assert result["llm_usage"]["calls"] == 1


def test_invalid_period_stops_before_planning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        workflow,
        "_invoke_groq",
        lambda *_args, **_kwargs: pytest.fail("invalid period must be rejected before planning"),
    )
    result = workflow.validate_input({"user_question": "Research Amazon", "period": "13 months"})
    assert result["input_status"] == "invalid"
    assert "Unsupported period" in result["execution_errors"][0]


def test_news_evidence_deduplicates_and_limits_excerpts() -> None:
    rows = [
        {
            "title": "Amazon earnings",
            "url": "https://news.example/story?id=1",
            "source": "news.example",
            "published": "2026-10-28",
            "snippet": "AMZN " + "x" * 100,
        },
        {
            "title": "Amazon earnings duplicate",
            "url": "https://news.example/story?id=2#section",
            "source": "news.example",
            "published": "2026-10-29",
            "snippet": "AMZN update",
        },
        {
            "title": "SEC filing",
            "url": "https://www.sec.gov/filing",
            "source": "sec.gov",
            "published": "2026-10-27",
            "snippet": "AMZN filing",
        },
    ]

    evidence = workflow.prepare_news_evidence(rows, ["AMZN"], ["Amazon"], limit=1, excerpt_length=20)
    assert len(evidence) == 1
    assert evidence[0]["url"] == "https://www.sec.gov/filing"
    assert len(evidence[0]["excerpt"]) <= 20


def test_news_evidence_cleans_metadata_and_deduplicates_syndicated_stories() -> None:
    rows = [
        {
            "title": "Amazon Q3 results: revenue grows 12 percent year over year",
            "url": "https://www.reuters.com/business/amazon-q3-results",
            "source": "www.reuters.com",
            "published": "2026-10-29T08:30:00Z",
            "snippet": "Amazon's net sales rose 12% in Q3. " + "context " * 35,
        },
        {
            "title": "Amazon Q3 results - revenue grows 12 percent year over year",
            "url": "https://finance.example.com/amazon-q3-results",
            "source": "finance.example.com",
            "published": "Wed, 29 Oct 2026 09:00:00 GMT",
            "snippet": "Syndicated version of the same earnings story.",
        },
        {
            "title": "Amazon raises planned capital spending for next year",
            "url": "https://www.reuters.com/business/amazon-capex",
            "source": "www.reuters.com",
            "published": "2026-10-28",
            "snippet": "The company expects higher capital expenditures next year.",
        },
    ]

    evidence = workflow.prepare_news_evidence(rows, ["AMZN"], ["Amazon"], limit=5, excerpt_length=100)

    assert len(evidence) == 2
    earnings = next(item for item in evidence if "Q3 results" in item["title"])
    capex = next(item for item in evidence if "capital spending" in item["title"])
    assert earnings["source"] == "Reuters"
    assert earnings["published"] == "2026-10-29"
    assert len(earnings["excerpt"]) <= 100
    assert "earnings_results" in earnings["categories"]
    assert "revenue_increase" in earnings["categories"]
    assert "capital_spending_increase" in capex["categories"]
    assert workflow._clean_publication_date("2 days ago")


def test_news_categories_follow_excerpt_facts_not_polarity_words() -> None:
    growth_and_risk = workflow._classify_article({
        "title": "Amazon cloud growth accelerates",
        "snippet": "AWS grows strongly while free cash flow turns negative amid higher capital spending.",
    })
    generic_positive_headline = workflow._classify_article({
        "title": "Amazon shares shine",
        "snippet": "Investors discuss the company during the session.",
    })

    assert "cloud_growth" in growth_and_risk
    assert "cash_flow_pressure" in growth_and_risk
    assert "capital_spending_increase" in growth_and_risk
    assert generic_positive_headline == ["company_update"]


def test_generic_news_pages_are_removed_from_selected_evidence() -> None:
    rows = [
        {
            "title": "NVDA Latest Stock News",
            "url": "https://example.com/nvda-news",
            "source": "example.com",
            "published": "",
            "snippet": "Get the latest news and real-time alerts from NVDA stock.",
        },
        {
            "title": "NVIDIA schedules quarterly results call",
            "url": "https://nvidia.example.com/quarterly-call",
            "source": "nvidia.example.com",
            "published": "2026-10-29",
            "snippet": "NVIDIA will discuss its fiscal third-quarter financial results on November 17.",
        },
    ]

    evidence = workflow.prepare_news_evidence(rows, ["NVDA"], ["NVIDIA"], limit=5, excerpt_length=160)
    assert len(evidence) == 1
    assert evidence[0]["title"] == "NVIDIA schedules quarterly results call"


def test_oversized_input_is_rejected_before_provider_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_INPUT_TOKEN_BUDGET", "10")
    monkeypatch.setenv("GROQ_CONTEXT_WINDOW", "100")
    monkeypatch.setenv("GROQ_MAX_TOKENS", "10")
    monkeypatch.setattr(workflow, "llm", lambda *_args, **_kwargs: pytest.fail("LLM must not be constructed"))

    with pytest.raises(workflow.PromptTooLargeError):
        workflow._invoke_groq("x" * 100)


def test_rate_limit_retries_are_bounded_and_usage_is_tracked(monkeypatch: pytest.MonkeyPatch) -> None:
    class RateLimitError(Exception):
        status_code = 429
        response = SimpleNamespace(status_code=429, headers={"retry-after": "0"})

    class FakeClient:
        calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            if self.calls == 1:
                raise RateLimitError("rate limited")
            return SimpleNamespace(
                content="ok",
                usage_metadata={"input_tokens": 12, "output_tokens": 8},
                response_metadata={},
            )

    client = FakeClient()
    monkeypatch.setattr(workflow, "llm", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(workflow.time, "sleep", lambda _delay: None)
    monkeypatch.setenv("GROQ_INPUT_TOKEN_BUDGET", "100")
    monkeypatch.setenv("GROQ_CONTEXT_WINDOW", "100")
    monkeypatch.setenv("GROQ_MAX_TOKENS", "10")
    monkeypatch.setenv("GROQ_MAX_RETRIES", "1")

    response, usage = workflow._invoke_groq("short prompt")
    assert response.content == "ok"
    assert client.calls == 2
    assert usage["calls"] == 2
    assert usage["retry_attempts"] == 1
    assert usage["actual_input_tokens"] == 12
    assert usage["actual_output_tokens"] == 8


def test_non_retryable_context_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    class TooLargeError(Exception):
        status_code = 413

    class FakeClient:
        calls = 0

        def invoke(self, _prompt):
            self.calls += 1
            raise TooLargeError("request too large")

    client = FakeClient()
    monkeypatch.setattr(workflow, "llm", lambda *_args, **_kwargs: client)
    monkeypatch.setenv("GROQ_INPUT_TOKEN_BUDGET", "100")
    monkeypatch.setenv("GROQ_CONTEXT_WINDOW", "100")
    monkeypatch.setenv("GROQ_MAX_TOKENS", "10")
    monkeypatch.setenv("GROQ_MAX_RETRIES", "3")

    with pytest.raises(workflow.GroqRequestError) as error:
        workflow._invoke_groq("short prompt")
    assert client.calls == 1
    assert error.value.usage["calls"] == 1


def test_validation_warnings_and_correction_route_are_bounded() -> None:
    state = {
        "report": "EXECUTIVE SUMMARY\n\nIncomplete.",
        "tickers": ["AMZN"],
        "market_summary": {},
        "news_evidence": [],
        "correction_attempts": 0,
    }
    result = workflow.validate_report(state)
    assert result["validation_status"] == "correction_needed"
    assert workflow.route_validation_result({**state, **result}) == "correct"
    after_one_attempt = {**state, **result, "correction_attempts": 1}
    assert workflow.validate_report(after_one_attempt)["validation_status"] == "failed"
    assert workflow.route_validation_result({**after_one_attempt, "validation_status": "failed"}) == "finish"


def test_report_validation_checks_news_citation_coverage() -> None:
    report = "\n\n".join(f"{section}\n\nEvidence." for section in workflow.REQUIRED_SECTIONS)
    state = {"tickers": ["AMZN"], "market_summary": {}, "news_evidence": [{"id": "news_1"}]}
    assert any(
        "need [News N] citations" in warning
        for warning in workflow._validate_report(report, state)
    )
    assert workflow._validate_report(report + "\n[News 1]", state) == []
    warnings = workflow._validate_report(report + "\n[News 9]", state)
    assert any("unsupported news citations" in warning for warning in warnings)


def test_markdown_report_sections_are_replaced_deduplicated_and_validated() -> None:
    report = "\n\n".join(
        f"**{section} (updated with citations)**\n\nModel text. [News 1]" if section == "EXECUTIVE SUMMARY"
        else f"**{section}**\n\nInvented SMA at $247. Unsupported selling-pressure claim." if section == "PRICE & VOLUME ANALYSIS"
        else f"**{section}**\n\nModel text."
        for section in workflow.REQUIRED_SECTIONS
    )
    report += "\n\n## **MARKET SNAPSHOT**\n\nDuplicate market data."
    report += "\n\n**HISTORICAL PERFORMANCE**\n\nDuplicate returns."
    state = {
        "tickers": ["AMZN"],
        "period": "1y",
        "market_summary": {
            "AMZN": {
                "company_name": "Amazon",
                "close": 243.79,
                "date": "2026-10-29",
                "daily_change_percent": -4.25,
                "open": 246.949996,
                "high": 249.139999,
                "low": 241.240005,
                "volume": 83_559_600,
                "average_volume_20d": 38_026_355,
                "returns": {},
            }
        },
        "news_evidence": [{
            "id": "news_1",
            "title": "Amazon earnings release",
            "source": "example.com",
            "published": "2026-10-29",
            "url": "https://example.com/amazon",
            "excerpt": "Example market evidence.",
        }],
    }

    updated = workflow._insert_deterministic_sections(report, state)
    assert updated.count("MARKET SNAPSHOT") == 1
    assert updated.count("HISTORICAL PERFORMANCE") == 1
    assert updated.count("RECENT NEWS & EVENTS") == 1
    assert updated.count("PRICE & VOLUME ANALYSIS") == 1
    assert "Invented SMA" not in updated
    assert "selling-pressure claim" not in updated
    assert "2.20x average" in updated
    assert "$246.95" in updated
    assert "83,559,600" in updated
    assert "[News 1]" in updated
    assert workflow._validate_report(updated, state) == []


def test_correction_requests_only_missing_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    report = "\n\n".join(
        f"{section}\n\nExisting text." for section in workflow.REQUIRED_SECTIONS
        if section != "COMPANY OVERVIEW"
    )
    prompts = []

    def fake_invoke(prompt):
        prompts.append(prompt)
        return SimpleNamespace(content="COMPANY OVERVIEW\n\nInsufficient company evidence."), {
            "calls": 1,
            "retry_attempts": 0,
            "estimated_input_tokens": 20,
            "actual_input_tokens": 20,
            "actual_output_tokens": 10,
        }

    monkeypatch.setattr(workflow, "_invoke_groq", fake_invoke)
    state = {
        "report": report,
        "tickers": ["AMZN"],
        "period": "1y",
        "market_summary": {},
        "news_evidence": [],
        "validation_warnings": ["Required report section is missing: COMPANY OVERVIEW."],
        "correction_attempts": 0,
    }

    result = workflow.correct_report(state)

    assert len(prompts) == 1
    assert "COMPANY OVERVIEW" in prompts[0]
    assert "EXECUTIVE SUMMARY" not in prompts[0]
    assert "Existing text." not in prompts[0]
    assert result["report"].count("COMPANY OVERVIEW") == 1
    assert result["correction_attempts"] == 1


def test_incomplete_llm_report_gets_suggestion_and_all_sections_without_correction() -> None:
    state = {
        "tickers": ["AMZN"],
        "period": "1y",
        "user_question": "amazon stock history plus analysis",
        "market_summary": {
            "AMZN": {
                "ticker": "AMZN",
                "company_name": "Amazon",
                "date": "2026-10-29",
                "close": 243.79,
                "daily_change_percent": -4.25,
                "volume": 83_559_600,
                "average_volume_20d": 38_026_355,
                "returns": {
                    "1-day": {"return_percent": -4.25, "start_price": 254.60, "end_price": 243.79, "start_date": "2026-10-28", "end_date": "2026-10-29"},
                    "1-year": {"return_percent": 5.86, "start_price": 230.30, "end_price": 243.79, "start_date": "2025-10-29", "end_date": "2026-10-29"},
                },
            }
        },
        "financial_metrics": {},
        "news_evidence": [{
            "id": "news_1",
            "title": "Amazon reports negative free cash flow",
            "source": "example.com",
            "published": "",
            "url": "https://example.com/amazon",
            "excerpt": "Amazon's negative free cash flow ballooned during the third quarter.",
        }],
        "warnings": [],
    }

    report = workflow._insert_deterministic_sections(
        "RECENT NEWS & EVENTS\n\n[News 1] Amazon reported negative free cash flow.",
        state,
    )

    assert workflow._validate_report(report, state) == []
    assert "FINAL ASSESSMENT" in report
    assert "What to check next:" in report
    assert "general research screen, not personalized investment advice" in report
    assert "RISKS & BEARISH FACTORS" in report
    assert "[News 1]" in report
    assert report.count("1-day:") <= 1
    assert "Evidence-based screen:" in report


def test_analyst_summary_connects_dated_fundamentals_and_price_trend() -> None:
    state = {
        "tickers": ["AMZN"],
        "market_summary": {
            "AMZN": {
                "company_name": "Amazon",
                "close": 200.0,
                "returns": {
                    "1-month": {"return_percent": -4.0},
                    "1-year": {"return_percent": 12.0},
                },
            }
        },
        "financial_metrics": {
            "AMZN": {
                "quarterly_profit_margin": 0.1,
                "pe": 30.0,
                "statement_metrics": {
                    "quarterly_revenue_growth_yoy": {
                        "value": 15.0,
                        "period_end": "2026-09-30",
                    },
                    "quarterly_free_cash_flow": {
                        "value": -50_000_000,
                        "period_end": "2026-09-30",
                    },
                },
            }
        },
        "news_evidence": [],
    }

    summary = workflow._analyst_summary_for_ticker(state, "AMZN")
    final = workflow._final_assessment_for_ticker(state, "AMZN")

    assert "recent weakness sits against a positive one-year trend" in summary
    assert "revenue growth is 15.00%" in summary
    assert "cash outflow after operating and investment activity" in summary
    assert "without a peer or historical comparison" in summary
    assert "What this means for this stock" not in summary
    assert "What to check next:" in final


def test_research_stance_requires_dated_statement_support() -> None:
    state = {
        "tickers": ["NVDA"],
        "market_summary": {
            "NVDA": {
                "close": 100.0,
                "returns": {"1-year": {"return_percent": 3.44}},
            }
        },
        "financial_metrics": {
            "NVDA": {
                "pe": 25.0,
                "quarterly_profit_margin": 0.2,
                "statement_metrics": {
                    "quarterly_revenue_growth_yoy": {"value": 20.0, "period_end": "2026-09-30"},
                    "quarterly_profit_margin": {"value": 0.2, "period_end": "2026-09-30"},
                },
            }
        },
        "news_evidence": [{
            "id": "news_1",
            "categories": "earnings_results",
            "excerpt": "NVIDIA scheduled a conference call to discuss results.",
        }],
    }

    cautious = workflow._research_stance(state)
    assert cautious[0].startswith("LIMITED DATA COVERAGE")
    assert "free cash flow" in cautious[3]
    assert "trailing P/E 25.00x" in cautious[3]

    state["news_evidence"] = [{
        "id": "news_2",
        "categories": "revenue_increase",
        "excerpt": "Reported revenue growth of 20 percent year over year.",
    }]
    state["financial_metrics"]["NVDA"]["statement_metrics"]["quarterly_free_cash_flow"] = {
        "value": 10_000_000,
        "period_end": "2026-09-30",
    }
    constructive = workflow._research_stance(state)
    assert constructive[0].startswith("CONSTRUCTIVE FUNDAMENTALS")
    assert "Free cash flow:" in constructive[3]
    assert "supports the operating case" in constructive[3]


def test_research_stance_does_not_call_positive_fundamentals_a_buy_without_valuation() -> None:
    state = {
        "tickers": ["NVDA"],
        "market_summary": {"NVDA": {"close": 100.0}},
        "financial_metrics": {
            "NVDA": {
                "quarterly_profit_margin": 0.2,
                "statement_metrics": {
                    "quarterly_revenue_growth_yoy": {"value": 20.0, "period_end": "2026-09-30"},
                    "quarterly_profit_margin": {"value": 0.2, "period_end": "2026-09-30"},
                    "quarterly_free_cash_flow": {"value": 10_000_000, "period_end": "2026-09-30"},
                },
            }
        },
        "news_evidence": [],
    }

    stance = workflow._research_stance(state)
    assert stance[0].startswith("CONSTRUCTIVE FUNDAMENTALS")
    assert "trailing P/E" in stance[3]
    assert "not scored" in stance[3]
    assert "BUY" not in stance[0]


def test_score_reports_fixed_and_available_ranges_with_factor_basis() -> None:
    state = {
        "tickers": ["AMZN"],
        "market_summary": {"AMZN": {"close": 100.0}},
        "financial_metrics": {
            "AMZN": {
                "quarterly_profit_margin": 0.2,
                "statement_metrics": {
                    "quarterly_revenue_growth_yoy": {
                        "value": 10.0,
                        "period_end": "2026-09-30",
                    },
                    "quarterly_profit_margin": {
                        "value": 0.2,
                        "period_end": "2026-09-30",
                    },
                    "quarterly_free_cash_flow": {
                        "value": -10_000_000,
                        "period_end": "2026-09-30",
                    },
                },
            }
        },
    }

    evidence = workflow._screen_evidence(state, "AMZN")

    assert evidence["score"] == 1
    assert (evidence["minimum_score"], evidence["maximum_score"]) == (-5, 5)
    assert evidence["available_max_score"] == 3
    assert evidence["cash_flow_score_basis"] == "quarterly free cash flow fallback"
    assert [factor["name"] for factor in evidence["factor_breakdown"]] == [
        "Revenue growth",
        "Operating-income growth",
        "Profit margin",
        "Free cash flow",
    ]
    assert evidence["factor_breakdown"][1]["evidence"] == "Not available"
    assert evidence["factor_breakdown"][1]["points"] == 0
    assert evidence["factor_breakdown"][3]["weight"] == 1


def test_ttm_free_cash_flow_uses_two_point_score_weight() -> None:
    evidence = workflow._screen_evidence({
        "tickers": ["AMZN"],
        "market_summary": {"AMZN": {"close": 100.0}},
        "financial_metrics": {
            "AMZN": {
                "statement_metrics": {
                    "trailing_12_month_free_cash_flow": {
                        "value": 20_000_000,
                        "period_start": "2025-10-01",
                        "period_end": "2026-09-30",
                    },
                },
            }
        },
    }, "AMZN")

    assert evidence["factor_breakdown"][3]["weight"] == 2
    assert (evidence["minimum_score"], evidence["maximum_score"]) == (-6, 6)
    assert evidence["cash_flow_score_basis"] == "trailing-12-month free cash flow"


def test_bearish_section_uses_dated_facts_and_cleans_escaped_markdown() -> None:
    state = {
        "tickers": ["AMZN"],
        "market_summary": {
            "AMZN": {
                "close": 200.0,
                "date": "2026-11-01",
                "daily_change_percent": -1.0,
            },
        },
        "financial_metrics": {
            "AMZN": {
                "statement_metrics": {
                    "trailing_12_month_free_cash_flow": {
                        "value": -42_400_000_000,
                        "period_start": "2025-10-01",
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                    },
                },
            },
        },
        "news_evidence": [],
    }
    report = workflow._insert_deterministic_sections(
        "RISKS & BEARISH FACTORS\n\n\\*\\*Old uncited claim\\*\\*",
        state,
    )
    section = workflow._extract_report_sections(report)["RISKS & BEARISH FACTORS"]

    assert section.startswith("- ")
    assert "2026-09-30" in section
    assert "[SEC EDGAR]" in section
    assert "\\*" not in section
    assert "Old uncited claim" not in section


def test_financial_sections_show_latest_reporting_period_and_actual_source() -> None:
    state = {
        "tickers": ["AMZN"],
        "market_summary": {
            "AMZN": {
                "company_name": "Amazon",
                "close": 200.0,
                "date": "2026-11-01",
                "daily_change_percent": 1.0,
                "returns": {},
            },
        },
        "financial_metrics": {
            "AMZN": {
                "company_name": "Amazon",
                "statement_metrics": {
                    "quarterly_free_cash_flow": {
                        "value": -34_837_000_000,
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                    },
                    "trailing_12_month_free_cash_flow": {
                        "value": -46_892_000_000,
                        "period_start": "2025-10-01",
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                        "basis": (
                            "SEC EDGAR: calculated from quarterly operating cash flow less "
                            "payments to acquire productive assets"
                        ),
                    },
                },
            },
        },
        "news_evidence": [],
    }

    report = workflow._insert_deterministic_sections(
        "EXECUTIVE SUMMARY\n\nAMZN FCF is current at -$2.47B through March 2026 [YFINANCE].\n\n"
        "RISKS & BEARISH FACTORS\n\n\\*\\*Old FCF: -$2.47B\\*\\*",
        state,
    )
    sections = workflow._extract_report_sections(report)

    for title in (
        "EXECUTIVE SUMMARY",
        "COMPANY OVERVIEW",
        "RISKS & BEARISH FACTORS",
        "FINAL ASSESSMENT",
    ):
        assert "2026-09-30" in sections[title]
        assert "[SEC EDGAR]" in sections[title]
        assert "2026-03-31" not in sections[title]
    assert "\\*" not in sections["RISKS & BEARISH FACTORS"]


def test_bullet_formatter_unescapes_stray_markdown_asterisks() -> None:
    assert workflow._as_bullet_list(r"\*\*Risk\*\*. Cash flow declined.") == (
        "- **Risk**.\n- Cash flow declined."
    )


def test_report_prompt_requests_concise_analytical_sections_without_repeating_data() -> None:
    prompt = workflow._report_prompt({
        "tickers": ["AMZN"],
        "period": "1y",
        "market_summary": {},
        "financial_metrics": {},
        "news_evidence": [],
        "warnings": [],
    })
    assert "under 360 words" in prompt
    assert "do not repeat the deterministic market snapshot" in prompt.lower()
    assert "compact analyst thesis" in prompt.lower()
    assert "technical analyst tone" in prompt.lower()
    assert "connect each cited fact to its business implication" in prompt.lower()
    assert "2-4 concise markdown bullet points per section" in prompt.lower()
    assert "avoid canned phrases" in prompt.lower()
    assert "EXECUTIVE SUMMARY" in prompt
    assert "COMPANY OVERVIEW" not in prompt.split("Evidence JSON:", 1)[0].split("exactly these headings:", 1)[1]


def test_stock_specific_explanations_interpret_current_data_without_changing_facts() -> None:
    state = {
        "tickers": ["AMZN"],
        "period": "1y",
        "market_summary": {
            "AMZN": {
                "company_name": "Amazon",
                "close": 100.0,
                "date": "2026-10-29",
                "daily_change_percent": 2.0,
                "returns": {
                    "1-month": {
                        "return_percent": -3.0,
                        "start_price": 103.09,
                        "end_price": 100.0,
                        "start_date": "2026-09-29",
                        "end_date": "2026-10-29",
                    },
                    "1-year": {
                        "return_percent": 5.0,
                        "start_price": 95.24,
                        "end_price": 100.0,
                        "start_date": "2025-10-29",
                        "end_date": "2026-10-29",
                    }
                },
            }
        },
        "financial_metrics": {},
        "news_evidence": [],
    }

    report = workflow._insert_deterministic_sections(
        "EXECUTIVE SUMMARY\n\nAMZN has a positive one-year return.",
        state,
    )

    assert "AMZN has a positive one-year return." not in report
    assert "**Bottom line:** Amazon" in report
    assert "close $100.00 on 2026-10-29; daily change +2.00%" in report
    assert "1-year: +5.00%" in report
    for section in workflow.REQUIRED_SECTIONS:
        assert f"{section}\n\n" in report
    assert "**What this means for this stock:**" not in report
    fallback_summary = workflow._fallback_section("EXECUTIVE SUMMARY", state)
    assert "recent weakness sits against a positive one-year trend" in fallback_summary
    assert "not a personalized buy or sell recommendation" in fallback_summary
    assert "BUY CANDIDATE" not in report
    assert workflow._validate_report(report, state) == []


def test_company_overview_keeps_business_focus_concise_and_complete() -> None:
    description = "Amazon operates online stores and cloud-computing services. " * 10
    report = workflow._insert_deterministic_sections(
        "COMPANY OVERVIEW\n\nShort generic model text.",
        {
            "tickers": ["AMZN"],
            "market_summary": {
                "AMZN": {
                    "company_name": "Amazon.com, Inc.",
                    "close": 100.0,
                    "date": "2026-10-29",
                }
            },
            "financial_metrics": {
                "AMZN": {
                    "sector": "Consumer Cyclical",
                    "industry": "Internet Retail",
                    "business_summary": description,
                    "revenue_growth": 0.12,
                }
            },
            "news_evidence": [],
        },
    )
    overview = report.split("COMPANY OVERVIEW", 1)[1].split("BULLISH FACTORS", 1)[0]
    assert "Business focus: Amazon operates online stores and cloud-computing services." in overview
    assert len(overview.split("Business focus: ", 1)[1].split(" [YFINANCE]", 1)[0]) <= 360
    assert "Consumer Cyclical / Internet Retail" in overview
    assert "Revenue growth: 12.00%" not in overview


def test_invalid_input_stops_before_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow, "retrieve_market", lambda _state: pytest.fail("market retrieval must not run"))
    monkeypatch.setattr(workflow, "retrieve_news", lambda _state: pytest.fail("news retrieval must not run"))
    result = workflow.build_graph().invoke({"user_question": ""})
    assert result["input_status"] == "invalid"
    assert result["execution_errors"]
    assert result["execution_time_seconds"] >= 0


@pytest.mark.parametrize("ticker", ["AMZN", "NVDA"])
def test_graph_runs_one_primary_report_call_with_mocked_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    ticker: str,
) -> None:
    date = "2026-10-29"
    monkeypatch.setattr(
        workflow,
        "retrieve_market",
        lambda state: {
            "market_summary": {
                state["tickers"][0]: {
                    "ticker": state["tickers"][0],
                    "company_name": state["tickers"][0],
                    "date": date,
                    "close": 100.25,
                    "daily_change_percent": 1.5,
                    "open": 99.0,
                    "high": 101.0,
                    "low": 98.0,
                    "volume": 1000,
                    "average_volume_20d": 900,
                    "returns": {},
                }
            },
            "financial_metrics": {},
            "market_errors": [],
        },
    )
    monkeypatch.setattr(
        workflow,
        "retrieve_news",
        lambda _state: {"news_evidence": [], "news_errors": []},
    )
    headings = "\n\n".join(f"### **{section}**\n\nEvidence is limited." for section in workflow.REQUIRED_SECTIONS)
    calls = []

    def fake_invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(
            content=headings,
            usage_metadata={"input_tokens": 40, "output_tokens": 60},
            response_metadata={},
        ), {
            "calls": 1,
            "retry_attempts": 0,
            "estimated_input_tokens": workflow.estimate_tokens(prompt),
            "actual_input_tokens": 40,
            "actual_output_tokens": 60,
        }

    monkeypatch.setattr(workflow, "_invoke_groq", fake_invoke)
    events = list(workflow.build_graph().stream(
        {"ticker": ticker, "period": "1y"},
        stream_mode="updates",
    ))
    result = {}
    completed_nodes = set()
    for event in events:
        for node, update in event.items():
            completed_nodes.add(node)
            result.update(update or {})

    assert result["validation_status"] == "passed"
    assert result.get("correction_attempts", 0) == 0
    assert result["llm_usage"]["calls"] == 1
    assert len(calls) == 1
    assert {"retrieve_market", "retrieve_news", "generate_report", "validate_report"} <= completed_nodes
    assert result["execution_time_seconds"] >= 0
    assert f"{ticker}: market" not in result["report"]
    assert "$100.25" in result["report"]


def test_research_screen_is_independent_for_each_ticker() -> None:
    def metrics(growth: float, margin: float, cash_flow: float, pe: float) -> dict:
        return {
            "quarterly_profit_margin": margin,
            "statement_metrics": {
                "quarterly_revenue_growth_yoy": {"value": growth, "period_end": "2026-09-30"},
                "quarterly_profit_margin": {"value": margin, "period_end": "2026-09-30"},
                "quarterly_free_cash_flow": {"value": cash_flow, "period_end": "2026-09-30"},
            },
            "pe": pe,
        }

    state = {
        "tickers": ["AMZN", "AMD"],
        "market_summary": {"AMZN": {"close": 100.0}, "AMD": {"close": 150.0}},
        "financial_metrics": {
            "AMZN": metrics(0.1, 0.2, -10.0, 30.0),
            "AMD": metrics(0.2, 0.3, 500.0, 35.0),
        },
        "news_evidence": [],
    }

    amzn = workflow._research_stance(state, "AMZN")
    amd = workflow._research_stance(state, "AMD")
    combined = workflow._research_stance(state)

    assert "MIXED FUNDAMENTALS" in amzn[0]
    assert "CONSTRUCTIVE FUNDAMENTALS" in amd[0]
    assert "PER-TICKER" in combined[0]
    assert "AMZN: MIXED FUNDAMENTALS" in combined[3]
    assert "AMD: CONSTRUCTIVE FUNDAMENTALS" in combined[3]


def test_news_search_is_scoped_and_attributed_to_each_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries = []

    class FakeSearch:
        def text(self, query, **_kwargs):
            queries.append(query)
            scope = "business" if "industry business news" in query else "stock"
            ticker = next(
                (symbol for symbol in ("AMZN", "AMD") if symbol in query),
                "AMZN" if "cloud" in query else "AMD",
            )
            return [{
                "title": f"{ticker} {scope} quarterly results",
                "href": f"https://news.example/{ticker.lower()}/{scope}",
                "date": "2026-10-29",
                "body": f"{ticker} reported its {scope} quarterly results.",
            }]

    monkeypatch.setattr(workflow, "DDGS", lambda **_kwargs: FakeSearch())
    result = workflow.retrieve_news({
        "tickers": ["AMZN", "AMD"],
        "companies": [],
        "financial_metrics": {
            "AMZN": {"industry": "cloud computing"},
            "AMD": {"industry": "semiconductors"},
        },
    })

    assert len(queries) == 4
    assert sum("industry business news" in query for query in queries) == 2
    assert sum("stock latest news" in query for query in queries) == 2
    assert {tuple(article["related_tickers"]) for article in result["news_evidence"]} == {
        ("AMZN",), ("AMD",)
    }
    assert {article["news_scope"] for article in result["news_evidence"]} == {
        "stock", "business",
    }
    assert not result["news_errors"]


def test_report_citations_normalize_valid_ids_and_flag_unknown_ids() -> None:
    state = {
        "tickers": ["AMZN"],
        "market_summary": {},
        "financial_metrics": {},
        "news_evidence": [{
            "id": "news_1",
            "title": "Quarterly revenue",
            "source": "example.com",
            "published": "2026-10-29",
            "url": "https://example.com/revenue",
            "excerpt": "Quarterly revenue increased.",
        }],
    }
    report = workflow._normalize_news_citations(
        "Revenue increased【news_1】. AWS revenue increased 10%【News\u202f1】. "
        "Unsupported claim【news_9】.",
        state,
    )

    assert "Revenue increased[News 1]." in report
    assert "AWS revenue increased 10%[News 1]." in report
    assert "[Unsupported News 9]" in report
    warnings = workflow._validate_report(report, state)
    assert any("do not match retrieved evidence" in warning for warning in warnings)


def test_multi_ticker_report_rejects_cross_ticker_news_citations() -> None:
    state = {
        "tickers": ["AMZN", "AMD"],
        "market_summary": {},
        "financial_metrics": {},
        "news_evidence": [{
            "id": "news_1",
            "title": "AMD quarterly results",
            "source": "example.com",
            "published": "2026-10-29",
            "url": "https://example.com/amd",
            "excerpt": "AMD revenue increased.",
            "related_tickers": ["AMD"],
        }],
    }
    report = "BULLISH FACTORS\n\nAMZN revenue increased [News 1]."

    warnings = workflow._validate_report(report, state)
    assert any("not associated with that evidence" in warning for warning in warnings)


def test_company_overview_prefers_dated_statement_metrics_over_profile_values() -> None:
    state = {
        "tickers": ["AMZN"],
        "market_summary": {"AMZN": {"company_name": "Amazon", "close": 100.0, "date": "2026-10-29"}},
        "financial_metrics": {
            "AMZN": {
                "revenue_growth": 0.99,
                "profit_margin": 0.9,
                "free_cash_flow": 99_000_000_000,
                "quarterly_profit_margin": 0.1,
                "statement_metrics": {
                    "quarterly_revenue_growth_yoy": {
                        "value": 12.0,
                        "period_end": "2026-09-30",
                    },
                    "quarterly_profit_margin": {
                        "value": 0.1,
                        "period_end": "2026-09-30",
                        "source": "SEC EDGAR company facts",
                    },
                    "quarterly_free_cash_flow": {
                        "value": -50_000_000,
                        "period_end": "2026-09-30",
                    },
                },
            }
        },
        "news_evidence": [],
    }

    overview = workflow._company_overview_section(state)

    assert "Quarterly revenue growth year over year: +12.00%" in overview
    assert "Profit margin: 10.00%" in overview
    assert "Quarterly free cash flow: -$50.00M" in overview
    assert "Revenue growth: 99.00%" not in overview
    assert "Profit margin: 90.00%" not in overview
    assert "profile values with a different or unclear reporting date" in overview


def test_dashboard_searches_company_names_and_streams_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSearch:
        def __init__(self, query, **kwargs):
            assert query == "Amazon"
            assert kwargs["max_results"] == 10
            self.quotes = [
                {"symbol": "AMZN", "longname": "Amazon.com, Inc.", "quoteType": "EQUITY", "exchDisp": "NasdaqGS"},
                {"symbol": "AMZN", "longname": "Amazon duplicate", "quoteType": "EQUITY"},
                {"symbol": "AMZN.BOND", "longname": "Amazon Bond", "quoteType": "BOND"},
            ]

    monkeypatch.setattr(dashboard.workflow.yf, "Search", FakeSearch)
    candidates = dashboard.search_companies("Amazon")
    assert candidates == [{
        "symbol": "AMZN",
        "name": "Amazon.com, Inc.",
        "exchange": "NasdaqGS",
    }]
    with pytest.raises(ValueError):
        dashboard.search_companies(" ")

    calls = []

    class FakeGraph:
        def stream(self, inputs, stream_mode):
            calls.append((inputs, stream_mode))
            yield {"begin_retrieval": None}
            yield {"validate_input": {"tickers": [inputs["requested_ticker"]], "input_status": "valid"}}
            yield {"generate_report": {"report": f"Report for {inputs['user_question']}"}}

    monkeypatch.setattr(dashboard.workflow, "build_graph", lambda: FakeGraph())

    class FakeStatus:
        messages = []

        def write(self, message):
            self.messages.append(message)

    result = dashboard.run_ticker("Amazon.com, Inc.", "AMZN", "2y", FakeStatus())
    assert calls == [(
        {
            "user_question": "Research Amazon.com, Inc. stock",
            "company_name": "Amazon.com, Inc.",
            "requested_ticker": "AMZN",
            "period": "2y",
        },
        "updates",
    )]
    assert result["tickers"] == ["AMZN"]


def test_dashboard_renders_a_complete_report_without_external_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeElement:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __getattr__(self, name):
            def invoke(*args, **_kwargs):
                calls.append((name, args))
                if name in {"columns", "tabs"}:
                    count = args[0] if isinstance(args[0], int) else len(args[0])
                    return [FakeElement() for _ in range(count)]
                if name == "expander":
                    return FakeElement()
                return None

            return invoke

    monkeypatch.setattr(dashboard, "st", FakeElement())
    report = "\n\n".join(
        f"{section}\n\nVerified section text: $2.47B."
        for section in workflow.REQUIRED_SECTIONS
    )
    result = {
        "market_summary": {
            "AMZN": {
                "company_name": "Amazon",
                "close": 200.0,
                "date": "2026-10-29",
                "daily_change_percent": 1.0,
                "price_basis": "Unadjusted Close",
                "returns": {"1-year": {"return_percent": 10.0}},
            }
        },
        "financial_metrics": {
            "AMZN": {
                "company_name": "Amazon",
                "market_cap": 1_000_000_000,
                "pe": 25.0,
                "statement_metrics": {
                    "quarterly_revenue_growth_yoy": {"value": 0.1, "period_end": "2026-09-30"}
                },
            }
        },
        "news_evidence": [{
            "id": "news_1",
            "title": "Amazon results",
            "source": "example.com",
            "published": "2026-10-29",
            "excerpt": "Revenue increased.",
            "url": "https://example.com/amazon",
            "related_tickers": ["AMZN"],
        }],
        "report": report,
        "validation_status": "passed",
        "validation_warnings": [],
        "llm_usage": {"calls": 1},
    }

    dashboard.render_dashboard("AMZN", result)

    rendered_calls = {name for name, _args in calls}
    assert {"subheader", "success", "metric", "bar_chart", "dataframe", "link_button"} <= rendered_calls
    markdown_text = [
        args[0]
        for name, args in calls
        if name == "markdown" and args and isinstance(args[0], str)
    ]
    assert any("&#36;2.47B" in text for text in markdown_text)
    assert all("$2.47B" not in text for text in markdown_text)
    assert any("0 points (missing comparable data)" in text for text in markdown_text)


def test_currency_markdown_escaping_only_escapes_numeric_amounts() -> None:
    text = r"FCF was -$2.47B; range $240-247B; escaped \$3; variable $x$."

    assert dashboard._escape_currency_markdown(text) == (
        r"FCF was -&#36;2.47B; range &#36;240-247B; escaped \$3; variable $x$."
    )


def test_money_formatting_uses_currency_symbol() -> None:
    assert dashboard._money_with_currency(1_000_000_000, "INR") == "₹1.00B"
