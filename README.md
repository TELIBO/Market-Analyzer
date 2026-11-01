# Stock research dashboard

Install the workflow and dashboard dependencies from this folder:

```powershell
python -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in the project `.env` file, then start the dashboard:

```powershell
python -m streamlit run streamlit_app.py
```

Enter a company name (for example, `Amazon` or `NVIDIA`) in the dashboard. Select the matching public listing from Yahoo Finance search results, then run the research. The selected listing resolves the company name to its market symbol; users do not need to know or enter a ticker. The dashboard streams completed workflow stages and shows market data, dated financial metrics, cited news, validation notes, and the report.

The workflow remains importable from `stock_research_workflow.py` for compatibility. Statement-based financial normalization lives in `stock_research/financials.py`, while SEC EDGAR retrieval and dated-source reconciliation live in `stock_research/sec.py`.

The summary leads with a plain-language analyst thesis, followed by the evidence that supports it and the main counterpoints. Interpretations should connect reported facts to their business significance; missing or incomparable data are called out rather than hidden behind generic “what this means” text.

The financial screen prefers comparable, sufficiently recent, dated statement values, including annual values only when appropriate; undated profile growth, margin, and cash-flow values are omitted from scoring rather than presented as current. Its weighted fundamental score explains revenue growth (±1), operating-income growth (±1 for a modest change, ±2 at 15% growth/decline), profit margin (±1), and free cash flow (±2 for trailing 12 months or ±1 for quarterly fallback). The range is -6 to +6 with TTM free cash flow or -5 to +5 with quarterly fallback; available measures and dated-data coverage are shown separately. When fewer than three measures or fewer than two dated measures are available, the result is labeled limited coverage instead of a directional screen. Valuation is shown separately and is not scored because the workflow does not currently compare P/E with peers or historical ranges. This screen is not a buy/sell recommendation.

Yahoo Finance via yfinance remains the broad-coverage data source. For U.S. SEC filers, official SEC EDGAR company facts are also retrieved when configured. Set `SEC_USER_AGENT` in the environment (or project `.env`) to an identifying application/organization name and a contact email, as required by the SEC. When SEC facts provide the dated core metrics, the workflow avoids redundant Yahoo statement requests while retaining Yahoo for market prices and profile/company context. Statement-period end dates are compared per metric: the newest sufficiently recent dated value is used, preferring SEC when sources report the same metric for the same period, and period/source differences are reported. SEC cash-flow facts are converted from year-to-date values to standalone quarters when needed, including deriving Q4 from full-year less year-to-date data. SEC TTM free cash flow is calculated from four consecutive quarterly operating cash flow values less reported payments to acquire productive assets. When a dated company-owned release reports issuer-defined TTM free cash flow for the same filing period, the report shows both definitions and uses the issuer-reported value for the FCF score; it describes known definition differences without presenting the reconciliation gap as a verified proceeds amount. If no matching official figure is found, the SEC calculation remains in use and the limitation is disclosed. The workflow uses TTM values rather than treating a single quarter as TTM. If a newer quarter exists but the data does not support refreshing TTM, that mismatch is called out. Debt-to-equity uses debt and equity from a shared balance-sheet date and labels that date; when no SEC aggregate debt fact exists, debt components are combined only for matching dates. News retrieval runs company-specific and business-area searches in parallel, targets three company articles and two business-area articles per ticker, filters dated articles older than 45 days, and deduplicates results without merging evidence across unrelated tickers or scopes. Undated news may still be included when no publication date is available. Search and official-page requests use bounded timeouts, and official company news pages have bounded response sizes to recover publication dates and focused excerpts; generic financial-statement directory pages are excluded. If SEC verification is not configured or available, the report explicitly says so and identifies the Yahoo statement period used. SEC EDGAR does not cover every non-U.S. listing; those require the relevant local regulator or company investor-relations source, so Yahoo remains the fallback. Do not commit the `.env` file.

Install `requirements-dev.txt` to run the mocked workflow tests without using external API quota.

Run the mocked regression suite with:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
