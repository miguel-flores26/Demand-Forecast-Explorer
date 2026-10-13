# Demand Forecast Explorer

[![tests](https://github.com/miguel-flores26/demand-forecast-explorer/actions/workflows/tests.yml/badge.svg)](https://github.com/miguel-flores26/demand-forecast-explorer/actions/workflows/tests.yml)

**Live app: [demand-forecast-explorer.streamlit.app](https://demand-forecast-explorer.streamlit.app/)**
(free hosting: if the app has been idle, click "Yes, get this app back up!" and wait about a minute)

An interactive web app for a common planning question:
**how much demand should we plan for, how uncertain is that, and does a more complex model
actually beat a simple one?**

- Pick one of seven U.S. retail categories, or **upload your own monthly, weekly, or daily
  sales CSV**.
- Compare three forecasting methods side by side, each with a prediction interval.
- See a **rolling backtest**: how each method would have performed historically, whether it
  beat a simple baseline, and whether its uncertainty ranges were trustworthy.
- Download the forecasts as CSV.

## Why this exists

Many forecasting demos fit one model and plot a confident line into the future. This app
asks whether the model would have been right in the past, whether it beats a simple
"same month last year" baseline, and whether its stated uncertainty holds up.
Sometimes the simple baseline wins, and the app says so.

## Results

Rolling-origin backtest on each built-in series: 12 forecast origins from Oct 2022 to
Jul 2025, 3 months apart, each forecasting 12 months ahead (so forecasts are checked against
actuals through Jul 2026). Data retrieved from FRED in October 2026.

**Forecast accuracy: mean absolute percentage error (MAPE), and change in mean absolute error
(MAE) relative to the seasonal naive baseline** (negative = more error than the baseline)

| Category | Seasonal naive | ETS | SARIMA | Most accurate |
|---|---|---|---|---|
| Total retail and food services | 3.4% | 1.5% (−56%) | 1.7% (−50%) | ETS |
| Grocery stores | 2.2% | 1.5% (−32%) | 1.8% (−18%) | ETS |
| Clothing and accessories | 3.7% | 5.4% (+45%) | 6.3% (+68%) | Seasonal naive |
| General merchandise | 2.5% | 2.1% (−18%) | 2.4% (−5%) | ETS |
| Building materials and garden | 3.6% | 3.8% (+6%) | 4.7% (+31%) | Seasonal naive |
| Electronic shopping and mail-order | 8.3% | 2.2% (−73%) | 2.6% (−69%) | ETS |
| Food services and drinking places | 6.2% | 3.6% (−40%) | 2.1% (−66%) | SARIMA |

**80% prediction interval coverage** (share of actual values that fell inside the interval;
about 80% is ideal)

| Category | Seasonal naive | ETS | SARIMA |
|---|---|---|---|
| Total retail and food services | 93% | 99% | 98% |
| Grocery stores | 88% | 88% | 81% |
| Clothing and accessories | 100% | 100% | 100% |
| General merchandise | 81% | 81% | 77% |
| Building materials and garden | 88% | 85% | 77% |
| Electronic shopping and mail-order | 19% | 100% | 99% |
| Food services and drinking places | 72% | 99% | 99% |

**What this shows**

- **No single method wins everywhere.** ETS or SARIMA reduced error relative to the baseline
  in 5 of 7 categories, sometimes substantially. In clothing and building materials, the
  simple baseline was more accurate than both.
- **A possible explanation**, not tested here: the models gained most where sales grew
  steadily (e-commerce, food services), which "same month last year" cannot anticipate, and
  did worst in categories where an earlier trend may have reversed (clothing, building
  materials). Confirming this would need further analysis.
- **Interval calibration is uneven.** The model intervals were often wider than needed
  (98–100% coverage in four categories). A likely contributor is the 2020–2021 COVID
  disruption in the training data, which inflates the models' estimate of how noisy sales
  are. The baseline's intervals were far too narrow for fast-growing e-commerce (19%).

These results cover one evaluation period on seven aggregated national series. They are a
careful comparison, not evidence about which method is best in general.

## Data

- **Built-in series:** U.S. Census Bureau, Monthly Retail Trade Survey, monthly sales by kind
  of business in millions of dollars, retrieved from
  [FRED](https://fred.stlouisfed.org/) (Federal Reserve Bank of St. Louis). Real, public data.
  - Seven categories: total retail and food services, grocery, clothing, general merchandise,
    building materials, electronic shopping/mail-order, and food services and drinking places.
  - **Not seasonally adjusted**, on purpose: modeling seasonality is part of the forecasting task.
  - Retrieved as FRED's public CSV download, so no API key is needed. The app refreshes data
    at most once a day.
  - Categories follow the NAICS classification used by these FRED series. The Census Bureau
    is moving its surveys to the 2022 NAICS, which may eventually change or end some series.
- **User uploads:** any CSV with a date column and a numeric value column (see below).

## Methods

| Method | Specification | Role |
|---|---|---|
| Seasonal naive | Each month = same month one year earlier | Baseline every other model must beat |
| Exponential smoothing | ETS(A, Ad, A): additive error, damped additive trend, additive seasonality, on log(sales) | Standard trend + seasonality method |
| Seasonal ARIMA | ARIMA(0,1,1)(0,1,1)[12] ("airline" model), on log(sales) | Classical statistical model |

All three return a point forecast and a prediction interval (80% or 95%).

- **Log scale.** ETS and SARIMA are fitted to log(sales) because retail seasonal swings grow
  with the level of sales. Back-transforming keeps forecasts positive, but it means their
  point forecasts are medians rather than means.
- **Fixed specifications.** SARIMA uses one fixed, well-known specification rather than
  automatic order selection per series. That keeps it fast and predictable, but it may not be
  the best-fitting ARIMA for every series.
- **Baseline intervals** use the standard seasonal-naive formula from Hyndman & Athanasopoulos,
  *Forecasting: Principles and Practice*.

### Evaluation: rolling-origin backtest

Each method is refit from several past "origins" using only data available at that point,
then its forecasts are compared with what actually happened. By default: 12 origins, spaced
3 months apart and ending at the latest month that leaves a full horizon of actuals. Shorter
uploaded series use fewer origins, and the app says so.

Reported per method:
- **MAE, MAPE, RMSE.** Average error in units, in percent, and with large misses weighted more.
- **Error vs. baseline.** How much lower (or higher) the method's MAE is than seasonal naive's.
  A method has to cut MAE by at least 5% to be reported as beating the baseline; smaller
  differences are reported as a tie. This threshold is a rule of thumb, not a significance test.
- **Interval coverage.** How often the actual value fell inside the prediction interval.

A test confirms that no method ever sees data from after its origin.

## Uploading your own data

The app accepts a CSV with one date column and one value column (it guesses which is which,
and you can override the guess). It handles common date formats, values like `$1,200`,
comma/semicolon/tab delimiters, and UTF-8 or Latin-1 encodings. It then:

- sums daily or weekly data into monthly totals, dropping incomplete first and last months;
- sums multiple rows for the same month;
- skips a small number of unreadable rows (up to 10%);
- fills up to 3 missing months by straight-line interpolation.

Every adjustment is listed on the page. Files that can't be used safely are refused with an
explanation: fewer than 36 months, quarterly or yearly data, zero or negative monthly totals,
or larger gaps. Uploaded files are not written to disk.

## Limitations

- Forecasts assume past patterns continue; they cannot anticipate shocks.
- The built-in series are national totals. A single business's sales can behave very
  differently.
- Dollar sales mix price and volume, so growth partly reflects inflation.
- **COVID-19.** The backtest origins start in Oct 2022, so the 2020–2021 disruption appears
  only in the training data, not in the evaluation windows. It still affects the fitted
  models, most visibly through wider prediction intervals.
- On a few backtest origins, the SARIMA optimizer stops with a convergence warning (a
  line-search stop, not an iteration limit). The estimates at those points were inspected and
  look typical for this model, but they are not guaranteed to be optimal.
- Results describe one evaluation period; other periods could rank the methods differently.
- This is a portfolio project, not a production forecasting system or financial advice.

## Project structure

```
app.py                         # Streamlit web app (layout and controls)
forecast_explorer/
  data.py                      # load and validate FRED retail series (with local caching)
  models.py                    # seasonal naive, ETS, SARIMA behind one shared interface
  backtest.py                  # rolling-origin backtest, metrics, plain-language verdict
  upload.py                    # CSV reading, cleaning, and validation for user uploads
  app_helpers.py               # tables and formatting for the app (testable without Streamlit)
  charts.py                    # Plotly charts
tests/                         # pytest tests (104), run on every push and pull request
.streamlit/config.toml         # app settings (upload size limit, headless mode)
.github/workflows/tests.yml    # GitHub Actions: lint (ruff) + tests (pytest)
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
```

To launch the app:

```bash
streamlit run app.py
```

## Development workflow

Features were built on separate branches and merged through pull requests. Every push runs
linting (`ruff`) and the test suite (`pytest`) via GitHub Actions. Several bugs were caught by
running the code on live data and fixed with regression tests (for example, an integer/float
type issue in the loader and a chart tooltip that mixed values from different months).

## How this was built

Developed with the help of an AI coding assistant. I reviewed and ran all code, tested it on
live data, investigated issues such as the SARIMA convergence warnings, and made and
documented the design decisions above.
