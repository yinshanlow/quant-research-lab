# Pre-registration — ML-Augmented Cross-Sectional Momentum vs. a Hand-Crafted Baseline

**Status: frozen.** This document is committed to git before any real market
data has entered this repository and before any in-sample or out-of-sample
result has been computed. `DEVIATIONS.md` is the append-only log for any
departure from what's written here — nothing below is edited silently after
this commit.

## 1. Hypothesis

**H1:** A gradient-boosted (XGBoost) model combining five standard
cross-sectional equity factors produces genuine out-of-sample predictive
skill for next-month stock returns, beyond what a single, unfit, hand-crafted
3-month momentum rule achieves, net of realistic transaction costs.

This directly retests a finding from the sibling `quant-finance-toolkit`
repo's `ai_augmented/signal_model.py`, where an XGBoost signal did **not**
beat a hand-crafted baseline on a 4-ticker, 5-year sample — too small a
universe and sample to draw a confident conclusion either way. This study
exists to test the same question properly: a real cross-sectional universe,
pre-registered gates, and the falsification discipline this repo is built
around, rather than a single train/test split.

## 2. Universe (fixed, pre-declared, screened before any return data was inspected)

30 large-cap US equities, screened only for (a) being a real, still-listed
company and (b) having Yahoo Finance daily price history back to at least
2007 (verified via a row-count/date-range feasibility check on 2026-08-05,
**before** any factor, return, or performance computation ran) — never
screened by past performance. Six sectors, five names each:

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, ORCL, IBM, INTC |
| Financials | JPM, BAC, WFC, GS, C |
| Healthcare | JNJ, PFE, MRK, ABT, UNH |
| Energy | XOM, CVX, COP, SLB, CSCO* |
| Consumer | PG, KO, PEP, WMT, HD |
| Industrials | MCD, BA, CAT, GE, HON |

*CSCO is grouped under Technology in spirit but listed once, in the Energy
row, purely to keep the table at 5 names/sector without a 31st slot — this
is a labeling artifact only and changes nothing about which 30 tickers are
in the universe (see `src/universe.py`, the single source of truth).

**SPY** is fetched alongside the universe as a diagnostic-only benchmark. It
is never a candidate in the long/short book and never enters the signal or
label. Its only use is an optional, non-gating context chart in the final
report (e.g. confirming the long-short book isn't secretly a leveraged
market-beta bet).

No tickers are added, removed, or substituted after this commit.

## 3. Sample window

- Prices fetched: full available daily history up to the fetch date
  (2026-08-05), back to the earliest common date across all 31 tickers
  (2006-08-04 per the feasibility check).
- Monthly panel built at each calendar month-end, factors computed causally
  (trailing windows only — see §4), from the month-end where every factor's
  lookback (max 126 trading days) is fully populated.
- **In-sample (IS): 2008-01 through 2020-12** (13 years — deliberately
  includes the 2008 financial crisis).
- **Out-of-sample (OOS): 2021-01 through the latest available month**
  (includes the 2022 rate-hiking bear market) — **locked, single-use**,
  opened only after every gate below is finalized and the promotion
  decision logic is committed in code.

## 4. Signals

All five factors are computed from trailing price history only, at each
month-end date, using only data up to and including that date
(`src/features.py::compute_daily_features` — every window is `.rolling(n)`
or `.shift(n)`, never centered or forward-looking). Truncation-invariance is
enforced by `tests/test_features_no_lookahead.py`, not merely asserted.

| Factor | Definition |
|---|---|
| `mom_21` | 21-trading-day (≈1 month) price return |
| `mom_63` | 63-trading-day (≈3 month) price return |
| `mom_126` | 126-trading-day (≈6 month) price return |
| `vol_21` | 21-day realized volatility of daily returns |
| `reversal_5` | negative 5-day return (higher = more recently "oversold") |

**Label:** forward 21-trading-day return from month-end t to t+21 trading
days (`src/features.py::build_monthly_panel`'s `fwd_return` column).

**Baseline signal:** the cross-sectional z-score of `mom_63` alone
(`src/baseline.py`). Zero fitted parameters — cannot overfit by
construction. This is the bar H1's ML signal must clear.

## 5. ML model and hyperparameter search

**Model:** `XGBRegressor`, predicting `fwd_return` from the five factors in
§4, `objective="reg:squarederror"`.

**Hyperparameter grid — the entire search space, nothing outside this grid
is tried at any point in this study:**

| Config | max_depth | n_estimators | learning_rate |
|---|---|---|---|
| A | 2 | 100 | 0.05 |
| B | 2 | 200 | 0.05 |
| C | 3 | 100 | 0.05 |
| D | 3 | 200 | 0.05 |

**Walk-forward refit protocol (`src/ml_signal.py::walk_forward_predict`):**
refit annually on an expanding window using all panel rows strictly before
January 1 of the predict year (`date < cutoff`), then predict every month in
that year before rolling forward. Minimum 3 years of training history
required before any prediction is made.

**Hyperparameter selection (in-sample only, never touches OOS):** all four
configs are walk-forward evaluated on an **inner IS validation window,
2017-01 through 2020-12** (trained on an expanding window starting from
2008, exactly as OOS will be). For each config, compute the monthly
cross-sectional Spearman IC series over that window, its bootstrap p-value
(`src/stats.py::bootstrap_p_value`, 10,000 resamples, monthly block), and
apply Benjamini-Hochberg FDR at **q = 0.10** across the four p-values. The
config selection rule:

1. If one or more configs survive BH-FDR, select the single **highest-IC**
   config among the survivors.
2. If none survive BH-FDR, select the single highest-IC config anyway, but
   the final report must state plainly that no configuration cleared
   selection-stage significance — this does not block the OOS test (the
   promotion gates in §6 are what decide the actual verdict), it only
   affects how much weight the write-up gives the "some config might have
   been better than baseline in-sample" framing.

Only the single selected config is walk-forward refit and evaluated on OOS.
No further configs are tried after OOS is opened.

## 6. Portfolio construction and costs

Monthly rebalance, dollar-neutral tercile long-short
(`src/backtest.py::long_short_returns`): rank the universe by the signal
each month-end, long the top tercile / short the bottom tercile, equal
weight within each leg (long leg sums to +1.0, short leg to −1.0). A flat
**10 bps round-trip cost** is charged on each ticker's turnover (weight
change from the prior month) every rebalance — a conservative estimate for
this liquid large-cap universe.

## 7. Promotion gates (evaluated on OOS, 2021-01 onward, exactly once)

H1 is **promoted** (judged to show genuine incremental edge over the
baseline) only if **all** of G1, G2, and G3 pass, with G4 required to pass
as a validity precondition:

- **G1 — IC significance.** The OOS monthly Spearman IC series' block
  bootstrap (10,000 resamples, resampling whole months) 95% CI excludes
  zero.
- **G2 — Beats baseline.** The OOS long-short portfolio's annualized Sharpe
  ratio (net of costs) exceeds the baseline signal's OOS Sharpe ratio,
  computed on the identical OOS months and cost model.
- **G3 — Paired significance.** A paired bootstrap (`src/stats.py::paired_bootstrap_diff_ci`,
  resampling months, preserving the pairing) of the monthly return
  difference (ML portfolio − baseline portfolio) gives a 95% CI that
  excludes zero.
- **G4 — Leakage/validity sanity check.** A label-permutation null — shuffle
  `fwd_return` **within each date's cross-section** (never across dates)
  before every walk-forward refit, using the selected config — must produce
  an OOS IC whose bootstrap 95% CI **includes** zero. If G4 fails (the
  permuted-label model also shows significant IC), that indicates a leakage
  bug in the pipeline; no promotion decision can be trusted regardless of
  G1-G3, and the study is halted pending investigation, logged in
  `DEVIATIONS.md`.

**If H1 is not promoted, that is a complete, reportable, first-class result**
— the honest re-test of the toolkit's original finding, now with a real
cross-sectional universe and proper statistical discipline behind it, either
direction.

## 8. What this study deliberately does not test

- No claim about small/mid-cap, non-US, or non-equity universes.
- No claim about horizons other than the 21-trading-day forward window.
- No intraday execution modeling — costs are a flat turnover-based estimate,
  not a fill simulation.
- No Deflated Sharpe Ratio — only one final configuration is ever evaluated
  on OOS (hyperparameter selection happens entirely pre-OOS, gated by
  BH-FDR at the selection stage), so there is only one OOS "trial" to
  deflate, which a single bootstrap CI already covers honestly.
- The five factors in §4 are a fixed, standard, off-the-shelf set — not
  claimed to be an exhaustive or optimal factor library.

## 9. Reproducibility commitment

Every source module (`src/`) and the full test suite (`tests/`, synthetic
fixtures only, no real data) is committed in the **same commit** as this
document, before `runners/run_00_data.py` is ever executed. The git history
is the audit trail that no result influenced this design.
