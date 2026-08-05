"""Streamlit presentation layer for the quant-research-lab study.

Deliberately NOT an interactive "twiddle the hyperparameters and re-run"
tool — the whole point of a pre-registered, locked out-of-sample window is
that it's opened exactly once. This app reads and presents the actual,
immutable artifacts the runner scripts already produced
(preregistration/PREREGISTRATION.md, reports/*.md, reports/*.json,
reports/figures/*.png) rather than recomputing anything. The one exception
is the "Explore the Universe" tab, which live-fetches real price data for
browsing — clearly separated from, and never feeding back into, the frozen
verdict.

Run with: streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fetch_universe_prices
from src.features import FEATURE_COLUMNS, compute_daily_features
from src.universe import BENCHMARK_TICKER, TICKERS

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
PREREG = ROOT / "preregistration"

st.set_page_config(page_title="Quant Research Lab", page_icon="\U0001F52C", layout="wide")

st.title("\U0001F52C Quant Research Lab")
st.caption(
    "A pre-registered, falsification-first test of whether an ML signal beats a hand-crafted momentum "
    "baseline. This page presents the study's actual, immutable results — it does not let you re-run the "
    "locked out-of-sample test, since that would defeat the point of locking it. "
    "[Full source & pre-registration on GitHub](https://github.com/yinshanlow/quant-research-lab)."
)

primary_report = json.loads((REPORTS / "primary_report.json").read_text())
selected_config = primary_report["selected_config"]
verdict = primary_report["verdict"]

tabs = st.tabs([
    "Overview", "Pre-Registration", "Phase 1 · Selection", "Phase 2 · OOS Verdict",
    "Explore the Universe", "Reproduce",
])

# ---------------------------------------------------------------- Overview
with tabs[0]:
    st.header("Verdict")
    if verdict.startswith("NOT PROMOTED"):
        st.warning(f"**{verdict}**")
    elif verdict.startswith("PROMOTED"):
        st.success(f"**{verdict}**")
    else:
        st.error(f"**{verdict}**")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OOS window", primary_report["oos_window"])
    c2.metric("OOS months", primary_report["n_oos_months"])
    c3.metric("ML Sharpe (net of costs)", f"{primary_report['G2_beats_baseline']['ml_sharpe']:.3f}")
    c4.metric("Baseline Sharpe", f"{primary_report['G2_beats_baseline']['baseline_sharpe']:.3f}")

    st.markdown(
        """
**H1:** a gradient-boosted (XGBoost) model combining five standard cross-sectional equity factors
(1/3/6-month momentum, 21-day volatility, 5-day reversal) produces genuine out-of-sample predictive skill
for next-month returns, beyond a single, unfit, hand-crafted 3-month momentum rule, net of a 10bps
round-trip cost — tested on 30 large-cap US equities across 6 sectors, 2008–2026.

This retests a finding from the sibling
[`quant-finance-toolkit`](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit)
repo, where the same comparison ran on 4 tickers over 5 years — too small a sample to trust either way.
Design (universe, factor definitions, hyperparameter grid, promotion gates) was frozen and committed to
git **before** any real price data entered the repository.
        """
    )

    st.subheader("The four promotion gates")
    gate_rows = [
        {"Gate": "G1 · IC significance", "Result": f"mean IC {primary_report['G1_ic_significance']['mean_ic']:.4f}, "
         f"95% CI [{primary_report['G1_ic_significance']['ci'][0]:.4f}, {primary_report['G1_ic_significance']['ci'][1]:.4f}]",
         "Pass?": "✅" if primary_report["G1_ic_significance"]["pass"] else "❌"},
        {"Gate": "G2 · Beats baseline", "Result": f"ML {primary_report['G2_beats_baseline']['ml_sharpe']:.3f} vs. "
         f"baseline {primary_report['G2_beats_baseline']['baseline_sharpe']:.3f}",
         "Pass?": "✅" if primary_report["G2_beats_baseline"]["pass"] else "❌"},
        {"Gate": "G3 · Paired significance", "Result": f"mean Δreturn {primary_report['G3_paired_significance']['mean_diff']:.5f}, "
         f"95% CI [{primary_report['G3_paired_significance']['ci'][0]:.5f}, {primary_report['G3_paired_significance']['ci'][1]:.5f}]",
         "Pass?": "✅" if primary_report["G3_paired_significance"]["pass"] else "❌"},
        {"Gate": "G4 · Permutation validity", "Result": f"permuted IC 95% CI [{primary_report['G4_permutation_validity']['ci'][0]:.4f}, "
         f"{primary_report['G4_permutation_validity']['ci'][1]:.4f}] (should include 0)",
         "Pass?": "✅ (no leakage)" if primary_report["G4_permutation_validity"]["pass"] else "❌ INVESTIGATE"},
    ]
    st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)
    st.caption(
        "Promotion requires G1 AND G2 AND G3, with G4 required as a validity precondition. "
        "G4 passing is what makes the null result trustworthy rather than a possible symptom of a leakage bug."
    )

# ---------------------------------------------------------------- Pre-Registration
with tabs[1]:
    st.header("The frozen design")
    st.info(
        "Committed to git **before** any real price data entered this repository. "
        "`DEVIATIONS.md` is the append-only log for any departure — empty means nothing changed after this freeze."
    )
    st.markdown((PREREG / "PREREGISTRATION.md").read_text())
    with st.expander("DEVIATIONS.md"):
        st.markdown((PREREG / "DEVIATIONS.md").read_text())

# ---------------------------------------------------------------- Phase 1
with tabs[2]:
    st.header("Phase 1 · Hyperparameter Selection (IS-only, 2017–2020)")
    st.caption("This stage never touches 2021+ (OOS) data.")
    st.markdown((REPORTS / "SELECTION_REPORT.md").read_text())

# ---------------------------------------------------------------- Phase 2
with tabs[3]:
    st.header("Phase 2 · The Locked Out-of-Sample Test")
    st.caption(f"Selected config: **{selected_config['selected_config_id']}** = `{selected_config['params']}` — {selected_config['selection_note']}")

    fig_col1, fig_col2 = st.columns(2)
    fig_col1.image(str(REPORTS / "figures" / "oos_equity_curves.png"), caption="Figure 1 — OOS equity curves, net of costs")
    fig_col2.image(str(REPORTS / "figures" / "oos_ic_comparison.png"), caption="Figure 2 — OOS Information Coefficient")

    st.markdown(
        """
**An honest nuance worth surfacing, not smoothing over:** the raw OOS mean IC is actually *higher* for the
ML signal (0.022) than for the baseline (0.006) — but neither CI excludes zero, and the ML signal's
realized long-short portfolio Sharpe is *worse* than the baseline's despite the nominally higher IC. IC
measures rank correlation across the whole cross-section; the tercile long-short book only cares about
getting the extremes right. That's exactly why this study pre-registered three different gates rather than
picking whichever single metric told the nicer story after the fact.
        """
    )
    st.markdown((REPORTS / "PRIMARY_REPORT.md").read_text())

# ---------------------------------------------------------------- Explore
with tabs[4]:
    st.header("Explore the Universe")
    st.caption(
        "Live data, browsed for context only — this tab does not feed into, and cannot change, the frozen "
        "verdict above. Fetched fresh from Yahoo Finance and cached for this session."
    )

    @st.cache_data(show_spinner="Fetching universe prices (Yahoo Finance)...")
    def _load_prices():
        return fetch_universe_prices(TICKERS + [BENCHMARK_TICKER], range_="20y", interval="1d")

    prices = _load_prices()
    st.success(f"Loaded {prices.shape[0]} common trading days, {prices.index.min().date()} to {prices.index.max().date()}, {len(TICKERS)} tickers + benchmark.")

    ticker = st.selectbox("Pick a ticker", TICKERS)
    st.line_chart(prices[ticker], height=300)

    daily_features = compute_daily_features(prices[TICKERS])
    latest_date = prices.index[-1]
    latest_row = {name: daily_features[name].loc[latest_date] for name in FEATURE_COLUMNS}
    latest_df = pd.DataFrame(latest_row)
    latest_df.index.name = "ticker"

    st.subheader(f"Latest causal factor values — {latest_date.date()}")
    st.dataframe(latest_df.style.format("{:.4f}"), use_container_width=True)
    st.caption(
        "These are the same five factors — computed the same causal, trailing-window way — that feed both "
        "the baseline and ML signals in the frozen study. Shown here purely for illustration of what the "
        "pipeline actually computes."
    )

# ---------------------------------------------------------------- Reproduce
with tabs[5]:
    st.header("Reproduce this study locally")
    st.code(
        """git clone https://github.com/yinshanlow/quant-research-lab.git
cd quant-research-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -v                      # 18 tests, synthetic fixtures, a few seconds

python -m runners.run_00_fetch_data             # pulls real Yahoo Finance data (~10s)
python -m runners.run_01_select_hyperparams     # IS-only, 2017-2020
python -m runners.run_02_oos_promotion          # THE locked OOS test — opens 2021+ once
python -m runners.run_03_figures""",
        language="bash",
    )
    st.markdown(
        """
Sibling project: [`quant-finance-toolkit`](https://github.com/yinshanlow/data_ai_project/tree/main/quant-finance-toolkit)
covers options pricing, portfolio risk, and AI-augmented research more broadly — this repo goes deep on a
single hypothesis with the falsification discipline a breadth-first toolkit can't fit into one module.
        """
    )
