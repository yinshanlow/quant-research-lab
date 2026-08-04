"""Phase 3 — regenerates the report figures from the already-computed, immutable
Phase 1/2 artifacts. Written before Phase 2 was ever run, so the choice of
what to plot and how cannot have been influenced by which way the OOS
verdict came out.

Run: python -m runners.run_03_figures
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import long_short_returns
from src.baseline import baseline_signal
from src.ml_signal import walk_forward_predict
from src.stats import spearman_ic_by_date

ROOT = Path(__file__).resolve().parent.parent
PANEL_CACHE = ROOT / "data" / "monthly_panel.parquet"
SELECTED_CONFIG_PATH = ROOT / "reports" / "selected_config.json"
FIG_DIR = ROOT / "reports" / "figures"
COST_BPS = 10.0
OOS_START_YEAR = 2021

if __name__ == "__main__":
    panel = pd.read_parquet(PANEL_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])
    oos_end_year = int(panel["date"].dt.year.max())

    selected = json.loads(SELECTED_CONFIG_PATH.read_text())
    params = selected["params"]

    preds_full = walk_forward_predict(panel, params, OOS_START_YEAR, oos_end_year, min_train_years=3)
    valid_index = preds_full.index[preds_full.notna()]
    oos_panel = panel.loc[valid_index].reset_index(drop=True)
    ml_signal_oos = pd.Series(preds_full.loc[valid_index].values)
    baseline_signal_oos = baseline_signal(oos_panel)

    ml_returns = long_short_returns(oos_panel, ml_signal_oos, cost_bps=COST_BPS)
    baseline_returns = long_short_returns(oos_panel, baseline_signal_oos, cost_bps=COST_BPS)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: OOS equity curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    (1 + ml_returns).cumprod().plot(ax=ax, label="ML signal (long/short)", color="steelblue")
    (1 + baseline_returns).cumprod().plot(ax=ax, label="Baseline (3-month momentum)", color="darkorange")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Growth of $1 (net of costs)")
    ax.set_title("OOS long/short equity curves — ML signal vs. baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "oos_equity_curves.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: OOS IC comparison, ML vs baseline, with bootstrap CIs ---
    from src.stats import block_bootstrap_ci

    ml_ic = spearman_ic_by_date(oos_panel, ml_signal_oos)
    baseline_ic = spearman_ic_by_date(oos_panel, baseline_signal_oos)
    ml_mean, ml_lo, ml_hi = block_bootstrap_ci(ml_ic, seed=1)
    base_mean, base_lo, base_hi = block_bootstrap_ci(baseline_ic, seed=2)

    fig, ax = plt.subplots(figsize=(6, 5))
    labels = ["Baseline\n(3-month momentum)", "ML signal\n(XGBoost)"]
    means = [base_mean, ml_mean]
    errs = [[base_mean - base_lo, ml_mean - ml_lo], [base_hi - base_mean, ml_hi - ml_mean]]
    ax.bar(labels, means, yerr=errs, capsize=6, color=["darkorange", "steelblue"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean OOS monthly IC (with 95% bootstrap CI)")
    ax.set_title("Out-of-sample Information Coefficient")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "oos_ic_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures to {FIG_DIR}")
