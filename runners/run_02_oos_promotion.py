"""Phase 2 — THE locked out-of-sample test, opened exactly once.

Refits the single hyperparameter config selected in Phase 1 (IS-only) walk-
forward across 2021+, evaluates the four pre-registered promotion gates
(PREREGISTRATION.md §7), and writes the final, immutable verdict to
reports/PRIMARY_REPORT.md. This script is run once; if it is ever re-run
with different code after being read, that is itself a deviation and must be
logged in preregistration/DEVIATIONS.md before re-running.

Run: python -m runners.run_02_oos_promotion
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import long_short_returns
from src.baseline import baseline_signal
from src.ml_signal import walk_forward_predict
from src.stats import annualized_sharpe, block_bootstrap_ci, paired_bootstrap_diff_ci, spearman_ic_by_date

ROOT = Path(__file__).resolve().parent.parent
PANEL_CACHE = ROOT / "data" / "monthly_panel.parquet"
SELECTED_CONFIG_PATH = ROOT / "reports" / "selected_config.json"
COST_BPS = 10.0
OOS_START_YEAR = 2021
PERMUTATION_SEED = 20210101


def _predict_and_align(panel: pd.DataFrame, params: dict, oos_start: int, oos_end: int, permute_seed: int | None = None):
    preds_full = walk_forward_predict(panel, params, oos_start, oos_end, min_train_years=3, permute_labels_seed=permute_seed)
    valid_index = preds_full.index[preds_full.notna()]
    aligned_panel = panel.loc[valid_index].reset_index(drop=True)
    aligned_signal = pd.Series(preds_full.loc[valid_index].values)
    return aligned_panel, aligned_signal


if __name__ == "__main__":
    panel = pd.read_parquet(PANEL_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])
    oos_end_year = int(panel["date"].dt.year.max())

    selected = json.loads(SELECTED_CONFIG_PATH.read_text())
    params = selected["params"]
    print(f"Selected config: {selected['config_id']} = {params} ({selected['selection_note']})")

    # --- ML signal, walk-forward across OOS ---
    oos_panel, ml_signal_oos = _predict_and_align(panel, params, OOS_START_YEAR, oos_end_year)
    n_oos_months = oos_panel["date"].nunique()
    print(f"OOS window: {oos_panel['date'].min().date()} to {oos_panel['date'].max().date()} ({n_oos_months} months)")

    # --- Baseline signal, same OOS rows, same dates ---
    baseline_signal_oos = baseline_signal(oos_panel)

    # --- G1: OOS IC significance ---
    ml_ic_series = spearman_ic_by_date(oos_panel, ml_signal_oos)
    g1_mean, g1_lo, g1_hi = block_bootstrap_ci(ml_ic_series, n_boot=10_000, seed=1)
    g1_pass = not (g1_lo <= 0 <= g1_hi)

    # --- G2: OOS Sharpe vs baseline Sharpe ---
    ml_returns = long_short_returns(oos_panel, ml_signal_oos, cost_bps=COST_BPS)
    baseline_returns = long_short_returns(oos_panel, baseline_signal_oos, cost_bps=COST_BPS)
    ml_sharpe = annualized_sharpe(ml_returns)
    baseline_sharpe = annualized_sharpe(baseline_returns)
    g2_pass = ml_sharpe > baseline_sharpe

    # --- G3: paired bootstrap on the return difference ---
    g3_mean, g3_lo, g3_hi = paired_bootstrap_diff_ci(ml_returns, baseline_returns, n_boot=10_000, seed=2)
    g3_pass = not (g3_lo <= 0 <= g3_hi)

    # --- G4: label-permutation validity sanity check ---
    perm_panel, perm_signal = _predict_and_align(panel, params, OOS_START_YEAR, oos_end_year, permute_seed=PERMUTATION_SEED)
    perm_ic_series = spearman_ic_by_date(perm_panel, perm_signal)
    g4_mean, g4_lo, g4_hi = block_bootstrap_ci(perm_ic_series, n_boot=10_000, seed=3)
    g4_pass = g4_lo <= 0 <= g4_hi  # PASS means the permuted model shows NO skill, as it should

    if not g4_pass:
        verdict = "INVALID — G4 (leakage sanity check) failed; pipeline requires investigation before any verdict can be trusted"
    elif g1_pass and g2_pass and g3_pass:
        verdict = "PROMOTED — H1 confirmed: the ML signal shows genuine incremental OOS edge over the baseline"
    else:
        verdict = "NOT PROMOTED — H1 not confirmed on this OOS sample"

    report = {
        "selected_config": selected, "oos_window": f"{oos_panel['date'].min().date()} to {oos_panel['date'].max().date()}",
        "n_oos_months": n_oos_months,
        "G1_ic_significance": {"mean_ic": g1_mean, "ci": [g1_lo, g1_hi], "pass": g1_pass},
        "G2_beats_baseline": {"ml_sharpe": ml_sharpe, "baseline_sharpe": baseline_sharpe, "pass": g2_pass},
        "G3_paired_significance": {"mean_diff": g3_mean, "ci": [g3_lo, g3_hi], "pass": g3_pass},
        "G4_permutation_validity": {"mean_ic": g4_mean, "ci": [g4_lo, g4_hi], "pass": g4_pass},
        "verdict": verdict,
    }

    report_json_path = ROOT / "reports" / "primary_report.json"
    report_json_path.write_text(json.dumps(report, indent=2, default=float))

    lines = [
        "# Phase 2 — Primary Report (OOS, opened once)",
        "",
        f"**Verdict: {verdict}**",
        "",
        f"Selected config: **{selected['config_id']}** = `{params}`",
        f"OOS window: {report['oos_window']} ({n_oos_months} months)",
        "",
        "| Gate | Result | Pass? |",
        "|---|---|---|",
        f"| G1 — IC significance | mean IC {g1_mean:.4f}, 95% CI [{g1_lo:.4f}, {g1_hi:.4f}] | {'YES' if g1_pass else 'no'} |",
        f"| G2 — Beats baseline | ML Sharpe {ml_sharpe:.3f} vs baseline Sharpe {baseline_sharpe:.3f} | {'YES' if g2_pass else 'no'} |",
        f"| G3 — Paired significance | mean Δreturn {g3_mean:.5f}, 95% CI [{g3_lo:.5f}, {g3_hi:.5f}] | {'YES' if g3_pass else 'no'} |",
        f"| G4 — Permutation validity | permuted mean IC {g4_mean:.4f}, 95% CI [{g4_lo:.4f}, {g4_hi:.4f}] | {'YES (no leakage)' if g4_pass else 'FAIL — investigate'} |",
        "",
        "Promotion requires G1 AND G2 AND G3, with G4 required as a validity precondition — see `PREREGISTRATION.md` §7.",
    ]
    report_md_path = ROOT / "reports" / "PRIMARY_REPORT.md"
    report_md_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {report_md_path} and {report_json_path}")
