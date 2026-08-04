"""Phase 1 — hyperparameter selection, IS-only (2017-2020 inner validation window).
This script never touches 2021+ (OOS) data. Selects one config per
PREREGISTRATION.md §5's rule and writes reports/SELECTION_REPORT.md.

Run: python -m runners.run_01_select_hyperparams
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml_signal import HYPERPARAM_GRID, walk_forward_predict
from src.stats import block_bootstrap_ci, bh_fdr, bootstrap_p_value

ROOT = Path(__file__).resolve().parent.parent
PANEL_CACHE = ROOT / "data" / "monthly_panel.parquet"
SELECTED_CONFIG_PATH = ROOT / "reports" / "selected_config.json"
CV_START_YEAR = 2017
CV_END_YEAR = 2020

if __name__ == "__main__":
    panel = pd.read_parquet(PANEL_CACHE)
    panel["date"] = pd.to_datetime(panel["date"])

    rows = []
    ic_series_by_config = []
    for i, params in enumerate(HYPERPARAM_GRID):
        preds = walk_forward_predict(panel, params, CV_START_YEAR, CV_END_YEAR, min_train_years=3)
        from src.stats import spearman_ic_by_date
        ic_series = spearman_ic_by_date(panel, preds)
        ic_series = ic_series[(ic_series.index.year >= CV_START_YEAR) & (ic_series.index.year <= CV_END_YEAR)]

        mean_ic, lo, hi = block_bootstrap_ci(ic_series, n_boot=10_000, seed=100 + i)
        p_value = bootstrap_p_value(ic_series, n_boot=10_000, seed=100 + i)

        rows.append({
            "config_id": chr(ord("A") + i), "params": params,
            "mean_ic": mean_ic, "ci_low": lo, "ci_high": hi, "p_value": p_value,
        })
        ic_series_by_config.append(ic_series)

    p_values = [r["p_value"] for r in rows]
    survives_fdr = bh_fdr(p_values, q=0.10)
    for r, s in zip(rows, survives_fdr):
        r["survives_bh_fdr_q10"] = bool(s)

    survivors = [r for r in rows if r["survives_bh_fdr_q10"]]
    candidates = survivors if survivors else rows
    selected = max(candidates, key=lambda r: r["mean_ic"])

    selection_note = (
        "selected from BH-FDR survivors (highest IC among significant configs)"
        if survivors else
        "NO config survived BH-FDR at the selection stage — falling back to the "
        "highest-IC config per the pre-registered rule; this is disclosed, not hidden"
    )

    SELECTED_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_CONFIG_PATH.write_text(json.dumps({
        "selected_config_id": selected["config_id"], "params": selected["params"],
        "selection_note": selection_note,
    }, indent=2))

    report_lines = [
        "# Phase 1 — Hyperparameter Selection (IS-only, 2017-2020)",
        "",
        f"Selection window: {CV_START_YEAR}-01 through {CV_END_YEAR}-12. **No OOS (2021+) data was used.**",
        "",
        "| Config | max_depth | n_estimators | learning_rate | Mean IC | 95% CI | Bootstrap p | Survives BH-FDR q=0.10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        p = r["params"]
        report_lines.append(
            f"| {r['config_id']} | {p['max_depth']} | {p['n_estimators']} | {p['learning_rate']} | "
            f"{r['mean_ic']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['p_value']:.4f} | "
            f"{'YES' if r['survives_bh_fdr_q10'] else 'no'} |"
        )
    report_lines += [
        "",
        f"**Selected: Config {selected['config_id']}** ({selection_note}).",
        "",
        f"This selection is now frozen for the OOS test — `run_02_oos_promotion.py` refits only "
        f"this configuration, walk-forward, across 2021+.",
    ]

    report_path = ROOT / "reports" / "SELECTION_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    print(f"\nWrote {report_path} and {SELECTED_CONFIG_PATH}")
