# Phase 1 — Hyperparameter Selection (IS-only, 2017-2020)

Selection window: 2017-01 through 2020-12. **No OOS (2021+) data was used.**

| Config | max_depth | n_estimators | learning_rate | Mean IC | 95% CI | Bootstrap p | Survives BH-FDR q=0.10 |
|---|---|---|---|---|---|---|---|
| A | 2 | 100 | 0.05 | -0.0538 | [-0.1151, 0.0066] | 0.0794 | no |
| B | 2 | 200 | 0.05 | -0.0171 | [-0.0747, 0.0400] | 0.5624 | no |
| C | 3 | 100 | 0.05 | -0.0383 | [-0.0971, 0.0181] | 0.1910 | no |
| D | 3 | 200 | 0.05 | -0.0313 | [-0.0972, 0.0344] | 0.3402 | no |

**Selected: Config B** (NO config survived BH-FDR at the selection stage — falling back to the highest-IC config per the pre-registered rule; this is disclosed, not hidden).

This selection is now frozen for the OOS test — `run_02_oos_promotion.py` refits only this configuration, walk-forward, across 2021+.
