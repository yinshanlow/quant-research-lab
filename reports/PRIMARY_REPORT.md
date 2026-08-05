# Phase 2 — Primary Report (OOS, opened once)

**Verdict: NOT PROMOTED — H1 not confirmed on this OOS sample**

Selected config: **B** = `{'max_depth': 2, 'n_estimators': 200, 'learning_rate': 0.05}`
OOS window: 2021-01-29 to 2026-06-30 (66 months)

| Gate | Result | Pass? |
|---|---|---|
| G1 — IC significance | mean IC 0.0218, 95% CI [-0.0309, 0.0770] | no |
| G2 — Beats baseline | ML Sharpe -0.089 vs baseline Sharpe 0.029 | no |
| G3 — Paired significance | mean Δreturn -0.00159, 95% CI [-0.01887, 0.01671] | no |
| G4 — Permutation validity | permuted mean IC 0.0129, 95% CI [-0.0467, 0.0734] | YES (no leakage) |

Promotion requires G1 AND G2 AND G3, with G4 required as a validity precondition — see `PREREGISTRATION.md` §7.
