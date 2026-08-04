"""The ML signal: a gradient-boosted (XGBoost) model combining the five factors
in `features.FEATURE_COLUMNS` to predict next-month cross-sectional forward return.

Walk-forward only: at every refit point, the model is trained exclusively on
data strictly before the refit date (`panel["date"] < refit_date`), and its
predictions are only ever used for dates after that refit — enforced by
`walk_forward_predict`'s date masks, and checked directly by
`tests/test_ml_signal_no_lookahead.py`.

The hyperparameter grid below is the full, pre-registered search space
(`preregistration/PREREGISTRATION.md` §4) — nothing outside this grid was
tried, and the grid was fixed before any model saw real data.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features import FEATURE_COLUMNS

HYPERPARAM_GRID: list[dict] = [
    {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.05},
    {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05},
    {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.05},
    {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05},
]


def _fit_one_model(train_df: pd.DataFrame, params: dict, seed: int = 0) -> XGBRegressor:
    model = XGBRegressor(
        max_depth=params["max_depth"], n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"], objective="reg:squarederror",
        random_state=seed, verbosity=0,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df["fwd_return"])
    return model


def walk_forward_predict(
    panel: pd.DataFrame, params: dict, first_predict_year: int, last_predict_year: int,
    min_train_years: int = 3, permute_labels_seed: int | None = None,
) -> pd.Series:
    """Refits annually on an expanding window (all data strictly before Jan 1 of the
    predict year), predicts every month in that year, then rolls forward.

    permute_labels_seed: if set, shuffles fwd_return WITHIN each training date
    cross-section (never across dates) before fitting — used only by the label-
    permutation leakage gate (G4), never by the real signal.
    """
    panel = panel.dropna(subset=FEATURE_COLUMNS + ["fwd_return"]).copy()
    preds = pd.Series(np.nan, index=panel.index)

    for year in range(first_predict_year, last_predict_year + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        train_df = panel[panel["date"] < cutoff]
        if train_df["date"].dt.year.nunique() < min_train_years:
            continue

        if permute_labels_seed is not None:
            rng = np.random.default_rng(permute_labels_seed + year)
            train_df = train_df.copy()
            train_df["fwd_return"] = train_df.groupby("date")["fwd_return"].transform(
                lambda x: rng.permutation(x.values)
            )

        model = _fit_one_model(train_df, params, seed=permute_labels_seed or 0)
        predict_mask = panel["date"].dt.year == year
        if predict_mask.any():
            preds.loc[panel.index[predict_mask]] = model.predict(panel.loc[predict_mask, FEATURE_COLUMNS])

    return preds
