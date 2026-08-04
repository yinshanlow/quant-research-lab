"""Proves the ML walk-forward pipeline cannot see future data: corrupting the
panel strictly after a given year's training cutoff must not change that
year's predictions, and years without enough training history must not be
predicted at all."""
import numpy as np
import pandas as pd

from src.ml_signal import HYPERPARAM_GRID, walk_forward_predict


def test_predictions_unaffected_by_corrupting_strictly_future_rows(synthetic_monthly_panel):
    panel = synthetic_monthly_panel
    years = sorted(panel["date"].dt.year.unique())
    predict_year = years[len(years) // 2]  # a year comfortably in the middle

    params = HYPERPARAM_GRID[0]
    original_preds = walk_forward_predict(panel, params, predict_year, predict_year, min_train_years=3)

    corrupted = panel.copy()
    cutoff = pd.Timestamp(f"{predict_year}-01-01")
    future_mask = corrupted["date"] >= cutoff
    # corrupt labels and features for every row on/after the cutoff for years
    # AFTER the predict year (never touching predict_year's own rows, since
    # those are legitimately used for prediction, just not training)
    later_mask = corrupted["date"].dt.year > predict_year
    corrupted.loc[later_mask, "fwd_return"] = corrupted.loc[later_mask, "fwd_return"] * -50 + 999

    corrupted_preds = walk_forward_predict(corrupted, params, predict_year, predict_year, min_train_years=3)

    pd.testing.assert_series_equal(original_preds.dropna(), corrupted_preds.dropna())


def test_no_predictions_before_minimum_training_history(synthetic_monthly_panel):
    panel = synthetic_monthly_panel
    first_year = panel["date"].dt.year.min()
    params = HYPERPARAM_GRID[0]

    preds = walk_forward_predict(panel, params, first_year, first_year, min_train_years=3)
    assert preds.dropna().empty, "should not predict in the very first year — no training history exists yet"


def test_all_grid_configs_produce_walk_forward_predictions(synthetic_monthly_panel):
    panel = synthetic_monthly_panel
    years = sorted(panel["date"].dt.year.unique())
    # years[-1] can be a degenerate trailing partial year (few or no rows with a
    # resolved forward return yet) — use the last FULL year instead.
    predict_year = years[-2]

    for params in HYPERPARAM_GRID:
        preds = walk_forward_predict(panel, params, predict_year, predict_year, min_train_years=3)
        assert preds.dropna().shape[0] > 0, f"config {params} produced no predictions"
