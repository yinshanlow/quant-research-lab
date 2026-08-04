import numpy as np
import pandas as pd

from src.baseline import baseline_signal


def test_baseline_signal_matches_manual_zscore():
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-31")] * 4,
        "ticker": ["A", "B", "C", "D"],
        "mom_63": [0.10, -0.05, 0.02, 0.01],
    })
    result = baseline_signal(panel)

    manual = (panel["mom_63"] - panel["mom_63"].mean()) / panel["mom_63"].std(ddof=0)
    np.testing.assert_allclose(result.values, manual.values)


def test_baseline_signal_is_zero_mean_within_each_date():
    panel = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-31")] * 3 + [pd.Timestamp("2020-02-29")] * 3,
        "ticker": ["A", "B", "C"] * 2,
        "mom_63": [0.1, 0.2, 0.3, -0.1, 0.0, 0.4],
    })
    result = baseline_signal(panel)
    panel = panel.assign(_z=result)
    means = panel.groupby("date")["_z"].mean()
    assert (means.abs() < 1e-9).all()
