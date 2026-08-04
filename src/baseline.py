"""The baseline signal: cross-sectional rank of 3-month (63-trading-day) momentum.

Deliberately a single, unfit, zero-parameter rule — nothing here is estimated
from data, so it cannot overfit by construction. This is the bar the ML
signal (`ml_signal.py`) has to clear, not just outperform on a single
in-sample fit.
"""
import pandas as pd

from src.features import cross_sectional_zscore


def baseline_signal(panel: pd.DataFrame) -> pd.Series:
    """Returns a Series aligned to `panel`'s index: the cross-sectional z-score
    of mom_63 on each date. No fitting, no lookback beyond mom_63 itself."""
    return cross_sectional_zscore(panel, "mom_63")
