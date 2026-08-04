"""Causal (no-look-ahead) cross-sectional factor construction.

Every factor at date t is computed using only price history up to and
including t — enforced structurally by using only trailing pandas rolling
windows (`.rolling(n)`, never `.rolling(n, center=True)` and never a forward
shift) and verified by `tests/test_features_no_lookahead.py`'s
truncation-invariance tests: recomputing a factor on a truncated price
history `prices.loc[:t]` must reproduce the exact value the full-history
computation gives at t.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["mom_21", "mom_63", "mom_126", "vol_21", "reversal_5"]


def compute_daily_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """prices: date-indexed DataFrame, one column per ticker.
    Returns a dict of feature_name -> DataFrame (same shape as prices), each entry
    computed causally (trailing-window only) at every daily date.
    """
    returns = prices.pct_change()

    mom_21 = prices / prices.shift(21) - 1.0
    mom_63 = prices / prices.shift(63) - 1.0
    mom_126 = prices / prices.shift(126) - 1.0
    vol_21 = returns.rolling(21).std()
    # sign-flipped 5-day return: higher value = more "oversold", the direction a
    # reversal signal predicts positively
    reversal_5 = -(prices / prices.shift(5) - 1.0)

    return {
        "mom_21": mom_21, "mom_63": mom_63, "mom_126": mom_126,
        "vol_21": vol_21, "reversal_5": reversal_5,
    }


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last available trading date in each calendar month present in `index`."""
    s = pd.Series(index, index=index)
    return s.resample("ME").last().dropna().values


def build_monthly_panel(prices: pd.DataFrame, forward_days: int = 21) -> pd.DataFrame:
    """Returns a tidy (long-format) monthly panel: one row per (month_end_date, ticker),
    with the causal factor values as of that month-end plus the forward `forward_days`-day
    return as the label (NaN for the final rows where the forward window doesn't exist yet
    — those rows are dropped by the caller before use, never imputed).
    """
    daily_features = compute_daily_features(prices)
    m_ends = month_end_dates(prices.index)

    forward_return = prices.shift(-forward_days) / prices - 1.0

    rows = []
    for date in m_ends:
        for ticker in prices.columns:
            row = {"date": date, "ticker": ticker}
            for feat_name, feat_df in daily_features.items():
                row[feat_name] = feat_df.loc[date, ticker]
            row["fwd_return"] = forward_return.loc[date, ticker]
            rows.append(row)

    panel = pd.DataFrame(rows)
    return panel


def cross_sectional_zscore(panel: pd.DataFrame, column: str) -> pd.Series:
    """Within each date, standardize `column` to zero mean / unit std across tickers.
    NaNs within a date are excluded from that date's mean/std and left as NaN."""
    return panel.groupby("date")[column].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
