"""Dollar-neutral tercile long-short portfolio construction, net of transaction costs.

Monthly rebalance: rank the universe by the signal within each date, go long
the top tercile / short the bottom tercile, equal-weighted within each leg,
dollar-neutral (long leg sums to +1, short leg to −1). Costs are charged on
turnover — the change in each ticker's weight from the prior month — at a
flat conservative rate, applied whether the change is opening a new position
or closing an old one.
"""
import numpy as np
import pandas as pd


def _monthly_weights(panel: pd.DataFrame, signal: pd.Series, top_frac: float, bottom_frac: float) -> dict[pd.Timestamp, pd.Series]:
    df = panel.assign(_signal=signal.values)
    weights_by_date = {}
    for date, group in df.groupby("date"):
        valid = group.dropna(subset=["_signal"])
        n = len(valid)
        if n < 6:  # need enough names for a meaningful tercile split
            continue
        ranked = valid.sort_values("_signal")
        n_bottom = max(1, int(round(n * bottom_frac)))
        n_top = max(1, int(round(n * top_frac)))
        shorts = ranked.iloc[:n_bottom]
        longs = ranked.iloc[-n_top:]

        w = pd.Series(0.0, index=valid["ticker"])
        w.loc[longs["ticker"]] = 1.0 / n_top
        w.loc[shorts["ticker"]] = -1.0 / n_bottom
        weights_by_date[date] = w
    return weights_by_date


def long_short_returns(
    panel: pd.DataFrame, signal: pd.Series, top_frac: float = 1 / 3, bottom_frac: float = 1 / 3,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Returns a monthly net-of-cost return Series, indexed by date."""
    weights_by_date = _monthly_weights(panel, signal, top_frac, bottom_frac)
    dates = sorted(weights_by_date.keys())
    if not dates:
        return pd.Series(dtype=float)

    fwd_return_lookup = panel.set_index(["date", "ticker"])["fwd_return"]

    net_returns = {}
    prev_weights = pd.Series(dtype=float)
    for date in dates:
        w = weights_by_date[date]
        gross_return = 0.0
        for ticker, weight in w.items():
            r = fwd_return_lookup.get((date, ticker), np.nan)
            if not np.isnan(r):
                gross_return += weight * r

        all_tickers = w.index.union(prev_weights.index)
        turnover = (w.reindex(all_tickers, fill_value=0.0) - prev_weights.reindex(all_tickers, fill_value=0.0)).abs().sum()
        cost = turnover * (cost_bps / 10_000)

        net_returns[date] = gross_return - cost
        prev_weights = w

    return pd.Series(net_returns).sort_index()
