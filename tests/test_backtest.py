import pandas as pd

from src.backtest import long_short_returns


def _make_panel(n_dates: int, n_tickers: int, fwd_return_fn, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-31", periods=n_dates, freq="ME")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            rows.append({"date": date, "ticker": ticker, "fwd_return": fwd_return_fn(i, rng)})
    return pd.DataFrame(rows)


def test_long_short_is_dollar_neutral_each_month():
    panel = _make_panel(6, 9, lambda i, rng: rng.standard_normal())
    signal = pd.Series(panel["fwd_return"].values)  # perfectly informative signal
    weights_by_date = None
    from src.backtest import _monthly_weights

    weights_by_date = _monthly_weights(panel, signal, top_frac=1 / 3, bottom_frac=1 / 3)
    for date, w in weights_by_date.items():
        assert abs(w[w > 0].sum() - 1.0) < 1e-9
        assert abs(w[w < 0].sum() + 1.0) < 1e-9


def test_zero_turnover_when_same_names_stay_top_and_bottom():
    """If the same tickers are top/bottom tercile every month, turnover (and cost)
    should be zero from the second month onward."""
    n_tickers = 9
    dates = pd.bdate_range("2020-01-31", periods=4, freq="ME")
    fixed_signal_rank = list(range(n_tickers))  # ticker i always has signal value i (fixed ranking)

    rows = []
    for date in dates:
        for i in range(n_tickers):
            rows.append({"date": date, "ticker": f"T{i}", "fwd_return": 0.01})
    panel = pd.DataFrame(rows)
    signal = pd.Series(fixed_signal_rank * len(dates), dtype=float)

    returns = long_short_returns(panel, signal, top_frac=1 / 3, bottom_frac=1 / 3, cost_bps=10.0)
    # first month pays cost to open the book; every month after should cost ~0 since
    # the same names stay top/bottom, so returns should be identical from month 2 onward
    assert abs(returns.iloc[1] - returns.iloc[2]) < 1e-9
    assert abs(returns.iloc[2] - returns.iloc[3]) < 1e-9


def test_perfectly_informative_signal_beats_a_random_signal_net_of_costs():
    n_tickers = 15
    panel = _make_panel(10, n_tickers, lambda i, rng: rng.normal(loc=(i - n_tickers / 2) * 0.01, scale=0.02), seed=11)
    perfect_signal = panel["fwd_return"]  # signal == the realized outcome (best possible, in-sample oracle)

    import numpy as np
    rng = np.random.default_rng(99)
    random_signal = pd.Series(rng.standard_normal(len(panel)))

    perfect_returns = long_short_returns(panel, perfect_signal, cost_bps=5.0)
    random_returns = long_short_returns(panel, random_signal, cost_bps=5.0)

    assert perfect_returns.mean() > random_returns.mean()
