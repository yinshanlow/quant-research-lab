import numpy as np
import pandas as pd
import pytest

from src.features import build_monthly_panel


def _make_synthetic_prices(n_days: int, n_tickers: int = 10, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n_tickers)]
    dates = pd.bdate_range("2008-01-01", periods=n_days)

    n = len(tickers)
    corr = np.full((n, n), 0.3)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    vol = rng.uniform(0.15, 0.35, n)
    drift = rng.uniform(0.02, 0.10, n)
    dt = 1 / 252
    z = rng.standard_normal((n_days, n)) @ chol.T
    daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    log_prices = np.log(100.0) + np.cumsum(daily_ret, axis=0)
    prices = np.exp(log_prices)
    return pd.DataFrame(prices, index=dates, columns=tickers)


@pytest.fixture
def synthetic_monthly_panel():
    """~9 years of synthetic monthly panel data (causal factors + forward returns)
    — long enough to exercise walk-forward refitting with min_train_years=3."""
    prices = _make_synthetic_prices(n_days=2350, n_tickers=12, seed=7)
    return build_monthly_panel(prices)


@pytest.fixture
def synthetic_prices():
    """Deterministic, correlated GBM price paths for a small fake universe —
    used by every test in this suite. Never real market data."""
    rng = np.random.default_rng(42)
    n_days = 900
    tickers = [f"T{i}" for i in range(10)]
    dates = pd.bdate_range("2015-01-01", periods=n_days)

    n = len(tickers)
    corr = np.full((n, n), 0.3)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)

    vol = rng.uniform(0.15, 0.35, n)
    drift = rng.uniform(0.02, 0.10, n)
    dt = 1 / 252
    z = rng.standard_normal((n_days, n)) @ chol.T
    daily_ret = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    log_prices = np.log(100.0) + np.cumsum(daily_ret, axis=0)
    prices = np.exp(log_prices)

    return pd.DataFrame(prices, index=dates, columns=tickers)
