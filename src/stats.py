"""Statistical validation toolbox: block bootstrap, BH-FDR, IC, Sharpe.

The bootstrap unit throughout is the MONTH, never the individual
(date, ticker) row — cross-sectional observations within the same month
share common-date noise and are not independent of each other, so
resampling individual rows would understate the true uncertainty. Resampling
whole months (with replacement) is the panel-data equivalent of the
stationary block bootstrap used for time series in the sibling research this
study borrows its discipline from.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def spearman_ic_by_date(panel: pd.DataFrame, signal: pd.Series, label_col: str = "fwd_return") -> pd.Series:
    """Cross-sectional Spearman rank correlation between `signal` and the forward
    return, computed separately within each date. Returns a Series indexed by date."""
    df = panel.assign(_signal=signal.values)
    ics = {}
    for date, group in df.groupby("date"):
        valid = group[["_signal", label_col]].dropna()
        if len(valid) < 5:
            continue
        ic, _ = spearmanr(valid["_signal"], valid[label_col])
        ics[date] = ic
    return pd.Series(ics).sort_index()


def block_bootstrap_ci(values: pd.Series, n_boot: int = 10_000, seed: int | None = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Resamples `values` (one observation per month, e.g. per-month IC or per-month
    portfolio return) as whole blocks with replacement. Returns (mean, ci_low, ci_high)."""
    values = values.dropna()
    rng = np.random.default_rng(seed)
    n = len(values)
    arr = values.values
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def bootstrap_p_value(values: pd.Series, n_boot: int = 10_000, seed: int | None = 0) -> float:
    """Two-sided bootstrap p-value for H0: mean(values) == 0, via the proportion of
    bootstrap resample means that fall on the opposite side of zero from the observed mean."""
    values = values.dropna()
    rng = np.random.default_rng(seed)
    arr = values.values
    n = len(arr)
    observed = arr.mean()
    boot_means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_boot)])
    if observed >= 0:
        p = 2 * min((boot_means <= 0).mean(), 0.5)
    else:
        p = 2 * min((boot_means >= 0).mean(), 0.5)
    return float(min(p, 1.0))


def paired_bootstrap_diff_ci(series_a: pd.Series, series_b: pd.Series, n_boot: int = 10_000, seed: int | None = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """CI for mean(series_a - series_b), resampling paired (same-month) observations
    together so the pairing is preserved in every bootstrap draw."""
    both = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    diff = both["a"] - both["b"]
    return block_bootstrap_ci(diff, n_boot=n_boot, seed=seed, alpha=alpha)


def bh_fdr(p_values: list[float], q: float = 0.10) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns a boolean list, same order as input,
    True where the hypothesis survives at the given false-discovery-rate threshold."""
    if not p_values:
        return []
    reject, _, _, _ = multipletests(p_values, alpha=q, method="fdr_bh")
    return list(reject)


def annualized_sharpe(monthly_returns: pd.Series, periods_per_year: int = 12) -> float:
    monthly_returns = monthly_returns.dropna()
    if monthly_returns.std(ddof=1) == 0 or len(monthly_returns) < 2:
        return 0.0
    return float(monthly_returns.mean() / monthly_returns.std(ddof=1) * np.sqrt(periods_per_year))
