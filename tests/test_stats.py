"""Sanity checks for the statistical toolbox itself — the same kind of
mandatory self-test used elsewhere for this style of validation machinery:
feed it pure noise and it must not find a signal; feed it a real planted
effect and it must find it."""
import numpy as np
import pandas as pd

from src.stats import annualized_sharpe, bh_fdr, block_bootstrap_ci, bootstrap_p_value, paired_bootstrap_diff_ci


def test_pure_noise_ci_includes_zero():
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.standard_normal(120))  # 10 years of monthly noise
    mean, lo, hi = block_bootstrap_ci(noise, n_boot=3000, seed=1)
    assert lo < 0 < hi, "a zero-mean noise series should have a CI spanning zero"


def test_planted_effect_ci_excludes_zero():
    rng = np.random.default_rng(2)
    planted = pd.Series(rng.standard_normal(120) * 0.5 + 2.0)  # clearly nonzero mean, modest noise
    mean, lo, hi = block_bootstrap_ci(planted, n_boot=3000, seed=2)
    assert lo > 0, "a strongly nonzero series should have a CI excluding zero"


def test_bootstrap_p_value_small_for_planted_effect():
    rng = np.random.default_rng(3)
    planted = pd.Series(rng.standard_normal(200) * 0.3 + 1.5)
    p = bootstrap_p_value(planted, n_boot=3000, seed=3)
    assert p < 0.01


def test_bootstrap_p_value_large_for_pure_noise():
    rng = np.random.default_rng(4)
    noise = pd.Series(rng.standard_normal(200))
    p = bootstrap_p_value(noise, n_boot=3000, seed=4)
    assert p > 0.10


def test_bh_fdr_controls_false_discoveries_under_pure_noise():
    """20 independent pure-noise p-values (uniform[0,1]) — BH-FDR at q=0.10 should
    reject only a small minority, not a large fraction, of purely null hypotheses."""
    rng = np.random.default_rng(5)
    p_values = list(rng.uniform(0, 1, 20))
    rejected = bh_fdr(p_values, q=0.10)
    assert sum(rejected) <= 4, "BH-FDR should not reject most hypotheses when all are truly null"


def test_bh_fdr_detects_a_clearly_planted_signal_among_noise():
    rng = np.random.default_rng(6)
    p_values = [0.0001] + list(rng.uniform(0.2, 1.0, 19))  # one clear signal, 19 clean nulls
    rejected = bh_fdr(p_values, q=0.10)
    assert bool(rejected[0]) is True


def test_paired_bootstrap_detects_a_real_paired_difference():
    rng = np.random.default_rng(7)
    a = pd.Series(rng.standard_normal(150) * 0.4 + 1.0)
    b = pd.Series(rng.standard_normal(150) * 0.4 + 0.2)
    mean, lo, hi = paired_bootstrap_diff_ci(a, b, n_boot=3000, seed=7)
    assert lo > 0, "a should be reliably higher than b, so the CI on the difference should exclude zero"


def test_annualized_sharpe_matches_hand_calculation():
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.00, 0.01])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(12)
    assert annualized_sharpe(returns) == expected
