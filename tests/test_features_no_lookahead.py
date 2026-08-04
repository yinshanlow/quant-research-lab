"""Truncation-invariance tests: every factor's value at date t must be identical
whether it's computed on the full price history or on a history truncated to
end at t. If any of these fail, the factor is leaking future information."""
import pandas as pd

from src.features import compute_daily_features


def test_momentum_and_vol_features_are_truncation_invariant(synthetic_prices):
    full_features = compute_daily_features(synthetic_prices)
    cutoff = synthetic_prices.index[500]
    truncated_prices = synthetic_prices.loc[:cutoff]
    truncated_features = compute_daily_features(truncated_prices)

    for name in full_features:
        full_value_at_cutoff = full_features[name].loc[cutoff]
        truncated_value_at_cutoff = truncated_features[name].loc[cutoff]
        pd.testing.assert_series_equal(
            full_value_at_cutoff, truncated_value_at_cutoff, check_names=False,
            obj=f"feature={name} at cutoff",
        )


def test_features_change_when_a_future_price_is_corrupted(synthetic_prices):
    """The complementary check: corrupting a price strictly AFTER the cutoff must
    NOT change the feature value AT the cutoff — proving the invariance above isn't
    trivially true because the feature ignores price data altogether."""
    cutoff = synthetic_prices.index[500]
    corrupted = synthetic_prices.copy()
    future_dates = corrupted.index[corrupted.index > cutoff]
    corrupted.loc[future_dates] *= 5.0  # wildly change every price after the cutoff

    original_features = compute_daily_features(synthetic_prices)
    corrupted_features = compute_daily_features(corrupted)

    for name in original_features:
        pd.testing.assert_series_equal(
            original_features[name].loc[cutoff], corrupted_features[name].loc[cutoff],
            check_names=False, obj=f"feature={name} at cutoff after future corruption",
        )
