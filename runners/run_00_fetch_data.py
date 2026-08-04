"""Phase 0 — pull the pre-registered universe's price history and build the
causal monthly factor panel. No performance, IC, or return statistic is
computed here — this script only produces the raw inputs everything else reads.

Run: python -m runners.run_00_fetch_data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import fetch_universe_prices
from src.features import build_monthly_panel
from src.universe import BENCHMARK_TICKER, TICKERS

PANEL_CACHE = Path(__file__).resolve().parent.parent / "data" / "monthly_panel.parquet"

if __name__ == "__main__":
    print(f"Fetching {len(TICKERS)} universe tickers + benchmark ({BENCHMARK_TICKER})...")
    prices = fetch_universe_prices(TICKERS + [BENCHMARK_TICKER], range_="20y", interval="1d")
    print(f"Prices: {prices.shape[0]} common trading days, {prices.index.min().date()} to {prices.index.max().date()}")

    universe_prices = prices[TICKERS]
    panel = build_monthly_panel(universe_prices, forward_days=21)
    print(f"Monthly panel: {panel.shape[0]} rows ({panel['date'].nunique()} month-ends x up to {len(TICKERS)} tickers)")

    PANEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_CACHE)
    print(f"Saved panel to {PANEL_CACHE}")

    prices[[BENCHMARK_TICKER]].to_csv(Path(__file__).resolve().parent.parent / "data" / "benchmark_spy.csv")
