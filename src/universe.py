"""The fixed, pre-registered study universe. Single source of truth for
`preregistration/PREREGISTRATION.md` §2 — do not edit after the
pre-registration commit without a logged entry in `DEVIATIONS.md`.
"""

TICKERS: list[str] = [
    "AAPL", "MSFT", "ORCL", "IBM", "INTC",  # Technology
    "JPM", "BAC", "WFC", "GS", "C",         # Financials
    "JNJ", "PFE", "MRK", "ABT", "UNH",      # Healthcare
    "XOM", "CVX", "COP", "SLB", "CSCO",     # Energy row (CSCO is Technology; see PREREGISTRATION.md §2 note)
    "PG", "KO", "PEP", "WMT", "HD",         # Consumer
    "MCD", "BA", "CAT", "GE", "HON",        # Industrials
]

BENCHMARK_TICKER = "SPY"  # diagnostic-only, never part of the tradable universe or signal
