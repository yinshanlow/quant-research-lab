"""Price data acquisition for the study universe.

Fetches adjusted-close daily prices directly from Yahoo Finance's public
chart endpoint (the same approach validated in the sibling
`quant-finance-toolkit` repo — a plain HTTP GET with a browser User-Agent,
which is reliable in this environment where the `yfinance` library's
session/crumb handshake is not).

Every successful pull is cached to CSV and recorded in a provenance manifest
(ticker, rows, date range, fetch timestamp, SHA-256 of the cached file) so a
later reader can verify exactly what data a reported result was computed on
— the same discipline the `Related research` this study was built in
response to uses, scaled down to what a free public data source can support
(no vendor checksums are available for Yahoo's endpoint, so the manifest
hashes the file this repo actually persisted, not an upstream-issued one).
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
MANIFEST_PATH = Path(__file__).parent.parent / "data" / "MANIFEST.json"


def _fetch_one(ticker: str, range_: str, interval: str) -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    resp = requests.get(
        url, params={"range": range_, "interval": interval},
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        timeout=20,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    adj_close = result["indicators"]["adjclose"][0]["adjclose"]
    index = pd.to_datetime(timestamps, unit="s").normalize()
    return pd.Series(adj_close, index=index, name=ticker)


def fetch_universe_prices(
    tickers: list[str], range_: str = "20y", interval: str = "1d", use_cache: bool = True,
) -> pd.DataFrame:
    """Returns a DataFrame of adjusted-close prices, one column per ticker, inner-joined
    on date (only dates where every ticker has a print are kept). Writes/updates
    data/MANIFEST.json with provenance for every ticker fetched fresh this call.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    series_list = []

    for ticker in tickers:
        cache_path = CACHE_DIR / f"{ticker}_{range_}_{interval}.csv"
        if use_cache and cache_path.exists():
            s = pd.read_csv(cache_path, index_col=0, parse_dates=True)[ticker]
        else:
            s = _fetch_one(ticker, range_, interval)
            s.to_frame(name=ticker).to_csv(cache_path)
            manifest[ticker] = {
                "range": range_, "interval": interval,
                "n_rows": int(s.shape[0]),
                "start": str(s.index.min().date()), "end": str(s.index.max().date()),
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            }
        series_list.append(s)

    _save_manifest(manifest)
    return pd.concat(series_list, axis=1).dropna(how="any")


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))
