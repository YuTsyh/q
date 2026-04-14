"""Historical OHLCV and Funding Rate downloader for real exchange data.

Downloads authentic market data from exchanges (OKX REST API) for
strategy validation, backtesting, and research.  Includes local CSV
caching to avoid repeated downloads.

Usage
-----
>>> from quantbot.research.real_data import download_ohlcv, download_funding
>>> bars = download_ohlcv("BTC-USDT-SWAP", "1D", "2023-01-01", "2024-01-01")
>>> funding = download_funding("BTC-USDT-SWAP", "2023-01-01", "2024-01-01")

.. note::
   This module requires ``httpx`` (already in project dependencies).
   No API key is required for public market data endpoints (OKX).
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Literal

import httpx

from quantbot.research.data import FundingRate, OhlcvBar

_logger = logging.getLogger(__name__)

# OKX public REST API base
_OKX_BASE = "https://www.okx.com"
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "cache"

# Map human-readable bar sizes to OKX API bar parameter values
_BAR_SIZE_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1H": "1H",
    "4H": "4H",
    "1D": "1D",
    "1W": "1W",
}


def _ensure_cache_dir() -> Path:
    """Create cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _cache_path(inst_id: str, data_type: str, bar_size: str = "") -> Path:
    """Generate cache file path for a given instrument and data type."""
    safe_name = inst_id.replace("/", "_").replace("-", "_")
    suffix = f"_{bar_size}" if bar_size else ""
    return _ensure_cache_dir() / f"{safe_name}_{data_type}{suffix}.csv"


def _parse_ts(ts_ms: str | int) -> datetime:
    """Parse millisecond timestamp to datetime."""
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)


def _ts_to_ms(dt_str: str) -> int:
    """Convert 'YYYY-MM-DD' string to epoch milliseconds."""
    dt = datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def download_ohlcv(
    inst_id: str,
    bar_size: Literal["1m", "5m", "15m", "1H", "4H", "1D", "1W"] = "1D",
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    *,
    use_cache: bool = True,
) -> list[OhlcvBar]:
    """Download historical OHLCV bars from OKX.

    Parameters
    ----------
    inst_id : str
        OKX instrument ID, e.g. ``"BTC-USDT-SWAP"``.
    bar_size : str
        Candle period (default ``"1D"``).
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM-DD`` format.
    use_cache : bool
        If True, check local CSV cache before downloading.

    Returns
    -------
    list[OhlcvBar]
        Chronologically sorted list of OHLCV bars.
    """
    cache_file = _cache_path(inst_id, "ohlcv", bar_size)

    # Try cache first
    if use_cache and cache_file.exists():
        _logger.info("Loading cached OHLCV from %s", cache_file)
        return _read_ohlcv_cache(cache_file, inst_id)

    _logger.info("Downloading OHLCV for %s (%s) from %s to %s", inst_id, bar_size, start, end)

    okx_bar = _BAR_SIZE_MAP.get(bar_size, bar_size)
    all_bars: list[dict] = []
    after_ms = _ts_to_ms(end)
    before_ms = _ts_to_ms(start)

    with httpx.Client(timeout=30.0) as client:
        while True:
            params: dict = {
                "instId": inst_id,
                "bar": okx_bar,
                "after": str(after_ms),
                "limit": "100",
            }
            resp = client.get(f"{_OKX_BASE}/api/v5/market/history-candles", params=params)
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != "0":
                _logger.error("OKX API error: %s", data.get("msg", "unknown"))
                break

            candles = data.get("data", [])
            if not candles:
                break

            for c in candles:
                ts_ms = int(c[0])
                if ts_ms < before_ms:
                    candles = []  # signal end
                    break
                all_bars.append({
                    "ts": c[0], "open": c[1], "high": c[2],
                    "low": c[3], "close": c[4], "vol": c[5],
                })

            if not candles:
                break

            # OKX returns newest first; the last item is the oldest
            after_ms = int(candles[-1][0]) - 1

    # Convert to OhlcvBar
    all_bars.sort(key=lambda x: int(x["ts"]))
    bars = [
        OhlcvBar(
            inst_id=inst_id,
            ts=_parse_ts(b["ts"]),
            open=Decimal(b["open"]),
            high=Decimal(b["high"]),
            low=Decimal(b["low"]),
            close=Decimal(b["close"]),
            volume=Decimal(b["vol"]),
        )
        for b in all_bars
    ]

    # Cache to disk
    if bars:
        _write_ohlcv_cache(cache_file, bars)
        _logger.info("Cached %d bars to %s", len(bars), cache_file)

    return bars


def download_funding(
    inst_id: str,
    start: str = "2023-01-01",
    end: str = "2024-01-01",
    *,
    use_cache: bool = True,
) -> list[FundingRate]:
    """Download historical funding rates from OKX.

    Parameters
    ----------
    inst_id : str
        OKX instrument ID, e.g. ``"BTC-USDT-SWAP"``.
    start : str
        Start date in ``YYYY-MM-DD`` format.
    end : str
        End date in ``YYYY-MM-DD`` format.
    use_cache : bool
        If True, check local CSV cache before downloading.

    Returns
    -------
    list[FundingRate]
        Chronologically sorted list of funding rates.

    Note
    ----
    OKX public API only retains ~3 months of funding rate history.
    For older data, use a third-party data provider or cached CSV files.
    """
    cache_file = _cache_path(inst_id, "funding")

    if use_cache and cache_file.exists():
        _logger.info("Loading cached funding from %s", cache_file)
        return _read_funding_cache(cache_file, inst_id)

    _logger.info("Downloading funding rates for %s from %s to %s", inst_id, start, end)

    all_rates: list[dict] = []
    start_ms = _ts_to_ms(start)
    # OKX API: 'after' param returns records OLDER than the given timestamp.
    # Start from end date and paginate backward toward start date.
    cursor_ms = _ts_to_ms(end)

    with httpx.Client(timeout=30.0) as client:
        while True:
            params: dict = {
                "instId": inst_id,
                "after": str(cursor_ms),
                "limit": "100",
            }
            resp = client.get(
                f"{_OKX_BASE}/api/v5/public/funding-rate-history",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != "0":
                _logger.error("OKX API error: %s", data.get("msg", "unknown"))
                break

            records = data.get("data", [])
            if not records:
                break

            reached_start = False
            for r in records:
                ts_ms = int(r["fundingTime"])
                if ts_ms < start_ms:
                    reached_start = True
                    break
                all_rates.append({
                    "ts": r["fundingTime"],
                    "rate": r["fundingRate"],
                })

            if reached_start:
                break

            # Records are returned newest-first; last item is oldest
            oldest_ts = int(records[-1]["fundingTime"])
            if oldest_ts >= cursor_ms:
                break  # No progress, avoid infinite loop
            cursor_ms = oldest_ts

    all_rates.sort(key=lambda x: int(x["ts"]))
    rates = [
        FundingRate(
            inst_id=inst_id,
            funding_time=_parse_ts(r["ts"]),
            funding_rate=Decimal(r["rate"]),
        )
        for r in all_rates
    ]

    if rates:
        _write_funding_cache(cache_file, rates)
        _logger.info("Cached %d funding rates to %s", len(rates), cache_file)

    return rates


# ---------------------------------------------------------------------------
# CSV cache helpers
# ---------------------------------------------------------------------------

def _write_ohlcv_cache(path: Path, bars: list[OhlcvBar]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow([
                b.ts.isoformat(), str(b.open), str(b.high),
                str(b.low), str(b.close), str(b.volume),
            ])


def _read_ohlcv_cache(path: Path, inst_id: str) -> list[OhlcvBar]:
    bars: list[OhlcvBar] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bars.append(OhlcvBar(
                inst_id=inst_id,
                ts=datetime.fromisoformat(row["ts"]),
                open=Decimal(row["open"]),
                high=Decimal(row["high"]),
                low=Decimal(row["low"]),
                close=Decimal(row["close"]),
                volume=Decimal(row["volume"]),
            ))
    return bars


def _write_funding_cache(path: Path, rates: list[FundingRate]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "rate"])
        for r in rates:
            writer.writerow([r.funding_time.isoformat(), str(r.funding_rate)])


def _read_funding_cache(path: Path, inst_id: str) -> list[FundingRate]:
    rates: list[FundingRate] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rates.append(FundingRate(
                inst_id=inst_id,
                funding_time=datetime.fromisoformat(row["ts"]),
                funding_rate=Decimal(row["rate"]),
            ))
    return rates
