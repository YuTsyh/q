"""OKX Demo API Connectivity Verification Script.

Tests all integration points between the quant system and OKX Demo:
1. REST API authentication & instrument metadata fetch
2. Public market data (ticker via REST)
3. Historical OHLCV download via real_data.py
4. Historical funding rate download
5. BotConfig loading from .env

Usage:
    python scripts/verify_okx_connection.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _info(msg: str) -> None:
    print(f"  [..]   {msg}")


async def test_rest_authentication() -> bool:
    """Test 1: REST API authentication by fetching instrument metadata."""
    _header("Test 1: REST API Authentication & Instrument Fetch")

    from quantbot.exchange.okx.auth import OkxCredentials, OkxSigner
    from quantbot.exchange.okx.rest import OkxRestClient

    api_key = os.environ.get("OKX_API_KEY", "")
    api_secret = os.environ.get("OKX_API_SECRET", "")
    api_passphrase = os.environ.get("OKX_API_PASSPHRASE", "")

    if not all([api_key, api_secret, api_passphrase]):
        _fail("Missing OKX_API_KEY, OKX_API_SECRET, or OKX_API_PASSPHRASE in .env")
        return False

    _info(f"API Key: {api_key[:12]}...{api_key[-4:]}")

    credentials = OkxCredentials(
        api_key=api_key,
        api_secret=api_secret,
        passphrase=api_passphrase,
    )
    signer = OkxSigner(credentials)
    client = OkxRestClient(
        base_url="https://www.okx.com",
        signer=signer,
        simulated=True,  # Demo mode
    )

    try:
        instrument = await client.fetch_spot_instrument("BTC-USDT")
        _ok(f"Fetched instrument: {instrument.inst_id}")
        _info(f"  Type: {instrument.inst_type}")
        _info(f"  Tick Size: {instrument.tick_sz}")
        _info(f"  Lot Size: {instrument.lot_sz}")
        _info(f"  Min Size: {instrument.min_sz}")
        _info(f"  State: {instrument.state}")
        await client.aclose()
        return True
    except Exception as e:
        _fail(f"REST API error: {e}")
        await client.aclose()
        return False


async def test_public_ticker() -> bool:
    """Test 2: Fetch public ticker data (no auth required)."""
    _header("Test 2: Public Ticker Data (REST)")

    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://www.okx.com/api/v5/market/ticker",
                params={"instId": "BTC-USDT"},
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("code") != "0":
                _fail(f"API error: {data.get('msg')}")
                return False

            ticker = data["data"][0]
            last_price = ticker["last"]
            vol_24h = ticker.get("vol24h", "N/A")
            vol_ccy = ticker.get("volCcy24h", "N/A")
            high = ticker.get("high24h", "N/A")
            low = ticker.get("low24h", "N/A")

            _ok(f"BTC-USDT Last: ${last_price}")
            _info(f"  24h High: ${high}")
            _info(f"  24h Low:  ${low}")
            _info(f"  24h Vol (BTC): {vol_24h}")
            _info(f"  24h Vol (USDT): {vol_ccy}")
            return True
    except Exception as e:
        _fail(f"Ticker fetch error: {e}")
        return False


async def test_multiple_instruments() -> bool:
    """Test 3: Fetch metadata for multiple trading pairs."""
    _header("Test 3: Multi-Instrument Metadata")

    import httpx

    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "DOGE-USDT", "XRP-USDT"]

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for symbol in symbols:
                resp = await client.get(
                    "https://www.okx.com/api/v5/market/ticker",
                    params={"instId": symbol},
                )
                resp.raise_for_status()
                data = resp.json()

                if data.get("code") == "0" and data["data"]:
                    ticker = data["data"][0]
                    _ok(f"{symbol}: ${ticker['last']}  Vol24h={ticker.get('volCcy24h', 'N/A')}")
                else:
                    _fail(f"{symbol}: API error")
                    return False
        return True
    except Exception as e:
        _fail(f"Multi-instrument error: {e}")
        return False


def test_historical_ohlcv() -> bool:
    """Test 4: Download historical OHLCV via real_data.py."""
    _header("Test 4: Historical OHLCV Download (real_data.py)")

    from quantbot.research.real_data import download_ohlcv

    try:
        bars = download_ohlcv(
            inst_id="BTC-USDT",
            bar_size="1D",
            start="2025-01-01",
            end="2025-01-31",
            use_cache=False,
        )

        if not bars:
            _fail("No bars returned")
            return False

        _ok(f"Downloaded {len(bars)} daily bars for BTC-USDT")
        _info(f"  Date range: {bars[0].ts.date()} -> {bars[-1].ts.date()}")
        _info(f"  First bar: O={bars[0].open} H={bars[0].high} L={bars[0].low} C={bars[0].close}")
        _info(f"  First bar volume: {bars[0].volume}")
        _info(f"  Last bar:  O={bars[-1].open} H={bars[-1].high} L={bars[-1].low} C={bars[-1].close}")

        # Verify volume is NOT zero (the fix we applied)
        nonzero_vol = sum(1 for b in bars if b.volume > Decimal("0"))
        _info(f"  Bars with non-zero volume: {nonzero_vol}/{len(bars)}")
        if nonzero_vol > 0:
            _ok("Volume data confirmed present")
        else:
            _fail("All bars have zero volume -- check API response format")
        return True
    except Exception as e:
        _fail(f"OHLCV download error: {e}")
        return False


def test_historical_funding() -> bool:
    """Test 5: Download historical funding rates via real_data.py."""
    _header("Test 5: Historical Funding Rates Download")

    from quantbot.research.real_data import download_funding

    try:
        # OKX only retains ~3 months of funding rate history
        # Use dates within the recent window
        from datetime import timedelta
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=14)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")

        _info(f"Fetching funding rates from {start_str} to {end_str}")
        rates = download_funding(
            inst_id="BTC-USDT-SWAP",
            start=start_str,
            end=end_str,
            use_cache=False,
        )

        if not rates:
            _fail("No funding rates returned")
            return False

        _ok(f"Downloaded {len(rates)} funding rate records")
        _info(f"  Date range: {rates[0].funding_time} -> {rates[-1].funding_time}")
        _info(f"  First rate: {rates[0].funding_rate}")
        _info(f"  Last rate:  {rates[-1].funding_rate}")

        # Stats
        avg_rate = sum(float(r.funding_rate) for r in rates) / len(rates)
        _info(f"  Average rate: {avg_rate:.6f}")
        return True
    except Exception as e:
        _fail(f"Funding download error: {e}")
        return False


def test_config_loading() -> bool:
    """Test 6: Verify BotConfig loads from .env correctly."""
    _header("Test 6: BotConfig Loading from .env")

    from quantbot.config import BotConfig

    try:
        config = BotConfig.from_env()
        _ok("Config loaded successfully")
        _info(f"  Environment: {config.okx_env}")
        _info(f"  Symbol: {config.okx_symbol}")
        _info(f"  API Key: {config.okx_api_key[:12]}...")
        _info(f"  REST Base: {config.rest_base_url}")
        _info(f"  Public WS: {config.public_ws_url}")
        _info(f"  Private WS: {config.private_ws_url}")

        if config.okx_env != "demo":
            _fail("Environment is NOT demo -- SAFETY CHECK FAILED")
            return False
        _ok("Demo mode confirmed")
        return True
    except Exception as e:
        _fail(f"Config loading error: {e}")
        return False


async def main() -> None:
    print("=" * 58)
    print("      OKX Demo API Connectivity Verification")
    print("      quantbot System Integration Test")
    print(f"      {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 58)

    results: dict[str, bool] = {}

    # Test 1: Config loading
    results["Config Loading"] = test_config_loading()

    # Test 2: REST authentication
    results["REST Auth"] = await test_rest_authentication()

    # Test 3: Public ticker
    results["Public Ticker"] = await test_public_ticker()

    # Test 4: Multi-instrument
    results["Multi-Instrument"] = await test_multiple_instruments()

    # Test 5: Historical OHLCV
    results["Historical OHLCV"] = test_historical_ohlcv()

    # Test 6: Historical Funding
    results["Historical Funding"] = test_historical_funding()

    # Summary
    _header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status}  {name}")

    print(f"\n  Result: {passed}/{total} tests passed")
    if passed == total:
        print("  >>> All systems operational -- OKX Demo API fully connected!")
    else:
        print("  >>> Some tests failed -- check output above for details")
    print()


if __name__ == "__main__":
    asyncio.run(main())
