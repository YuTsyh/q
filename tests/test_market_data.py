from decimal import Decimal

from quantbot.market_data.okx import OkxTickerParser


def test_okx_ticker_parser_extracts_last_price_and_timestamp() -> None:
    message = {
        "arg": {"channel": "tickers", "instId": "BTC-USDT"},
        "data": [
            {
                "instId": "BTC-USDT",
                "last": "65000.1",
                "ts": "1710000000000",
            }
        ],
    }

    snapshot = OkxTickerParser.parse(message)

    assert snapshot.inst_id == "BTC-USDT"
    assert snapshot.last_price == Decimal("65000.1")
    assert snapshot.received_at.isoformat() == "2024-03-09T16:00:00+00:00"


def test_okx_ticker_parser_rejects_non_ticker_messages() -> None:
    message = {"event": "subscribe", "arg": {"channel": "tickers", "instId": "BTC-USDT"}}

    try:
        OkxTickerParser.parse(message)
    except ValueError as exc:
        assert "ticker data" in str(exc)
    else:
        raise AssertionError("non-ticker message should be rejected")

