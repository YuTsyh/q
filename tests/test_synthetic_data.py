"""Tests for synthetic market data generator."""

from __future__ import annotations



from quantbot.research.synthetic_data import (
    FULL_CYCLE_REGIMES,
    MarketRegime,
    generate_funding_rates,
    generate_multi_instrument_data,
    generate_ohlcv,
)


class TestGenerateOhlcv:
    def test_correct_length(self):
        regime = MarketRegime("test", 0.0, 0.01, 50)
        bars = generate_ohlcv("BTC-USDT", [regime], seed=42)
        assert len(bars) == 50

    def test_multiple_regimes(self):
        bars = generate_ohlcv("ETH-USDT", FULL_CYCLE_REGIMES, seed=42)
        expected = sum(r.duration_bars for r in FULL_CYCLE_REGIMES)
        assert len(bars) == expected

    def test_deterministic_with_seed(self):
        regime = MarketRegime("test", 0.001, 0.02, 20)
        b1 = generate_ohlcv("X", [regime], seed=123)
        b2 = generate_ohlcv("X", [regime], seed=123)
        assert [b.close for b in b1] == [b.close for b in b2]

    def test_prices_positive(self):
        regime = MarketRegime("volatile", -0.01, 0.1, 100)
        bars = generate_ohlcv("X", [regime], seed=42)
        for b in bars:
            assert b.close > 0
            assert b.high >= b.low

    def test_inst_id_preserved(self):
        regime = MarketRegime("test", 0.0, 0.01, 5)
        bars = generate_ohlcv("SOL-USDT", [regime])
        for b in bars:
            assert b.inst_id == "SOL-USDT"

    def test_timestamps_sequential(self):
        regime = MarketRegime("test", 0.0, 0.01, 10)
        bars = generate_ohlcv("X", [regime])
        for i in range(1, len(bars)):
            assert bars[i].ts > bars[i - 1].ts


class TestGenerateFundingRates:
    def test_correct_count(self):
        rates = generate_funding_rates("BTC-USDT", 30, seed=42)
        assert len(rates) == 30

    def test_deterministic(self):
        r1 = generate_funding_rates("X", 10, seed=99)
        r2 = generate_funding_rates("X", 10, seed=99)
        assert [r.funding_rate for r in r1] == [r.funding_rate for r in r2]

    def test_inst_id_preserved(self):
        rates = generate_funding_rates("ETH-USDT-SWAP", 5)
        for r in rates:
            assert r.inst_id == "ETH-USDT-SWAP"


class TestGenerateMultiInstrumentData:
    def test_returns_all_instruments(self):
        ids = ["BTC-USDT-SWAP", "ETH-USDT-SWAP"]
        bars, funding = generate_multi_instrument_data(ids)
        assert set(bars.keys()) == set(ids)
        assert set(funding.keys()) == set(ids)

    def test_instruments_have_different_data(self):
        ids = ["A-USDT-SWAP", "B-USDT-SWAP"]
        bars, _ = generate_multi_instrument_data(ids)
        a_closes = [b.close for b in bars["A-USDT-SWAP"]]
        b_closes = [b.close for b in bars["B-USDT-SWAP"]]
        assert a_closes != b_closes

    def test_bull_regime_generally_trends_up(self):
        ids = ["X-USDT-SWAP"]
        bull = MarketRegime("bull", drift=0.005, volatility=0.01, duration_bars=100)
        bars, _ = generate_multi_instrument_data(ids, [bull], seed_base=42)
        prices = [float(b.close) for b in bars["X-USDT-SWAP"]]
        assert prices[-1] > prices[0]  # Should trend up with strong drift
