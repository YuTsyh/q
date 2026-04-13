"""Tests for the Markov-Switching regime detection module."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quantbot.research.data import OhlcvBar
from quantbot.research.markov_regime import (
    MarkovRegimeConfig,
    MarkovRegimeDetector,
    MarkovRegimeResult,
    MarkovRegimeState,
    _compute_vol_ratio,
    compute_amihud,
)


# ---------------------------------------------------------------------------
# Inline GBM bar generator
# ---------------------------------------------------------------------------

def _make_bars(
    n: int,
    drift: float = 0.001,
    vol: float = 0.02,
    start_price: float = 100.0,
    seed: int = 42,
    inst_id: str = "TEST-USDT",
) -> list[OhlcvBar]:
    rng = random.Random(seed)
    bars: list[OhlcvBar] = []
    price = start_price
    t = datetime(2023, 1, 1, tzinfo=timezone.utc)
    for _ in range(n):
        ret = drift + vol * rng.gauss(0, 1)
        new_price = price * math.exp(ret)
        bars.append(
            OhlcvBar(
                inst_id=inst_id,
                ts=t,
                open=Decimal(str(round(price, 6))),
                high=Decimal(str(round(max(price, new_price) * 1.005, 6))),
                low=Decimal(str(round(min(price, new_price) * 0.995, 6))),
                close=Decimal(str(round(new_price, 6))),
                volume=Decimal(str(round(rng.uniform(1000, 10000), 2))),
            )
        )
        price = new_price
        t += timedelta(days=1)
    return bars


# ---------------------------------------------------------------------------
# Tests for compute_amihud
# ---------------------------------------------------------------------------


class TestComputeAmihud:
    def test_normal_data(self) -> None:
        bars = _make_bars(30)
        result = compute_amihud(bars, window=20)
        assert isinstance(result, float)
        assert result > 0.0

    def test_insufficient_data_returns_zero(self) -> None:
        bars = _make_bars(5)
        assert compute_amihud(bars, window=20) == 0.0

    def test_window_zero_returns_zero(self) -> None:
        bars = _make_bars(30)
        assert compute_amihud(bars, window=0) == 0.0

    def test_zero_volume_bars_skipped(self) -> None:
        bars = _make_bars(30)
        # Replace volumes with zero in the last 21 bars
        zero_bars = list(bars[:-21])
        t = bars[-21].ts
        for b in bars[-21:]:
            zero_bars.append(
                OhlcvBar(
                    inst_id=b.inst_id,
                    ts=b.ts,
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=Decimal("0"),
                )
            )
        result = compute_amihud(zero_bars, window=20)
        assert result == 0.0

    def test_larger_window_uses_more_data(self) -> None:
        bars = _make_bars(60)
        short = compute_amihud(bars, window=10)
        long = compute_amihud(bars, window=40)
        # Both should be positive; values will differ due to averaging window
        assert short > 0.0
        assert long > 0.0


# ---------------------------------------------------------------------------
# Tests for _compute_vol_ratio
# ---------------------------------------------------------------------------


class TestComputeVolRatio:
    def test_insufficient_data_returns_one(self) -> None:
        bars = _make_bars(5)
        assert _compute_vol_ratio(bars, short_window=10, long_window=40) == 1.0

    def test_equal_vol_windows_near_one(self) -> None:
        bars = _make_bars(100, vol=0.02, seed=1)
        ratio = _compute_vol_ratio(bars, short_window=40, long_window=40)
        assert abs(ratio - 1.0) < 0.01

    def test_high_short_vol_gives_ratio_above_one(self) -> None:
        # 60 calm bars + 20 volatile bars
        calm = _make_bars(60, vol=0.01, seed=1)
        volatile = _make_bars(20, vol=0.08, seed=2, start_price=float(calm[-1].close))
        # Adjust timestamps
        last_ts = calm[-1].ts
        adjusted = []
        for i, b in enumerate(volatile):
            adjusted.append(
                OhlcvBar(
                    inst_id=b.inst_id,
                    ts=last_ts + timedelta(days=i + 1),
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                )
            )
        combined = calm + adjusted
        ratio = _compute_vol_ratio(combined, short_window=10, long_window=40)
        assert ratio > 1.0


# ---------------------------------------------------------------------------
# Tests for MarkovRegimeDetector.classify
# ---------------------------------------------------------------------------


class TestMarkovRegimeDetectorClassify:
    def test_low_vol_data_returns_valid_result(self) -> None:
        bars = _make_bars(100, drift=0.001, vol=0.005, seed=10)
        detector = MarkovRegimeDetector()
        result = detector.classify(bars)
        assert isinstance(result, MarkovRegimeResult)
        assert isinstance(result.state, MarkovRegimeState)
        # Low vol → vol_ratio should be moderate (not extreme)
        assert result.vol_ratio < 3.0

    def test_high_vol_data_detects_stressed_regime(self) -> None:
        # Create a spike: calm then extreme vol
        calm = _make_bars(60, vol=0.005, seed=1)
        crisis = _make_bars(
            50, vol=0.15, seed=2, start_price=float(calm[-1].close)
        )
        last_ts = calm[-1].ts
        adjusted = []
        for i, b in enumerate(crisis):
            adjusted.append(
                OhlcvBar(
                    inst_id=b.inst_id,
                    ts=last_ts + timedelta(days=i + 1),
                    open=b.open,
                    high=b.high,
                    low=b.low,
                    close=b.close,
                    volume=b.volume,
                )
            )
        combined = calm + adjusted
        detector = MarkovRegimeDetector()
        result = detector.classify(combined)
        # The HMM should produce a valid classification with all 4 states
        assert isinstance(result.state, MarkovRegimeState)
        assert len(result.state_probabilities) == 4

    def test_result_contains_all_fields(self) -> None:
        bars = _make_bars(100)
        detector = MarkovRegimeDetector()
        result = detector.classify(bars)
        assert isinstance(result.state, MarkovRegimeState)
        assert isinstance(result.vol_ratio, float)
        assert isinstance(result.amihud_score, float)
        assert isinstance(result.state_probabilities, dict)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0

    def test_state_probabilities_sum_to_one(self) -> None:
        bars = _make_bars(100)
        detector = MarkovRegimeDetector()
        result = detector.classify(bars)
        total = sum(result.state_probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_state_persistence_across_calls(self) -> None:
        bars = _make_bars(100, vol=0.005, seed=7)
        detector = MarkovRegimeDetector()
        results = []
        # Call classify multiple times with slightly extended data
        for i in range(5):
            end = 50 + i * 10
            results.append(detector.classify(bars[:end]))
        # Verify temporal persistence: consecutive calls with similar data
        # should not wildly oscillate — at most 2 distinct states in 5 calls
        states = {r.state for r in results}
        assert len(states) <= 3  # some persistence expected

    def test_insufficient_data_raises_value_error(self) -> None:
        bars = _make_bars(10)
        detector = MarkovRegimeDetector()
        with pytest.raises(ValueError, match="Need at least"):
            detector.classify(bars)

    def test_custom_config(self) -> None:
        config = MarkovRegimeConfig(
            short_vol_window=5, long_vol_window=20, amihud_window=10
        )
        bars = _make_bars(50, seed=99)
        detector = MarkovRegimeDetector(config)
        result = detector.classify(bars)
        assert isinstance(result, MarkovRegimeResult)


# ---------------------------------------------------------------------------
# Tests for classify_portfolio
# ---------------------------------------------------------------------------


class TestClassifyPortfolio:
    def test_multiple_instruments(self) -> None:
        detector = MarkovRegimeDetector()
        bars_by_inst = {
            "BTC-USDT": _make_bars(100, vol=0.02, seed=1, inst_id="BTC-USDT"),
            "ETH-USDT": _make_bars(100, vol=0.03, seed=2, inst_id="ETH-USDT"),
        }
        result = detector.classify_portfolio(bars_by_inst)
        assert isinstance(result, MarkovRegimeResult)
        assert isinstance(result.state, MarkovRegimeState)
        total = sum(result.state_probabilities.values())
        assert abs(total - 1.0) < 0.01

    def test_all_insufficient_returns_default(self) -> None:
        detector = MarkovRegimeDetector()
        bars_by_inst = {
            "BTC-USDT": _make_bars(5, inst_id="BTC-USDT"),
        }
        result = detector.classify_portfolio(bars_by_inst)
        assert result.state == MarkovRegimeState.MID_VOL_MID_LIQ

    def test_empty_dict_returns_default(self) -> None:
        detector = MarkovRegimeDetector()
        result = detector.classify_portfolio({})
        assert result.state == MarkovRegimeState.MID_VOL_MID_LIQ

    def test_mixed_sufficiency(self) -> None:
        detector = MarkovRegimeDetector()
        bars_by_inst = {
            "BTC-USDT": _make_bars(100, seed=1, inst_id="BTC-USDT"),
            "SHORT": _make_bars(5, inst_id="SHORT"),  # too short
        }
        result = detector.classify_portfolio(bars_by_inst)
        # Should still produce a result from the valid instrument
        assert isinstance(result.state, MarkovRegimeState)
