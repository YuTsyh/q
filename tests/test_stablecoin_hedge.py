"""Tests for dynamic stablecoin hedging and volatility scaling."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quantbot.research.data import OhlcvBar
from quantbot.research.markov_regime import MarkovRegimeState
from quantbot.research.stablecoin_hedge import (
    AdaptivePortfolioConstructor,
    StablecoinHedgeConfig,
    StablecoinHedger,
    VolatilityScaler,
    VolatilityScalingConfig,
)


# ---------------------------------------------------------------------------
# Inline GBM bar generator
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")


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
# VolatilityScaler tests
# ---------------------------------------------------------------------------


class TestVolatilityScaler:
    def test_scale_weights_adjusts_by_vol(self) -> None:
        scaler = VolatilityScaler()
        weights = {"BTC": Decimal("0.5"), "ETH": Decimal("0.5")}
        bars_by_inst = {
            "BTC": _make_bars(50, vol=0.01, seed=1, inst_id="BTC"),
            "ETH": _make_bars(50, vol=0.05, seed=2, inst_id="ETH"),
        }
        scaled = scaler.scale_weights(weights, bars_by_inst)
        # Both should still exist and be non-zero
        assert "BTC" in scaled
        assert "ETH" in scaled
        assert scaled["BTC"] != _ZERO
        assert scaled["ETH"] != _ZERO
        # Gross exposure should be preserved
        original_gross = sum(abs(w) for w in weights.values())
        scaled_gross = sum(abs(w) for w in scaled.values())
        assert abs(float(scaled_gross - original_gross)) < 1e-6

    def test_empty_weights_returns_empty(self) -> None:
        scaler = VolatilityScaler()
        assert scaler.scale_weights({}, {}) == {}

    def test_zero_weight_stays_zero(self) -> None:
        scaler = VolatilityScaler()
        weights = {"BTC": Decimal("0.5"), "ETH": _ZERO}
        bars_by_inst = {
            "BTC": _make_bars(50, seed=1, inst_id="BTC"),
            "ETH": _make_bars(50, seed=2, inst_id="ETH"),
        }
        scaled = scaler.scale_weights(weights, bars_by_inst)
        assert scaled["ETH"] == _ZERO

    def test_missing_bars_leaves_weight_unchanged(self) -> None:
        scaler = VolatilityScaler()
        weights = {"BTC": Decimal("0.5"), "MISSING": Decimal("0.5")}
        bars_by_inst = {
            "BTC": _make_bars(50, seed=1, inst_id="BTC"),
        }
        scaled = scaler.scale_weights(weights, bars_by_inst)
        assert "MISSING" in scaled

    def test_compute_realized_vol_reasonable(self) -> None:
        scaler = VolatilityScaler()
        bars = _make_bars(50, vol=0.02)
        vol = scaler.compute_realized_vol(bars)
        assert isinstance(vol, float)
        assert vol > 0.0
        # Annualised vol from daily 2% vol ≈ 0.02 * sqrt(365) ≈ 0.38
        assert 0.05 <= vol <= 1.5

    def test_compute_realized_vol_single_bar_returns_floor(self) -> None:
        scaler = VolatilityScaler()
        bars = _make_bars(1)
        vol = scaler.compute_realized_vol(bars)
        assert vol == scaler.config.vol_floor

    def test_custom_config(self) -> None:
        cfg = VolatilityScalingConfig(target_volatility=0.30, vol_lookback=10)
        scaler = VolatilityScaler(cfg)
        weights = {"BTC": Decimal("1.0")}
        bars_by_inst = {"BTC": _make_bars(30, seed=1, inst_id="BTC")}
        scaled = scaler.scale_weights(weights, bars_by_inst)
        assert "BTC" in scaled


# ---------------------------------------------------------------------------
# StablecoinHedger tests
# ---------------------------------------------------------------------------


class TestStablecoinHedger:
    def _weights(self) -> dict[str, Decimal]:
        return {"BTC": Decimal("0.5"), "ETH": Decimal("0.5")}

    def test_crisis_80_percent_stablecoin(self) -> None:
        hedger = StablecoinHedger()
        result = hedger.apply_hedge(self._weights(), MarkovRegimeState.CRISIS)
        stable_ids = set(hedger.config.stablecoin_ids)
        stable_weight = sum(abs(result[k]) for k in result if k in stable_ids)
        gross = sum(abs(v) for v in result.values())
        assert abs(float(stable_weight / gross) - 0.80) < 0.01

    def test_high_vol_40_percent_stablecoin(self) -> None:
        hedger = StablecoinHedger()
        result = hedger.apply_hedge(self._weights(), MarkovRegimeState.HIGH_VOL_LOW_LIQ)
        stable_ids = set(hedger.config.stablecoin_ids)
        stable_weight = sum(abs(result[k]) for k in result if k in stable_ids)
        gross = sum(abs(v) for v in result.values())
        assert abs(float(stable_weight / gross) - 0.40) < 0.01

    def test_mid_vol_10_percent_stablecoin(self) -> None:
        hedger = StablecoinHedger()
        result = hedger.apply_hedge(self._weights(), MarkovRegimeState.MID_VOL_MID_LIQ)
        stable_ids = set(hedger.config.stablecoin_ids)
        stable_weight = sum(abs(result[k]) for k in result if k in stable_ids)
        gross = sum(abs(v) for v in result.values())
        assert abs(float(stable_weight / gross) - 0.10) < 0.01

    def test_low_vol_0_percent_stablecoin(self) -> None:
        hedger = StablecoinHedger()
        result = hedger.apply_hedge(self._weights(), MarkovRegimeState.LOW_VOL_HIGH_LIQ)
        stable_ids = set(hedger.config.stablecoin_ids)
        stable_weight = sum(abs(result.get(k, _ZERO)) for k in stable_ids)
        # 0% target; change is 0% vs. 0% current → below threshold → returns unchanged
        # The original weights had no stablecoins → current is 0% → target 0% → no change
        assert stable_weight == _ZERO or float(stable_weight) < 0.01

    def test_empty_weights_returns_empty(self) -> None:
        hedger = StablecoinHedger()
        assert hedger.apply_hedge({}, MarkovRegimeState.CRISIS) == {}

    def test_gross_exposure_preserved(self) -> None:
        hedger = StablecoinHedger()
        weights = self._weights()
        original_gross = sum(abs(w) for w in weights.values())
        result = hedger.apply_hedge(weights, MarkovRegimeState.CRISIS)
        result_gross = sum(abs(w) for w in result.values())
        assert abs(float(result_gross - original_gross)) < 0.01


# ---------------------------------------------------------------------------
# AdaptivePortfolioConstructor tests
# ---------------------------------------------------------------------------


class TestAdaptivePortfolioConstructor:
    def test_construct_full_pipeline(self) -> None:
        constructor = AdaptivePortfolioConstructor()
        raw_weights = {"BTC": Decimal("0.5"), "ETH": Decimal("0.5")}
        bars_by_inst = {
            "BTC": _make_bars(50, vol=0.02, seed=1, inst_id="BTC"),
            "ETH": _make_bars(50, vol=0.03, seed=2, inst_id="ETH"),
        }
        result = constructor.construct(
            raw_weights, MarkovRegimeState.MID_VOL_MID_LIQ, bars_by_inst
        )
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_construct_with_crisis_adds_stablecoins(self) -> None:
        constructor = AdaptivePortfolioConstructor()
        raw_weights = {"BTC": Decimal("0.5"), "ETH": Decimal("0.5")}
        bars_by_inst = {
            "BTC": _make_bars(50, seed=1, inst_id="BTC"),
            "ETH": _make_bars(50, seed=2, inst_id="ETH"),
        }
        result = constructor.construct(
            raw_weights, MarkovRegimeState.CRISIS, bars_by_inst
        )
        # Crisis should inject stablecoin allocations
        stable_ids = {"USDT", "USDC"}
        has_stable = any(k in stable_ids for k in result)
        assert has_stable

    def test_empty_weights_returns_empty(self) -> None:
        constructor = AdaptivePortfolioConstructor()
        result = constructor.construct(
            {}, MarkovRegimeState.LOW_VOL_HIGH_LIQ, {}
        )
        assert result == {}

    def test_construct_preserves_instrument_ids(self) -> None:
        constructor = AdaptivePortfolioConstructor()
        raw_weights = {"BTC": Decimal("0.5"), "ETH": Decimal("0.5")}
        bars_by_inst = {
            "BTC": _make_bars(50, seed=1, inst_id="BTC"),
            "ETH": _make_bars(50, seed=2, inst_id="ETH"),
        }
        result = constructor.construct(
            raw_weights, MarkovRegimeState.HIGH_VOL_LOW_LIQ, bars_by_inst
        )
        # Original instruments should still be present
        assert "BTC" in result
        assert "ETH" in result
