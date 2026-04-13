"""Crypto-native alpha factors for digital asset strategy construction.

Extends the base factor set with crypto-specific signals:
- NvtRatioFactor: Network Value to Transactions proxy
- OrderImbalanceFactor: Volume-weighted buy/sell pressure
- FundingRateSpreadFactor: Two-tiered funding rate arbitrage signal
- AmihudIlliquidityFactor: Amihud (2002) illiquidity measure
- VolatilityOfVolatilityFactor: Instability of volatility regime
- RsiFactory: Relative Strength Index centered around zero
- BollingerBandWidthFactor: Bollinger Band compression detection
- VwapDeviationFactor: Price deviation from VWAP
- OBVTrendFactor: On-Balance Volume trend slope
- CTrendAggregateFactor: Elastic Net regularised multi-factor aggregate
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from quantbot.research.data import FundingRate, OhlcvBar

if TYPE_CHECKING:
    from quantbot.research.factors import Factor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO = Decimal("0")
_ONE = Decimal("1")


def _returns(bars: list[OhlcvBar], n: int) -> list[float]:
    """Compute simple returns for the last *n* periods."""
    out: list[float] = []
    for i in range(-n, 0):
        prev = bars[i - 1].close
        curr = bars[i].close
        if prev > 0:
            out.append(float(curr / prev - 1))
    return out


def _std(values: list[float]) -> float:
    """Sample standard deviation (Bessel-corrected)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# Factor implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NvtRatioFactor:
    """Network Value to Transactions ratio proxy.

    Approximates NVT using ``price * lookback / volume_sum``.  A lower NVT
    implies the asset is undervalued relative to its on-chain transaction
    volume, so the factor returns **-NVT** so that cheaper assets rank higher
    in a descending sort.
    """

    lookback: int = 14
    name: str = "nvt_ratio"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback:
            raise ValueError("not enough bars for nvt_ratio")
        recent = bars[-self.lookback :]
        volume_sum = sum((b.volume for b in recent), _ZERO)
        if volume_sum <= 0:
            return _ZERO
        price = bars[-1].close
        nvt = price * Decimal(self.lookback) / volume_sum
        return -nvt


@dataclass(frozen=True)
class OrderImbalanceFactor:
    """Approximate order-flow imbalance from OHLCV data.

    For each bar the full bar volume is attributed to buyers when
    ``close > open`` and to sellers when ``close < open``.  The imbalance is
    ``(buy_vol - sell_vol) / total_vol`` over the lookback window.
    """

    lookback: int = 10
    name: str = "order_imbalance"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback:
            raise ValueError("not enough bars for order_imbalance")
        recent = bars[-self.lookback :]
        buy_vol = _ZERO
        sell_vol = _ZERO
        for b in recent:
            if b.close > b.open:
                buy_vol += b.volume
            elif b.close < b.open:
                sell_vol += b.volume
            # close == open contributes to neither side
        total = buy_vol + sell_vol
        if total <= 0:
            return _ZERO
        return (buy_vol - sell_vol) / total


@dataclass(frozen=True)
class FundingRateSpreadFactor:
    """Two-tiered funding rate spread.

    Computes the spread between the short-window average funding rate and
    its longer-term average.  A positive spread (recent funding > long-term)
    signals an overheated market, so the factor returns **-spread** to favour
    assets where the current rate is below average (funding arbitrage signal).
    """

    short_window: int = 3
    long_window: int = 14
    name: str = "funding_spread"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(funding) < self.long_window:
            raise ValueError("not enough funding observations for funding_spread")
        short = funding[-self.short_window :]
        long = funding[-self.long_window :]
        short_avg = sum((f.funding_rate for f in short), _ZERO) / Decimal(len(short))
        long_avg = sum((f.funding_rate for f in long), _ZERO) / Decimal(len(long))
        spread = short_avg - long_avg
        return -spread


@dataclass(frozen=True)
class AmihudIlliquidityFactor:
    """Amihud (2002) illiquidity ratio.

    Defined as ``mean(|return_i| / volume_i)`` over the lookback window.
    More liquid assets have lower Amihud values; the factor returns
    **-illiquidity** so that liquid assets rank higher.
    """

    lookback: int = 20
    name: str = "amihud_illiquidity"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback + 1:
            raise ValueError("not enough bars for amihud_illiquidity")
        ratios: list[float] = []
        for i in range(-self.lookback, 0):
            prev = bars[i - 1].close
            curr = bars[i].close
            vol = bars[i].volume
            if prev > 0 and vol > 0:
                abs_ret = abs(float(curr / prev - 1))
                ratios.append(abs_ret / float(vol))
        if not ratios:
            return _ZERO
        illiquidity = sum(ratios) / len(ratios)
        return Decimal(str(-illiquidity))


@dataclass(frozen=True)
class VolatilityOfVolatilityFactor:
    """Standard deviation of rolling volatility estimates.

    High vol-of-vol indicates an unstable volatility regime, which is a
    negative signal.  The factor returns **-vol_of_vol**.
    """

    lookback: int = 20
    inner_window: int = 5
    name: str = "vol_of_vol"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        required = self.lookback + self.inner_window
        if len(bars) < required:
            raise ValueError("not enough bars for vol_of_vol")
        # Compute rolling volatility for each position in the lookback window
        vols: list[float] = []
        for offset in range(self.lookback):
            end = len(bars) - self.lookback + offset + 1
            start = end - self.inner_window - 1
            window_returns: list[float] = []
            for j in range(start + 1, end):
                prev = bars[j - 1].close
                curr = bars[j].close
                if prev > 0:
                    window_returns.append(float(curr / prev - 1))
            if len(window_returns) >= 2:
                vols.append(_std(window_returns))
        if len(vols) < 2:
            return _ZERO
        return Decimal(str(-_std(vols)))


@dataclass(frozen=True)
class RsiFactory:
    """Relative Strength Index (RSI).

    Standard Wilder RSI calculation.  Returns ``(RSI - 50) / 50`` to centre
    the signal around zero: positive when bullish, negative when bearish.
    """

    lookback: int = 14
    name: str = "rsi"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback + 1:
            raise ValueError("not enough bars for rsi")
        gains: list[float] = []
        losses: list[float] = []
        for i in range(-self.lookback, 0):
            change = float(bars[i].close - bars[i - 1].close)
            if change >= 0:
                gains.append(change)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(-change)
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0.0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
        return Decimal(str((rsi - 50.0) / 50.0))


@dataclass(frozen=True)
class BollingerBandWidthFactor:
    """Bollinger Band width as a volatility-compression signal.

    Width = ``(upper - lower) / middle``.  Tighter bands signal potential
    breakouts; the factor returns **-width** so compressed states rank
    higher.
    """

    lookback: int = 20
    num_std: float = 2.0
    name: str = "bb_width"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback:
            raise ValueError("not enough bars for bb_width")
        recent = bars[-self.lookback :]
        closes = [float(b.close) for b in recent]
        middle = sum(closes) / len(closes)
        if middle == 0.0:
            return _ZERO
        std = _std(closes)
        upper = middle + self.num_std * std
        lower = middle - self.num_std * std
        width = (upper - lower) / middle
        return Decimal(str(-width))


@dataclass(frozen=True)
class VwapDeviationFactor:
    """Volume-weighted average price deviation.

    ``VWAP = sum(typical_price * volume) / sum(volume)``
    ``deviation = (close - VWAP) / VWAP``

    Positive deviation means the asset is trading above its VWAP (bullish).
    """

    lookback: int = 10
    name: str = "vwap_deviation"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback:
            raise ValueError("not enough bars for vwap_deviation")
        recent = bars[-self.lookback :]
        tp_vol_sum = _ZERO
        vol_sum = _ZERO
        for b in recent:
            typical = (b.high + b.low + b.close) / Decimal("3")
            tp_vol_sum += typical * b.volume
            vol_sum += b.volume
        if vol_sum <= 0:
            return _ZERO
        vwap = tp_vol_sum / vol_sum
        if vwap == 0:
            return _ZERO
        return (bars[-1].close - vwap) / vwap


@dataclass(frozen=True)
class OBVTrendFactor:
    """On-Balance Volume trend (regression slope normalised).

    Accumulates OBV over the lookback window then estimates a linear trend
    via a simple least-squares slope normalised by the mean absolute OBV.
    Positive slope = accumulation, negative = distribution.
    """

    lookback: int = 20
    name: str = "obv_trend"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback + 1:
            raise ValueError("not enough bars for obv_trend")
        # Build OBV series over the lookback window
        obv_series: list[float] = [0.0]
        for i in range(-self.lookback, 0):
            prev_close = bars[i - 1].close
            curr_close = bars[i].close
            vol = float(bars[i].volume)
            if curr_close > prev_close:
                obv_series.append(obv_series[-1] + vol)
            elif curr_close < prev_close:
                obv_series.append(obv_series[-1] - vol)
            else:
                obv_series.append(obv_series[-1])

        n = len(obv_series)
        # Linear regression slope: slope = cov(x,y) / var(x)
        x_mean = (n - 1) / 2.0
        y_mean = sum(obv_series) / n
        cov_xy = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(obv_series))
        var_x = sum((i - x_mean) ** 2 for i in range(n))
        if var_x == 0:
            return _ZERO
        slope = cov_xy / var_x

        # Normalise by mean absolute OBV to make comparable across assets
        mean_abs_obv = sum(abs(y) for y in obv_series) / n
        if mean_abs_obv < 1e-12:
            return _ZERO
        return Decimal(str(slope / mean_abs_obv))


@dataclass(frozen=True)
class CTrendAggregateFactor:
    """Aggregates multiple sub-factors using Elastic Net-style regularisation.

    For each sub-factor, a weight is computed as::

        w_i = 1 / (1 + l1_penalty * |prev_w_i| + l2_penalty * prev_w_i^2)

    where ``prev_w_i`` defaults to the prior iteration's normalised weight
    (initialised to ``1/N``).  After one pass the weights are normalised to
    sum to 1 and the final score is the weighted average of sub-factor values.

    This provides a simple shrinkage mechanism that down-weights factors
    whose previous contribution was large (L1 sparsity) while also penalising
    very large weights (L2 ridge).
    """

    factors: tuple[Factor, ...] = ()
    l1_penalty: float = 0.1
    l2_penalty: float = 0.1
    name: str = "ctrend_aggregate"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if not self.factors:
            return _ZERO

        # Compute all sub-factor values
        values: list[Decimal] = []
        for f in self.factors:
            values.append(f.compute(bars, funding))

        n = len(values)
        # Initialise equal weights
        prev_weights = [1.0 / n] * n

        # Single-pass Elastic Net weight update
        raw_weights: list[float] = []
        for pw in prev_weights:
            denom = 1.0 + self.l1_penalty * abs(pw) + self.l2_penalty * pw * pw
            raw_weights.append(1.0 / denom)

        weight_sum = sum(raw_weights)
        if weight_sum < 1e-15:
            return _ZERO
        norm_weights = [w / weight_sum for w in raw_weights]

        result = _ZERO
        for w, v in zip(norm_weights, values):
            result += Decimal(str(w)) * v
        return result
