"""Volatility-adjusted factors for robust strategy construction.

Extends the base factor set with:
- VolatilityFactor: Historical annualised volatility
- VolAdjMomentumFactor: Risk-adjusted momentum (momentum / volatility)
- MeanReversionFactor: Distance from moving average (oversold detection)
- DualMomentumFactor: Combines absolute and relative momentum
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from quantbot.research.data import FundingRate, OhlcvBar


@dataclass(frozen=True)
class VolatilityFactor:
    """Historical close-to-close volatility.

    Lower volatility scores higher (inverse, for rank-based systems
    the scorer handles direction via _rank_desc).
    """

    lookback: int
    name: str = "volatility"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback + 1:
            raise ValueError("not enough bars for volatility")
        returns = []
        for i in range(-self.lookback, 0):
            prev = bars[i - 1].close
            curr = bars[i].close
            if prev > 0:
                returns.append(float(curr / prev - 1))
        if len(returns) < 2:
            return Decimal("0")
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        vol = math.sqrt(var)
        # Return negative so lower volatility ranks higher in _rank_desc
        return Decimal(str(-vol))


@dataclass(frozen=True)
class VolAdjMomentumFactor:
    """Volatility-adjusted momentum: momentum / volatility.

    Based on the insight that risk-adjusted returns are better predictors
    of future performance than raw momentum. Reference:
    Moskowitz, Ooi & Pedersen (2012) "Time Series Momentum"
    """

    lookback: int
    name: str = "vol_adj_momentum"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback + 1:
            raise ValueError("not enough bars for vol-adj momentum")

        # Raw momentum
        start = bars[-self.lookback - 1].close
        end = bars[-1].close
        if start == 0:
            return Decimal("0")
        momentum = float(end / start - 1)

        # Volatility over lookback
        returns = []
        for i in range(-self.lookback, 0):
            prev = bars[i - 1].close
            curr = bars[i].close
            if prev > 0:
                returns.append(float(curr / prev - 1))
        if len(returns) < 2:
            return Decimal("0")
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        vol = math.sqrt(var)

        if vol < 1e-12:
            return Decimal("0")
        return Decimal(str(momentum / vol))


@dataclass(frozen=True)
class MeanReversionFactor:
    """Mean reversion: how far price has deviated below its moving average.

    Higher values indicate more oversold conditions (potential long entry).
    Formula: SMA / close - 1  (positive when price below SMA)
    """

    lookback: int
    name: str = "mean_reversion"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback:
            raise ValueError("not enough bars for mean reversion")
        recent = bars[-self.lookback:]
        sma = sum((b.close for b in recent), Decimal("0")) / Decimal(len(recent))
        if bars[-1].close == 0:
            return Decimal("0")
        # Positive when price below SMA (oversold)
        return sma / bars[-1].close - Decimal("1")


@dataclass(frozen=True)
class DualMomentumFactor:
    """Dual momentum: combines absolute and relative momentum.

    Based on Gary Antonacci's Dual Momentum framework:
    - Absolute momentum: Is the asset trending up? (momentum > 0)
    - Cross-sectional: Which asset is trending most?
    Returns momentum value only if absolute momentum is positive,
    otherwise returns a penalty score.
    """

    lookback: int
    name: str = "dual_momentum"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        if len(bars) < self.lookback + 1:
            raise ValueError("not enough bars for dual momentum")
        start = bars[-self.lookback - 1].close
        end = bars[-1].close
        if start == 0:
            return Decimal("-1")
        momentum = end / start - Decimal("1")
        # If absolute momentum is negative, penalise heavily
        if momentum < 0:
            return Decimal("-1") + momentum  # Deep penalty
        return momentum


@dataclass(frozen=True)
class TrendStrengthFactor:
    """Trend strength using price relative to multiple moving averages.

    Counts how many MAs the price is above, normalised.
    Inspired by multi-timeframe trend confirmation.
    """

    lookbacks: tuple[int, ...] = (5, 10, 20)
    name: str = "trend_strength"

    def compute(self, bars: list[OhlcvBar], funding: list[FundingRate]) -> Decimal:
        max_lb = max(self.lookbacks)
        if len(bars) < max_lb:
            raise ValueError("not enough bars for trend strength")
        current = bars[-1].close
        above_count = 0
        for lb in self.lookbacks:
            recent = bars[-lb:]
            sma = sum((b.close for b in recent), Decimal("0")) / Decimal(len(recent))
            if current > sma:
                above_count += 1
        return Decimal(str(above_count / len(self.lookbacks)))
