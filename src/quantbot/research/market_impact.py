"""Non-linear market impact model and enhanced execution simulator.

Implements the Almgren-Chriss (2001) framework for optimal execution with
square-root impact following Bouchaud et al. (2009).

References
----------
- Almgren, R. & Chriss, N. (2001). "Optimal execution of portfolio
  transactions." *Journal of Risk*, 3(2), 5-39.
- Bouchaud, J.-P., Farmer, J.D. & Lillo, F. (2009). "How markets slowly
  digest changes in supply and demand." In *Handbook of Financial Markets:
  Dynamics and Evolution*, 57-160. Elsevier.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from quantbot.research.data import OhlcvBar

_ZERO = Decimal("0")
_ONE = Decimal("1")
_QUANTIZE_PRICE = Decimal("0.00000001")
_ADV_VOLUME_FALLBACK_RATIO = Decimal("0.05")  # When bar_volume=0, use 5% of ADV

_logger = logging.getLogger(__name__)


def _sign(value: Decimal) -> int:
    """Return +1, 0, or -1 matching the sign of *value*."""
    if value > _ZERO:
        return 1
    if value < _ZERO:
        return -1
    return 0


def _decimal_sqrt(value: float) -> Decimal:
    """Square-root via math, returned as Decimal."""
    return Decimal(str(math.sqrt(value)))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketImpactConfig:
    """Parameters for the Almgren-Chriss non-linear impact model.

    Attributes
    ----------
    permanent_impact_coeff : float
        Gamma – permanent impact coefficient (Almgren & Chriss 2001).
    temporary_impact_coeff : float
        Eta – temporary impact coefficient.
    impact_exponent : float
        Exponent of the square-root impact law (Bouchaud et al. 2009).
        The canonical value is 0.5, giving impact proportional to the
        square root of the volume participation rate.
    maker_fee_rate : Decimal
        Exchange maker fee rate.
    taker_fee_rate : Decimal
        Exchange taker fee rate.
    max_volume_participation : float
        Maximum fraction of bar volume that a single trade may consume.
    funding_rate_interval_hours : int
        Hours between successive funding-rate settlements for perpetual
        futures (typically 8 on most crypto exchanges).
    """

    permanent_impact_coeff: float = 0.1
    temporary_impact_coeff: float = 0.01
    impact_exponent: float = 0.5
    maker_fee_rate: Decimal = Decimal("0.0002")
    taker_fee_rate: Decimal = Decimal("0.0005")
    max_volume_participation: float = 0.05
    funding_rate_interval_hours: int = 8


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketImpactResult:
    """Detailed breakdown of market-impact and execution costs.

    All monetary fields are denominated in quote currency.
    """

    temporary_impact: Decimal
    permanent_impact: Decimal
    total_slippage: Decimal
    effective_price: Decimal
    maker_fee: Decimal
    taker_fee: Decimal
    funding_cost: Decimal
    total_cost: Decimal
    volume_participation_rate: float
    was_capped: bool


@dataclass(frozen=True)
class EnhancedSimulatedFill:
    """A single simulated fill with full cost attribution."""

    inst_id: str
    notional: Decimal
    effective_price: Decimal
    fee: Decimal
    slippage: Decimal
    funding_cost: Decimal
    total_cost: Decimal
    volume_participation_rate: float
    was_capped: bool


# ---------------------------------------------------------------------------
# Core impact computation
# ---------------------------------------------------------------------------


def compute_market_impact(
    *,
    trade_notional: Decimal,
    price: Decimal,
    bar_volume: Decimal,
    adv: Decimal,
    config: MarketImpactConfig,
    funding_rate: Decimal = _ZERO,
    hours_held: float = 0.0,
) -> MarketImpactResult:
    """Compute non-linear market impact using the Almgren-Chriss model.

    The temporary impact follows the square-root law of Bouchaud et al.
    (2009):  ``temp = eta * sign(Q) * (|Q| / (P * V))^delta * P``

    The permanent impact captures the information content of the trade:
    ``perm = gamma * sign(Q) * (|Q| / (P * ADV))^delta * P``

    Parameters
    ----------
    trade_notional : Decimal
        Signed notional of the trade (positive = buy, negative = sell).
    price : Decimal
        Reference price at execution time.
    bar_volume : Decimal
        Volume (in quote currency) of the current bar.
    adv : Decimal
        Average Daily Volume in quote currency.
    config : MarketImpactConfig
        Model calibration parameters.
    funding_rate : Decimal
        Periodic funding rate for perpetual futures (default 0).
    hours_held : float
        Hours the position has been / will be held (for funding cost).

    Returns
    -------
    MarketImpactResult
    """
    if price <= _ZERO:
        raise ValueError("price must be positive")

    abs_notional = abs(trade_notional)
    sign = _sign(trade_notional)

    # --- handle degenerate volume -----------------------------------------
    if abs_notional == _ZERO:
        return MarketImpactResult(
            temporary_impact=_ZERO,
            permanent_impact=_ZERO,
            total_slippage=_ZERO,
            effective_price=price,
            maker_fee=_ZERO,
            taker_fee=_ZERO,
            funding_cost=_ZERO,
            total_cost=_ZERO,
            volume_participation_rate=0.0,
            was_capped=False,
        )

    # Volume fallback: when bar_volume is zero, use ADV * ratio as proxy
    effective_bar_volume = bar_volume
    if bar_volume <= _ZERO:
        if adv > _ZERO:
            effective_bar_volume = adv * _ADV_VOLUME_FALLBACK_RATIO
            _logger.warning(
                "bar_volume=0 for trade_notional=%s, using ADV fallback=%s",
                trade_notional, effective_bar_volume,
            )
        else:
            # Both bar_volume and ADV are zero — apply flat penalty
            _logger.warning(
                "Both bar_volume and ADV are zero; applying taker fee only"
            )
            taker_fee = (abs_notional * config.taker_fee_rate).quantize(
                _QUANTIZE_PRICE, rounding=ROUND_HALF_UP
            )
            return MarketImpactResult(
                temporary_impact=_ZERO,
                permanent_impact=_ZERO,
                total_slippage=_ZERO,
                effective_price=price,
                maker_fee=_ZERO,
                taker_fee=taker_fee,
                funding_cost=_ZERO,
                total_cost=taker_fee,
                volume_participation_rate=0.0,
                was_capped=False,
            )

    effective_adv = adv if adv > _ZERO else effective_bar_volume

    # --- volume participation rate ----------------------------------------
    volume_notional = price * effective_bar_volume
    raw_participation = float(abs_notional / volume_notional)
    was_capped = raw_participation > config.max_volume_participation
    participation_rate = min(raw_participation, config.max_volume_participation)

    # --- Almgren-Chriss temporary impact ----------------------------------
    temp_basis = participation_rate ** config.impact_exponent
    temporary_impact = (
        Decimal(str(config.temporary_impact_coeff))
        * Decimal(str(sign))
        * Decimal(str(temp_basis))
        * price
    ).quantize(_QUANTIZE_PRICE, rounding=ROUND_HALF_UP)

    # --- permanent impact -------------------------------------------------
    adv_notional = price * effective_adv
    adv_ratio = float(abs_notional / adv_notional)
    perm_basis = adv_ratio ** config.impact_exponent
    permanent_impact = (
        Decimal(str(config.permanent_impact_coeff))
        * Decimal(str(sign))
        * Decimal(str(perm_basis))
        * price
    ).quantize(_QUANTIZE_PRICE, rounding=ROUND_HALF_UP)

    # --- aggregated slippage & effective price ----------------------------
    total_slippage = temporary_impact + permanent_impact
    effective_price = price + total_slippage

    # --- fees (assume aggressive / taker) ---------------------------------
    taker_fee = (abs_notional * config.taker_fee_rate).quantize(
        _QUANTIZE_PRICE, rounding=ROUND_HALF_UP
    )
    maker_fee = (abs_notional * config.maker_fee_rate).quantize(
        _QUANTIZE_PRICE, rounding=ROUND_HALF_UP
    )

    # --- funding cost for perpetuals --------------------------------------
    funding_cost = _ZERO
    if hours_held > 0 and funding_rate != _ZERO:
        intervals = Decimal(str(hours_held / config.funding_rate_interval_hours))
        funding_cost = (abs_notional * funding_rate * intervals).quantize(
            _QUANTIZE_PRICE, rounding=ROUND_HALF_UP
        )

    total_cost = abs(total_slippage * abs_notional / price) + taker_fee + abs(funding_cost)
    total_cost = total_cost.quantize(_QUANTIZE_PRICE, rounding=ROUND_HALF_UP)

    return MarketImpactResult(
        temporary_impact=temporary_impact,
        permanent_impact=permanent_impact,
        total_slippage=total_slippage,
        effective_price=effective_price,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        funding_cost=funding_cost,
        total_cost=total_cost,
        volume_participation_rate=participation_rate,
        was_capped=was_capped,
    )


# ---------------------------------------------------------------------------
# ADV helper
# ---------------------------------------------------------------------------


def compute_adv(bars: list[OhlcvBar], *, lookback: int = 20) -> Decimal:
    """Compute Average Daily Volume from recent OHLCV bars.

    Parameters
    ----------
    bars : list[OhlcvBar]
        Historical bars sorted in chronological order.
    lookback : int
        Number of most-recent bars to average over (default 20).

    Returns
    -------
    Decimal
        Average volume over the lookback window.  Returns ``Decimal("0")``
        when *bars* is empty.
    """
    if not bars:
        return _ZERO
    recent = bars[-lookback:]
    total = sum((bar.volume for bar in recent), _ZERO)
    return (total / Decimal(str(len(recent)))).quantize(_QUANTIZE_PRICE, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enhanced simulator
# ---------------------------------------------------------------------------


class EnhancedExecutionSimulator:
    """Execution simulator with Almgren-Chriss non-linear impact.

    Unlike :class:`~quantbot.research.simulator.ReplayExecutionSimulator`,
    which uses a flat BPS slippage model, this simulator computes
    volume-dependent, non-linear slippage and deducts perpetual-futures
    funding costs.
    """

    def __init__(self, config: MarketImpactConfig) -> None:
        self._config = config

    # -- public API --------------------------------------------------------

    def rebalance(
        self,
        *,
        equity: Decimal,
        current_weights: dict[str, Decimal],
        target_weights: dict[str, Decimal],
        prices: dict[str, Decimal],
        volumes: dict[str, Decimal],
        adv: dict[str, Decimal],
        funding_rates: dict[str, Decimal] | None = None,
        hours_held: float = 0.0,
    ) -> list[EnhancedSimulatedFill]:
        """Simulate a portfolio rebalance with non-linear market impact.

        Parameters
        ----------
        equity : Decimal
            Current portfolio equity in quote currency.
        current_weights : dict[str, Decimal]
            Mapping of inst_id → current portfolio weight.
        target_weights : dict[str, Decimal]
            Mapping of inst_id → desired portfolio weight.
        prices : dict[str, Decimal]
            Mapping of inst_id → current mid-price.
        volumes : dict[str, Decimal]
            Mapping of inst_id → bar volume (quote currency).
        adv : dict[str, Decimal]
            Mapping of inst_id → average daily volume.
        funding_rates : dict[str, Decimal] | None
            Mapping of inst_id → periodic funding rate.  ``None`` when
            not trading perpetual futures.
        hours_held : float
            Hours the existing positions have been held (used for
            funding cost attribution on carried positions).

        Returns
        -------
        list[EnhancedSimulatedFill]
        """
        funding_rates = funding_rates or {}
        fills: list[EnhancedSimulatedFill] = []

        for inst_id, target_weight in target_weights.items():
            current_weight = current_weights.get(inst_id, _ZERO)
            delta_notional = equity * (target_weight - current_weight)
            if delta_notional == _ZERO:
                continue

            inst_price = prices[inst_id]
            inst_volume = volumes.get(inst_id, _ZERO)
            inst_adv = adv.get(inst_id, _ZERO)
            inst_funding = funding_rates.get(inst_id, _ZERO)

            # funding cost on the *held* portion (current weight)
            held_funding = self._compute_funding_cost(
                weight=current_weight,
                equity=equity,
                funding_rate=inst_funding,
                hours_held=hours_held,
            )

            impact = compute_market_impact(
                trade_notional=delta_notional,
                price=inst_price,
                bar_volume=inst_volume,
                adv=inst_adv,
                config=self._config,
                funding_rate=inst_funding,
                hours_held=hours_held,
            )

            total_funding = held_funding + abs(impact.funding_cost)
            total_cost = (
                impact.total_cost - abs(impact.funding_cost) + total_funding
            ).quantize(_QUANTIZE_PRICE, rounding=ROUND_HALF_UP)

            fills.append(
                EnhancedSimulatedFill(
                    inst_id=inst_id,
                    notional=delta_notional,
                    effective_price=impact.effective_price,
                    fee=impact.taker_fee,
                    slippage=impact.total_slippage,
                    funding_cost=total_funding,
                    total_cost=total_cost,
                    volume_participation_rate=impact.volume_participation_rate,
                    was_capped=impact.was_capped,
                )
            )

        return fills

    # -- internal helpers --------------------------------------------------

    @staticmethod
    def _compute_funding_cost(
        *,
        weight: Decimal,
        equity: Decimal,
        funding_rate: Decimal,
        hours_held: float,
        funding_interval_hours: int = 8,
    ) -> Decimal:
        """Compute periodic funding cost on an existing perpetual position.

        Parameters
        ----------
        weight : Decimal
            Portfolio weight of the position.
        equity : Decimal
            Current portfolio equity.
        funding_rate : Decimal
            Per-interval funding rate (e.g. 0.0001 = 1 bps).
        hours_held : float
            Duration the position has been held in hours.
        funding_interval_hours : int
            Interval between funding settlements (default 8h).

        Returns
        -------
        Decimal
            Absolute funding cost (always non-negative).
        """
        if weight == _ZERO or funding_rate == _ZERO or hours_held <= 0:
            return _ZERO
        position_notional = abs(weight * equity)
        intervals = Decimal(str(hours_held / funding_interval_hours))
        return (position_notional * abs(funding_rate) * intervals).quantize(
            _QUANTIZE_PRICE, rounding=ROUND_HALF_UP
        )
