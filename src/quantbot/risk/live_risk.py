"""Live deployment risk manager with kill switch, daily loss limits, and position sizing.

Extends the existing RiskEngine for production deployment with:
- Per-trade risk limit (1% of equity)
- Daily loss circuit breaker
- Maximum leverage control
- Kill switch with automatic recovery
- Position sizing based on account equity and volatility
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, ROUND_DOWN


@dataclass(frozen=True)
class LiveRiskConfig:
    """Risk parameters for live deployment."""

    max_risk_per_trade_pct: Decimal = Decimal("0.01")  # 1% per trade
    max_daily_loss_pct: Decimal = Decimal("0.03")  # 3% daily loss limit
    max_leverage: Decimal = Decimal("3.0")
    max_position_pct: Decimal = Decimal("0.25")  # 25% max single position
    max_open_orders: int = 10
    kill_switch_cooldown: timedelta = timedelta(hours=1)
    max_drawdown_pct: Decimal = Decimal("0.25")  # 25% max DD kill switch
    # Tiered drawdown scaling: (dd_threshold, exposure_multiplier)
    # At 10% DD → keep 50% exposure, at 15% → keep 25%, at 20%+ → full flat
    dd_tiers: tuple[tuple[float, float], ...] = (
        (0.10, 0.50),  # 10% DD → 50% exposure
        (0.15, 0.25),  # 15% DD → 25% exposure
        (0.20, 0.0),   # 20% DD → flat
    )


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""

    size: Decimal
    notional: Decimal
    risk_amount: Decimal
    stop_distance_pct: Decimal


@dataclass
class DailyPnL:
    """Tracks daily P&L for circuit breaker."""

    date: datetime
    realised_pnl: Decimal = Decimal("0")
    unrealised_pnl: Decimal = Decimal("0")

    @property
    def total_pnl(self) -> Decimal:
        return self.realised_pnl + self.unrealised_pnl


@dataclass
class LiveRiskManager:
    """Production risk manager with comprehensive safety controls."""

    config: LiveRiskConfig
    _equity: Decimal = Decimal("100000")
    _peak_equity: Decimal = Decimal("100000")
    _daily_pnl: DailyPnL = field(default_factory=lambda: DailyPnL(date=datetime.now(UTC)))
    _kill_switch_active: bool = False
    _kill_switch_until: datetime | None = None
    _open_order_count: int = 0

    def update_equity(self, equity: Decimal) -> None:
        """Update current equity and check drawdown limits."""
        self._equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Check max drawdown kill switch
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown >= self.config.max_drawdown_pct:
                self.activate_kill_switch("max_drawdown_exceeded")

    def get_drawdown_exposure_scale(self) -> float:
        """Return exposure multiplier based on tiered drawdown levels.

        Provides graduated de-risking instead of a binary kill switch.
        Strategies should multiply their target weights by this value.

        Returns
        -------
        float
            Exposure multiplier between 0.0 and 1.0.
        """
        dd = float(self.current_drawdown)
        # Tiers must be sorted ascending by threshold
        # Find the highest tier that the current DD exceeds
        scale = 1.0
        for threshold, multiplier in self.config.dd_tiers:
            if dd >= threshold:
                scale = multiplier
        return scale

    def compute_position_size(
        self,
        price: Decimal,
        stop_loss_pct: Decimal,
        lot_sz: Decimal = Decimal("0.00001"),
    ) -> PositionSizeResult:
        """Compute position size based on risk-per-trade and stop distance.

        Uses the formula: size = (equity * risk_pct) / (price * stop_distance)
        Then floors to lot_sz.
        """
        if stop_loss_pct <= 0 or price <= 0:
            return PositionSizeResult(
                size=Decimal("0"),
                notional=Decimal("0"),
                risk_amount=Decimal("0"),
                stop_distance_pct=stop_loss_pct,
            )

        risk_amount = self._equity * self.config.max_risk_per_trade_pct
        raw_size = risk_amount / (price * stop_loss_pct)

        # Apply max position constraint
        max_notional = self._equity * self.config.max_position_pct
        max_size = max_notional / price if price > 0 else Decimal("0")
        size = min(raw_size, max_size)

        # Floor to lot size
        if lot_sz > 0:
            units = (size / lot_sz).to_integral_value(rounding=ROUND_DOWN)
            size = units * lot_sz

        notional = size * price

        return PositionSizeResult(
            size=size,
            notional=notional,
            risk_amount=risk_amount,
            stop_distance_pct=stop_loss_pct,
        )

    def check_order_allowed(self) -> tuple[bool, str]:
        """Check whether a new order is allowed under current risk limits."""
        # Kill switch check
        if self._kill_switch_active:
            if self._kill_switch_until and datetime.now(UTC) > self._kill_switch_until:
                self._kill_switch_active = False
                self._kill_switch_until = None
            else:
                return False, "kill_switch_active"

        # Daily loss limit
        self._refresh_daily_pnl()
        daily_loss_limit = self._equity * self.config.max_daily_loss_pct
        if self._daily_pnl.total_pnl < -daily_loss_limit:
            self.activate_kill_switch("daily_loss_limit_exceeded")
            return False, "daily_loss_limit_exceeded"

        # Open order limit
        if self._open_order_count >= self.config.max_open_orders:
            return False, "max_open_orders_exceeded"

        return True, "allowed"

    def record_fill(self, pnl: Decimal) -> None:
        """Record a fill's P&L for daily tracking."""
        self._refresh_daily_pnl()
        self._daily_pnl.realised_pnl += pnl

    def record_order_opened(self) -> None:
        """Track an open order."""
        self._open_order_count += 1

    def record_order_closed(self) -> None:
        """Track a closed order."""
        self._open_order_count = max(0, self._open_order_count - 1)

    def activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch with cooldown."""
        self._kill_switch_active = True
        self._kill_switch_until = datetime.now(UTC) + self.config.kill_switch_cooldown

    def deactivate_kill_switch(self) -> None:
        """Manually deactivate kill switch."""
        self._kill_switch_active = False
        self._kill_switch_until = None

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def current_drawdown(self) -> Decimal:
        if self._peak_equity <= 0:
            return Decimal("0")
        return (self._peak_equity - self._equity) / self._peak_equity

    def _refresh_daily_pnl(self) -> None:
        """Reset daily P&L if date has changed."""
        now = datetime.now(UTC)
        if now.date() != self._daily_pnl.date.date():
            self._daily_pnl = DailyPnL(date=now)
