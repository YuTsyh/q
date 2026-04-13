from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Callable, Protocol

from quantbot.core.models import MarketSnapshot, OrderIntent, OrderSide, OrderType
from quantbot.research.data import FundingRate, OhlcvBar


class Strategy(Protocol):
    def on_market(self, snapshot: MarketSnapshot) -> OrderIntent | None:
        """Consume market data and optionally return a single order intent."""


StrategyAllocator = Callable[
    [dict[str, list[OhlcvBar]], dict[str, list[FundingRate]]],
    dict[str, Decimal],
]


class AllocatorStrategyAdapter:
    """Bridge allocator-based strategies to the :class:`Strategy` Protocol.

    Allocator strategies (``trend_following``, ``adaptive_momentum``,
    ``ensemble``, etc.) produce target portfolio weights from historical
    bars.  This adapter accumulates incoming :class:`MarketSnapshot`
    ticks, builds OHLCV bars, and periodically calls the wrapped
    allocator to produce :class:`OrderIntent` objects consumable by
    :class:`TradingEngine`.

    Parameters
    ----------
    allocator :
        A strategy allocator callable (e.g. from
        ``create_trend_following_allocator``).
    inst_ids :
        Instruments the allocator should manage.
    bar_interval_seconds :
        Duration of one synthetic bar (default 86400 = 1 day).
    rebalance_every_n_bars :
        How many bars between rebalance calls.
    order_size_quote :
        Default order size in quote currency per instrument when the
        allocator signals a new position.
    client_order_prefix :
        Prefix for generated ``client_order_id`` values.

    .. note::
       Bars built from snapshots will have ``volume=0`` because tick-level
       volume is not available via :class:`MarketSnapshot`.  If the
       downstream backtest engine uses volume-dependent market impact
       (``use_market_impact=True``), the impact model will treat volumes
       as zero and produce zero slippage.  This is acceptable for live
       trading where actual exchange order books handle execution.
    """

    def __init__(
        self,
        allocator: StrategyAllocator,
        inst_ids: list[str],
        *,
        bar_interval_seconds: int = 86_400,
        rebalance_every_n_bars: int = 1,
        order_size_quote: Decimal = Decimal("100"),
        client_order_prefix: str = "ALLOC",
    ) -> None:
        self._allocator = allocator
        self._inst_ids = inst_ids
        self._bar_interval = bar_interval_seconds
        self._rebalance_n = rebalance_every_n_bars
        self._order_size = order_size_quote
        self._prefix = client_order_prefix

        # Accumulated bars per instrument
        self._bars: dict[str, list[OhlcvBar]] = defaultdict(list)
        self._funding: dict[str, list[FundingRate]] = defaultdict(list)
        self._current_weights: dict[str, Decimal] = {}

        # Intra-bar state for building bars from snapshots
        self._bar_open: dict[str, Decimal] = {}
        self._bar_high: dict[str, Decimal] = {}
        self._bar_low: dict[str, Decimal] = {}
        self._bar_start_ts: dict[str, float] = {}

        self._bar_count = 0
        self._seq = 1

    def on_market(self, snapshot: MarketSnapshot) -> OrderIntent | None:
        """Accumulate tick → build bar → rebalance → emit order."""
        inst_id = snapshot.inst_id
        if inst_id not in self._inst_ids:
            return None

        price = snapshot.last_price
        ts_epoch = snapshot.received_at.timestamp()

        # Update intra-bar OHLC
        if inst_id not in self._bar_open:
            self._bar_open[inst_id] = price
            self._bar_high[inst_id] = price
            self._bar_low[inst_id] = price
            self._bar_start_ts[inst_id] = ts_epoch
        else:
            self._bar_high[inst_id] = max(self._bar_high[inst_id], price)
            self._bar_low[inst_id] = min(self._bar_low[inst_id], price)

        # Check if bar is complete
        if ts_epoch - self._bar_start_ts.get(inst_id, ts_epoch) < self._bar_interval:
            return None

        # Finalize bar
        bar = OhlcvBar(
            inst_id=inst_id,
            ts=snapshot.received_at,
            open=self._bar_open[inst_id],
            high=self._bar_high[inst_id],
            low=self._bar_low[inst_id],
            close=price,
            volume=Decimal("0"),  # Volume not available from snapshots
        )
        self._bars[inst_id].append(bar)
        # Reset intra-bar state
        del self._bar_open[inst_id]
        del self._bar_high[inst_id]
        del self._bar_low[inst_id]
        del self._bar_start_ts[inst_id]

        self._bar_count += 1
        if self._bar_count % self._rebalance_n != 0:
            return None

        # Call allocator
        try:
            target = self._allocator(dict(self._bars), dict(self._funding))
        except (ValueError, ZeroDivisionError):
            return None

        # Determine the first instrument with a weight change
        for iid in self._inst_ids:
            new_w = target.get(iid, Decimal("0"))
            old_w = self._current_weights.get(iid, Decimal("0"))
            if new_w > old_w and iid == inst_id:
                self._current_weights = target
                size = self._order_size / price if price > 0 else Decimal("0")
                intent = OrderIntent(
                    client_order_id=f"{self._prefix}{self._seq:06d}",
                    inst_id=iid,
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    price=price,
                    size=size,
                )
                self._seq += 1
                return intent
            if new_w < old_w and iid == inst_id:
                self._current_weights = target
                size = self._order_size / price if price > 0 else Decimal("0")
                intent = OrderIntent(
                    client_order_id=f"{self._prefix}{self._seq:06d}",
                    inst_id=iid,
                    side=OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    price=price,
                    size=size,
                )
                self._seq += 1
                return intent

        self._current_weights = target
        return None

