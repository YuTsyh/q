from decimal import Decimal

from quantbot.research.simulator import ExecutionAssumptions, ReplayExecutionSimulator


def test_replay_execution_simulator_applies_fee_slippage_and_partial_fill() -> None:
    simulator = ReplayExecutionSimulator(
        ExecutionAssumptions(
            taker_fee_rate=Decimal("0.0005"),
            slippage_bps=Decimal("2"),
            partial_fill_ratio=Decimal("0.5"),
        )
    )

    fills = simulator.rebalance(
        equity=Decimal("10000"),
        current_weights={},
        target_weights={"BTC-USDT-SWAP": Decimal("0.2")},
        prices={"BTC-USDT-SWAP": Decimal("50000")},
    )

    assert len(fills) == 1
    assert fills[0].inst_id == "BTC-USDT-SWAP"
    assert fills[0].notional == Decimal("1000.0")
    assert fills[0].price == Decimal("50010.0000")
    assert fills[0].fee == Decimal("0.50000")

