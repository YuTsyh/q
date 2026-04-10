from decimal import Decimal

from quantbot.research.portfolio import EqualWeightTopNConstructor


def test_equal_weight_top_n_constructor_caps_positions_and_gross() -> None:
    constructor = EqualWeightTopNConstructor(
        top_n=2,
        gross_exposure=Decimal("0.8"),
        max_symbol_weight=Decimal("0.5"),
    )

    weights = constructor.construct(
        {
            "BTC-USDT-SWAP": Decimal("0.9"),
            "ETH-USDT-SWAP": Decimal("0.8"),
            "SOL-USDT-SWAP": Decimal("0.1"),
        }
    )

    assert weights == {
        "BTC-USDT-SWAP": Decimal("0.4"),
        "ETH-USDT-SWAP": Decimal("0.4"),
    }

