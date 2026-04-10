# Execution Assumptions

## Replay Simulator

The first replay simulator applies:

- Taker fee rate.
- Slippage in basis points.
- Partial fill ratio.

The fill notional is:

```text
equity * (target_weight - current_weight) * partial_fill_ratio
```

The buy-side fill price assumption is:

```text
reference_price * (1 + slippage_bps / 10000)
```

The fee is:

```text
abs(fill_notional) * taker_fee_rate
```

## Limitations

This is a deterministic research assumption model, not a full exchange simulator. It does not yet model queue position, order book depth, maker fill probability, funding cashflows, liquidation, margin requirements, or trading halts.

## Next Required Extensions

- Funding cashflow accounting for perpetual swaps.
- Rebalance schedule accounting for 4h and 1d modes.
- Turnover and fee attribution by rebalance.
- Conservative missing-data handling.
- Walk-forward reporting.
