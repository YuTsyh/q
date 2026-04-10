# Research Plan

## Phase A: Data Layer

Implement local research records for OHLCV, funding, instrument metadata, fees, tick/lot/minimum size, minimum notional, and quote-volume based universe selection.

Use OKX official endpoints as the source of truth when wiring live data ingestion:

- Candles and historical candles for OHLCV.
- Funding rate and funding history for carry.
- Instruments for tick size, lot size, min size, state, and product type.
- Fee-rate endpoints or explicit fee configuration for backtest assumptions.

## Phase B: Factor Research

Start with price and funding factors only:

- Momentum.
- Trend.
- Carry/funding.

Use cross-sectional ranking before portfolio construction. Avoid alternative data and intraday microstructure factors until the base pipeline is stable.

## Phase C: Replay Simulation

Use deterministic execution assumptions:

- Fee rate.
- Slippage in basis points.
- Partial fill ratio.
- Scheduled rebalance at 4h or 1d.

The simulator should make assumptions explicit and conservative rather than optimize PnL by hiding execution costs.

