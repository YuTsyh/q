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

Reference implementation ideas from `awesome-systematic-trading`:

- Use vectorized research ergonomics from vectorbt as a benchmark, but keep the MVP dependency-light.
- Use pysystemtrade as a process reference for conservative systematic strategy design.
- Use Freqtrade's pairlist and bias-analysis concepts as a reminder to add universe and lookahead checks.

## Phase C: Replay Simulation

Use deterministic execution assumptions:

- Fee rate.
- Slippage in basis points.
- Partial fill ratio.
- Scheduled rebalance at 4h or 1d.

The simulator should make assumptions explicit and conservative rather than optimize PnL by hiding execution costs.

## Phase D: Strategy Validation

Before any OKX Demo order lifecycle test is linked to the strategy, require:

- Data completeness checks for OHLCV and funding.
- Universe membership snapshots by rebalance timestamp.
- Lookahead checks: factor values at time `t` must only use data available before rebalance.
- Turnover, fee, slippage, and partial-fill attribution.
- Period-based metrics: return, volatility, drawdown, Sharpe-like ratio, hit rate, VaR/CVaR placeholder.
- Walk-forward split: train/selection window and out-of-sample evaluation window.
- Sensitivity checks for lookback, top-N, gross exposure, fee, slippage, and partial fill assumptions.

## Phase E: Adoption Decisions

Do not migrate to a third-party framework by default. Use external projects as references:

- vectorbt for fast research and walk-forward ergonomics.
- quantstats for metrics coverage.
- cvxportfolio/PyPortfolioOpt for later constrained portfolio construction.
- NautilusTrader for research-to-live parity principles.
