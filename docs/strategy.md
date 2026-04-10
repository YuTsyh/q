# Strategy MVP

## Scope

The first research strategy is a low-frequency, liquid USDT perpetual universe strategy. It does not use high-frequency signals, market making, cross-exchange arbitrage, or alternative data.

## Frequency

Default research frequencies:

- 4h for faster iteration and funding-aware tests.
- 1d for lower turnover production candidates.

## Signal Stack

The MVP uses:

- Momentum: trailing close-to-close return.
- Trend: latest close versus trailing simple moving average.
- Carry: negative trailing funding rate average, so lower funding is preferred for a long-biased perpetual book.

## Current Implementation

`MultiFactorPerpStrategy.default_low_frequency` computes factor features, combines cross-sectional ranks, and produces target weights through an equal-weight top-N constructor.

## Research References

The strategy track uses [awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading) as a source index. Current design references are documented in `docs/research-sources.md`.

Current constraints remain unchanged:

- No high-frequency strategy.
- No market making.
- No cross-exchange arbitrage.
- No alternative data.
- No dependency on real OKX connectivity during strategy research.
