# Research Sources

## Primary Index

Source: [wangzhe3224/awesome-systematic-trading](https://github.com/wangzhe3224/awesome-systematic-trading)

Use this repository as a curated research and tooling index. Do not treat any linked project as an exchange specification or directly deployable framework.

## Sources Reviewed For Current Strategy Track

### pysystemtrade

Source: [robcarver17/pysystemtrade](https://github.com/pst-group/pysystemtrade)

Relevant takeaways:

- Treat systematic trading as a complete process: research, backtest, risk, and live trading discipline.
- Keep risk warnings explicit; backtests do not imply future performance.
- Use this as a process reference, not as a direct dependency in the MVP.

### vectorbt

Source: [polakowo/vectorbt](https://github.com/polakowo/vectorbt)

Relevant takeaways:

- Fast vectorized research is useful for parameter sweeps and many-instrument tests.
- Backtests must include fees and should support robustness testing and walk-forward validation.
- Use as a benchmark for research ergonomics; do not add it as a dependency until the in-house data model is stable.

### NautilusTrader

Source: [nautechsystems/nautilus_trader](https://github.com/nautechsystems/nautilus_trader)

Relevant takeaways:

- Research/live parity reduces deployment risk.
- Deterministic event-driven semantics are valuable for replay.
- Use this as an architecture reference for simulator and execution parity, not as a replacement for the current minimal OKX adapter.

### Freqtrade

Source: [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade)

Relevant takeaways:

- Crypto strategy development needs data download, backtesting, strategy analysis, pairlist/whitelist management, and bias analysis.
- For this project, borrow the ideas of dynamic universe filters and lookahead/recursive analysis checks.
- Do not borrow Telegram/web UI or broad bot features for the MVP.

### quantstats

Source: [ranaroussi/quantstats](https://github.com/ranaroussi/quantstats)

Relevant takeaways:

- Strategy evaluation needs period-based risk and performance metrics, not just final PnL.
- Add Sharpe-like return/risk metrics, drawdown, volatility, VaR/CVaR, hit rate, turnover, and fee attribution.

### cvxportfolio and PyPortfolioOpt

Sources:

- [cvxgrp/cvxportfolio](https://github.com/cvxgrp/cvxportfolio)
- [PyPortfolio/PyPortfolioOpt](https://github.com/PyPortfolio/PyPortfolioOpt)

Relevant takeaways:

- Risk-aware portfolio construction is a later step after equal-weight top-N is validated.
- Candidate future constructors: volatility targeting, covariance-aware weighting, HRP, and constrained optimization.
- Do not introduce optimizer dependencies before the replay simulator has cost, turnover, and rebalance accounting.

## Explicitly Deferred

The following categories are not part of the current strategy development track:

- High-frequency order book modeling.
- Market making.
- Cross-exchange arbitrage.
- Alternative data.
- Reinforcement learning and LLM trading agents.
- Full framework migration to Freqtrade, NautilusTrader, Qlib, or vectorbt.

