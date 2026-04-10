# Portfolio Construction

## MVP Policy

The first constructor is equal-weight top-N over ranked USDT perpetual instruments.

Inputs:

- Cross-sectional factor scores.
- Gross exposure cap.
- Per-symbol max weight.
- Number of selected symbols.

Output:

- Target weights by instrument ID.

## Constraints

The first version is long-only and research-oriented. It intentionally avoids:

- Leverage optimization.
- Volatility targeting.
- Sector clustering.
- Cross-exchange allocation.
- Margin mode modeling.

Those belong after the replay simulator and OKX Demo lifecycle tests are stable.

## Future Research Candidates

References from the systematic trading source review:

- PyPortfolioOpt: prototype HRP and shrinkage covariance weighting after returns and turnover metrics exist.
- cvxportfolio: evaluate constrained optimization after the simulator models turnover, fees, and execution costs.

These are not current dependencies.
