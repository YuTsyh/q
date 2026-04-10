# Factor Specification

## Momentum

Formula:

```text
close[t] / close[t-lookback] - 1
```

Higher is better.

## Trend

Formula:

```text
close[t] / SMA(close[t-lookback+1:t]) - 1
```

Higher is better.

## Carry / Funding

Formula:

```text
-average(funding_rate over lookback observations)
```

For the first long-biased perpetual book, lower funding is preferred because long positions pay less funding or may receive funding when rates are negative.

## Ranking

Each factor is ranked cross-sectionally in descending order and converted to a score from 1 to 0. The default MVP weights are:

- Momentum: 40%
- Trend: 40%
- Carry: 20%

