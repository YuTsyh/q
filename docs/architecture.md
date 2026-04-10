# Architecture

## Target

The MVP targets OKX Spot Demo for a single spot instrument and a low-frequency strategy. It avoids leverage, margin, futures, swaps, options, multi-account routing, and cross-exchange logic.

## Data Flow

1. Load configuration and secrets from environment variables.
2. Fetch OKX server time and instrument metadata by REST.
3. Subscribe to public market data over WebSocket.
4. Convert strategy output into `OrderIntent`.
5. Validate the intent with the risk engine and instrument metadata.
6. Submit accepted orders through the OKX REST trade endpoint.
7. Consume private order updates from the OKX order channel.
8. Periodically reconcile open orders, fills, balances, and positions through REST.
9. Persist execution events and expose metrics.

## Implemented Slice

The current codebase implements the deterministic core for config safety gates, OKX signing, spot instrument parsing, order payload mapping, REST place/cancel/query methods, ticker parsing, order state transitions, reconciliation by order state query, event persistence, metrics counters, and a breakout strategy shell.

No live OKX Demo request is executed by default. Integration tests are credential-gated.

## Recovery

On restart, the bot must load local open orders, fetch OKX pending orders, compare by `ordId` and `clOrdId`, mark unknown local orders for reconciliation, and block new orders until the reconciliation pass finishes.
