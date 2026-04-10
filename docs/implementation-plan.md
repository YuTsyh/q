# OKX Spot Demo MVP Implementation Plan

## Goal

Build a production-oriented OKX Spot Demo trading bot skeleton that can safely grow into real execution after demo validation.

## Ordered Steps

1. Create Python project skeleton, docs, `.gitignore`, and `.env.example`.
2. Add deterministic tests for config, signing, metadata quantization, risk, order state, reconciliation, market data parsing, OKX order mapping, REST request construction, and the first strategy shell.
3. Implement only enough production code to pass those tests.
4. Add OKX Demo integration tests behind a credential gate.
5. Add persistence and metrics after the deterministic core is stable.
6. Add replay/paper runner before enabling any live production mode.
7. Add a separate live-trading gate before any production endpoint can be used.

## Current Status

Completed:

- Deterministic tests for config, signing, metadata, risk, order lifecycle, reconciliation, market data parsing, OKX order mapping, REST request construction, persistence, metrics, and engine coordination.
- OKX REST methods for place order, cancel order, get order state, and fetch one spot instrument.
- Credential-gated integration placeholder that skips without local OKX Demo credentials.

Remaining:

- Real OKX Demo connectivity test for server time and instrument metadata.
- WebSocket public/private client loop with reconnect and resubscribe behavior.
- Full balance/fill reconciliation.
- Replay/paper runner.
- Prometheus exporter and alerting integration.

## Acceptance Criteria

- Local unit tests pass.
- Demo config rejects production mode.
- OKX REST order requests include `x-simulated-trading: 1`.
- Order submission timeout/transport failure becomes `AmbiguousOrderError` and is not retried automatically.
- Reconciliation can mark missing local live orders as `UNKNOWN`.
- Strategy output is an `OrderIntent`; it cannot call the exchange directly.
