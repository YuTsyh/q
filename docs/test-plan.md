# Test Plan

## Local Deterministic Tests

- Config rejects production mode unless explicitly enabled by a future release gate.
- OKX REST signer generates the expected HMAC-SHA256 base64 signature.
- Instrument metadata quantizes prices and sizes using OKX tick and lot sizes.
- Risk engine rejects stale data, kill switch, duplicate client order ids, price band violations, and excessive notional.
- Order state machine handles accepted, partially filled, filled, canceled, rejected, and unknown states.
- OKX REST client sends demo signed place/cancel/query requests through mock transport.
- Trading engine quantizes strategy intents before risk evaluation and submission.

## OKX Demo Integration Tests

Run only with local demo credentials:

- Fetch server time.
- Fetch `SPOT` instruments and validate `BTC-USDT` metadata.
- Subscribe to ticker channel.
- Place a tiny post-only or low-risk limit order that will not cross the spread.
- Cancel it.
- Reconcile final order state through REST and private order channel.

The current integration gate is present but skipped unless OKX Demo credentials are available in the local environment.

## Promotion Criteria

Move from demo to small-size live only after repeated demo runs show no unknown orders, no reconciliation drift, no unhandled rejects, stable reconnect behavior, and correct metrics.
