# Risk Specification

The first version must enforce:

- Demo mode guard before any trading request.
- Max order notional.
- Max position notional.
- Max daily loss.
- Max order rate per rolling window.
- Price band check against the last trusted market price.
- Stale market data rejection.
- Kill switch.
- Duplicate client order id prevention.
- Unknown order state escalation after ambiguous submit/cancel outcomes.

For OKX Spot Demo, orders use `tdMode=cash`; leverage and position mode are out of scope for the first MVP.

