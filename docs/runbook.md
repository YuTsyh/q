# Runbook

## Start Preconditions

- `OKX_ENV=demo`
- `OKX_API_KEY`, `OKX_API_SECRET`, and `OKX_API_PASSPHRASE` are present in the local environment.
- The bot confirms the demo REST header `x-simulated-trading: 1`.
- The bot uses demo WebSocket endpoints.
- Instrument metadata has been fetched and validated.
- Reconciliation has completed before live order submission is enabled.

## Kill Switch

Set the configured kill switch flag before sending any new order. Once active, the execution engine must reject new open-risk intents and only allow reconciliation and risk-reducing cleanup.

## Ambiguous Order Handling

If an order request returns an ambiguous transport or exchange result, do not retry blindly. Mark the order as `UNKNOWN`, query by `clOrdId` and `ordId` when available, and wait for the OKX order channel or REST order state before submitting any replacement.

