# Deployment Guide

## MVP Environment

Deploy the first version as a single process on a locked-down workstation or small VM. Use OKX Spot Demo only.

## Secrets

Provide credentials through environment variables or a local secrets manager:

- `OKX_API_KEY`
- `OKX_API_SECRET`
- `OKX_API_PASSPHRASE`

Do not commit `.env` files, shell history with secrets, CI variables, or logs containing signed payloads.

## Promotion Flow

1. Run local deterministic tests.
2. Run OKX Demo connectivity checks.
3. Run event replay or paper trading.
4. Run OKX Demo order lifecycle validation.
5. Run small-size live only after a separate live-trading review gate is implemented.

## Rollback

Activate the kill switch, stop new order submission, cancel known open orders manually if automation is unhealthy, and reconcile orders and balances before restarting.

