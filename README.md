# OKX Spot Demo Bot

Production-oriented scaffold for an OKX Spot Demo quantitative trading bot.

The first deployable target is deliberately narrow:

- Exchange: OKX
- Environment: Spot Demo
- Product: Spot only
- Trading mode: `tdMode=cash`
- Strategy: low-frequency breakout shell
- Secrets: environment variables only

Do not run against production credentials until the demo validation checklist in `docs/test-plan.md` and `docs/runbook.md` has been completed.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Required environment variables

```text
OKX_API_KEY
OKX_API_SECRET
OKX_API_PASSPHRASE
OKX_ENV=demo
OKX_SYMBOL=BTC-USDT
```

The demo mode configuration must send `x-simulated-trading: 1` and use OKX demo WebSocket endpoints.

