from __future__ import annotations

from dataclasses import dataclass
from os import environ


@dataclass(frozen=True)
class BotConfig:
    okx_api_key: str
    okx_api_secret: str
    okx_api_passphrase: str
    okx_env: str
    okx_symbol: str
    rest_base_url: str = "https://us.okx.com"

    def __post_init__(self) -> None:
        env = self.okx_env.lower()
        if env != "demo":
            raise ValueError("Production trading is disabled for the OKX Spot Demo MVP")
        if not self.okx_symbol or "-" not in self.okx_symbol:
            raise ValueError("OKX symbol must be an instrument id like BTC-USDT")
        object.__setattr__(self, "okx_env", env)

    @classmethod
    def from_env(cls) -> BotConfig:
        return cls(
            okx_api_key=environ["OKX_API_KEY"],
            okx_api_secret=environ["OKX_API_SECRET"],
            okx_api_passphrase=environ["OKX_API_PASSPHRASE"],
            okx_env=environ.get("OKX_ENV", "demo"),
            okx_symbol=environ.get("OKX_SYMBOL", "BTC-USDT"),
        )

    @property
    def public_ws_url(self) -> str:
        return "wss://wspap.okx.com:8443/ws/v5/public"

    @property
    def private_ws_url(self) -> str:
        return "wss://wspap.okx.com:8443/ws/v5/private"

    @property
    def business_ws_url(self) -> str:
        return "wss://wspap.okx.com:8443/ws/v5/business"

    @property
    def rest_headers(self) -> dict[str, str]:
        return {"x-simulated-trading": "1"}

