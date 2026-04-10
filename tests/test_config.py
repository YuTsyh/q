import pytest

from quantbot.config import BotConfig


def test_demo_config_uses_simulated_trading_header_and_demo_ws_urls() -> None:
    config = BotConfig(
        okx_api_key="key",
        okx_api_secret="secret",
        okx_api_passphrase="pass",
        okx_env="demo",
        okx_symbol="BTC-USDT",
    )

    assert config.rest_base_url == "https://us.okx.com"
    assert config.rest_headers["x-simulated-trading"] == "1"
    assert "wspap.okx.com" in config.public_ws_url
    assert "wspap.okx.com" in config.private_ws_url


def test_production_mode_is_rejected_for_mvp() -> None:
    with pytest.raises(ValueError, match="Production trading is disabled"):
        BotConfig(
            okx_api_key="key",
            okx_api_secret="secret",
            okx_api_passphrase="pass",
            okx_env="prod",
            okx_symbol="BTC-USDT",
        )

