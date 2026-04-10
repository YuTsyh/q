from quantbot.exchange.okx.auth import OkxCredentials, OkxSigner


def test_okx_signer_uses_timestamp_method_path_and_body() -> None:
    credentials = OkxCredentials(api_key="key", api_secret="secret", passphrase="pass")
    signer = OkxSigner(credentials)

    signature = signer.sign(
        timestamp="2020-12-08T09:08:57.715Z",
        method="POST",
        request_path="/api/v5/trade/order",
        body='{"instId":"BTC-USDT","tdMode":"cash"}',
    )

    assert signature == "Qpc4u1EkC5yBa8Qp1+qYDqnBAGBlqxt98QLHdcLAhPc="


def test_okx_signer_headers_include_passphrase_and_simulated_trading() -> None:
    credentials = OkxCredentials(api_key="key", api_secret="secret", passphrase="pass")
    signer = OkxSigner(credentials)

    headers = signer.headers(
        timestamp="2020-12-08T09:08:57.715Z",
        method="GET",
        request_path="/api/v5/account/balance",
        body="",
        simulated=True,
    )

    assert headers["OK-ACCESS-KEY"] == "key"
    assert headers["OK-ACCESS-PASSPHRASE"] == "pass"
    assert headers["x-simulated-trading"] == "1"
