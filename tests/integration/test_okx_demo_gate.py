import os

import pytest


REQUIRED_ENV_VARS = ("OKX_API_KEY", "OKX_API_SECRET", "OKX_API_PASSPHRASE")


@pytest.mark.skipif(
    not all(os.environ.get(name) for name in REQUIRED_ENV_VARS),
    reason="OKX Demo credentials are not present in local environment",
)
def test_okx_demo_credentials_are_demo_scoped() -> None:
    assert os.environ.get("OKX_ENV", "demo") == "demo"

