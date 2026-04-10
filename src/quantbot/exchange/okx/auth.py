from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass


@dataclass(frozen=True)
class OkxCredentials:
    api_key: str
    api_secret: str
    passphrase: str


class OkxSigner:
    def __init__(self, credentials: OkxCredentials) -> None:
        self._credentials = credentials

    def sign(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        prehash = f"{timestamp}{method.upper()}{request_path}{body}"
        digest = hmac.new(
            self._credentials.api_secret.encode("utf-8"),
            prehash.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(digest).decode("ascii")

    def headers(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = "",
        *,
        simulated: bool,
    ) -> dict[str, str]:
        headers = {
            "OK-ACCESS-KEY": self._credentials.api_key,
            "OK-ACCESS-SIGN": self.sign(timestamp, method, request_path, body),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self._credentials.passphrase,
            "Content-Type": "application/json",
        }
        if simulated:
            headers["x-simulated-trading"] = "1"
        return headers

