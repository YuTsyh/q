from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from urllib.parse import urlencode

import httpx

from quantbot.core.models import OrderIntent, OrderStatus
from quantbot.exchange.models import InstrumentMetadata
from quantbot.exchange.okx.auth import OkxSigner
from quantbot.exchange.okx.trade import OkxOrderMapper


class OkxRestError(RuntimeError):
    """Base class for OKX REST adapter errors."""


class AmbiguousOrderError(OkxRestError):
    """Raised when OKX may have received an order request but no final state is known."""


class OkxRejectError(OkxRestError):
    """Raised when OKX returned a deterministic reject."""


@dataclass(frozen=True)
class OkxOrderAck:
    order_id: str
    client_order_id: str


@dataclass(frozen=True)
class OkxCancelAck:
    order_id: str
    client_order_id: str
    accepted: bool


@dataclass(frozen=True)
class OkxOrderState:
    order_id: str
    client_order_id: str
    status: OrderStatus
    filled_size: Decimal


class OkxRestClient:
    def __init__(
        self,
        *,
        base_url: str,
        signer: OkxSigner,
        simulated: bool,
        timeout: float = 5.0,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        timestamp_factory: Callable[[], str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._signer = signer
        self._simulated = simulated
        self._timestamp_factory = timestamp_factory or _utc_timestamp
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    async def place_order(self, intent: OrderIntent) -> OkxOrderAck:
        request_path = "/api/v5/trade/order"
        body = _json_body(OkxOrderMapper.to_place_order_payload(intent))
        response_payload = await self._request_order_once("POST", request_path, body=body)
        row = _first_data_row(response_payload)
        if row.get("sCode") != "0":
            raise OkxRejectError(f"OKX rejected order {intent.client_order_id}: {row.get('sMsg', '')}")
        return OkxOrderAck(order_id=row["ordId"], client_order_id=row["clOrdId"])

    async def cancel_order(self, *, inst_id: str, client_order_id: str) -> OkxCancelAck:
        request_path = "/api/v5/trade/cancel-order"
        body = _json_body({"instId": inst_id, "clOrdId": client_order_id})
        response_payload = await self._request_order_once("POST", request_path, body=body)
        row = _first_data_row(response_payload)
        if row.get("sCode") != "0":
            raise OkxRejectError(f"OKX rejected cancel {client_order_id}: {row.get('sMsg', '')}")
        return OkxCancelAck(
            order_id=row.get("ordId", ""),
            client_order_id=row["clOrdId"],
            accepted=True,
        )

    async def get_order_state(self, *, inst_id: str, client_order_id: str) -> OkxOrderState:
        response_payload = await self._request(
            "GET",
            "/api/v5/trade/order",
            query={"instId": inst_id, "clOrdId": client_order_id},
            ambiguous_on_transport=False,
        )
        row = _first_data_row(response_payload)
        return OkxOrderState(
            order_id=row.get("ordId", ""),
            client_order_id=row.get("clOrdId", client_order_id),
            status=OkxOrderMapper.to_domain_status(row.get("state", "")),
            filled_size=Decimal(row.get("accFillSz", "0")),
        )

    async def fetch_spot_instrument(self, inst_id: str) -> InstrumentMetadata:
        response_payload = await self._request(
            "GET",
            "/api/v5/public/instruments",
            query={"instType": "SPOT", "instId": inst_id},
            ambiguous_on_transport=False,
        )
        row = _first_data_row(response_payload)
        return InstrumentMetadata(
            inst_id=row["instId"],
            inst_type=row["instType"],
            tick_sz=Decimal(row["tickSz"]),
            lot_sz=Decimal(row["lotSz"]),
            min_sz=Decimal(row["minSz"]),
            state=row["state"],
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request_order_once(self, method: str, request_path: str, *, body: str) -> dict:
        return await self._request(method, request_path, body=body, ambiguous_on_transport=True)

    async def _request(
        self,
        method: str,
        request_path: str,
        *,
        body: str = "",
        query: dict[str, str] | None = None,
        ambiguous_on_transport: bool,
    ) -> dict:
        signed_path = _request_path_with_query(request_path, query)
        timestamp = self._timestamp_factory()
        headers = self._signer.headers(
            timestamp=timestamp,
            method=method,
            request_path=signed_path,
            body=body,
            simulated=self._simulated,
        )
        try:
            response = await self._client.request(
                method,
                request_path,
                params=query,
                content=body,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            if ambiguous_on_transport:
                raise AmbiguousOrderError(
                    "OKX order request timed out; order state must be reconciled"
                ) from exc
            raise OkxRestError("OKX REST request timed out") from exc
        except httpx.TransportError as exc:
            if ambiguous_on_transport:
                raise AmbiguousOrderError(
                    "OKX order transport failed; order state must be reconciled"
                ) from exc
            raise OkxRestError("OKX REST transport failed") from exc

        payload = response.json()
        if payload.get("code") != "0":
            raise OkxRejectError(f"OKX REST error {payload.get('code')}: {payload.get('msg')}")
        return payload


def _json_body(payload: dict[str, str]) -> str:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _first_data_row(payload: dict) -> dict:
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise OkxRestError("OKX response does not contain a data row")
    return data[0]


def _request_path_with_query(request_path: str, query: dict[str, str] | None) -> str:
    if not query:
        return request_path
    return f"{request_path}?{urlencode(query)}"


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
