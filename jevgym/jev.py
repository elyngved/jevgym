"""Jev client pinned to one model version.

Every call sends the pinned model ID and checks the model the response says answered. A mismatch
raises PinMismatchError (the call record is attached so it can still be logged). Cost is recorded
per call from billable input tokens; output tokens are free.

Talks to the HTTP API directly (POST /v1/systemone) so the exact request and response bodies can be
logged. Docs: https://docs.typesafe.ai/api/system-one.md
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

PIN = "jev-1.14.0"
# https://docs.typesafe.ai/models.md (checked 2026-11-07): $42 per billion input tokens; output free.
PRICE_USD_PER_INPUT_TOKEN = 42 / 1e9
DEFAULT_BASE_URL = "https://api.typesafe.ai"
API_KEY_ENV = "TYPESAFE_API_KEY"
RETRY_STATUSES = {429, 500, 502, 503, 504, 529}


class JevError(RuntimeError):
    pass


class PinMismatchError(JevError):
    def __init__(self, message: str, call: "CallRecord"):
        super().__init__(message)
        self.call = call


class JevAPIError(JevError):
    def __init__(self, message: str, call: "CallRecord"):
        super().__init__(message)
        self.call = call


@dataclass
class CallRecord:
    row_id: str
    request: dict[str, Any]
    response: dict[str, Any] | None
    status_code: int | None
    returned_model: str | None
    pin: str
    pin_ok: bool
    input_tokens: int
    output_tokens: int
    cost_usd: float
    attempts: int
    errors: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    started_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cost_for(input_tokens: int) -> float:
    return input_tokens * PRICE_USD_PER_INPUT_TOKEN


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        pin: str = PIN,
        max_concurrency: int = 8,
        max_attempts: int = 6,
        timeout_s: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise JevError(f"{API_KEY_ENV} is not set")
        self.base_url = base_url
        self.pin = pin
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self.max_concurrency = max_concurrency
        self._sem = asyncio.Semaphore(max_concurrency)
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=timeout_s,
            transport=transport,
        )

    def settings(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "pin": self.pin,
            "max_concurrency": self.max_concurrency,
            "max_attempts": self.max_attempts,
            "timeout_s": self.timeout_s,
            "price_usd_per_input_token": PRICE_USD_PER_INPUT_TOKEN,
        }

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "JevClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def system_one(self, row_id: str, request: dict[str, Any]) -> CallRecord:
        if request.get("model") != self.pin:
            raise JevError(f"request model {request.get('model')!r} is not the pin {self.pin!r}")
        errors: list[str] = []
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        async with self._sem:
            for attempt in range(1, self.max_attempts + 1):
                status = None
                try:
                    resp = await self._http.post("/v1/systemone", json=request)
                    status = resp.status_code
                    if status == 200:
                        body = resp.json()
                        returned = body.get("model")
                        usage = body.get("usage") or {}
                        tin = int(usage.get("input_tokens", 0))
                        rec = CallRecord(
                            row_id=row_id,
                            request=request,
                            response=body,
                            status_code=status,
                            returned_model=returned,
                            pin=self.pin,
                            pin_ok=returned == self.pin,
                            input_tokens=tin,
                            output_tokens=int(usage.get("output_tokens", 0)),
                            cost_usd=cost_for(tin),
                            attempts=attempt,
                            errors=errors,
                            latency_s=round(time.monotonic() - t0, 4),
                            started_utc=started,
                        )
                        if not rec.pin_ok:
                            raise PinMismatchError(
                                f"row {row_id}: returned model {returned!r} != pin {self.pin!r}", rec
                            )
                        return rec
                    errors.append(f"attempt {attempt}: HTTP {status}: {resp.text[:500]}")
                    if status not in RETRY_STATUSES:
                        break
                    retry_after = resp.headers.get("retry-after")
                    delay = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    errors.append(f"attempt {attempt}: {type(e).__name__}: {e}")
                    delay = None
                if attempt < self.max_attempts:
                    await asyncio.sleep(delay if delay is not None else min(30.0, 0.5 * 2 ** (attempt - 1)) * (0.5 + random.random()))
        rec = CallRecord(
            row_id=row_id,
            request=request,
            response=None,
            status_code=status,
            returned_model=None,
            pin=self.pin,
            pin_ok=False,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            attempts=len(errors),
            errors=errors,
            latency_s=round(time.monotonic() - t0, 4),
            started_utc=started,
        )
        raise JevAPIError(f"row {row_id}: call failed: {errors[-1] if errors else 'unknown'}", rec)
