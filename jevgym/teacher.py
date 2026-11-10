"""Teacher client: one pinned Claude model that writes Prompt parts, every call logged.

Each call is appended to the run directory's teacher_calls.jsonl with the system prompt, user prompt,
output schema, raw response text, stop reason, token usage, cost, request id and the model that
answered. A returned model other than the pin, a refusal, or a truncated answer stops the run
(no server-side fallback: a fallback model would silently break the pin).
"""

from __future__ import annotations

import json
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from jevgym.runlog import append_jsonl

TEACHER_CALLS = "teacher_calls.jsonl"


class TeacherError(RuntimeError):
    pass


@dataclass(frozen=True)
class TeacherConfig:
    model: str
    effort: str
    max_tokens: int
    usd_per_input_token: float
    usd_per_output_token: float
    price_source: str

    @classmethod
    def load(cls, path: str | Path) -> "TeacherConfig":
        with open(path, "rb") as f:
            t = tomllib.load(f)["teacher"]
        return cls(
            model=t["model"],
            effort=t["effort"],
            max_tokens=int(t["max_tokens"]),
            usd_per_input_token=t["usd_per_mtok_input"] / 1e6,
            usd_per_output_token=t["usd_per_mtok_output"] / 1e6,
            price_source=t["price_source"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "usd_per_input_token": self.usd_per_input_token,
            "usd_per_output_token": self.usd_per_output_token,
            "price_source": self.price_source,
        }


class Teacher:
    def __init__(self, config: TeacherConfig, client: anthropic.Anthropic | None = None):
        self.config = config
        self.client = client or anthropic.Anthropic()

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return input_tokens * self.config.usd_per_input_token + output_tokens * self.config.usd_per_output_token

    def count_input_tokens(self, system: str, user: str, schema: dict[str, Any]) -> int:
        r = self.client.messages.count_tokens(
            model=self.config.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.config.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        return r.input_tokens

    def write_json(self, system: str, user: str, schema: dict[str, Any], log_path: Path, purpose: str) -> dict[str, Any]:
        """One call with a JSON-schema output; returns the parsed object. Logged before any check fails."""
        output_config = {"effort": self.config.effort, "format": {"type": "json_schema", "schema": schema}}
        started = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()
        resp = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config,
        )
        latency = time.monotonic() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        record = {
            "purpose": purpose,
            "requested_model": self.config.model,
            "returned_model": resp.model,
            "model_ok": resp.model == self.config.model,
            "effort": self.config.effort,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "user": user,
            "output_config": output_config,
            "response_text": text,
            "stop_reason": resp.stop_reason,
            "usage": {
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", None),
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", None),
            },
            "cost_usd": self.cost(u.input_tokens, u.output_tokens),
            "request_id": getattr(resp, "_request_id", None),
            "started_utc": started,
            "latency_s": round(latency, 3),
        }
        append_jsonl(log_path, record)
        if not record["model_ok"]:
            raise TeacherError(f"Teacher answered as {resp.model!r}, pin is {self.config.model!r}")
        if resp.stop_reason != "end_turn":
            raise TeacherError(f"Teacher stopped with {resp.stop_reason!r}; see {log_path}")
        return json.loads(text)
