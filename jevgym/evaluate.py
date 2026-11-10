"""Evaluate an Arm on a split: one Jev call per row, everything logged to a run directory."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from jevgym import PACKAGE_ROOT
from jevgym.choice import Arm, build_request
from jevgym.data import Row, TestGuard
from jevgym.jev import PIN, PRICE_USD_PER_INPUT_TOKEN, JevAPIError, JevClient, PinMismatchError
from jevgym.report import build_summary
from jevgym.runlog import (
    CALLS,
    CONFIG,
    ERROR,
    EVALUATIONS,
    SUMMARY,
    append_jsonl,
    git_info,
    write_json,
    write_jsonl,
)

APPROVAL_THRESHOLD_USD = 1.0
# Token estimate per call = (request JSON chars / 3 + fixed server-side overhead) x safety margin.
# Calibrated on a 1,000-row run of short questions: billed input tokens ran 375-411 per call against
# chars/3 of 103-151, i.e. a fixed overhead of 252-279 tokens per call (mean 267). chars/3 alone was
# 3.3x too low. Recalibrate when a run's billed tokens exceed its estimate (the run summary shows both).
CHARS_PER_TOKEN_ESTIMATE = 3.0
OVERHEAD_TOKENS_PER_CALL = 280
SAFETY_MARGIN = 1.25


def estimate_cost(arm: Arm, rows: Sequence[Row], model: str = PIN) -> dict[str, Any]:
    chars = sum(len(json.dumps(build_request(arm.choice, arm.prompt, r.text, model))) for r in rows)
    raw = chars / CHARS_PER_TOKEN_ESTIMATE + OVERHEAD_TOKENS_PER_CALL * len(rows)
    tokens = int(raw * SAFETY_MARGIN) + 1
    return {
        "n_rows": len(rows),
        "n_calls": len(rows),
        "est_input_tokens": tokens,
        "usd": tokens * PRICE_USD_PER_INPUT_TOKEN,
        "method": (
            f"(request JSON chars / {CHARS_PER_TOKEN_ESTIMATE} + {OVERHEAD_TOKENS_PER_CALL} tokens/call) "
            f"x {SAFETY_MARGIN} at ${PRICE_USD_PER_INPUT_TOKEN * 1e6:.3f}/Mtok input"
        ),
    }


def build_config(
    arm: Arm,
    split: str,
    rows: Sequence[Row],
    dataset: dict[str, Any],
    client_settings: dict[str, Any],
    estimate: dict[str, Any],
    approval: dict[str, Any] | None,
    seed: int | None,
    argv: Sequence[str],
) -> dict[str, Any]:
    return {
        "arm": arm.to_dict(),
        "split": split,
        "n_rows": len(rows),
        "dataset": dataset,
        "model_pin": client_settings["pin"],
        "seed": seed,
        "git": git_info(),
        "core_git": git_info(PACKAGE_ROOT),
        "cost_estimate": estimate,
        "approval": approval,
        "client": client_settings,
        "command": list(argv),
    }


def evaluation_line(row: Row, qname: str, response: dict[str, Any]) -> dict[str, Any]:
    ans = response["answers"][qname]
    return {
        "row_id": row.id,
        "label": row.label,
        "prediction": ans["choice"],
        "correct": ans["choice"] == row.label,
        "probabilities": ans["probabilities"],
    }


async def run_eval(
    arm: Arm,
    split: str,
    rows: Sequence[Row],
    client: JevClient,
    run_dir: Path,
    config: dict[str, Any],
    guard: TestGuard | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Run every row, log as calls complete, then write evaluations and summary."""
    if split == "test":
        (guard or TestGuard()).record_test_run(run_dir)
    write_json(run_dir / CONFIG, config)
    calls_path = run_dir / CALLS
    calls_path.touch()

    qname = arm.choice.name
    responses: dict[str, dict[str, Any]] = {}

    async def one(row: Row):
        req = build_request(arm.choice, arm.prompt, row.text, client.pin)
        return row, await client.system_one(row.id, req)

    tasks = [asyncio.create_task(one(r)) for r in rows]
    done = 0
    try:
        for fut in asyncio.as_completed(tasks):
            try:
                row, rec = await fut
            except (PinMismatchError, JevAPIError) as e:
                append_jsonl(calls_path, e.call.to_dict())
                raise
            append_jsonl(calls_path, rec.to_dict())
            responses[row.id] = rec.response
            done += 1
            if progress and (done % 100 == 0 or done == len(rows)):
                print(f"  {done}/{len(rows)} calls", file=sys.stderr)
    except BaseException as e:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        write_json(run_dir / ERROR, {"error": type(e).__name__, "message": str(e), "calls_completed": done})
        raise

    write_jsonl(run_dir / EVALUATIONS, (evaluation_line(r, qname, responses[r.id]) for r in rows))
    summary = build_summary(run_dir)
    write_json(run_dir / SUMMARY, summary)
    return summary
