import asyncio
import json

import pytest

from conftest import fake_transport
from jevgym import cli
from jevgym.data import TestGuard
from jevgym.evaluate import build_config, estimate_cost, run_eval
from jevgym.jev import PIN, JevClient, PinMismatchError
from jevgym.report import build_summary
from jevgym.runlog import read_json, read_jsonl


def run(arm, rows, tmp_path, transport, split="dev"):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    async def go():
        async with JevClient(api_key="test", transport=transport) as c:
            cfg = build_config(arm, split, rows, {"dir": "x"}, c.settings(), estimate_cost(arm, rows), None, 1, ["t"])
            return await run_eval(arm, split, rows, c, run_dir, cfg, guard=TestGuard(tmp_path), progress=False)

    return run_dir, asyncio.run(go())


def test_run_dir_complete_and_report_reproduces(arm, rows, tmp_path):
    pick = {"How many feet in a mile ?": "NUM", "What does NASA stand for ?": "ABBR", "Why is the sky blue ?": "NUM"}
    run_dir, s = run(arm, rows, tmp_path, fake_transport(pick=pick))
    for f in ("config.json", "calls.jsonl", "evaluations.jsonl", "summary.json"):
        assert (run_dir / f).exists()
    cfg = read_json(run_dir / "config.json")
    for k in ("arm", "split", "model_pin", "seed", "git", "cost_estimate", "approval"):
        assert k in cfg
    assert cfg["arm"]["prompt"]["instructions"]
    calls = read_jsonl(run_dir / "calls.jsonl")
    assert len(calls) == 4 and all(c["returned_model"] == PIN for c in calls)
    assert all({"row_id", "request", "response", "returned_model", "cost_usd"} <= set(c) for c in calls)
    evals = read_jsonl(run_dir / "evaluations.jsonl")
    assert [e["row_id"] for e in evals] == [r.id for r in rows]
    # r1 NUM ok, r2 ABBR ok, r3 predicted NUM (wrong), r4 defaults to ABBR (wrong)
    assert s["n_correct"] == 2 and s["accuracy"] == 0.5
    assert s["total_cost_usd"] == pytest.approx(4 * 50 * 42e-9)
    assert s["all_calls_pinned"]
    assert build_summary(run_dir) == read_json(run_dir / "summary.json")
    assert cli.main(["report", str(run_dir)]) == 0


def test_report_detects_tampered_evaluations(arm, rows, tmp_path):
    run_dir, _ = run(arm, rows, tmp_path, fake_transport())
    evals = read_jsonl(run_dir / "evaluations.jsonl")
    evals[0]["prediction"] = "DESC"
    (run_dir / "evaluations.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evals))
    with pytest.raises(Exception):
        build_summary(run_dir)


def test_pin_mismatch_stops_run_and_logs(arm, rows, tmp_path):
    with pytest.raises(PinMismatchError):
        run(arm, rows, tmp_path, fake_transport(model="jev-1.15.0"))
    run_dir = tmp_path / "run"
    assert (run_dir / "error.json").exists()
    calls = read_jsonl(run_dir / "calls.jsonl")
    assert calls and calls[0]["returned_model"] == "jev-1.15.0"
    assert not (run_dir / "summary.json").exists()
