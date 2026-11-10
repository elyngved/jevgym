"""Build summary.json from a run directory's other files, offline.

The eval command writes summary.json with this same function, so `jevgym report` must reproduce it
exactly. It also cross-checks evaluations.jsonl against the raw responses in calls.jsonl, and
recomputes cost from token counts and the price recorded in config.json.

Runs with config "backend": "local" (non-LLM arms such as B2 naive Bayes) make no Jev calls: their
calls.jsonl is empty, cost is 0, and the pin fields are null (not applicable).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from jevgym import metrics
from jevgym.runlog import CALLS, CONFIG, EVALUATIONS, read_json, read_jsonl

TEACHER_CALLS = "teacher_calls.jsonl"  # same name as jevgym.teacher.TEACHER_CALLS (kept import-free here)


class ReportError(RuntimeError):
    pass


def build_summary(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    config = read_json(run_dir / CONFIG)
    calls = read_jsonl(run_dir / CALLS)
    evals = read_jsonl(run_dir / EVALUATIONS)
    local = config.get("backend", "jev") == "local"
    if local and calls:
        raise ReportError("local run has Jev calls in calls.jsonl")

    option_ids = config["arm"]["choice"]["option_ids"]
    qname = config["arm"]["choice"]["name"]
    pin = config["model_pin"]
    price = 0.0 if local else config["client"]["price_usd_per_input_token"]

    calls_by_row = {c["row_id"]: c for c in calls}
    if len(calls_by_row) != len(calls):
        raise ReportError("duplicate row ids in calls.jsonl")
    if len(evals) != config["n_rows"]:
        raise ReportError(f"{len(evals)} evaluations but config says {config['n_rows']} rows")

    labels, preds, prob_rows, label_idx = [], [], [], []
    max_dev = 0.0
    for e in evals:
        if not local:
            c = calls_by_row.get(e["row_id"])
            if c is None:
                raise ReportError(f"no call for row {e['row_id']}")
            ans = c["response"]["answers"][qname]
            if ans["choice"] != e["prediction"] or ans["probabilities"] != e["probabilities"]:
                raise ReportError(f"row {e['row_id']}: evaluation does not match call response")
        raw = sum(float(v) for v in e["probabilities"].values())
        max_dev = max(max_dev, abs(raw - 1.0))
        labels.append(e["label"])
        preds.append(e["prediction"])
        prob_rows.append(metrics.normalize(e["probabilities"], option_ids))
        label_idx.append(option_ids.index(e["label"]))

    n = len(evals)
    correct = sum(y == p for y, p in zip(labels, preds))
    lo, hi = metrics.wilson_interval(correct, n)
    tokens = sum(int(c["input_tokens"]) for c in calls)
    returned = Counter(str(c["returned_model"]) for c in calls)
    out = {
        "arm": config["arm"]["name"],
        "split": config["split"],
        "n": n,
        "n_correct": correct,
        "accuracy": correct / n,
        "wilson_95": [lo, hi],
        "brier": metrics.brier(prob_rows, label_idx),
        "brier_definition": "mean over rows of sum over options of (p - y)^2; range 0-2",
        "log_loss": metrics.log_loss(prob_rows, label_idx),
        "log_loss_eps": metrics.LOG_LOSS_EPS,
        "log_loss_strict": metrics.log_loss(prob_rows, label_idx, eps=metrics.LOG_LOSS_EPS_STRICT),
        "log_loss_strict_eps": metrics.LOG_LOSS_EPS_STRICT,
        "n_true_label_prob_zero": sum(ps[y] == 0 for ps, y in zip(prob_rows, label_idx)),
        "prob_sum_max_abs_dev": max_dev,
        "n_calls": len(calls),
        "total_input_tokens": tokens,
        "total_cost_usd": tokens * price,
        "cost_estimate_usd": config["cost_estimate"]["usd"],
        "returned_models": dict(sorted(returned.items())),
        "all_calls_pinned": None if local else all(c["returned_model"] == pin for c in calls),
        "confusion": {"option_ids": option_ids, "rows_true_cols_pred": metrics.confusion(labels, preds, option_ids)},
    }
    teacher_path = run_dir / TEACHER_CALLS
    if teacher_path.exists():  # only runs with Teacher calls get these keys, so older summaries are unchanged
        tc = read_jsonl(teacher_path)
        out["teacher_calls"] = len(tc)
        out["teacher_models"] = dict(sorted(Counter(c["returned_model"] for c in tc).items()))
        out["teacher_all_pinned"] = all(c["model_ok"] for c in tc)
        out["teacher_input_tokens"] = sum(c["usage"]["input_tokens"] for c in tc)
        out["teacher_output_tokens"] = sum(c["usage"]["output_tokens"] for c in tc)
        out["teacher_cost_usd"] = sum(c["cost_usd"] for c in tc)
        out["total_cost_with_teacher_usd"] = out["total_cost_usd"] + out["teacher_cost_usd"]
    return out
