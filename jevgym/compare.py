"""Paired comparison of two runs on the same rows: Wilson intervals, gap, exact McNemar, label.

Labels, checked in this order:
  1. "within noise floor": |gap| <= the floor in runs/noise-floor.json.
  2. "not yet resolved":   the exact McNemar p-value is >= 0.05.
  3. "resolved":           otherwise.
The 95% Wilson intervals are reported for both runs but do not gate the label: they mostly measure how
hard the sample of rows is, which affects both runs equally and cancels out in a paired comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jevgym import metrics
from jevgym.noise import load_floor
from jevgym.report import build_summary
from jevgym.runlog import EVALUATIONS, read_jsonl

ALPHA = 0.05
WITHIN_FLOOR = "within noise floor"
NOT_RESOLVED = "not yet resolved"
RESOLVED = "resolved"


class CompareError(RuntimeError):
    pass


def label(gap_pts: float, floor_pts: float, p: float) -> tuple[str, str]:
    """Label for the gap and the reason for it."""
    if round(abs(gap_pts), 9) <= round(floor_pts, 9):
        return WITHIN_FLOOR, f"|gap| {abs(gap_pts):.1f} pts <= floor {floor_pts:.1f} pts"
    if p >= ALPHA:
        return NOT_RESOLVED, f"gap above floor, but McNemar p = {p:.3g} >= {ALPHA}"
    return RESOLVED, f"gap above floor and McNemar p = {p:.3g} < {ALPHA}"


def compare(run_a: str | Path, run_b: str | Path, floor: dict[str, Any] | None = None) -> dict[str, Any]:
    run_a, run_b = Path(run_a), Path(run_b)
    sa, sb = build_summary(run_a), build_summary(run_b)
    ea, eb = read_jsonl(run_a / EVALUATIONS), read_jsonl(run_b / EVALUATIONS)
    if sa["split"] != sb["split"]:
        raise CompareError(f"different splits: {sa['split']} vs {sb['split']}")
    by_b = {e["row_id"]: e for e in eb}
    if set(by_b) != {e["row_id"] for e in ea}:
        raise CompareError("runs are not on the same rows; a paired comparison is impossible")
    ca = [e["correct"] for e in ea]
    cb = [by_b[e["row_id"]]["correct"] for e in ea]
    cells = metrics.paired_counts(ca, cb)
    p = metrics.mcnemar_exact(cells["only_a"], cells["only_b"])

    floor = floor if floor is not None else load_floor()
    floor_pts = float(floor["noise_floor_points"])
    warn = []
    if floor.get("split") != sa["split"] or floor.get("n_rows") != sa["n"]:
        warn.append(f"noise floor was measured on {floor.get('split')} n={floor.get('n_rows')}, "
                    f"this comparison is {sa['split']} n={sa['n']}")
    gap_pts = (sb["accuracy"] - sa["accuracy"]) * 100
    verdict, reason = label(gap_pts, floor_pts, p)
    return {
        "a": {"run": run_a.name, "arm": sa["arm"], "accuracy": sa["accuracy"], "wilson_95": sa["wilson_95"], "n": sa["n"]},
        "b": {"run": run_b.name, "arm": sb["arm"], "accuracy": sb["accuracy"], "wilson_95": sb["wilson_95"], "n": sb["n"]},
        "gap_points": gap_pts,
        "noise_floor_points": floor_pts,
        "noise_floor_source": floor.get("source_runs"),
        "paired_cells": cells,
        "mcnemar_p": p,
        "mcnemar_test": "exact two-sided binomial on discordant rows",
        "label": verdict,
        "reason": reason,
        "warnings": warn,
    }


def format_text(r: dict[str, Any]) -> str:
    a, b, c = r["a"], r["b"], r["paired_cells"]
    lines = [
        f"A {a['arm']:<4} {a['run']}: {a['accuracy']:.1%}  95% Wilson [{a['wilson_95'][0]:.1%}, {a['wilson_95'][1]:.1%}]",
        f"B {b['arm']:<4} {b['run']}: {b['accuracy']:.1%}  95% Wilson [{b['wilson_95'][0]:.1%}, {b['wilson_95'][1]:.1%}]",
        f"Gap B - A: {r['gap_points']:+.1f} pts   noise floor: {r['noise_floor_points']:.1f} pts",
        f"Paired rows (n={a['n']}): both right {c['both']}, only A right {c['only_a']}, "
        f"only B right {c['only_b']}, both wrong {c['neither']}",
        f"Exact McNemar on {c['only_a'] + c['only_b']} discordant rows: p = {r['mcnemar_p']:.3g}",
        f"Label: {r['label']} ({r['reason']})",
    ]
    lines += [f"Warning: {w}" for w in r["warnings"]]
    return "\n".join(lines)
