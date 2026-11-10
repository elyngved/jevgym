"""Noise floor: how much Dev accuracy moves when the same Prompt is evaluated again.

The floor is the max - min spread of accuracy across repeat runs of one identical config, in
percentage points. It is written to runs/noise-floor.json, and every later comparison reads it from
there. Also reported: the sample standard deviation of accuracy and the per-row flip rate (rows whose
prediction differs in at least one repeat).
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Sequence

from jevgym import WORKSPACE
from jevgym.report import build_summary
from jevgym.runlog import CONFIG, EVALUATIONS, read_json, read_jsonl

NOISE_FLOOR_FILE = WORKSPACE / "runs" / "noise-floor.json"
METHOD = "max - min of Dev accuracy across repeat runs of one identical config, in points"


class NoiseError(RuntimeError):
    pass


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(p)


def _identity(config: dict[str, Any]) -> dict[str, Any]:
    """The parts of a config that must match for runs to count as repeats."""
    return {
        "arm": config["arm"],
        "split": config["split"],
        "n_rows": config["n_rows"],
        "split_file_sha256": config["dataset"]["split_file_sha256"],
        "model_pin": config["model_pin"],
    }


def load_repeats(run_dirs: Sequence[str | Path]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Summaries (recomputed offline) and evaluations per run; checks the runs really are repeats."""
    run_dirs = [Path(d) for d in run_dirs]
    if len(run_dirs) < 2:
        raise NoiseError("need at least 2 repeat runs")
    configs = [read_json(d / CONFIG) for d in run_dirs]
    ident = _identity(configs[0])
    for d, c in zip(run_dirs, configs):
        if _identity(c) != ident:
            raise NoiseError(f"{d.name} is not a repeat of {run_dirs[0].name} (arm, split, data or pin differs)")
    groups = {c.get("noise_group") for c in configs}
    if len(groups) != 1:
        raise NoiseError(f"runs carry different noise groups: {sorted(map(str, groups))}")
    summaries = [build_summary(d) for d in run_dirs]
    evals = {d.name: read_jsonl(d / EVALUATIONS) for d in run_dirs}
    order = [e["row_id"] for e in evals[run_dirs[0].name]]
    for name, ev in evals.items():
        if [e["row_id"] for e in ev] != order:
            raise NoiseError(f"{name}: rows differ from {run_dirs[0].name}")
    return summaries, evals


def row_flips(evals: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Rows whose prediction is not the same in every repeat, with each repeat's prediction."""
    names = list(evals)
    out = []
    for i, first in enumerate(evals[names[0]]):
        preds = [evals[n][i]["prediction"] for n in names]
        if len(set(preds)) > 1:
            out.append({
                "row_id": first["row_id"],
                "label": first["label"],
                "predictions": preds,
                "n_correct": sum(p == first["label"] for p in preds),
            })
    return out


def compute(run_dirs: Sequence[str | Path]) -> dict[str, Any]:
    run_dirs = [Path(d) for d in run_dirs]
    summaries, evals = load_repeats(run_dirs)
    accs = [s["accuracy"] for s in summaries]
    flips = row_flips(evals)
    n = summaries[0]["n"]
    labels = [e["label"] for e in next(iter(evals.values()))]
    by_class: dict[str, dict[str, int]] = {}
    for y in sorted(set(labels)):
        by_class[y] = {"n": labels.count(y), "flipped": sum(f["label"] == y for f in flips)}
    floor_pts = (max(accs) - min(accs)) * 100
    group = read_json(run_dirs[0] / CONFIG).get("noise_group")
    return {
        "noise_floor_points": round(floor_pts, 6),
        "method": METHOD,
        "noise_group": group,
        "arm": summaries[0]["arm"],
        "split": summaries[0]["split"],
        "n_rows": n,
        "n_repeats": len(run_dirs),
        "accuracies": accs,
        "mean_accuracy": statistics.fmean(accs),
        "sd_points": statistics.stdev(accs) * 100,
        "sd_definition": "sample standard deviation (n-1) of accuracy across repeats, in points",
        "n_rows_flipped": len(flips),
        "flip_rate": len(flips) / n,
        "flip_definition": "share of rows whose top prediction differs in at least one repeat",
        "flips_by_class": by_class,
        "all_floor_zero_note": "floor is 0: all repeats agree exactly; open question whether the floor needs a minimum"
        if floor_pts == 0 else None,
        "source_runs": [_rel(d) for d in run_dirs],
    }


def load_floor(path: Path = NOISE_FLOOR_FILE) -> dict[str, Any]:
    if not path.exists():
        raise NoiseError(f"{_rel(path)} does not exist; measure the floor first (jevgym noise ...)")
    return read_json(path)
