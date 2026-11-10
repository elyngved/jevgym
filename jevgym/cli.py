"""jevgym command line.

  jevgym eval --arm <arm> --split dev         print the cost estimate, then run (stops above $1 without approval)
  jevgym eval --arm <arm> --split dev --estimate   print only the estimate
  jevgym report <run_dir>                     rebuild summary.json offline and compare with the stored one
  jevgym noise <run_dir>... [--write]         noise floor from repeat runs (--write saves runs/noise-floor.json)
  jevgym leakcheck <run_dir>...                Teacher prompts: no Dev text, every example a Train row
  jevgym compare <run_a> <run_b>              paired comparison: Wilson intervals, gap, McNemar, label
  jevgym plot confusion <run_dir> --out <stem>        confusion matrix -> <stem>.png and <stem>.svg
  jevgym plot noise-repeats <run_dir>... --out <stem> repeat accuracies with Wilson intervals and floor band
  jevgym plot row-flips <run_dir>... --out <stem>     rows whose prediction changes across repeats
  jevgym plot arms <run_dir>... --out <stem>          accuracy (Wilson, floor band) and Brier per arm
  jevgym plot confusion-change <run_a> <run_b> --out <stem> [--best-of N]   confusion of B minus A, same rows
  jevgym plot seeds <run_dir>... --out <stem> [--ref <arm>]  per-seed accuracy, best of N marked, floor band

The workspace is $JEVGYM_HOME or the current directory; see jevgym/__init__.py for jevgym.toml.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from jevgym import WORKSPACE, workspace_config
from jevgym.data import TestGuard, load_split, sha256_file
from jevgym.evaluate import APPROVAL_THRESHOLD_USD, build_config, estimate_cost, run_eval
from jevgym.jev import PIN, JevClient
from jevgym.report import build_summary
from jevgym.runlog import SUMMARY, new_run_dir, read_json, write_json

EXIT_NEEDS_APPROVAL = 3


def _experiment(name: str):
    # Experiments live in the workspace, not in this package.
    if str(WORKSPACE) not in sys.path:
        sys.path.insert(0, str(WORKSPACE))
    return importlib.import_module(f"experiments.{name}.arms")


def cmd_eval(a: argparse.Namespace) -> int:
    a.experiment = a.experiment or workspace_config().get("experiment")
    if not a.experiment:
        print("give --experiment <name> or set experiment in jevgym.toml", file=sys.stderr)
        return 2
    try:
        exp = _experiment(a.experiment)
    except ModuleNotFoundError as err:
        if err.name not in ("experiments", f"experiments.{a.experiment}", f"experiments.{a.experiment}.arms"):
            raise
        print(f"no experiment {a.experiment!r} in workspace {WORKSPACE} (expected experiments/{a.experiment}/arms.py)", file=sys.stderr)
        return 2
    if a.arm not in exp.ARMS:
        print(f"unknown arm {a.arm!r}; known: {sorted(exp.ARMS)}", file=sys.stderr)
        return 2
    arm = exp.ARMS[a.arm]
    guard = TestGuard()
    rows = load_split(exp.DATASET_DIR, a.split, guard=guard)
    if a.limit:
        rows = rows[: a.limit]
    est = estimate_cost(arm, rows)
    print(
        f"Cost estimate for {arm.name} on {a.split}: {est['n_calls']} Jev calls, "
        f"~{est['est_input_tokens']:,} input tokens, ~${est['usd']:.4f} ({est['method']})"
    )
    if a.estimate:
        return 0
    approval = None
    if est["usd"] > APPROVAL_THRESHOLD_USD:
        if a.approve_usd is None or a.approve_usd < est["usd"]:
            print(
                f"Estimate is above ${APPROVAL_THRESHOLD_USD:.2f}. Stopping for approval. "
                f"Rerun with --approve-usd <amount> --approval-note '<who/when>' once approved.",
                file=sys.stderr,
            )
            return EXIT_NEEDS_APPROVAL
        approval = {"approved_usd": a.approve_usd, "note": a.approval_note}

    split_file = Path(exp.DATASET_DIR) / f"{a.split}.jsonl"
    dataset = {
        "dir": str(Path(exp.DATASET_DIR).relative_to(WORKSPACE)),
        "split_file_sha256": sha256_file(split_file),
        "limit": a.limit,
    }

    async def go() -> dict:
        async with JevClient(pin=PIN, max_concurrency=a.concurrency) as client:
            seed = getattr(exp, "SPLIT_SEED", None)
            run_dir = new_run_dir(f"{arm.slug}-{a.split}" + (f"-{a.tag}" if a.tag else ""))
            print(f"Run dir: {run_dir.relative_to(WORKSPACE)}")
            config = build_config(arm, a.split, rows, dataset, client.settings(), est, approval, seed, sys.argv)
            config["noise_group"] = a.noise_group or None
            return await run_eval(arm, a.split, rows, client, run_dir, config, guard=guard)

    s = asyncio.run(go())
    lo, hi = s["wilson_95"]
    print(
        f"{s['arm']} {s['split']}: accuracy {s['accuracy']:.4f} ({s['n_correct']}/{s['n']}), "
        f"95% Wilson [{lo:.4f}, {hi:.4f}], Brier {s['brier']:.4f}, log loss {s['log_loss']:.4f}, "
        f"cost ${s['total_cost_usd']:.6f}, all calls pinned: {s['all_calls_pinned']}"
    )
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir)
    rebuilt = build_summary(run_dir)
    path = run_dir / SUMMARY
    if path.exists() and not a.write:
        stored = read_json(path)
        if stored == rebuilt:
            print(f"IDENTICAL: rebuilt summary matches {path}")
            print(json.dumps({k: rebuilt[k] for k in ("accuracy", "wilson_95", "brier", "log_loss", "total_cost_usd")}))
            return 0
        diff = sorted(k for k in set(stored) | set(rebuilt) if stored.get(k) != rebuilt.get(k))
        print(f"DIFFERS: keys {diff}", file=sys.stderr)
        return 1
    write_json(path, rebuilt)
    print(f"wrote {path}")
    return 0


def cmd_noise(a: argparse.Namespace) -> int:
    from jevgym import noise

    r = noise.compute(a.run_dirs)
    print(
        f"{r['arm']} {r['split']} x{r['n_repeats']}: accuracies {', '.join(f'{x:.3f}' for x in r['accuracies'])}\n"
        f"noise floor {r['noise_floor_points']:.2f} pts (max - min), sd {r['sd_points']:.2f} pts, "
        f"{r['n_rows_flipped']} rows flip ({r['flip_rate']:.1%})"
    )
    if r["all_floor_zero_note"]:
        print(r["all_floor_zero_note"])
    if a.write:
        write_json(noise.NOISE_FLOOR_FILE, r)
        print(f"wrote {noise.NOISE_FLOOR_FILE.relative_to(WORKSPACE)}")
    return 0


def cmd_leakcheck(a: argparse.Namespace) -> int:
    from jevgym import leakcheck

    bad = 0
    for d in a.run_dirs:
        d = Path(d)
        r = leakcheck.check_run(d, leakcheck.dataset_dir_of(d))
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{status} {r['run']}: {r['teacher_calls']} Teacher calls, {r.get('examples_checked', 0)} examples "
              f"all Train, Dev scan (exact + {r.get('ngram', '-')}-gram) "
              + ("clean" if r["ok"] else json.dumps(r["problems"])))
        bad += not r["ok"]
    return 1 if bad else 0


def cmd_compare(a: argparse.Namespace) -> int:
    from jevgym import compare

    r = compare.compare(a.run_a, a.run_b)
    print(json.dumps(r, indent=2) if a.json else compare.format_text(r))
    return 0


def cmd_plot(a: argparse.Namespace) -> int:
    from jevgym import plot

    run_dirs = [Path(d) for d in a.run_dirs]
    stem = Path(a.out)
    kw = {}
    if a.kind == "seeds":
        kw["ref"] = a.ref
    if a.kind == "confusion-change" and a.best_of:
        kw["best_of"] = a.best_of
    paths = getattr(plot, a.kind.replace("-", "_"))(run_dirs, stem, **kw)
    for p in paths:
        print(f"wrote {p}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Secrets (TYPESAFE_API_KEY) live in the gitignored workspace .env; real env vars win.
    load_dotenv(WORKSPACE / ".env", override=False)
    p = argparse.ArgumentParser(prog="jevgym")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("eval", help="evaluate an arm on a split")
    e.add_argument("--arm", required=True)
    e.add_argument("--split", required=True, choices=["train", "dev", "test"])
    e.add_argument("--experiment", default=None, help="experiments/<name>/arms.py in the workspace (default: jevgym.toml)")
    e.add_argument("--estimate", action="store_true", help="print the cost estimate only")
    e.add_argument("--approve-usd", type=float, default=None, help="approved spend for runs over $1")
    e.add_argument("--approval-note", default="", help="who approved and when")
    e.add_argument("--limit", type=int, default=0, help="first N rows only (smoke tests)")
    e.add_argument("--concurrency", type=int, default=8)
    e.add_argument("--tag", default="", help="suffix for the run directory name")
    e.add_argument("--noise-group", default="", help="tag repeat runs of one identical config (recorded in config.json)")
    e.set_defaults(func=cmd_eval)

    r = sub.add_parser("report", help="rebuild summary.json offline from a run directory")
    r.add_argument("run_dir")
    r.add_argument("--write", action="store_true", help="overwrite summary.json instead of comparing")
    r.set_defaults(func=cmd_report)

    nz = sub.add_parser("noise", help="noise floor from repeat runs of one identical config")
    nz.add_argument("run_dirs", nargs="+")
    nz.add_argument("--write", action="store_true", help="save runs/noise-floor.json")
    nz.set_defaults(func=cmd_noise)

    lk = sub.add_parser("leakcheck", help="fail if a Teacher prompt holds Dev text or non-Train examples")
    lk.add_argument("run_dirs", nargs="+")
    lk.set_defaults(func=cmd_leakcheck)

    c = sub.add_parser("compare", help="paired comparison of two runs on the same rows")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_compare)

    pl = sub.add_parser("plot", help="charts from run directories")
    pl.add_argument("kind", choices=["confusion", "noise-repeats", "row-flips", "arms", "confusion-change", "seeds"])
    pl.add_argument("run_dirs", nargs="+")
    pl.add_argument("--out", required=True, help="output path stem (without extension)")
    pl.add_argument("--ref", default=None, help="seeds: arm the noise-floor band is centred on (default: the floor's arm)")
    pl.add_argument("--best-of", type=int, default=0, help="confusion-change: run B was picked as best of N")
    pl.set_defaults(func=cmd_plot)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
