"""Run directories: runs/<UTC timestamp>-<slug>/ with config, calls, evaluations and summary."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from jevgym import WORKSPACE

CONFIG = "config.json"
CALLS = "calls.jsonl"
EVALUATIONS = "evaluations.jsonl"
SUMMARY = "summary.json"
ERROR = "error.json"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_run_dir(slug: str, runs_root: str | Path | None = None) -> Path:
    runs_root = Path(runs_root) if runs_root else WORKSPACE / "runs"
    d = runs_root / f"{utc_stamp()}-{slug}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def git_info(root: Path = WORKSPACE) -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status), "dirty_files": status.splitlines()}


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, obj: Any) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, objs: Iterable[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Any]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
