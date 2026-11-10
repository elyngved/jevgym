"""JSONL datasets, split loading, and the test-once guard.

Each dataset directory holds train.jsonl, dev.jsonl and test.jsonl with rows {id, text, label}.

Test is guarded:
- while `test.lock` exists in the workspace, loading Test raises TestLockedError;
- once a Test run has been recorded (`test-run.json` in the workspace), loading Test again raises
  TestAlreadyRunError, even after the lock is removed.
The generic JSONL loader refuses any file named test.jsonl, so Test is only reachable through
`TestGuard.load_test`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from jevgym import WORKSPACE

TEST_FILENAME = "test.jsonl"
SPLITS = ("train", "dev", "test")


class TestLockedError(RuntimeError):
    """Test is locked: test.lock exists."""

    __test__ = False


class TestAlreadyRunError(RuntimeError):
    """A Test run has already been recorded; Test runs once."""

    __test__ = False


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "text": self.text, "label": self.label}


def _read_jsonl(path: Path) -> list[Row]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            try:
                rows.append(Row(id=str(d["id"]), text=str(d["text"]), label=str(d["label"])))
            except KeyError as e:
                raise ValueError(f"{path}:{n}: missing field {e}") from None
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate row ids")
    return rows


def load_jsonl(path: str | Path) -> list[Row]:
    """Load a JSONL dataset file. Refuses Test files; use TestGuard.load_test for those."""
    path = Path(path)
    if path.name == TEST_FILENAME:
        raise TestLockedError(f"{path} is a Test file; it can only be loaded through TestGuard.load_test")
    return _read_jsonl(path)


def write_jsonl(path: str | Path, rows: Iterable[Row]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class TestGuard:
    """The test-once guard. `root` is the workspace (overridable for tests)."""

    __test__ = False

    def __init__(self, root: str | Path = WORKSPACE):
        self.root = Path(root)
        self.lock_path = self.root / "test.lock"
        self.record_path = self.root / "test-run.json"

    def check(self) -> None:
        if self.lock_path.exists():
            raise TestLockedError(
                f"Test is locked ({self.lock_path} exists). Remove it only once the candidates to test are frozen."
            )
        if self.record_path.exists():
            raise TestAlreadyRunError(
                f"A Test run is already recorded in {self.record_path}. Test runs once."
            )

    def load_test(self, dataset_dir: str | Path) -> list[Row]:
        self.check()
        return _read_jsonl(Path(dataset_dir) / TEST_FILENAME)

    def record_test_run(self, run_dir: str | Path, note: str = "") -> None:
        """Record that a Test run started. Exclusive create: a second record fails."""
        self.check()
        with open(self.record_path, "x", encoding="utf-8") as f:
            json.dump(
                {"run_dir": str(run_dir), "started_utc": datetime.now(timezone.utc).isoformat(), "note": note},
                f,
                indent=2,
            )


def load_split(dataset_dir: str | Path, split: str, guard: TestGuard | None = None) -> list[Row]:
    """Load train/dev directly; test only through the guard."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if split == "test":
        return (guard or TestGuard()).load_test(dataset_dir)
    return load_jsonl(Path(dataset_dir) / f"{split}.jsonl")
