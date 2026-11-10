"""Leak check for Teacher calls: the Teacher sees Train only.

`jevgym leakcheck <run_dir>` fails when any Teacher prompt in the run's teacher_calls.jsonl
  1. contains a Dev question, by normalized exact match (lowercased, punctuation dropped), or
  2. shares a run of NGRAM consecutive words with a Dev question (long n-gram overlap), or
  3. quotes an example question that is not an exact Train row with that Train label (provenance).

Matches that Train itself explains are not counted: if a Dev question's word sequence also occurs inside
a Train question, it cannot be told apart from legitimate Train content (those are reported as
"ambiguous", not failed).

Test is not read: it stays behind test.lock. Check 3 covers it instead: when the splits are built so
that Train holds no Test text, a prompt made of a fixed template plus verified Train rows holds no Test
text either.

Example questions must be written in the canonical format from `format_examples`, so check 3 can find them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

from jevgym.data import Row, load_split
from jevgym.runlog import read_json, read_jsonl
from jevgym.teacher import TEACHER_CALLS

NGRAM = 8
_EX_BLOCK = re.compile(r'<examples option="([^"]+)">\n(.*?)\n</examples>', re.S)


def format_examples(option_id: str, texts: Sequence[str]) -> str:
    """Canonical example block for Teacher prompts: one Train question per line."""
    return f'<examples option="{option_id}">\n' + "\n".join(f"- {t}" for t in texts) + "\n</examples>"


def parse_examples(prompt: str) -> list[tuple[str, str]]:
    out = []
    for opt, body in _EX_BLOCK.findall(prompt):
        for line in body.split("\n"):
            if not line.startswith("- "):
                out.append((opt, f"<malformed line: {line[:40]!r}>"))
            else:
                out.append((opt, line[2:]))
    return out


def norm_tokens(text: str) -> list[str]:
    return re.sub(r"[^0-9a-z]+", " ", text.lower()).split()


def _windows(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class LeakIndex:
    """Dev question sequences and n-grams, minus what Train explains."""

    def __init__(self, dev: Sequence[Row], train: Sequence[Row], ngram: int = NGRAM):
        self.ngram = ngram
        self.train = {(r.text, r.label) for r in train}
        train_toks = [norm_tokens(r.text) for r in train]
        dev_seqs = {tuple(norm_tokens(r.text)): r.id for r in dev}
        lengths = {len(s) for s in dev_seqs}
        train_windows: set[tuple[str, ...]] = set()
        for t in train_toks:
            for n in lengths | {ngram}:
                train_windows |= _windows(t, n)
        self.exact = {s: rid for s, rid in dev_seqs.items() if s not in train_windows}
        self.ambiguous = len(dev_seqs) - len(self.exact)
        self.lengths = {len(s) for s in self.exact}
        self.ngrams: dict[tuple[str, ...], str] = {}
        for s, rid in dev_seqs.items():
            for g in _windows(list(s), ngram):
                if g not in train_windows:
                    self.ngrams.setdefault(g, rid)

    def scan(self, text: str) -> dict[str, list[str]]:
        toks = norm_tokens(text)
        exact = sorted({self.exact[w] for n in self.lengths for w in _windows(toks, n) if w in self.exact})
        grams = sorted({self.ngrams[g] for g in _windows(toks, self.ngram) if g in self.ngrams})
        return {"exact_dev_rows": exact, "ngram_dev_rows": grams}


def check_run(run_dir: str | Path, dataset_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    path = run_dir / TEACHER_CALLS
    if not path.exists():
        return {"run": run_dir.name, "teacher_calls": 0, "ok": True, "note": "no Teacher calls"}
    calls = read_jsonl(path)
    idx = LeakIndex(load_split(dataset_dir, "dev"), load_split(dataset_dir, "train"))
    problems = []
    n_examples = 0
    for i, c in enumerate(calls):
        prompt = f"{c['system']}\n{c['user']}"
        hits = idx.scan(prompt)
        if hits["exact_dev_rows"] or hits["ngram_dev_rows"]:
            problems.append({"call": i, "kind": "dev_text", **hits})
        examples = parse_examples(c["user"])
        n_examples += len(examples)
        bad = [(o, t) for o, t in examples if (t, o) not in idx.train]
        if bad:
            problems.append({"call": i, "kind": "not_train", "examples": bad[:5], "n": len(bad)})
    return {
        "run": run_dir.name,
        "teacher_calls": len(calls),
        "examples_checked": n_examples,
        "dev_questions_ambiguous_with_train": idx.ambiguous,
        "ngram": idx.ngram,
        "test": "not read (locked); covered by Train provenance of every example",
        "problems": problems,
        "ok": not problems,
    }


def dataset_dir_of(run_dir: Path) -> Path:
    from jevgym import WORKSPACE

    return WORKSPACE / read_json(run_dir / "config.json")["dataset"]["dir"]
