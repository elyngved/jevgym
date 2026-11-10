# JevGym

Measure and improve prompts for [Jev](https://typesafe.ai) Choice questions, with statistics you can trust.

JevGym evaluates a Prompt (Instructions plus one Description per option) on a labelled dataset and reports accuracy with a 95% Wilson interval, Brier score and log loss. It compares two Prompts on the same rows with a paired McNemar test, measured against the run-to-run noise floor. Every run is logged and can be rebuilt offline.

**Status:** early. Evaluation, comparison, noise floor, charts and a Teacher client (a Claude model that writes Prompt parts) work. The optimizer is not in yet.

## Setup

```
uv sync
export TYPESAFE_API_KEY=...      # or put it in a .env file in your workspace
```

Requires Python 3.12+. Every Jev call is checked against the pinned model (`jev-1.14.0`), and a run stops if a different model answers.

## Workspace

JevGym works in a **workspace**: the current directory, or `$JEVGYM_HOME`. It holds:

```
data/<dataset>/{train,dev,test}.jsonl   rows: {"id": ..., "text": ..., "label": ...}
experiments/<name>/arms.py              DATASET_DIR and ARMS = {"<arm>": Arm(...)}
runs/                                   one directory per run (written by jevgym)
test.lock                               while present, Test cannot be loaded
jevgym.toml                             optional: experiment = "<name>", colors = "<table.md>"
```

An arm is a named Prompt for a Choice:

```python
from jevgym import WORKSPACE
from jevgym.choice import Arm, Choice, Prompt

DATASET_DIR = WORKSPACE / "data" / "tickets"
TICKETS = Choice(name="team", option_ids=("BILLING", "BUG", "OTHER"), state_field="ticket")
ARMS = {
    "a1": Arm(name="A1", slug="a1-plain", choice=TICKETS, prompt=Prompt(
        instructions="Which team should handle `ticket`?",
        descriptions={"BILLING": "payments, invoices, refunds", "BUG": "something is broken", "OTHER": "anything else"},
    )),
}
```

Test is guarded: the loader refuses it while `test.lock` exists, and it can be run only once (recorded in `test-run.json`).

## Commands

```
jevgym eval --arm <arm> --split dev       cost estimate, then the run (stops above $1 without --approve-usd)
jevgym report <run_dir>                   rebuild summary.json offline and check it matches
jevgym noise <run_dir>... --write         noise floor from repeat runs of one config
jevgym compare <run_a> <run_b>            paired comparison: gap, Wilson intervals, McNemar, label
jevgym leakcheck <run_dir>...             Teacher prompts hold no Dev text, only Train examples
jevgym plot <kind> <run_dir>... --out <stem>   charts (PNG and SVG) from run directories
```

`jevgym --help` lists every option.

## Tests

```
uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
