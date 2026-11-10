"""JevGym: prompt optimizer for Jev Choice questions (generic core)."""

import os
from pathlib import Path

# Workspace: where runs/, test.lock, .env, experiments/ and figures live. It is $JEVGYM_HOME if
# set, else the current directory. Paths stored in run configs are relative to it.
WORKSPACE = Path(os.environ.get("JEVGYM_HOME") or Path.cwd()).resolve()

# Optional workspace settings, read from <workspace>/jevgym.toml:
#   experiment = "name"   default for `jevgym eval --experiment` (experiments/<name>/arms.py)
#   colors = "path.md"    markdown table of arm colors, relative to the workspace
CONFIG_FILE = WORKSPACE / "jevgym.toml"


def workspace_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    import tomllib

    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


# Source checkout of this package, recorded next to the workspace commit in every run config.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
