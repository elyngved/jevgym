"""House style behind every `jevgym plot` chart.

- Title says what is plotted; subtitle says how to read it (`header`).
- Noise-floor band and 95% Wilson intervals wherever accuracies are compared (`noise_band`, `wilson_point`).
- Best-of-N points are marked as such (`best_of_n`).
- Run directories and the command go into PNG/SVG metadata, not onto the image (`save`). A visible footer
  appears only to cite an external source (`external_footer`).
- Arm colors are read from the markdown table named by `colors` in jevgym.toml (`ARM_COLORS`), never hard-coded here.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from jevgym import WORKSPACE, workspace_config  # noqa: E402

COLORS_FILE = WORKSPACE / workspace_config().get("colors", "COLORS.md")
INK = "#1f1f1e"
MUTED = "#6b6a66"
FAINT = "#e4e3df"
SURFACE = "#ffffff"
FLOOR_BAND = "#d9d7d0"
FALLBACK = "#6b6a66"
# Diagnostic diverging pair for "change vs. previous best" charts; not an arm color.
BETTER = "#6a4fc2"  # purple
WORSE = "#d0453a"  # red

plt.rcParams.update(
    {
        "svg.hashsalt": "jevgym",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def read_arm_colors(path: Path = COLORS_FILE) -> dict[str, str]:
    """Arm -> hex from COLORS.md rows like "| B3 Minimal | blue | `#2a78d6` |" (arm IDs like B3 only)."""
    colors = {}
    if not path.exists():  # without a colors table every arm gets the fallback color
        return colors
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*([A-Z]+\d+)\b[^|]*\|[^|]*\|\s*`(#[0-9a-fA-F]{6})`\s*\|", line)
        if m:
            colors[m.group(1)] = m.group(2).lower()
    if not colors:
        raise ValueError(f"no arm colors found in {path}")
    return colors


ARM_COLORS = read_arm_colors()


def arm_color(arm: str) -> str:
    return ARM_COLORS.get(arm, FALLBACK)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(WORKSPACE))
    except ValueError:
        return str(p)


def header(fig: plt.Figure, title: str, subtitle: str | Sequence[str], top_in: float = 0.29) -> None:
    """Bold title, then one or more muted subtitle lines, left-aligned; spacing in inches from the top."""
    h = fig.get_figheight()
    fig.text(0.03, 1 - top_in / h, title, fontsize=13, fontweight="bold", ha="left")
    lines = [subtitle] if isinstance(subtitle, str) else list(subtitle)
    for i, line in enumerate(lines):
        fig.text(0.03, 1 - (top_in + 0.27 + 0.205 * i) / h, line, fontsize=8.5, color=MUTED, ha="left")


def noise_band(ax: plt.Axes, lo: float, hi: float, text: str, horizontal: bool = True) -> None:
    """Shaded band for the noise floor; accuracies inside it are not distinguishable from rerun noise."""
    if horizontal:
        ax.axhspan(lo, hi, color=FLOOR_BAND, alpha=0.7, lw=0, zorder=0)
        ax.annotate(text, xy=(1.0, (lo + hi) / 2), xycoords=("axes fraction", "data"), xytext=(-4, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8, color=MUTED)
    else:
        ax.axvspan(lo, hi, color=FLOOR_BAND, alpha=0.7, lw=0, zorder=0)
        ax.annotate(text, xy=(hi, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -4),
                    textcoords="offset points", ha="left", va="top", fontsize=8, color=MUTED)


def wilson_point(ax: plt.Axes, x: float, acc: float, wilson: Sequence[float], color: str, **kw) -> None:
    """Accuracy as a dot with its 95% Wilson interval as a whisker."""
    ax.errorbar([x], [acc], yerr=[[acc - wilson[0]], [wilson[1] - acc]], fmt="o", color=color,
                ecolor=color, elinewidth=1.6, capsize=4, markersize=7, zorder=3, **kw)


def best_of_n(ax: plt.Axes, x: float, y: float, n: int) -> None:
    """Mark a point picked from N runs or seeds; selection makes it optimistic."""
    ax.scatter([x], [y], s=180, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
    ax.annotate(f"best of {n}", xy=(x, y), xytext=(8, 8), textcoords="offset points", fontsize=8, color=INK)


def external_footer(fig: plt.Figure, sources: list[str]) -> None:
    """Visible footer, only for citing external sources (papers, datasets). Run dirs go in metadata."""
    if sources:
        fig.text(0.03, 0.025, "Source: " + "; ".join(sources), fontsize=7.5, color=MUTED, ha="left")


def save(fig: plt.Figure, out_stem: Path, run_dirs: Sequence[Path], command: str) -> list[Path]:
    """Save PNG and SVG; the run directories and the command are recorded in file metadata."""
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = out_stem.with_suffix(".png"), out_stem.with_suffix(".svg")
    source = " ".join(_rel(Path(d)) for d in run_dirs) + f" · {command}"
    fig.savefig(png, dpi=200, facecolor=SURFACE, metadata={"Software": None, "Source": source})
    fig.savefig(svg, facecolor=SURFACE, metadata={"Date": None, "Creator": None, "Source": source})
    plt.close(fig)
    return [png, svg]
