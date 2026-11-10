"""Charts generated from run directories only. Every chart goes through jevgym/style.py (house style)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from jevgym import noise
from jevgym.report import build_summary
from jevgym.runlog import CALLS, CONFIG, read_json, read_jsonl
from jevgym.style import FAINT, INK, MUTED, SURFACE, arm_color, header, noise_band, plt, save, wilson_point


def confusion(run_dirs: Sequence[str | Path], out_stem: str | Path) -> list[Path]:
    """Confusion matrix for one run: cells shaded by share of the true class, labelled with counts."""
    if len(run_dirs) != 1:
        raise ValueError("confusion takes exactly one run directory")
    run_dir = Path(run_dirs[0])
    config = read_json(run_dir / CONFIG)
    s = build_summary(run_dir)  # recomputed from run files, not read from summary.json
    opts = s["confusion"]["option_ids"]
    m = s["confusion"]["rows_true_cols_pred"]
    arm = s["arm"]
    cmap = LinearSegmentedColormap.from_list(f"{arm}-seq", [SURFACE, arm_color(arm)])

    shares = [[c / max(1, sum(row)) for c in row] for row in m]
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.85, bottom=0.09)
    ax.imshow(shares, cmap=cmap, vmin=0, vmax=1)
    for i, row in enumerate(m):
        for j, c in enumerate(row):
            dark = shares[i][j] > 0.55
            ax.text(
                j, i, f"{c}\n{shares[i][j]:.0%}" if c else "·",
                ha="center", va="center", fontsize=9,
                color=SURFACE if dark else (INK if c else MUTED),
                fontweight="bold" if i == j else "normal",
            )
    n_true = [sum(row) for row in m]
    ax.set_xticks(range(len(opts)), opts)
    ax.set_yticks(range(len(opts)), [f"{o}  (n={n})" for o, n in zip(opts, n_true)])
    ax.set_xlabel("Predicted (Jev's top option)")
    ax.set_ylabel("True label")
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(opts))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(opts))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)

    lo, hi = s["wilson_95"]
    header(fig, f"{arm} confusion matrix on {s['split'].capitalize()} (n={s['n']})", [
        "Rows are true classes, columns are predictions; each cell shows the count and the share of its row.",
        f"Diagonal = correct. Accuracy {s['accuracy']:.1%} (95% Wilson {lo:.1%}–{hi:.1%}), "
        f"Brier {s['brier']:.3f}, model {config['model_pin']}.",
    ])
    return save(fig, Path(out_stem), [run_dir], "jevgym plot confusion")


def noise_repeats(run_dirs: Sequence[str | Path], out_stem: str | Path) -> list[Path]:
    """Accuracy of each repeat run with its Wilson interval; the noise floor as a band."""
    run_dirs = [Path(d) for d in run_dirs]
    nf = noise.compute(run_dirs)
    summaries = [build_summary(d) for d in run_dirs]
    arm = nf["arm"]
    color = arm_color(arm)
    accs = nf["accuracies"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.72, bottom=0.08)
    noise_band(ax, min(accs), max(accs), f"noise floor\n{nf['noise_floor_points']:.1f} pts\n(max − min)")
    for i, s in enumerate(summaries, start=1):
        wilson_point(ax, i, s["accuracy"], s["wilson_95"], color)
        ax.annotate(f"{s['accuracy']:.1%}", xy=(i, s["accuracy"]), xytext=(9, -3), textcoords="offset points",
                    fontsize=8.5, color=INK)
    ax.set_xticks(range(1, len(summaries) + 1), [f"run {i}" for i in range(1, len(summaries) + 1)])
    ax.set_xlim(0.5, len(summaries) + 1.25)
    lo = min(s["wilson_95"][0] for s in summaries)
    hi = max(s["wilson_95"][1] for s in summaries)
    ax.set_ylim(lo - 0.006, hi + 0.008)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}" if round(v * 100, 6) % 1 == 0 else f"{v:.1%}"))
    ax.set_ylabel(f"{nf['split'].capitalize()} accuracy (n={nf['n_rows']})")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=FAINT, linewidth=0.8)
    ax.set_axisbelow(True)

    header(fig, f"{arm_title(run_dirs[0])} on {nf['split'].capitalize()}: the same Prompt run {nf['n_repeats']} times", [
        "Dots are accuracy, whiskers the 95% Wilson interval. The grey band spans the lowest to highest run:",
        "that is the noise floor. Gaps narrower than the band are rerun noise; wider gaps go to a paired McNemar test.",
        f"Same config, model {read_json(run_dirs[0] / CONFIG)['model_pin']}; {nf['n_rows_flipped']} of "
        f"{nf['n_rows']} rows change prediction in at least one run.",
    ])
    return save(fig, Path(out_stem), run_dirs, "jevgym plot noise-repeats")


def _questions(run_dir: Path) -> dict[str, str]:
    """Question text per row, read from the logged requests."""
    out = {}
    for c in read_jsonl(run_dir / CALLS):
        state = c["request"]["state"]
        out[c["row_id"]] = next(iter(state.values()))
    return out


def row_flips(run_dirs: Sequence[str | Path], out_stem: str | Path) -> list[Path]:
    """Grid of the rows whose prediction changes across repeats: rows grouped by true class, one column per run."""
    run_dirs = [Path(d) for d in run_dirs]
    nf = noise.compute(run_dirs)
    _, evals = noise.load_repeats(run_dirs)
    flips = noise.row_flips(evals)
    questions = _questions(run_dirs[0])
    arm = nf["arm"]
    color = arm_color(arm)
    option_ids = read_json(run_dirs[0] / CONFIG)["arm"]["choice"]["option_ids"]
    flips.sort(key=lambda f: (option_ids.index(f["label"]), -f["n_correct"], f["row_id"]))
    k = len(run_dirs)

    n = len(flips)
    fig_h = 1.3 + 0.27 * (n + len(option_ids) + 1)
    fig = plt.figure(figsize=(9.0, fig_h))
    top = 1 - 0.95 / fig_h
    ax = fig.add_axes([0.03, 0.25 / fig_h, 0.94, top - 0.25 / fig_h])
    text_w = 6.4  # grid units for the question column
    ax.set_xlim(-text_w, k + 1.9)

    y = 0
    for cls in option_ids:
        group = [f for f in flips if f["label"] == cls]
        st = nf["flips_by_class"].get(cls, {"n": 0, "flipped": 0})
        ax.text(-text_w, y + 0.5, f"True {cls}: {st['flipped']} of {st['n']} rows flip",
                fontsize=9, fontweight="bold", va="center", ha="left", color=INK)
        y += 1
        for f in group:
            q = questions.get(f["row_id"], f["row_id"])
            q = q if len(q) <= 62 else q[:60] + "…"
            ax.text(-text_w, y + 0.5, q, fontsize=8, va="center", ha="left", color=INK)
            for j, p in enumerate(f["predictions"]):
                ok = p == f["label"]
                ax.add_patch(Rectangle((j + 0.06, y + 0.08), 0.88, 0.84, facecolor=color if ok else FAINT,
                                       edgecolor="none"))
                ax.text(j + 0.5, y + 0.5, p, fontsize=7.5, ha="center", va="center",
                        color=SURFACE if ok else INK, fontweight="bold" if ok else "normal")
            ax.text(k + 0.2, y + 0.5, f"{f['n_correct']}/{k}", fontsize=8, va="center", ha="left", color=MUTED)
            y += 1
    for j in range(k):
        ax.text(j + 0.5, -0.15, f"run {j + 1}", fontsize=8, ha="center", va="bottom", color=MUTED)
    ax.text(k + 0.2, -0.15, "right", fontsize=8, ha="left", va="bottom", color=MUTED)
    ax.set_ylim(y, -0.9)
    ax.axis("off")

    header(fig, f"Which {nf['split'].capitalize()} rows flip when {arm_title(run_dirs[0])} is rerun", [
        f"Each line is one of the {n} questions (of {nf['n_rows']}) whose top prediction is not the same in all {k} runs, "
        "grouped by true class.",
        f"Cells show the predicted class per run; filled = correct. The other {nf['n_rows'] - n} rows "
        f"get the same answer every time.",
    ])
    return save(fig, Path(out_stem), run_dirs, "jevgym plot row-flips")


def arms(run_dirs: Sequence[str | Path], out_stem: str | Path) -> list[Path]:
    """Accuracy (Wilson whiskers, noise-floor band) and Brier per arm, one run per arm, same split.

    The floor band is centred on the arm the floor was measured on (from runs/noise-floor.json); that
    arm's repeat runs are drawn as small faint dots so its run-to-run spread is visible.
    """
    run_dirs = [Path(d) for d in run_dirs]
    summaries = [build_summary(d) for d in run_dirs]
    nf = noise.load_floor()
    floor = nf["noise_floor_points"] / 100
    ref = next((s for s in summaries if s["arm"] == nf["arm"]), summaries[0])
    repeats = [build_summary(noise.WORKSPACE / d) for d in nf["source_runs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.9), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.72, bottom=0.1, wspace=0.32)
    xs = range(len(summaries))
    noise_band(ax1, ref["accuracy"] - floor, ref["accuracy"] + floor,
               f"{ref['arm']} ± floor\n({nf['noise_floor_points']:.1f} pts)")
    for x, s in zip(xs, summaries):
        c = arm_color(s["arm"])
        wilson_point(ax1, x, s["accuracy"], s["wilson_95"], c)
        ax1.annotate(f"{s['accuracy']:.1%}", xy=(x, s["accuracy"]), xytext=(9, -3), textcoords="offset points",
                     fontsize=8.5, color=INK)
        ax2.scatter([x], [s["brier"]], s=60, color=c, zorder=3)
        ax2.annotate(f"{s['brier']:.3f}", xy=(x, s["brier"]), xytext=(9, -3), textcoords="offset points",
                     fontsize=8.5, color=INK)
        if s["arm"] == nf["arm"]:
            for r in repeats:
                ax1.scatter([x - 0.22], [r["accuracy"]], s=10, color=c, alpha=0.45, zorder=2, lw=0)
                ax2.scatter([x - 0.22], [r["brier"]], s=10, color=c, alpha=0.45, zorder=2, lw=0)
    labels = [s["arm"] for s in summaries]
    for ax in (ax1, ax2):
        ax.set_xticks(list(xs), labels)
        ax.set_xlim(-0.6, len(summaries) - 0.1)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=FAINT, linewidth=0.8)
        ax.set_axisbelow(True)
    ax1.set_xlim(-0.6, len(summaries) + 0.35)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_ylabel(f"{ref['split'].capitalize()} accuracy (n={ref['n']})")
    ax1.set_title("Accuracy (higher is better)", fontsize=9.5, loc="left", color=INK)
    ax2.set_ylabel("Brier score (0 = perfect, 2 = worst)")
    ax2.set_title("Brier (lower is better)", fontsize=9.5, loc="left", color=INK)
    lo = min(s["brier"] for s in summaries + repeats)
    hi = max(s["brier"] for s in summaries + repeats)
    ax2.set_ylim(lo - 0.04, hi + 0.04)

    header(fig, f"Arms on {ref['split'].capitalize()}: accuracy and Brier per arm", [
        "Left: dots are accuracy, whiskers the 95% Wilson interval, grey band = "
        f"{ref['arm']} ± the noise floor. Right: Brier, the squared",
        "error of the probabilities. One run per arm; the small faint dots are "
        f"{ref['arm']}'s {len(repeats)} repeat runs. Paired McNemar tests are not shown.",
    ])
    extra = [noise.WORKSPACE / d for d in nf["source_runs"] if (noise.WORKSPACE / d).resolve() not in
             {d.resolve() for d in run_dirs}]
    return save(fig, Path(out_stem), run_dirs + extra, "jevgym plot arms")


def confusion_change(run_dirs: Sequence[str | Path], out_stem: str | Path, best_of: int = 0) -> list[Path]:
    """Confusion matrix of run B minus run A (same rows): where errors move to or away from."""
    from jevgym.style import BETTER, WORSE

    if len(run_dirs) != 2:
        raise ValueError("confusion-change takes two run directories: A (before) and B (after)")
    a_dir, b_dir = (Path(d) for d in run_dirs)
    sa, sb = build_summary(a_dir), build_summary(b_dir)
    if sa["confusion"]["option_ids"] != sb["confusion"]["option_ids"] or sa["n"] != sb["n"]:
        raise ValueError("runs are not on the same options and rows")
    opts = sa["confusion"]["option_ids"]
    ma, mb = sa["confusion"]["rows_true_cols_pred"], sb["confusion"]["rows_true_cols_pred"]
    d = [[mb[i][j] - ma[i][j] for j in range(len(opts))] for i in range(len(opts))]
    vmax = max(abs(v) for row in d for v in row) or 1

    fig, ax = plt.subplots(figsize=(6.8, 6.6))
    fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.09)
    for i in range(len(opts)):
        for j in range(len(opts)):
            v = d[i][j]
            good = (v > 0) == (i == j)
            color = BETTER if good else WORSE
            alpha = 0.12 + 0.88 * abs(v) / vmax if v else 0
            ax.add_patch(Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, facecolor=color if v else SURFACE,
                                   alpha=alpha, edgecolor=FAINT if i == j else "none", lw=1.2))
            dark = v and abs(v) / vmax > 0.5
            ax.text(j, i, f"{v:+d}" if v else "·", ha="center", va="center", fontsize=9.5,
                    color=SURFACE if dark else (INK if v else MUTED), fontweight="bold" if i == j else "normal")
    n_true = [sum(r) for r in ma]
    ax.set_xlim(-0.5, len(opts) - 0.5)
    ax.set_ylim(len(opts) - 0.5, -0.5)
    ax.set_xticks(range(len(opts)), opts)
    ax.set_yticks(range(len(opts)), [f"{o}  (n={n})" for o, n in zip(opts, n_true)])
    ax.set_xlabel("Predicted (Jev's top option)")
    ax.set_ylabel("True label")
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    la, lb = run_label(a_dir), run_label(b_dir)
    lb_full = f"{lb} (best of {best_of})" if best_of else lb
    header(fig, f"{lb_full} vs. {la}: where errors move ({sa['split'].capitalize()})", [
        f"Same rows. Each cell is {lb}'s count minus {la}'s count for that true/predicted pair.",
        "Purple = better (more on the diagonal, fewer off it); red = worse. Diagonal cells are outlined.",
        f"Accuracy {la} {sa['accuracy']:.1%} → {lb} {sb['accuracy']:.1%}"
        + (f"; {lb} was picked as the best of {best_of} seeds, so its gain is optimistic." if best_of else "."),
    ])
    return save(fig, Path(out_stem), [a_dir, b_dir], "jevgym plot confusion-change")


_SEED_SUFFIX = re.compile(r"-s(\d+)$")


def run_label(run_dir: Path) -> str:
    """Arm name, plus the seed when the arm's slug ends in -s<N> (slug "b4-teacher-s5" -> "B4 s5")."""
    arm = read_json(Path(run_dir) / CONFIG)["arm"]
    m = _SEED_SUFFIX.search(arm["slug"])
    return f"{arm['name']} s{m.group(1)}" if m else arm["name"]


def arm_title(run_dir: Path) -> str:
    """Arm name plus the rest of its slug, for titles (name "B3", slug "b3-minimal" -> "B3 Minimal")."""
    arm = read_json(Path(run_dir) / CONFIG)["arm"]
    rest = _SEED_SUFFIX.sub("", arm["slug"]).removeprefix(arm["name"].lower()).strip("-")
    return f"{arm['name']} {rest.replace('-', ' ').capitalize()}" if rest else arm["name"]


def seeds(run_dirs: Sequence[str | Path], out_stem: str | Path, ref: str | None = None) -> list[Path]:
    """Accuracy per run with Wilson whiskers; runs of one arm over several seeds sit side by side,
    the best of them marked "best of N"; the noise-floor band is centred on the `ref` arm."""
    from jevgym.style import best_of_n

    run_dirs = [Path(d) for d in run_dirs]
    summaries = [build_summary(d) for d in run_dirs]
    labels = [run_label(d) for d in run_dirs]
    nf = noise.load_floor()
    floor = nf["noise_floor_points"] / 100
    ref = ref or nf["arm"]
    ref_s = next(s for s in summaries if s["arm"] == ref)
    by_arm: dict[str, list[int]] = {}
    for i, s in enumerate(summaries):
        by_arm.setdefault(s["arm"], []).append(i)

    xs, x = [], 0.0
    prev = None
    for s in summaries:
        x += 0.55 if s["arm"] == prev else 1.0
        xs.append(x)
        prev = s["arm"]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.73, bottom=0.11)
    noise_band(ax, ref_s["accuracy"] - floor, ref_s["accuracy"] + floor,
               f"{ref} ± floor\n({nf['noise_floor_points']:.1f} pts)")
    for xi, s in zip(xs, summaries):
        wilson_point(ax, xi, s["accuracy"], s["wilson_95"], arm_color(s["arm"]))
        if len(by_arm[s["arm"]]) > 1:  # seeds sit close together: label under the whisker
            ax.annotate(f"{s['accuracy']:.1%}", xy=(xi, s["wilson_95"][0]), xytext=(0, -4), textcoords="offset points",
                        fontsize=8, color=INK, ha="center", va="top")
        else:
            ax.annotate(f"{s['accuracy']:.1%}", xy=(xi, s["accuracy"]), xytext=(8, -3), textcoords="offset points",
                        fontsize=8, color=INK)
    for arm, idx in by_arm.items():
        if len(idx) > 1:
            best = max(idx, key=lambda i: summaries[i]["accuracy"])
            best_of_n(ax, xs[best], summaries[best]["accuracy"], len(idx))
            mean = sum(summaries[i]["accuracy"] for i in idx) / len(idx)
            ax.hlines(mean, xs[idx[0]] - 0.2, xs[idx[-1]] + 0.2, color=arm_color(arm), lw=1, ls="--", zorder=1)
            ax.annotate(f"mean of {len(idx)}: {mean:.1%}", xy=(xs[idx[-1]] + 0.2, mean), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MUTED)
    ax.set_xticks(xs, labels)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 1.3)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel(f"{ref_s['split'].capitalize()} accuracy (n={ref_s['n']})")
    ax.grid(axis="y", color=FAINT, linewidth=0.8)
    ax.set_axisbelow(True)
    multi = [a for a, idx in by_arm.items() if len(idx) > 1]
    header(fig, f"{' and '.join(multi) or 'Arms'} across seeds vs. other arms ({ref_s['split'].capitalize()})", [
        f"Dots are accuracy, whiskers the 95% Wilson interval; grey band = {ref} ± the noise floor. Each seed is a",
        "separate Teacher pass scored once; the dashed line is the seed mean. The ringed point is the best seed,",
        "picked after seeing the scores, so it is optimistic: "
        + ", ".join(f"best of {len(by_arm[a])}" for a in multi) + ".",
    ])
    extra = [noise.WORKSPACE / d for d in nf["source_runs"]]
    return save(fig, Path(out_stem), run_dirs + [d for d in extra if d.resolve() not in {r.resolve() for r in run_dirs}],
                "jevgym plot seeds")
