import asyncio

import pytest

from conftest import fake_transport
from jevgym import compare, noise, style
from jevgym.data import TestGuard
from jevgym.evaluate import build_config, estimate_cost, run_eval
from jevgym.jev import JevClient


def make_run(arm, rows, tmp_path, name, pick, group="g1"):
    run_dir = tmp_path / name
    run_dir.mkdir()

    async def go():
        async with JevClient(api_key="test", transport=fake_transport(pick=pick)) as c:
            cfg = build_config(arm, "dev", rows, {"dir": "x", "split_file_sha256": "abc"}, c.settings(),
                               estimate_cost(arm, rows), None, 1, ["t"])
            cfg["noise_group"] = group
            await run_eval(arm, "dev", rows, c, run_dir, cfg, guard=TestGuard(tmp_path), progress=False)

    asyncio.run(go())
    return run_dir


ALL_RIGHT = {"How many feet in a mile ?": "NUM", "What does NASA stand for ?": "ABBR",
             "Why is the sky blue ?": "DESC", "How far is the moon ?": "NUM"}


def test_noise_floor_spread_sd_and_flips(arm, rows, tmp_path):
    r1 = make_run(arm, rows, tmp_path, "r1", ALL_RIGHT)                                   # 4/4
    r2 = make_run(arm, rows, tmp_path, "r2", {**ALL_RIGHT, "Why is the sky blue ?": "NUM"})  # 3/4
    r3 = make_run(arm, rows, tmp_path, "r3", ALL_RIGHT)                                   # 4/4
    nf = noise.compute([r1, r2, r3])
    assert nf["accuracies"] == [1.0, 0.75, 1.0]
    assert nf["noise_floor_points"] == pytest.approx(25.0)
    # sample sd of (1, .75, 1): mean 11/12, squared deviations sum 1/24, /2 -> sqrt(1/48)
    assert nf["sd_points"] == pytest.approx(100 * (1 / 48) ** 0.5)
    assert nf["n_rows_flipped"] == 1 and nf["flip_rate"] == 0.25
    assert nf["flips_by_class"]["DESC"] == {"n": 1, "flipped": 1}
    assert nf["all_floor_zero_note"] is None


def test_zero_floor_raises_open_question(arm, rows, tmp_path):
    runs = [make_run(arm, rows, tmp_path, f"r{i}", ALL_RIGHT) for i in range(2)]
    nf = noise.compute(runs)
    assert nf["noise_floor_points"] == 0 and nf["all_floor_zero_note"]


def test_noise_rejects_non_repeats(arm, rows, tmp_path):
    a = make_run(arm, rows, tmp_path, "a", ALL_RIGHT, group="g1")
    b = make_run(arm, rows, tmp_path, "b", ALL_RIGHT, group="g2")
    with pytest.raises(noise.NoiseError):
        noise.compute([a, b])


def test_compare_paired_cells_and_floor_label(arm, rows, tmp_path):
    a = make_run(arm, rows, tmp_path, "a", ALL_RIGHT)
    b = make_run(arm, rows, tmp_path, "b", {**ALL_RIGHT, "Why is the sky blue ?": "NUM"})
    floor = {"noise_floor_points": 30.0, "split": "dev", "n_rows": 4, "source_runs": []}
    r = compare.compare(a, b, floor=floor)
    assert r["paired_cells"] == {"both": 3, "only_a": 1, "only_b": 0, "neither": 0}
    assert r["mcnemar_p"] == 1.0
    assert r["gap_points"] == pytest.approx(-25.0)
    assert r["label"] == compare.WITHIN_FLOOR
    r = compare.compare(a, b, floor={**floor, "noise_floor_points": 0.5})
    assert r["label"] == compare.NOT_RESOLVED


def test_label_rules():
    # gap at the floor counts as within the floor, however small p is
    assert compare.label(0.5, 0.5, 0.001)[0] == compare.WITHIN_FLOOR
    assert compare.label(-0.5, 0.5, 0.001)[0] == compare.WITHIN_FLOOR
    # above the floor, McNemar decides; the S2 example: 20 fixed vs 5 broken = +1.5 pts, p = 0.004
    from jevgym.metrics import mcnemar_exact
    assert compare.label(1.5, 0.5, mcnemar_exact(5, 20))[0] == compare.RESOLVED
    assert compare.label(1.5, 0.5, 0.05)[0] == compare.NOT_RESOLVED
    assert compare.label(8.0, 0.5, 0.2)[0] == compare.NOT_RESOLVED


def test_arm_colors_read_from_colors_md(tmp_path):
    f = tmp_path / "COLORS.md"
    f.write_text("| Arm | Color | Hex |\n| --- | --- | --- |\n| B1 Hand-written | green | `#1BAF7A` |\n| B12 Extra | red | `#d0453a` |\n")
    assert style.read_arm_colors(f) == {"B1": "#1baf7a", "B12": "#d0453a"}
    assert style.read_arm_colors(tmp_path / "missing.md") == {}
