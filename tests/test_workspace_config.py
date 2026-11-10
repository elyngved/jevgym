"""Workspace settings (jevgym.toml) and arm labels taken from run configs."""

import jevgym
import jevgym.cli as cli
from jevgym import plot
from jevgym.runlog import write_json


def test_eval_needs_an_experiment(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(jevgym, "CONFIG_FILE", tmp_path / "jevgym.toml")
    assert cli.main(["eval", "--arm", "x", "--split", "dev"]) == 2
    assert "give --experiment" in capsys.readouterr().err


def test_experiment_default_comes_from_jevgym_toml(monkeypatch, tmp_path, capsys):
    (tmp_path / "jevgym.toml").write_text('experiment = "nope"\n')
    monkeypatch.setattr(jevgym, "CONFIG_FILE", tmp_path / "jevgym.toml")
    monkeypatch.setattr(cli, "WORKSPACE", tmp_path)
    assert cli.main(["eval", "--arm", "x", "--split", "dev"]) == 2
    assert "no experiment 'nope'" in capsys.readouterr().err


def _run(tmp_path, name, slug):
    d = tmp_path / slug
    d.mkdir()
    write_json(d / "config.json", {"arm": {"name": name, "slug": slug}})
    return d


def test_labels_come_from_the_arm_slug(tmp_path):
    seeded = _run(tmp_path, "B4", "b4-teacher-s5")
    plain = _run(tmp_path, "B3", "b3-minimal")
    bare = _run(tmp_path, "X", "x")
    assert plot.run_label(seeded) == "B4 s5"
    assert plot.run_label(plain) == "B3"
    assert plot.arm_title(seeded) == "B4 Teacher"
    assert plot.arm_title(plain) == "B3 Minimal"
    assert plot.arm_title(bare) == "X"
