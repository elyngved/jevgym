"""The CLI loads secrets from the workspace .env; real environment variables win."""

import jevgym.cli as cli


def _run_cli(monkeypatch, tmp_path, env_text):
    (tmp_path / ".env").write_text(env_text)
    monkeypatch.setattr(cli, "WORKSPACE", tmp_path)
    # Empty workspace (no experiments/): main returns 2 right after loading .env and parsing args,
    # without any Jev call.
    return cli.main(["eval", "--arm", "no-such-arm", "--split", "dev"])


def test_env_file_is_loaded(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "placeholder")  # so teardown restores the original state
    monkeypatch.delenv("TYPESAFE_API_KEY")
    assert _run_cli(monkeypatch, tmp_path, "TYPESAFE_API_KEY=from-dotenv\n") == 2
    import os

    assert os.environ["TYPESAFE_API_KEY"] == "from-dotenv"


def test_real_env_wins_over_env_file(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-shell")
    assert _run_cli(monkeypatch, tmp_path, "TYPESAFE_API_KEY=from-dotenv\n") == 2
    import os

    assert os.environ["TYPESAFE_API_KEY"] == "from-shell"
