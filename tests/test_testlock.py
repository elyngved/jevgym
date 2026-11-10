import pytest

from jevgym.data import (
    Row,
    TestAlreadyRunError,
    TestGuard,
    TestLockedError,
    load_jsonl,
    load_split,
    write_jsonl,
)


@pytest.fixture
def repo(tmp_path):
    ds = tmp_path / "data"
    for split in ("train", "dev", "test"):
        write_jsonl(ds / f"{split}.jsonl", [Row(f"{split}-1", f"{split} question ?", "NUM")])
    return tmp_path, ds


def test_refuses_test_while_lock_exists(repo):
    root, ds = repo
    (root / "test.lock").write_text("locked")
    guard = TestGuard(root)
    with pytest.raises(TestLockedError):
        guard.load_test(ds)
    with pytest.raises(TestLockedError):
        load_split(ds, "test", guard=guard)


def test_generic_loader_refuses_test_file(repo):
    _, ds = repo
    with pytest.raises(TestLockedError):
        load_jsonl(ds / "test.jsonl")


def test_train_and_dev_load_while_locked(repo):
    root, ds = repo
    (root / "test.lock").write_text("locked")
    guard = TestGuard(root)
    assert load_split(ds, "dev", guard=guard)[0].id == "dev-1"
    assert load_split(ds, "train", guard=guard)[0].id == "train-1"


def test_refuses_second_test_run(repo):
    root, ds = repo
    guard = TestGuard(root)  # no lock: Test has been unlocked
    assert guard.load_test(ds)[0].id == "test-1"
    guard.record_test_run(root / "runs" / "first")
    with pytest.raises(TestAlreadyRunError):
        guard.load_test(ds)
    with pytest.raises(TestAlreadyRunError):
        guard.record_test_run(root / "runs" / "second")


def test_record_refused_while_locked(repo):
    root, _ = repo
    (root / "test.lock").write_text("locked")
    with pytest.raises(TestLockedError):
        TestGuard(root).record_test_run(root / "runs" / "x")
    assert not (root / "test-run.json").exists()
