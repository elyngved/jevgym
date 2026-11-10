import math

import numpy as np
import pytest

from jevgym import nb
from jevgym.report import build_summary
from jevgym.runlog import write_json, write_jsonl


def test_features_unigrams_and_bigrams():
    assert nb.features("Who is it ?") == [
        "who", "is", "it", "?", "<s> who", "who is", "is it", "it ?", "? </s>"]


def test_naive_bayes_hand_computed():
    # Two classes, one training text each. Vocab (unigrams + bigrams):
    # A "x": x, <s> x, x </s>      B "y": y, <s> y, y </s>   -> |V| = 6, 3 features per class.
    m = nb.NaiveBayes(("A", "B"), alpha=1.0).fit(["x", "y"], ["A", "B"])
    assert np.allclose(np.exp(m.log_prior), [0.5, 0.5])
    # P(x | A) = (1 + 1) / (3 + 6) = 2/9 ; P(x | B) = (0 + 1) / 9 = 1/9
    assert math.exp(m.log_lik[0, m.vocab["x"]]) == pytest.approx(2 / 9)
    assert math.exp(m.log_lik[1, m.vocab["x"]]) == pytest.approx(1 / 9)
    # "x" has 3 features, each twice as likely under A: odds 2^3 = 8 -> P(A) = 8/9
    assert m.predict_proba(["x"])[0] == pytest.approx([8 / 9, 1 / 9])
    # a text whose features were all unseen in Train falls back to the prior
    assert m.predict_proba(["zzz"])[0] == pytest.approx([0.5, 0.5])


def test_stratified_folds_cover_rows_once():
    labels = ["a"] * 7 + ["b"] * 3
    folds = nb.stratified_folds(labels, 3, seed=1)
    assert sorted(i for f in folds for i in f) == list(range(10))
    assert all(sum(labels[i] == "b" for i in f) == 1 for f in folds)


def test_pick_alpha_ties_go_to_lower_log_loss():
    cv = [{"alpha": 0.1, "cv_accuracy": 0.8, "cv_log_loss": 0.7},
          {"alpha": 1.0, "cv_accuracy": 0.8, "cv_log_loss": 0.6},
          {"alpha": 3.0, "cv_accuracy": 0.7, "cv_log_loss": 0.1}]
    assert nb.pick_alpha(cv) == 1.0


def test_local_run_summary_has_no_calls_and_zero_cost(tmp_path):
    cfg = {"arm": {"name": "B2", "choice": {"name": "t", "option_ids": ["A", "B"], "state_field": "q"}},
           "backend": "local", "split": "dev", "n_rows": 2, "model_pin": None, "client": None,
           "cost_estimate": {"usd": 0.0}}
    write_json(tmp_path / "config.json", cfg)
    (tmp_path / "calls.jsonl").touch()
    write_jsonl(tmp_path / "evaluations.jsonl", [
        {"row_id": "r1", "label": "A", "prediction": "A", "correct": True, "probabilities": {"A": 0.9, "B": 0.1}},
        {"row_id": "r2", "label": "B", "prediction": "A", "correct": False, "probabilities": {"A": 0.6, "B": 0.4}},
    ])
    s = build_summary(tmp_path)
    assert s["accuracy"] == 0.5 and s["total_cost_usd"] == 0 and s["n_calls"] == 0
    assert s["all_calls_pinned"] is None and s["returned_models"] == {}
