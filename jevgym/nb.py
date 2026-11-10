"""Multinomial naive Bayes on word and bigram counts: a non-LLM baseline.

Written out by hand (numpy only) so every step is inspectable:

  P(class | text)  ∝  P(class) * Π over features f in text of P(f | class) ^ count(f)
  P(f | class)     =  (count of f in class + alpha) / (total feature count in class + alpha * |vocab|)

alpha is additive (Laplace/Lidstone) smoothing: without it, one feature never seen in a class
would force that class's probability to 0. alpha is chosen by k-fold cross-validation on Train only.
Features never seen in Train are ignored at prediction time.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

BOS, EOS = "<s>", "</s>"


def features(text: str) -> list[str]:
    """Lowercased whitespace tokens plus adjacent-pair bigrams (with start/end markers).

    Expects pre-tokenized text (spaces around punctuation), so splitting on whitespace is enough.
    """
    toks = text.lower().split()
    padded = [BOS, *toks, EOS]
    return toks + [f"{a} {b}" for a, b in zip(padded, padded[1:])]


@dataclass
class NaiveBayes:
    classes: tuple[str, ...]
    alpha: float = 1.0
    vocab: dict[str, int] = field(default_factory=dict)
    log_prior: np.ndarray | None = None
    log_lik: np.ndarray | None = None  # [class, feature]

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "NaiveBayes":
        feats = [features(t) for t in texts]
        vocab: dict[str, int] = {}
        for fs in feats:
            for f in fs:
                vocab.setdefault(f, len(vocab))
        counts = np.zeros((len(self.classes), len(vocab)))
        cidx = {c: i for i, c in enumerate(self.classes)}
        n_per_class = np.zeros(len(self.classes))
        for fs, y in zip(feats, labels):
            k = cidx[y]
            n_per_class[k] += 1
            for f, n in Counter(fs).items():
                counts[k, vocab[f]] += n
        self.vocab = vocab
        self.log_prior = np.log(n_per_class / n_per_class.sum())
        smoothed = counts + self.alpha
        self.log_lik = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.classes)))
        for i, t in enumerate(texts):
            score = self.log_prior.copy()
            for f, n in Counter(features(t)).items():
                j = self.vocab.get(f)
                if j is not None:
                    score += n * self.log_lik[:, j]
            score -= score.max()  # stable softmax
            p = np.exp(score)
            out[i] = p / p.sum()
        return out

    def top_features(self, k: int = 8) -> dict[str, list[str]]:
        """Per class, the features whose likelihood is highest relative to the other classes."""
        inv = {j: f for f, j in self.vocab.items()}
        out = {}
        for c, row in enumerate(self.log_lik):
            others = np.delete(self.log_lik, c, axis=0).max(axis=0)
            out[self.classes[c]] = [inv[j] for j in np.argsort(-(row - others))[:k]]
        return out


def stratified_folds(labels: Sequence[str], k: int, seed: int) -> list[list[int]]:
    """Row indices split into k folds, each label spread evenly across folds."""
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    by_label: dict[str, list[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(y, []).append(i)
    offset = 0
    for y in sorted(by_label):
        idx = by_label[y][:]
        rng.shuffle(idx)
        for n, i in enumerate(idx):
            folds[(n + offset) % k].append(i)
        offset += len(idx)
    return [sorted(f) for f in folds]


def cross_validate(texts: Sequence[str], labels: Sequence[str], classes: Sequence[str],
                   alphas: Sequence[float], k: int = 5, seed: int = 0) -> list[dict[str, Any]]:
    """Mean accuracy and log loss (clip 0.005) per alpha over k stratified folds of the given rows."""
    folds = stratified_folds(labels, k, seed)
    cidx = {c: i for i, c in enumerate(classes)}
    results = []
    for a in alphas:
        accs, lls = [], []
        for f in folds:
            held = set(f)
            tr = [i for i in range(len(texts)) if i not in held]
            m = NaiveBayes(tuple(classes), a).fit([texts[i] for i in tr], [labels[i] for i in tr])
            P = m.predict_proba([texts[i] for i in f])
            y = [cidx[labels[i]] for i in f]
            accs.append(float(np.mean(P.argmax(axis=1) == np.array(y))))
            lls.append(float(np.mean([-math.log(max(P[r, yy], 0.005)) for r, yy in enumerate(y)])))
        results.append({"alpha": a, "cv_accuracy": float(np.mean(accs)), "cv_accuracy_folds": accs,
                        "cv_log_loss": float(np.mean(lls))})
    return results


def pick_alpha(cv: list[dict[str, Any]]) -> float:
    """Highest mean CV accuracy; ties go to the lower CV log loss."""
    return min(cv, key=lambda r: (-round(r["cv_accuracy"], 12), r["cv_log_loss"]))["alpha"]
