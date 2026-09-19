"""Unit tests for ``ovos_m2v_pipeline.strategies``."""
from __future__ import annotations

import numpy as np
import pytest

from ovos_m2v_pipeline import PrototypeIntentStore
from ovos_m2v_pipeline.strategies import (
    PrototypeStrategy,
    score_labels,
    select_anchors,
)

ALL_STRATEGIES = list(PrototypeStrategy)


class _FakeModel:
    """Deterministic encoder: each sentence → hash-derived embedding."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, sentences, **kwargs):
        import hashlib
        out = []
        for s in sentences:
            seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            out.append(v)
        return np.asarray(out)


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# select_anchors: anchor counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy,expected", [
    (PrototypeStrategy.MEAN_CENTROID, 1),
    (PrototypeStrategy.MEDOID, 1),
    (PrototypeStrategy.MAX_OVER_ALL, 5),       # capped at k
    (PrototypeStrategy.FARTHEST_POINT, 5),
    (PrototypeStrategy.KMEANS_CENTERS, 5),
    (PrototypeStrategy.TOP_K_MEAN, 12),        # keep all samples
    (PrototypeStrategy.SOFTMAX_WEIGHTED, 12),
])
def test_select_anchors_counts(strategy, expected):
    rng = np.random.default_rng(0)
    embs = _l2(rng.standard_normal((12, 16)).astype(np.float32))
    out = select_anchors(embs, strategy, k=5)
    assert len(out) == expected


def test_select_anchors_returns_normalised():
    rng = np.random.default_rng(1)
    embs = _l2(rng.standard_normal((20, 8)).astype(np.float32))
    for s in ALL_STRATEGIES:
        out = select_anchors(embs, s, k=4)
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"{s} produced non-unit anchors"


def test_select_anchors_empty():
    embs = np.empty((0, 8), dtype=np.float32)
    for s in ALL_STRATEGIES:
        assert len(select_anchors(embs, s, k=5)) == 0


# ---------------------------------------------------------------------------
# score_labels: each strategy returns one score per label
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_score_labels_one_per_label(strategy):
    rng = np.random.default_rng(2)
    embs = _l2(rng.standard_normal((30, 8)).astype(np.float32))
    labels = np.array(["a"] * 10 + ["b"] * 10 + ["c"] * 10, dtype=object)
    q = _l2(rng.standard_normal(8).astype(np.float32))
    out = score_labels(q, embs, labels, strategy)
    assert set(out) == {"a", "b", "c"}
    for v in out.values():
        assert -1.0001 <= v <= 1.0001


def test_score_labels_max_over_all_matches_naive():
    rng = np.random.default_rng(3)
    embs = _l2(rng.standard_normal((20, 8)).astype(np.float32))
    labels = np.array(["a"] * 10 + ["b"] * 10, dtype=object)
    q = _l2(rng.standard_normal(8).astype(np.float32))
    out = score_labels(q, embs, labels, PrototypeStrategy.MAX_OVER_ALL)
    sims = embs @ q
    assert out["a"] == pytest.approx(float(sims[:10].max()))
    assert out["b"] == pytest.approx(float(sims[10:].max()))


def test_score_labels_top_k_mean_bounded_by_max():
    rng = np.random.default_rng(4)
    embs = _l2(rng.standard_normal((10, 8)).astype(np.float32))
    labels = np.array(["x"] * 10, dtype=object)
    q = _l2(rng.standard_normal(8).astype(np.float32))
    max_s = score_labels(q, embs, labels, PrototypeStrategy.MAX_OVER_ALL)["x"]
    topk = score_labels(q, embs, labels, PrototypeStrategy.TOP_K_MEAN, top_k=3)["x"]
    assert topk <= max_s + 1e-6


def test_score_labels_softmax_low_tau_approaches_max():
    rng = np.random.default_rng(5)
    embs = _l2(rng.standard_normal((20, 8)).astype(np.float32))
    labels = np.array(["x"] * 20, dtype=object)
    q = _l2(rng.standard_normal(8).astype(np.float32))
    max_s = score_labels(q, embs, labels, PrototypeStrategy.MAX_OVER_ALL)["x"]
    soft = score_labels(q, embs, labels, PrototypeStrategy.SOFTMAX_WEIGHTED, tau=0.01)["x"]
    assert abs(soft - max_s) < 0.05


def test_score_labels_empty():
    embs = np.empty((0, 8), dtype=np.float32)
    labels = np.array([], dtype=object)
    q = np.zeros(8, dtype=np.float32)
    for s in ALL_STRATEGIES:
        assert score_labels(q, embs, labels, s) == {}


# ---------------------------------------------------------------------------
# PrototypeIntentStore end-to-end with each strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_store_add_and_score_each_strategy(strategy):
    model = _FakeModel()
    store = PrototypeIntentStore(strategy=strategy, top_k=2, tau=0.1)
    store.add(model, "weather", [f"weather example {i}" for i in range(10)], k=5)
    store.add(model, "music",   [f"music example {i}"   for i in range(10)], k=5)
    assert set(store.unique_labels) == {"weather", "music"}
    q = model.encode(["weather example 0"])[0]
    out = store.scores(q)
    assert set(out) == {"weather", "music"}
    # The "weather example 0" embedding should rank weather >= music.
    assert out["weather"] >= out["music"]


def test_store_default_strategy_is_max_over_all_backcompat():
    """The store default preserves pre-strategy behaviour."""
    store = PrototypeIntentStore()
    assert store.strategy is PrototypeStrategy.MAX_OVER_ALL


def test_store_centroid_collapses_to_one_anchor():
    model = _FakeModel()
    store = PrototypeIntentStore(strategy=PrototypeStrategy.MEAN_CENTROID)
    store.add(model, "lbl", [f"hello {i}" for i in range(10)], k=5)
    assert len(store) == 1
