"""Unit tests for :class:`HierarchicalIntentClassifier`."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


from ovos_m2v_pipeline.hierarchical_classifier import HierarchicalIntentClassifier


def _toy_corpus(dim: int = 16):
    """Two domains (`media`, `home`), two intents each, separable embeddings."""
    rng = np.random.default_rng(0)
    proto = {
        "media:play":   np.eye(dim, dtype=np.float32)[0],
        "media:pause":  np.eye(dim, dtype=np.float32)[1],
        "home:lights":  np.eye(dim, dtype=np.float32)[2],
        "home:thermo":  np.eye(dim, dtype=np.float32)[3],
    }
    X, y = [], []
    for label, vec in proto.items():
        for _ in range(8):
            X.append(vec + 0.01 * rng.standard_normal(dim).astype(np.float32))
            y.append(label)
    return np.asarray(X, dtype=np.float32), y, proto


def test_construction_defaults():
    clf = HierarchicalIntentClassifier()
    assert clf.domain_threshold == 0.0
    assert clf.domains == []
    assert len(clf) == 0


def test_train_creates_domain_and_intent_classifiers():
    X, y, _ = _toy_corpus()
    clf = HierarchicalIntentClassifier.train(X, y)
    assert set(clf.domains) == {"media", "home"}
    assert set(clf.intent_classifiers) == {"media", "home"}


def test_predict_routes_to_correct_intent():
    X, y, proto = _toy_corpus()
    clf = HierarchicalIntentClassifier.train(X, y)
    for label, vec in proto.items():
        pred, conf = clf.predict(vec)
        assert pred == label
        assert 0.0 < conf <= 1.0


def test_predict_proba_returns_only_routed_domain():
    X, y, proto = _toy_corpus()
    clf = HierarchicalIntentClassifier.train(X, y)
    out = clf.predict_proba(proto["media:play"])
    assert set(out) == {"media:play", "media:pause"}


def test_domain_threshold_rejects_low_confidence():
    X, y, proto = _toy_corpus()
    clf = HierarchicalIntentClassifier.train(X, y, domain_threshold=2.0)
    # Threshold above any softmax score forces a rejection.
    pred, conf = clf.predict(proto["media:play"])
    assert pred is None
    assert conf == 0.0
    assert clf.predict_proba(proto["media:play"]) == {}


def test_single_domain_label_treated_as_self_domain():
    # No '.' or ':' — domain == label.
    X = np.eye(8, dtype=np.float32)[:4]
    y = ["a", "a", "b", "b"]
    clf = HierarchicalIntentClassifier.train(X, y)
    assert set(clf.domains) == {"a", "b"}


def test_save_and_load_round_trip():
    X, y, proto = _toy_corpus()
    clf = HierarchicalIntentClassifier.train(X, y, domain_threshold=0.1)
    with tempfile.TemporaryDirectory() as d:
        bundle = os.path.join(d, "bundle")
        clf.save(bundle)
        assert os.path.isfile(os.path.join(bundle, "manifest.json"))
        loaded = HierarchicalIntentClassifier.load(bundle)
    assert loaded.domain_threshold == pytest.approx(0.1)
    assert set(loaded.domains) == set(clf.domains)
    for label, vec in proto.items():
        assert loaded.predict(vec)[0] == label


def test_domain_is_the_full_skill_id_of_a_dotted_label():
    # a skill id carries dots; the domain is everything before the ':'
    X = np.eye(8, dtype=np.float32)[:4]
    y = ["ovos-skill-news.openvoiceos:news", "ovos-skill-news.openvoiceos:news",
         "ovos-skill-news.openvoiceos:global", "ovos-skill-weather.openvoiceos:now"]
    clf = HierarchicalIntentClassifier.train(X, y)
    assert set(clf.domains) == {"ovos-skill-news.openvoiceos",
                                "ovos-skill-weather.openvoiceos"}
