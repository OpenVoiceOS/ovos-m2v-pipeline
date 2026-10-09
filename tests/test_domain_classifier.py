"""Unit tests for :class:`DomainIntentClassifier`."""
from __future__ import annotations

import os
import tempfile

import numpy as np


from ovos_m2v_pipeline.domain_classifier import DomainIntentClassifier


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
    clf = DomainIntentClassifier()
    assert clf.domains == []
    assert len(clf) == 0


def test_train_creates_per_domain_classifiers_only():
    X, y, _ = _toy_corpus()
    clf = DomainIntentClassifier.train(X, y)
    assert set(clf.domains) == {"media", "home"}
    assert set(clf.intent_classifiers) == {"media", "home"}


def test_predict_picks_global_argmax_across_domains():
    X, y, proto = _toy_corpus()
    clf = DomainIntentClassifier.train(X, y)
    for label, vec in proto.items():
        pred, conf = clf.predict(vec)
        assert pred == label
        assert 0.0 < conf <= 1.0


def test_predict_proba_aggregates_across_every_domain():
    X, y, proto = _toy_corpus()
    clf = DomainIntentClassifier.train(X, y)
    out = clf.predict_proba(proto["media:play"])
    # Every intent across both domains should appear.
    assert set(out) == {"media:play", "media:pause", "home:lights", "home:thermo"}


def test_single_domain_label_treated_as_self_domain():
    X = np.eye(8, dtype=np.float32)[:4]
    y = ["a", "a", "b", "b"]
    clf = DomainIntentClassifier.train(X, y)
    assert set(clf.domains) == {"a", "b"}


def test_save_and_load_round_trip():
    X, y, proto = _toy_corpus()
    clf = DomainIntentClassifier.train(X, y)
    with tempfile.TemporaryDirectory() as d:
        bundle = os.path.join(d, "bundle")
        clf.save(bundle)
        assert os.path.isfile(os.path.join(bundle, "manifest.json"))
        # No domain/ subdir in this layout.
        assert not os.path.isdir(os.path.join(bundle, "domain"))
        loaded = DomainIntentClassifier.load(bundle)
    assert set(loaded.domains) == set(clf.domains)
    for label, vec in proto.items():
        assert loaded.predict(vec)[0] == label


def test_domain_is_the_full_skill_id_of_a_dotted_label():
    X = np.eye(8, dtype=np.float32)[:4]
    y = ["ovos-skill-news.openvoiceos:news", "ovos-skill-news.openvoiceos:global",
         "ovos-skill-weather.openvoiceos:now", "ovos-skill-weather.openvoiceos:now"]
    clf = DomainIntentClassifier.train(X, y)
    assert set(clf.domains) == {"ovos-skill-news.openvoiceos",
                                "ovos-skill-weather.openvoiceos"}
