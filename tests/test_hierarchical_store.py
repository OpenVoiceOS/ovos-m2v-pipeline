"""Unit tests for ``HierarchicalPrototypeIntentStore`` (two-stage routing)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ovos_m2v_pipeline import HierarchicalPrototypeIntentStore, PrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy

ALL_STRATEGIES = list(PrototypeStrategy)


class _FakeModel:
    """Deterministic encoder. Sentences that share an obvious *anchor word*
    map to embeddings that cluster together: enough to drive two-stage
    routing across several domains."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self._anchors = {
            "media":  self._direction(0),
            "home":   self._direction(1),
            "play":   self._direction(0),
            "pause":  self._direction(3),
            "lights": self._direction(1),
            "thermostat": self._direction(1),
            "ola": self._direction(2),
        }

    def _direction(self, axis: int) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[axis] = 1.0
        return v

    def encode(self, sentences, **kwargs):
        out = []
        for s in sentences:
            sl = s.lower()
            chosen = None
            for kw, vec in self._anchors.items():
                if kw in sl:
                    chosen = vec
                    break
            if chosen is None:
                seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)
                chosen = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
                chosen = chosen / (np.linalg.norm(chosen) + 1e-12)
            out.append(chosen)
        return np.asarray(out)


def _q(model, text):
    return model.encode([text])[0]


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

def test_default_is_max_over_all():
    s = HierarchicalPrototypeIntentStore()
    assert s.intent_strategy is PrototypeStrategy.MAX_OVER_ALL
    assert s.domain_threshold == 0.0
    assert s.domains == {}
    assert len(s) == 0


def test_default_router_is_mean_centroid():
    s = HierarchicalPrototypeIntentStore()
    assert s._domain_strategy is PrototypeStrategy.MEAN_CENTROID


def test_custom_intent_strategy_and_threshold():
    s = HierarchicalPrototypeIntentStore(
        intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
        intent_tau=0.05,
        intent_top_k=2,
        domain_threshold=0.3,
    )
    assert s.intent_strategy is PrototypeStrategy.SOFTMAX_WEIGHTED
    assert s.domain_threshold == 0.3


# ---------------------------------------------------------------------------
# add() / remove() lifecycle
# ---------------------------------------------------------------------------

def test_domain_is_the_skill_id_of_the_label():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    n = s.add(m, "media.skill:play", ["play song", "put on song"])
    assert n == 2
    assert list(s.domains) == ["media.skill"]
    assert list(s.domains["media.skill"].unique_labels) == ["media.skill:play"]
    assert list(s.unique_labels) == ["media.skill:play"]


def test_explicit_domain_overrides_the_skill_id():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "play", ["play song"], domain="media")
    s.add(m, "pause", ["pause"], domain="media")
    assert list(s.domains) == ["media"]
    assert set(s.domains["media"].unique_labels) == {"play", "pause"}


def test_reregistering_under_another_domain_moves_the_label():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "play", ["play song"], domain="media")
    s.add(m, "play", ["play song"], domain="music")
    assert list(s.domains) == ["music"]
    assert s.calc_domain(_q(m, "play africa")) == "music"


def test_remove_intent_keeps_domain():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    s.add(m, "media:pause", ["pause"])
    s.remove("media:play")
    assert list(s.domains["media"].unique_labels) == ["media:pause"]


def test_removing_the_last_intent_drops_the_domain_and_its_route():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    s.add(m, "home:lights", ["lights on"])
    s.remove("media:play")
    assert list(s.domains) == ["home"]
    # the router no longer sends a media query to the removed domain
    assert s.calc_domain(_q(m, "play africa")) == "home"


def test_remove_skill_drops_only_that_skill():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    s.add(m, "media:pause", ["pause"])
    s.add(m, "home:lights", ["lights on"])
    s.remove_skill("media")
    assert list(s.domains) == ["home"]
    assert list(s.unique_labels) == ["home:lights"]
    assert s.scores(_q(m, "play africa")) == {"home:lights": pytest.approx(0.0, abs=1e-6)}


def test_remove_domain_drops_everything():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    s.add(m, "home:lights", ["lights on"])
    s.remove_domain("media")
    assert list(s.domains) == ["home"]
    assert s.calc_domain(_q(m, "play africa")) == "home"


def test_add_anchors_installs_prebuilt_rows_and_routes_them():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    internal = s._compose("media:play", "en-US")
    rows = m.encode(["play song"]).astype(np.float32)
    assert s._add_anchors(internal, rows) == 1
    assert s.calc_domain(_q(m, "play africa"), lang="en-US") == "media"
    assert s.scores(_q(m, "play africa"), lang="en-US") == {"media:play": pytest.approx(1.0)}


# ---------------------------------------------------------------------------
# Stage 1: domain routing
# ---------------------------------------------------------------------------

def test_calc_domain_picks_single_best_domain():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    s.add(m, "home:lights", ["lights on", "turn on lights"])
    assert s.calc_domain(_q(m, "lights on please")) == "home"
    assert s.calc_domain(_q(m, "play africa")) == "media"


def test_calc_domain_empty_store_returns_none():
    s = HierarchicalPrototypeIntentStore()
    assert s.calc_domain(np.zeros(16, dtype=np.float32)) is None


def test_calc_domain_below_threshold_returns_none():
    m = _FakeModel()
    # A threshold above any achievable cosine rejects everything.
    s = HierarchicalPrototypeIntentStore(domain_threshold=2.0)
    s.add(m, "media:play", ["play song"])
    assert s.calc_domain(_q(m, "play africa")) is None


def test_routing_is_per_language():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"], lang="en-US")
    s.add(m, "home:lights", ["lights on"], lang="en-US")
    # a Portuguese greeting in "home" must not pull English queries there
    s.add(m, "home:greet", ["ola"], lang="pt-PT")
    q = 0.8 * m._direction(2) + 0.6 * m._direction(0)
    assert s.calc_domain(q, lang="pt-PT") == "home"
    assert s.calc_domain(q, lang="en-US") == "media"
    assert s.scores(q, lang="en-US") == {"media:play": pytest.approx(0.6)}


# ---------------------------------------------------------------------------
# Stage 2: two-stage scoring
# ---------------------------------------------------------------------------

def test_scores_route_to_single_domain():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    s.add(m, "home:lights", ["lights on", "turn on lights"])
    out = s.scores(_q(m, "lights on please"))
    # Only the routed domain's intents are scored.
    assert out == {"home:lights": pytest.approx(1.0)}


def test_calc_intent_two_stage_argmax():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    s.add(m, "media:pause", ["pause"])
    s.add(m, "home:lights", ["lights on"])
    assert s.calc_intent(_q(m, "play africa")) == "media:play"


def test_scores_below_threshold_returns_empty():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore(domain_threshold=2.0)
    s.add(m, "media:play", ["play song"])
    assert s.scores(_q(m, "play africa")) == {}
    assert s.calc_intent(_q(m, "play africa")) is None


def test_scores_with_explicit_domain_bypasses_router():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    s.add(m, "home:lights", ["lights on"])
    out = s.scores(_q(m, "lights on please"), domain="media")
    assert set(out) == {"media:play"}


def test_scores_explicit_domain_bypasses_threshold():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore(domain_threshold=2.0)
    s.add(m, "media:play", ["play song"])
    assert set(s.scores(_q(m, "play"), domain="media")) == {"media:play"}


def test_scores_explicit_unknown_domain_returns_empty():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media:play", ["play song"])
    assert s.scores(_q(m, "play"), domain="nope") == {}


def test_scores_empty_store_returns_empty_dict():
    s = HierarchicalPrototypeIntentStore()
    q = np.zeros(16, dtype=np.float32)
    assert s.scores(q) == {}
    assert s.calc_intent(q) is None


def test_routed_scores_equal_the_flat_store_restricted_to_that_domain():
    m = _FakeModel()
    samples = {"media:play": ["play song", "play one"],
               "media:pause": ["pause", "pause it"],
               "home:lights": ["lights on"]}
    hier = HierarchicalPrototypeIntentStore()
    flat = PrototypeIntentStore()
    for label, sents in samples.items():
        hier.add(m, label, sents)
        flat.add(m, label, sents)
    q = _q(m, "play the news")
    expected = {k: v for k, v in flat.scores(q).items() if k.startswith("media:")}
    assert hier.scores(q) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Strategy round-trip: every PrototypeStrategy survives add+scores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strat", ALL_STRATEGIES)
def test_intent_strategy_round_trip(strat):
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore(intent_strategy=strat,
                                         intent_top_k=2, intent_tau=0.1)
    s.add(m, "media:play", ["play one", "play two", "play three"])
    s.add(m, "media:pause", ["pause one", "pause two", "pause three"])
    s.add(m, "home:lights", ["lights on", "turn on lights"])
    out = s.scores(_q(m, "play four"))
    assert out, f"{strat.value} produced no scores"
    assert max(out, key=out.get) == "media:play", f"{strat.value} mispredicted"
