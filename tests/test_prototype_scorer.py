"""Tests for the standalone, bus-free ``PrototypeScorer`` API.

The critical test here is the PARITY test: it registers the same
intents/samples through the real ``padatious:register_intent`` bus-handler
path on a ``Model2VecPrototypePipeline`` instance, and separately through
``PrototypeScorer.add_intent`` (no bus at all), then asserts the two
produce byte-identical ranked scores for a battery of utterances. That is
the whole point of this module: an external evaluator (ovoscope, the
arena) driving ``PrototypeScorer`` must be measuring the exact same
algorithm a live OVOS install dispatches on the bus, not a
reimplementation of it.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

# Import eagerly (real `model2vec`, not yet mocked) so `ovos_m2v_pipeline`'s
# own top-level `from model2vec.inference import StaticModelPipeline` is
# resolved once against the real package. Each test then patches
# `sys.modules["model2vec"]` only for the *lazy* `from model2vec import
# StaticModel` inside the deferred model-load path -- the already-imported
# `ovos_m2v_pipeline` module is unaffected.
import ovos_m2v_pipeline  # noqa: F401,E402

# Deterministic 4-D directions for the mock encoder -- identical to the
# ones tests/test_ovoscope_prototype_e2e.py uses, so cosine similarity
# between any two of these is 0 (orthogonal); identical inputs score 1.0.
_DIRECTIONS = {
    "lights": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "music": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "weather": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "stop": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}
# A fifth direction not tied to a keyword, so entity-filled samples (whose
# value words don't appear in `_DIRECTIONS`) still land somewhere non-zero
# and distinguishable per color.
_COLOR_DIRECTIONS = {
    "red": np.array([0.6, 0.0, 0.0, 0.8], dtype=np.float32),
    "blue": np.array([0.0, 0.6, 0.0, 0.8], dtype=np.float32),
}
_NOISE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _fake_encode(sentences, **kwargs):
    """Deterministic embeddings: first matching keyword direction wins."""
    out = []
    for s in sentences:
        sl = s.lower()
        vec = _NOISE.copy()
        for table in (_DIRECTIONS, _COLOR_DIRECTIONS):
            hit = False
            for key, direction in table.items():
                if key in sl:
                    vec = direction.copy()
                    hit = True
                    break
            if hit:
                break
        out.append(vec)
    return np.stack(out)


def _mock_static_model():
    model = MagicMock()
    model.encode.side_effect = _fake_encode
    model.dim = 4
    return model


def _fake_model2vec_module(mock_model):
    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = mock_model
    return fake_m2v


class TestPrototypeScorerParity(unittest.TestCase):
    """Bus-handler path vs. ``PrototypeScorer.add_intent`` must agree exactly."""

    INTENTS = {
        "skill_a:lights": ["turn on the lights", "switch on the lights", "lights on"],
        "skill_b:music": ["play some music", "start the music"],
        "skill_c:weather": ["what is the weather like", "tell me the weather"],
    }
    UTTERANCES = [
        "lights on now",
        "play some music please",
        "weather today",
        "glorpfest unrelated nonsense",
    ]

    def setUp(self):
        self._mock_model = _mock_static_model()
        self._patch = patch.dict(sys.modules, {"model2vec": _fake_model2vec_module(self._mock_model)})
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _bus_path_scores(self, utterance):
        """Register everything via the real bus handler, then score."""
        from ovos_m2v_pipeline import Model2VecPrototypePipeline
        pipeline = Model2VecPrototypePipeline(
            bus=FakeBus(),
            config={"model": "fake-model", "prototype_cache": False},
        )
        pipeline.model = self._mock_model
        for label, samples in self.INTENTS.items():
            skill_id, intent_name = label.split(":", 1)
            pipeline.bus.emit(Message(
                "padatious:register_intent",
                data={"skill_id": skill_id, "intent_name": intent_name,
                      "name": f"{skill_id}:{intent_name}.intent", "samples": samples},
            ))
        emb = pipeline.model.encode([utterance], use_multiprocessing=False)[0]
        return pipeline.prototype_store.scores(emb)

    def _scorer_path_scores(self, utterance):
        """Register everything via PrototypeScorer.add_intent, then score."""
        from ovos_m2v_pipeline import PrototypeScorer
        scorer = PrototypeScorer(model="fake-model", prototype_cache=False)
        for label, samples in self.INTENTS.items():
            scorer.add_intent(label, samples)
        return dict(scorer.score(utterance))

    def test_rankings_are_identical_across_paths(self):
        for utterance in self.UTTERANCES:
            with self.subTest(utterance=utterance):
                bus_scores = self._bus_path_scores(utterance)
                scorer_scores = self._scorer_path_scores(utterance)
                self.assertEqual(
                    set(bus_scores), set(scorer_scores),
                    f"label sets differ for {utterance!r}",
                )
                for label in bus_scores:
                    self.assertAlmostEqual(
                        bus_scores[label], scorer_scores[label], places=6,
                        msg=f"score for {label!r} on {utterance!r} diverged "
                            f"between the bus path and PrototypeScorer",
                    )


class TestPrototypeScorerTierSemantics(unittest.TestCase):
    """``match()``'s tier thresholds must behave like match_high/medium/low."""

    def setUp(self):
        self._mock_model = _mock_static_model()
        self._patch = patch.dict(sys.modules, {"model2vec": _fake_model2vec_module(self._mock_model)})
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _scorer(self, **overrides):
        from ovos_m2v_pipeline import PrototypeScorer
        cfg = {"model": "fake-model", "prototype_cache": False}
        cfg.update(overrides)
        return PrototypeScorer(**cfg)

    def test_exact_match_clears_every_tier(self):
        scorer = self._scorer()
        scorer.add_intent("skill_a:lights", ["turn on the lights"])
        for tier in ("high", "medium", "low"):
            with self.subTest(tier=tier):
                self.assertEqual(scorer.match("turn on the lights", tier=tier),
                                  "skill_a:lights")

    def test_zero_score_utterance_misses_every_tier(self):
        scorer = self._scorer()
        scorer.add_intent("skill_a:lights", ["turn on the lights"])
        for tier in ("high", "medium", "low"):
            with self.subTest(tier=tier):
                self.assertIsNone(scorer.match("glorpfest", tier=tier))

    def test_match_uses_same_thresholds_as_match_medium(self):
        """A score strictly below conf_medium is a miss on the medium tier,
        mirroring `Model2VecIntentPipeline.match_medium`'s
        `if prob < min_conf: return None`."""
        from ovos_m2v_pipeline import Model2VecPrototypePipeline
        pipeline = Model2VecPrototypePipeline(
            bus=FakeBus(),
            config={"model": "fake-model", "prototype_cache": False,
                    "conf_medium": 0.9},
        )
        pipeline.model = self._mock_model
        pipeline.bus.emit(Message(
            "padatious:register_intent",
            data={"skill_id": "skill_a", "intent_name": "lights",
                  "name": "skill_a:lights.intent", "samples": ["turn on the lights"]},
        ))
        msg = Message("recognizer_loop:utterance",
                       data={"utterances": ["lights on"], "lang": "en-US"})
        # "lights on" contains the "lights" keyword -> cosine 1.0 (exact
        # direction match under the mock encoder) even though conf_medium
        # is raised to 0.9, so match_medium should still find it.
        result = pipeline.match_medium(["lights on"], "en-US", msg)
        self.assertIsNotNone(result)

        scorer = self._scorer(conf_medium=0.9)
        scorer.add_intent("skill_a:lights", ["turn on the lights"])
        self.assertEqual(scorer.match("lights on", tier="medium"), "skill_a:lights")

    def test_unknown_tier_raises(self):
        scorer = self._scorer()
        scorer.add_intent("skill_a:lights", ["turn on the lights"])
        with self.assertRaises(ValueError):
            scorer.match("turn on the lights", tier="ultra")


class TestPrototypeScorerEntityExpansion(unittest.TestCase):
    """``add_intent(..., entities=...)`` must fill `{slot}` placeholders."""

    def setUp(self):
        self._mock_model = _mock_static_model()
        self._patch = patch.dict(sys.modules, {"model2vec": _fake_model2vec_module(self._mock_model)})
        self._patch.start()
        self.addCleanup(self._patch.stop)

    def _scorer(self):
        from ovos_m2v_pipeline import PrototypeScorer
        return PrototypeScorer(model="fake-model", prototype_cache=False)

    def test_entity_placeholder_expands_to_one_prototype_per_value(self):
        scorer = self._scorer()
        n = scorer.add_intent(
            "skill_a:paint",
            ["paint it {color}"],
            entities={"color": ["red", "blue"]},
        )
        self.assertEqual(n, 2)
        labels = list(scorer._pipeline.prototype_store.labels)
        self.assertEqual(labels.count("skill_a:paint"), 2)

    def test_expanded_prototypes_score_their_own_color_direction(self):
        scorer = self._scorer()
        scorer.add_intent(
            "skill_a:paint",
            ["paint it {color}"],
            entities={"color": ["red", "blue"]},
        )
        # "red" is in _COLOR_DIRECTIONS, distinct from "blue" -> exact
        # cosine match for a red utterance despite both being registered
        # under the same label / template.
        ranked = scorer.score("paint it red")
        self.assertEqual(ranked[0][0], "skill_a:paint")
        self.assertAlmostEqual(ranked[0][1], 1.0, places=5)

    def test_unregistered_entity_leaves_placeholder_literal(self):
        scorer = self._scorer()
        n = scorer.add_intent("skill_a:paint", ["paint it {unregistered}"])
        self.assertEqual(n, 1)
        labels = list(scorer._pipeline.prototype_store.labels)
        self.assertEqual(labels.count("skill_a:paint"), 1)


if __name__ == "__main__":
    unittest.main()
