import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline(config=None, intents=None, renormalize=True):
    """Helper: create a classifier-mode pipeline with a mocked model and FakeBus."""
    config = config or {}
    config.setdefault("model", "fake-model")
    config["renormalize"] = renormalize

    mock_model = MagicMock()
    mock_model.classes_ = np.array([])
    mock_model.predict_proba.return_value = np.array([[]])

    with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
         patch("ovos_m2v_pipeline.Configuration", return_value={}):
        MockSMP.from_pretrained.return_value = mock_model
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)

    pipeline.model = mock_model
    if intents is not None:
        pipeline.intents = set(intents)
    return pipeline


def _make_prototype_pipeline(config=None, intents=None, proto_store=None):
    """Create a pipeline in prototype mode with mocked StaticModel and FakeBus."""
    import sys
    config = config or {}
    config.setdefault("model", "fake-embed-model")
    config["mode"] = "prototype"

    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value = np.zeros((1, 4), dtype=np.float32)

    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)

    pipeline.model = mock_embed_model
    if proto_store is not None:
        pipeline.prototype_store = proto_store
    if intents is not None:
        pipeline.intents = set(intents)
    return pipeline


def _setup_model(pipeline, labels, probs):
    """Set numpy-typed classes and a single-row probability matrix."""
    pipeline.model.classes_ = np.array(labels)
    pipeline.model.predict_proba.return_value = np.array([probs])


# Alias used in some feature-branch tests
_setup_classifier = _setup_model


# ---------------------------------------------------------------------------
# PrototypeIntentStore unit tests
# ---------------------------------------------------------------------------

class TestPrototypeIntentStore(unittest.TestCase):
    def _store(self, embeddings, labels):
        from ovos_m2v_pipeline import PrototypeIntentStore
        return PrototypeIntentStore(np.array(embeddings, dtype=np.float32),
                                    np.array(labels))

    def test_embeddings_are_l2_normalised(self):
        store = self._store([[3.0, 4.0]], ["a"])
        norms = np.linalg.norm(store.embeddings, axis=1)
        np.testing.assert_allclose(norms, [1.0], atol=1e-6)

    def test_scores_returns_max_per_label(self):
        # Two prototypes for label "a", one for "b"
        store = self._store(
            [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]],
            ["a", "a", "b"],
        )
        query = np.array([1.0, 0.0], dtype=np.float32)
        scores = store.scores(query)
        # label "a": max(cos([1,0],[1,0])=1.0, cos([1,0],[0,1])=0.0) = 1.0
        self.assertAlmostEqual(scores["a"], 1.0, places=5)
        # label "b": cos([1,0],[0.707,0.707]) ~= 0.707
        self.assertAlmostEqual(scores["b"], 0.707, places=3)

    def test_scores_normalises_query(self):
        store = self._store([[1.0, 0.0]], ["a"])
        # Pass an unnormalized query; should still give cosine = 1.0
        scores = store.scores(np.array([5.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(scores["a"], 1.0, places=5)

    def test_save_load_roundtrip(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = self._store([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "proto.npz")
            store.save(path)
            loaded = PrototypeIntentStore.load(path)
        np.testing.assert_allclose(loaded.embeddings, store.embeddings, atol=1e-6)
        np.testing.assert_array_equal(loaded.labels, store.labels)

    def test_build_samples_k_per_label(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)

        sentences = [f"sent_{i}" for i in range(20)]
        labels = ["a"] * 10 + ["b"] * 10
        store = PrototypeIntentStore.build(mock_model, sentences, labels, k=3)
        counts = {lbl: (store.labels == lbl).sum() for lbl in ["a", "b"]}
        self.assertEqual(counts["a"], 3)
        self.assertEqual(counts["b"], 3)

    def test_build_keeps_all_when_fewer_than_k(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)

        sentences = ["only one"]
        labels = ["a"]
        store = PrototypeIntentStore.build(mock_model, sentences, labels, k=5)
        self.assertEqual((store.labels == "a").sum(), 1)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------

class TestInit(unittest.TestCase):
    def test_default_conf_thresholds(self):
        p = _make_pipeline()
        self.assertEqual(p.config.get("conf_high", 0.7), 0.7)
        self.assertEqual(p.config.get("conf_medium", 0.5), 0.5)
        self.assertEqual(p.config.get("conf_low", 0.15), 0.15)

    def test_ignore_labels_default_empty(self):
        p = _make_pipeline()
        self.assertEqual(p.ignore_labels, [])

    def test_ignore_labels_from_config(self):
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill:bad.intent"]})
        self.assertIn("skill:bad.intent", p.ignore_labels)

    def test_missing_model_raises(self):
        with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
             patch("ovos_m2v_pipeline.Configuration", return_value={}):
            from ovos_m2v_pipeline import Model2VecIntentPipeline
            from ovos_utils.fakebus import FakeBus
            with self.assertRaises(FileNotFoundError):
                Model2VecIntentPipeline(bus=FakeBus(), config={"model": ""})

    def test_prototype_mode_starts_with_empty_store(self):
        p = _make_prototype_pipeline()
        self.assertIsNotNone(p.prototype_store)
        self.assertEqual(len(p.prototype_store), 0)

    def test_classifier_mode_has_no_prototype_store(self):
        p = _make_pipeline()
        self.assertIsNone(p.prototype_store)


# ---------------------------------------------------------------------------
# Intent sync tests
# ---------------------------------------------------------------------------

class TestGetAdaptIntents(unittest.TestCase):
    def test_returns_intent_names(self):
        p = _make_pipeline()
        fake_intents = [{"name": "skill_a:intent_one"}, {"name": "skill_b:intent_two"}]
        mock_response = Message("intent.service.adapt.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_adapt_intents()
        self.assertEqual(result, ["skill_a:intent_one", "skill_b:intent_two"])

    def test_filters_ignore_labels(self):
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill_a:intent_one"]})
        fake_intents = [{"name": "skill_a:intent_one"}, {"name": "skill_b:intent_two"}]
        mock_response = Message("intent.service.adapt.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_adapt_intents()
        self.assertNotIn("skill_a:intent_one", result)
        self.assertIn("skill_b:intent_two", result)

    def test_raises_on_no_response(self):
        p = _make_pipeline()
        p.bus.wait_for_response = MagicMock(return_value=None)
        with self.assertRaises(RuntimeError):
            p._get_adapt_intents()


class TestGetPadatiousIntents(unittest.TestCase):
    def test_returns_intent_names(self):
        p = _make_pipeline()
        fake_intents = ["skill_a:one.intent", "skill_b:two.intent"]
        mock_response = Message("intent.service.padatious.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_padatious_intents()
        self.assertEqual(result, fake_intents)

    def test_filters_ignore_labels(self):
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill_a:one.intent"]})
        fake_intents = ["skill_a:one.intent", "skill_b:two.intent"]
        mock_response = Message("intent.service.padatious.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_padatious_intents()
        self.assertNotIn("skill_a:one.intent", result)
        self.assertIn("skill_b:two.intent", result)

    def test_raises_on_no_response(self):
        p = _make_pipeline()
        p.bus.wait_for_response = MagicMock(return_value=None)
        with self.assertRaises(RuntimeError):
            p._get_padatious_intents()


class TestHandleSyncIntents(unittest.TestCase):
    def test_debounce_while_syncing(self):
        p = _make_pipeline()
        p._syncing = True
        p._get_adapt_intents = MagicMock()
        p.handle_sync_intents(Message("test"))
        p._get_adapt_intents.assert_not_called()

    def test_syncs_intents(self):
        p = _make_pipeline()
        p._get_adapt_intents = MagicMock(return_value=["skill:adapt_intent"])
        p._get_padatious_intents = MagicMock(return_value=["skill:pad_intent"])
        with patch("ovos_m2v_pipeline.time") as mock_time:
            mock_time.sleep = MagicMock()
            p.handle_sync_intents(Message("test"))
        self.assertIn("skill:adapt_intent", p.intents)
        self.assertIn("skill:pad_intent", p.intents)
        self.assertFalse(p._syncing)

    def test_handles_runtime_error_gracefully(self):
        p = _make_pipeline()
        p._get_adapt_intents = MagicMock(side_effect=RuntimeError("bus timeout"))
        with patch("ovos_m2v_pipeline.time") as mock_time:
            mock_time.sleep = MagicMock()
            p.handle_sync_intents(Message("test"))  # must not raise
        self.assertFalse(p._syncing)


# ---------------------------------------------------------------------------
# Classifier _match tests
# ---------------------------------------------------------------------------

class TestMatchClassifier(unittest.TestCase):
    def test_registered_intent_yielded(self):
        p = _make_pipeline(intents=["skill_a:my.intent"], renormalize=False)
        _setup_classifier(p, ["skill_a:my.intent"], [0.9])
        results = list(p._match("turn on the lights"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob, slots = results[0]
        self.assertEqual(label, "skill_a:my.intent")
        self.assertEqual(skill_id, "skill_a")
        self.assertAlmostEqual(prob, 0.9)

    def test_unregistered_intent_discarded(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_classifier(p, ["skill_a:my.intent"], [0.9])
        self.assertEqual(list(p._match("something")), [])

    def test_ocp_special_case(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_classifier(p, ["ocp:play"], [0.95])
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play.play_search")

    def test_common_query_special_case(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_classifier(p, ["common_query:common_query"], [0.8])
        results = list(p._match("what is the capital of France"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "common_query.openvoiceos")
        self.assertEqual(label, "common_query.question")

    def test_stop_special_case(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_classifier(p, ["stop:stop"], [0.85])
        results = list(p._match("stop"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "stop.openvoiceos")
        self.assertEqual(label, "mycroft.stop")

    def test_multiple_candidates_sorted_by_prob(self):
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=False)
        _setup_classifier(p, ["skill_a:a.intent", "skill_b:b.intent"], [0.3, 0.6])
        results = list(p._match("test"))
        self.assertEqual(results[0][2], 0.6)
        self.assertEqual(results[1][2], 0.3)


# ---------------------------------------------------------------------------
# Classifier renormalization tests (dev additions)
# ---------------------------------------------------------------------------

class TestMatch(unittest.TestCase):
    def test_ignore_labels_not_yielded(self):
        # ignore_labels are excluded at sync time; they won't be in self.intents
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill_a:my.intent"]},
                           renormalize=False)
        p.intents = set()
        _setup_model(p, ["skill_a:my.intent"], [0.9])
        results = list(p._match("something"))
        self.assertEqual(results, [])

    def test_renormalize_sums_to_one(self):
        # Two registered intents with raw probs that don't sum to 1 after masking
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=True)
        # Model has 3 classes; only 2 are registered. Raw probs: [0.6, 0.2, 0.2]
        _setup_model(p, ["skill_a:a.intent", "skill_b:b.intent", "skill_c:c.intent"],
                     [0.6, 0.2, 0.2])
        results = list(p._match("test"))
        total = sum(prob for _, _, prob, _ in results)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_renormalize_false_preserves_raw_probs(self):
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=False)
        _setup_model(p, ["skill_a:a.intent", "skill_b:b.intent", "skill_c:c.intent"],
                     [0.6, 0.2, 0.2])
        results = list(p._match("test"))
        probs = [prob for _, _, prob, _ in results]
        # Only the 2 registered intents survive; probs are raw (0.6 and 0.2)
        self.assertAlmostEqual(sorted(probs, reverse=True)[0], 0.6, places=6)
        self.assertAlmostEqual(sorted(probs, reverse=True)[1], 0.2, places=6)

    def test_renormalize_redistributes_masked_probability(self):
        # One registered intent gets 0.3 raw; 0.7 is in unregistered classes.
        # After renormalization it should become 1.0.
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, ["skill_a:a.intent", "skill_b:b.intent"], [0.3, 0.7])
        results = list(p._match("test"))
        _, _, prob, _ = results[0]
        self.assertAlmostEqual(prob, 1.0, places=6)

    def test_renormalize_no_division_by_zero(self):
        # All masked probs are 0 -> should not raise, probs stay 0
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, ["skill_a:a.intent"], [0.0])
        results = list(p._match("test"))
        _, _, prob, _ = results[0]
        self.assertFalse(np.isnan(prob))
        self.assertAlmostEqual(prob, 0.0)

    def test_special_labels_included_in_normalization(self):
        # ocp:play + one registered intent: both survive masking
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, ["skill_a:a.intent", "ocp:play", "skill_b:b.intent"],
                     [0.3, 0.4, 0.3])
        results = list(p._match("test"))
        total = sum(prob for _, _, prob, _ in results)
        self.assertAlmostEqual(total, 1.0, places=6)
        # ocp:play (0.4) + skill_a (0.3) -> renorm sum = 0.7
        labels = {label for _, label, _, _ in results}
        self.assertIn("ovos.common_play.play_search", labels)
        self.assertIn("skill_a:a.intent", labels)


# ---------------------------------------------------------------------------
# Prototype _match tests
# ---------------------------------------------------------------------------

class TestMatchPrototype(unittest.TestCase):
    def _make_store(self, label_embeddings: dict):
        """label_embeddings: {label: embedding_vector}"""
        from ovos_m2v_pipeline import PrototypeIntentStore
        labels = list(label_embeddings.keys())
        embs = np.array([label_embeddings[l] for l in labels], dtype=np.float32)
        return PrototypeIntentStore(embs, np.array(labels))

    def _make_proto_pipeline(self, store, intents=None):
        p = _make_prototype_pipeline(intents=intents, proto_store=store)
        return p

    def test_registered_intent_yielded(self):
        store = self._make_store({"skill_a:my.intent": [1.0, 0.0, 0.0, 0.0]})
        p = self._make_proto_pipeline(store, intents=["skill_a:my.intent"])
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("turn on lights"))
        self.assertEqual(len(results), 1)
        skill_id, label, score, slots = results[0]
        self.assertEqual(label, "skill_a:my.intent")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_all_store_labels_yielded(self):
        # Prototype mode yields every label in the store (store = registered intents)
        store = self._make_store({"skill_a:my.intent": [1.0, 0.0, 0.0, 0.0]})
        p = self._make_proto_pipeline(store)
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("something"))
        self.assertEqual(len(results), 1)

    def test_ignore_labels_not_yielded(self):
        store = self._make_store({"skill_a:bad.intent": [1.0, 0.0, 0.0, 0.0]})
        p = self._make_proto_pipeline(store)
        p.ignore_labels = ["skill_a:bad.intent"]
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        self.assertEqual(list(p._match("something")), [])

    def test_ocp_special_case(self):
        store = self._make_store({"ocp:play": [1.0, 0.0, 0.0, 0.0]})
        p = self._make_proto_pipeline(store, intents=[])
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play.play_search")

    def test_common_query_special_case(self):
        store = self._make_store({"common_query:common_query": [1.0, 0.0, 0.0, 0.0]})
        p = self._make_proto_pipeline(store, intents=[])
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("what is X"))
        self.assertEqual(results[0][1], "common_query.question")

    def test_stop_special_case(self):
        store = self._make_store({"stop:stop": [1.0, 0.0, 0.0, 0.0]})
        p = self._make_proto_pipeline(store, intents=[])
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("stop"))
        self.assertEqual(results[0][1], "mycroft.stop")

    def test_sorted_by_cosine_score(self):
        store = self._make_store({
            "skill_a:a.intent": [1.0, 0.0, 0.0, 0.0],
            "skill_b:b.intent": [0.0, 1.0, 0.0, 0.0],
        })
        p = self._make_proto_pipeline(store, intents=["skill_a:a.intent", "skill_b:b.intent"])
        # Query closer to skill_b
        p.model.encode.return_value = np.array([[0.1, 0.9, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("test"))
        self.assertGreater(results[0][2], results[1][2])
        self.assertIn("skill_b:b.intent", results[0][1])

    def test_valid_labels_not_applied_to_special_prototype_label(self):
        """`valid_labels` is a classifier-only allow-list (drawn from the
        model manifest's frozen training vocabulary). A runtime-registered
        prototype label such as `ocp:play` still matches and is mapped to
        its canonical bus topic regardless of `valid_labels` contents."""
        store = self._make_store({"ocp:play": [1.0, 0.0, 0.0, 0.0]})
        p = self._make_proto_pipeline(store, intents=[])
        p.valid_labels = ["ocp:play"]
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play.play_search")

    def test_valid_labels_does_not_drop_prototype_label_not_listed(self):
        """A prototype label absent from `valid_labels` still matches: the
        prototype store already IS the allow-list (it only ever holds
        runtime-registered labels), so the manifest allow-list does not
        apply here."""
        store = self._make_store({"ocp:play": [1.0, 0.0, 0.0, 0.0]})
        p = self._make_proto_pipeline(store, intents=[])
        p.valid_labels = ["stop:stop"]
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play.play_search")


# ---------------------------------------------------------------------------
# Confidence threshold tests (classifier mode, no renormalize)
# ---------------------------------------------------------------------------

class TestMatchConfidence(unittest.TestCase):
    """Test threshold logic with renormalize=False to isolate confidence from normalization."""

    def _setup(self, prob, intents=None):
        p = _make_pipeline(intents=["skill_a:my.intent"] if intents is None else intents,
                           renormalize=False)
        _setup_model(p, ["skill_a:my.intent"], [prob])
        return p, Message("recognizer_loop:utterance")

    def test_match_high_above_threshold(self):
        p, msg = self._setup(0.8)
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)
        self.assertAlmostEqual(result.match_data["confidence"], 0.8)

    def test_match_high_below_threshold_returns_none(self):
        p, msg = self._setup(0.6)
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_medium_above_threshold(self):
        p, msg = self._setup(0.55)
        result = p.match_medium(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)

    def test_match_medium_below_threshold_returns_none(self):
        p, msg = self._setup(0.4)
        result = p.match_medium(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_low_above_threshold(self):
        p, msg = self._setup(0.2)
        result = p.match_low(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)

    def test_match_low_below_threshold_returns_none(self):
        p, msg = self._setup(0.05)
        result = p.match_low(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_returns_none_when_no_intents(self):
        p, msg = self._setup(0.99, intents=[])
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_data_contains_utterance(self):
        p, msg = self._setup(0.9)
        result = p.match_high(["hello world"], "en", msg)
        self.assertEqual(result.match_data["utterance"], "hello world")
        self.assertEqual(result.utterance, "hello world")

    def test_custom_conf_high_from_config(self):
        p = _make_pipeline(config={"model": "fake", "conf_high": 0.95},
                           intents=["skill_a:my.intent"], renormalize=False)
        _setup_model(p, ["skill_a:my.intent"], [0.9])
        msg = Message("recognizer_loop:utterance")
        result = p.match_high(["test"], "en", msg)
        self.assertIsNone(result)  # 0.9 < 0.95


# ---------------------------------------------------------------------------
# PrototypeIntentStore mutation tests
# ---------------------------------------------------------------------------

class TestPrototypeIntentStoreMutable(unittest.TestCase):
    def _mock_model(self, dim=4):
        m = MagicMock()
        m.encode.side_effect = lambda sents, **kw: np.eye(len(sents), dim, dtype=np.float32)
        return m

    def test_add_populates_empty_store(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        n = store.add(self._mock_model(), "skill_a:intent", ["hello", "hi"], k=5)
        self.assertEqual(n, 2)
        self.assertEqual(len(store), 2)
        self.assertIn("skill_a:intent", store.unique_labels)

    def test_add_replaces_existing_label(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        store.add(self._mock_model(), "skill_a:intent", ["old1", "old2"], k=5)
        store.add(self._mock_model(), "skill_a:intent", ["new1"], k=5)
        self.assertEqual((store.labels == "skill_a:intent").sum(), 1)

    def test_add_samples_k_when_more_sentences(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        sentences = [f"sentence {i}" for i in range(20)]
        store.add(self._mock_model(), "skill_a:intent", sentences, k=3)
        self.assertEqual((store.labels == "skill_a:intent").sum(), 3)

    def test_add_empty_sentences_returns_zero(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        n = store.add(self._mock_model(), "skill_a:intent", [])
        self.assertEqual(n, 0)
        self.assertEqual(len(store), 0)

    def test_remove_deletes_label(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        store.add(self._mock_model(), "skill_a:intent", ["hello"], k=5)
        store.add(self._mock_model(), "skill_b:intent", ["bye"], k=5)
        store.remove("skill_a:intent")
        self.assertNotIn("skill_a:intent", store.unique_labels)
        self.assertIn("skill_b:intent", store.unique_labels)

    def test_remove_nonexistent_label_is_noop(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        store.remove("nonexistent:intent")  # must not raise
        self.assertEqual(len(store), 0)

    def test_remove_skill_deletes_matching_prefix(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        store.add(self._mock_model(), "myskill:intent_a", ["a1"], k=5)
        store.add(self._mock_model(), "myskill:intent_b", ["b1"], k=5)
        store.add(self._mock_model(), "otherskill:intent", ["c1"], k=5)
        store.remove_skill("myskill")
        self.assertNotIn("myskill:intent_a", store.unique_labels)
        self.assertNotIn("myskill:intent_b", store.unique_labels)
        self.assertIn("otherskill:intent", store.unique_labels)

    def test_remove_skill_empty_store_is_noop(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        store.remove_skill("myskill")  # must not raise


# ---------------------------------------------------------------------------
# _parse_intent_file tests
# ---------------------------------------------------------------------------

class TestParseIntentFile(unittest.TestCase):
    def _write(self, tmp, content):
        path = os.path.join(tmp, "test.intent")
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_returns_non_blank_lines(self):
        from ovos_m2v_pipeline import _parse_intent_file
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "turn on the lights\nplay music\n")
            result = _parse_intent_file(path)
        self.assertEqual(result, ["turn on the lights", "play music"])

    def test_skips_blank_lines(self):
        from ovos_m2v_pipeline import _parse_intent_file
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "hello\n\n   \nworld\n")
            result = _parse_intent_file(path)
        self.assertEqual(result, ["hello", "world"])

    def test_skips_comment_lines(self):
        from ovos_m2v_pipeline import _parse_intent_file
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "# this is a comment\nhello\n")
            result = _parse_intent_file(path)
        self.assertEqual(result, ["hello"])

    def test_expands_alternatives(self):
        from ovos_m2v_pipeline import _parse_intent_file
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "(turn on|switch on) the lights\n")
            result = _parse_intent_file(path)
        self.assertIn("turn on the lights", result)
        self.assertIn("switch on the lights", result)

    def test_expands_optional_words(self):
        from ovos_m2v_pipeline import _parse_intent_file
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "[please] play music\n")
            result = _parse_intent_file(path)
        self.assertIn("please play music", result)
        self.assertIn("play music", result)

    def test_missing_file_returns_empty(self):
        from ovos_m2v_pipeline import _parse_intent_file
        result = _parse_intent_file("/nonexistent/path.intent")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Prototype-mode bus handler tests
# ---------------------------------------------------------------------------

class TestPrototypeBusHandlers(unittest.TestCase):
    def _pipeline(self):
        return _make_prototype_pipeline()

    def test_handle_register_padatious_adds_prototypes(self):
        p = self._pipeline()
        with patch("ovos_m2v_pipeline._parse_intent_file", return_value=["hello", "hi"]):
            p.model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)
            p._handle_register_padatious(Message("padatious:register_intent", data={
                "name": "skill_a:greet.intent",
                "file_name": "/fake/greet.intent",
            }))
        self.assertIn("skill_a:greet", p.prototype_store.unique_labels)
        self.assertIn("skill_a:greet", p.intents)

    def test_handle_register_padatious_updates_existing_label(self):
        p = self._pipeline()
        with patch("ovos_m2v_pipeline._parse_intent_file", return_value=["v1"]):
            p.model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)
            p._handle_register_padatious(Message("padatious:register_intent", data={
                "name": "skill_a:greet.intent",
                "file_name": "/fake/greet.intent",
            }))
        with patch("ovos_m2v_pipeline._parse_intent_file", return_value=["v2"]):
            p.model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)
            p._handle_register_padatious(Message("padatious:register_intent", data={
                "name": "skill_a:greet.intent",
                "file_name": "/fake/greet.intent",
            }))
        self.assertEqual((p.prototype_store.labels == "skill_a:greet").sum(), 1)

    def test_handle_register_padatious_inline_samples(self):
        p = self._pipeline()
        p.model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)
        p._handle_register_padatious(Message("padatious:register_intent", data={
            "name": "skill_a:greet.intent",
            "samples": ["hello", "hi"],
        }))
        self.assertIn("skill_a:greet", p.prototype_store.unique_labels)
        self.assertIn("skill_a:greet", p.intents)

    def test_handle_register_padatious_inline_samples_expand_template(self):
        """Inline samples with template syntax are expanded."""
        p = self._pipeline()
        p.model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)
        p._handle_register_padatious(Message("padatious:register_intent", data={
            "name": "skill_a:lights.intent",
            "samples": ["(turn on|switch on) the lights"],
        }))
        # Two expanded variants should both be embedded
        n_protos = (p.prototype_store.labels == "skill_a:lights").sum()
        self.assertEqual(n_protos, 2)

    def test_handle_register_padatious_skips_missing_file(self):
        p = self._pipeline()
        with patch("ovos_m2v_pipeline._parse_intent_file", return_value=[]):
            p._handle_register_padatious(Message("padatious:register_intent", data={
                "name": "skill_a:greet.intent",
                "file_name": "",
            }))
        self.assertEqual(len(p.prototype_store), 0)
        self.assertNotIn("skill_a:greet.intent", p.intents)

    def test_handle_register_padatious_skips_ignored_label(self):
        p = self._pipeline()
        p.ignore_labels = ["skill_a:greet.intent"]
        with patch("ovos_m2v_pipeline._parse_intent_file", return_value=["hello"]):
            p._handle_register_padatious(Message("padatious:register_intent", data={
                "name": "skill_a:greet.intent",
                "file_name": "/fake/greet.intent",
            }))
        self.assertEqual(len(p.prototype_store), 0)

    def test_handle_register_adapt_adds_to_intents(self):
        p = self._pipeline()
        p._handle_register_adapt(Message("register_intent", data={"name": "skill_a:adapt.intent"}))
        self.assertIn("skill_a:adapt.intent", p.intents)
        self.assertEqual(len(p.prototype_store), 0)

    def test_handle_register_adapt_skips_ignored_label(self):
        p = self._pipeline()
        p.ignore_labels = ["skill_a:adapt.intent"]
        p._handle_register_adapt(Message("register_intent", data={"name": "skill_a:adapt.intent"}))
        self.assertNotIn("skill_a:adapt.intent", p.intents)

    def test_handle_detach_intent_removes_prototypes_and_intents(self):
        p = self._pipeline()
        with patch("ovos_m2v_pipeline._parse_intent_file", return_value=["hello"]):
            p.model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)
            p._handle_register_padatious(Message("padatious:register_intent", data={
                "name": "skill_a:greet.intent",
                "file_name": "/fake/greet.intent",
            }))
        p._handle_detach_intent(Message("detach_intent", data={"intent_name": "skill_a:greet.intent"}))
        self.assertNotIn("skill_a:greet.intent", p.prototype_store.unique_labels)
        self.assertNotIn("skill_a:greet.intent", p.intents)

    def test_handle_detach_skill_uses_context_skill_id(self):
        p = self._pipeline()
        p.model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)
        with patch("ovos_m2v_pipeline._parse_intent_file", return_value=["hello"]):
            p._handle_register_padatious(Message("padatious:register_intent",
                                                 data={"name": "skill_a:intent", "file_name": "/fake/x.intent"}))
        msg = Message("detach_skill", data={}, context={"skill_id": "skill_a"})
        p._handle_detach_skill(msg)
        self.assertNotIn("skill_a:intent", p.prototype_store.unique_labels)

    def test_handle_detach_skill_removes_all_skill_labels(self):
        p = self._pipeline()
        p.model.encode.side_effect = lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32)
        for name in ("skill_a:intent_x", "skill_a:intent_y"):
            with patch("ovos_m2v_pipeline._parse_intent_file", return_value=["example"]):
                p._handle_register_padatious(Message("padatious:register_intent",
                                                     data={"name": name, "file_name": "/fake/x.intent"}))
        with patch("ovos_m2v_pipeline._parse_intent_file", return_value=["keep"]):
            p._handle_register_padatious(Message("padatious:register_intent",
                                                 data={"name": "skill_b:intent", "file_name": "/fake/b.intent"}))

        p._handle_detach_skill(Message("detach_skill", data={"skill_id": "skill_a"}))
        for label in ("skill_a:intent_x", "skill_a:intent_y"):
            self.assertNotIn(label, p.prototype_store.unique_labels)
            self.assertNotIn(label, p.intents)
        self.assertIn("skill_b:intent", p.prototype_store.unique_labels)


# ---------------------------------------------------------------------------
# Initial intent sync tests (dev additions)
# ---------------------------------------------------------------------------

class TestInitialIntentSync(unittest.TestCase):
    """Skills that loaded BEFORE the pipeline never emit register_intent.
    The plugin must seed its intent list by querying the manifests on
    construction."""

    def test_seeds_intents_from_manifests_on_init(self):
        adapt_resp = Message("intent.service.adapt.manifest",
                             data={"intents": [{"name": "skill_a:foo"}]})
        pad_resp = Message("intent.service.padatious.manifest",
                           data={"intents": ["skill_b:bar"]})

        def fake_wait_for_response(msg, reply_type, timeout=1):
            if reply_type == "intent.service.adapt.manifest":
                return adapt_resp
            if reply_type == "intent.service.padatious.manifest":
                return pad_resp
            return None

        mock_model = MagicMock()
        mock_model.classes_ = np.array([])
        mock_model.predict_proba.return_value = np.array([[]])

        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}):
            MockSMP.from_pretrained.return_value = mock_model
            from ovos_m2v_pipeline import Model2VecIntentPipeline
            from ovos_utils.fakebus import FakeBus
            bus = FakeBus()
            bus.wait_for_response = MagicMock(side_effect=fake_wait_for_response)
            p = Model2VecIntentPipeline(bus=bus, config={"model": "fake"})

        self.assertIn("skill_a:foo", p.intents)
        self.assertIn("skill_b:bar", p.intents)

    def test_init_survives_missing_manifests(self):
        mock_model = MagicMock()
        mock_model.classes_ = np.array([])
        mock_model.predict_proba.return_value = np.array([[]])
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}):
            MockSMP.from_pretrained.return_value = mock_model
            from ovos_m2v_pipeline import Model2VecIntentPipeline
            from ovos_utils.fakebus import FakeBus
            bus = FakeBus()
            bus.wait_for_response = MagicMock(return_value=None)
            p = Model2VecIntentPipeline(bus=bus, config={"model": "fake"})
        self.assertEqual(len(p.intents), 0)


# ---------------------------------------------------------------------------
# Special label session gating tests (dev additions)
# ---------------------------------------------------------------------------

class TestSpecialLabelGating(unittest.TestCase):
    """`_allowed_special_labels` gates ocp/stop/common_query by session."""

    def _msg(self, pipeline):
        from ovos_bus_client.session import Session
        return Message("test", context={
            "session": Session(session_id="s", pipeline=pipeline).serialize(),
        })

    def test_allows_only_pipelines_present_in_session(self):
        p = _make_pipeline()
        msg = self._msg(["ovos-ocp-pipeline-plugin-high"])
        self.assertEqual(p._allowed_special_labels(msg), {"ocp:play"})

    def test_none_message_falls_back_to_all(self):
        p = _make_pipeline()
        self.assertEqual(
            p._allowed_special_labels(None),
            {"ocp:play", "common_query:common_query", "stop:stop"},
        )

    def test_match_filters_special_when_session_excludes_it(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, ["ocp:play"], [0.95])
        msg = self._msg(["ovos-m2v-pipeline"])  # no ocp pipeline
        self.assertEqual(list(p._match("play music", msg)), [])

    def test_match_passes_special_when_session_includes_it(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, ["ocp:play"], [0.95])
        msg = self._msg(["ovos-ocp-pipeline-plugin-high"])
        results = list(p._match("play music", msg))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "ovos.common_play.play_search")


if __name__ == "__main__":
    unittest.main()
