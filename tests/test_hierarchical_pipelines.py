"""The hierarchical and trained-bundle variants on the shared model.

Every variant takes its embedding from ``load_shared_model``: booted next
to the flat prototype plugin on the same model, the process holds one
``StaticModel`` and loads it once. The hierarchical prototype plugin
drives a ``HierarchicalPrototypeIntentStore`` through the ordinary
registration topics, deferred-load buffer included.
"""
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_m2v_pipeline import (HierarchicalIntentClassifier,
                               HierarchicalPrototypeIntentStore,
                               Model2VecDomainIntentPipeline,
                               Model2VecHierarchicalIntentPipeline,
                               Model2VecHierarchicalPrototypePipeline,
                               Model2VecPrototypePipeline,
                               DomainIntentClassifier, PrototypeIntentStore,
                               clear_shared_models, shared_model_stats)

DIM = 8
WORDS = {"play": 0, "pause": 1, "lights": 2, "heat": 3}


class _Encoder:
    """One axis per keyword; enough to separate the toy intents."""

    dim = DIM

    def encode(self, sentences, **kwargs):
        out = np.zeros((len(sentences), DIM), dtype=np.float32)
        for i, s in enumerate(sentences):
            for word, axis in WORDS.items():
                if word in s:
                    out[i, axis] = 1.0
            if not out[i].any():
                out[i, DIM - 1] = 1.0
        return out


def _register(pipe, skill_id, name, samples):
    pipe._handle_register_padatious(Message(
        "padatious:register_intent",
        {"name": f"{skill_id}:{name}.intent", "samples": samples, "lang": "en-US"},
        context={"skill_id": skill_id}))


class _SharedModelCase(unittest.TestCase):

    def setUp(self):
        clear_shared_models()
        self._tmp = TemporaryDirectory()
        self.model_dir = self._tmp.name
        self.encoder = _Encoder()
        self.m2v = MagicMock(name="model2vec")
        self.m2v.StaticModel.from_pretrained.return_value = self.encoder
        patches = [
            patch.dict(sys.modules, {"model2vec": self.m2v}),
            patch("ovos_m2v_pipeline.Configuration", return_value={}),
            patch("ovos_m2v_pipeline.StaticModelPipeline"),
        ]
        mocks = [p.start() for p in patches]
        self.static_model_pipeline = mocks[2]
        self.static_model_pipeline.from_pretrained.side_effect = AssertionError(
            "a variant loaded a classifier head of its own")
        for p in patches:
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(clear_shared_models)


class TestHierarchicalPrototypePipeline(_SharedModelCase):

    def _pipe(self, **config):
        return Model2VecHierarchicalPrototypePipeline(
            FakeBus(), {"model": self.model_dir, "prototype_cache": False, **config})

    def test_shares_the_one_model_with_the_flat_plugin(self):
        flat = Model2VecPrototypePipeline(
            FakeBus(), {"model": self.model_dir, "prototype_cache": False})
        hier = self._pipe()
        self.assertTrue(flat._ensure_model(background_ok=False))
        self.assertTrue(hier._ensure_model(background_ok=False))
        self.assertIs(hier.model, flat.model)
        self.assertIs(hier.model, self.encoder)
        self.assertEqual(shared_model_stats(), {"models": 1, "loads": 1})
        self.m2v.StaticModel.from_pretrained.assert_called_once_with(self.model_dir)

    def test_the_store_is_hierarchical_and_reads_its_config(self):
        hier = self._pipe(domain_threshold=0.4, intent_strategy="mean_centroid")
        self.assertIsInstance(hier.prototype_store, HierarchicalPrototypeIntentStore)
        self.assertEqual(hier.prototype_store.domain_threshold, 0.4)
        self.assertEqual(hier.prototype_store.intent_strategy.value, "mean_centroid")
        flat = Model2VecPrototypePipeline(
            FakeBus(), {"model": self.model_dir, "prototype_cache": False})
        self.assertIs(type(flat.prototype_store), PrototypeIntentStore)

    def test_defaults_route_on_centroids_and_score_on_all_anchors(self):
        hier = self._pipe()
        self.assertEqual(hier._prototype_strategy.value, "mean_centroid")
        self.assertEqual(hier.prototype_store._domain_strategy.value, "mean_centroid")
        self.assertEqual(hier.prototype_store.intent_strategy.value, "max_over_all")

    def test_buffered_registrations_reach_the_domain_stores(self):
        hier = self._pipe()
        _register(hier, "media.skill", "play", ["play a song"])
        _register(hier, "media.skill", "pause", ["pause it"])
        _register(hier, "home.skill", "lights", ["lights on"])
        # nothing encoded before the model exists
        self.assertEqual(len(hier.prototype_store), 0)
        self.assertTrue(hier._ensure_model(background_ok=False))
        self.assertEqual(sorted(hier.prototype_store.domains),
                         ["home.skill", "media.skill"])
        matches = list(hier._match("play something", Message("x"), "en-US"))
        self.assertEqual([(s, lbl) for s, lbl, _, _ in matches],
                         [("media.skill", "media.skill:play"),
                          ("media.skill", "media.skill:pause")])
        self.assertAlmostEqual(matches[0][2], 1.0, places=5)

    def test_detach_skill_removes_its_domain(self):
        hier = self._pipe()
        self.assertTrue(hier._ensure_model(background_ok=False))
        _register(hier, "media.skill", "play", ["play a song"])
        _register(hier, "home.skill", "lights", ["lights on"])
        hier._handle_detach_skill(Message("detach_skill", {"skill_id": "media.skill"}))
        self.assertEqual(list(hier.prototype_store.domains), ["home.skill"])
        labels = [lbl for _, lbl, _, _ in hier._match("play a song", Message("x"), "en-US")]
        self.assertEqual(labels, ["home.skill:lights"])

    def test_domain_threshold_rejects_an_off_topic_query(self):
        hier = self._pipe(domain_threshold=0.5)
        self.assertTrue(hier._ensure_model(background_ok=False))
        _register(hier, "media.skill", "play", ["play a song"])
        self.assertEqual(list(hier._match("what time is it", Message("x"), "en-US")), [])


class TestBundlePipelines(_SharedModelCase):

    def _bundle(self, cls):
        labels = ["media.skill:play", "media.skill:pause",
                  "home.skill:lights", "home.skill:heat"]
        words = ["play", "pause", "lights", "heat"]
        X = np.concatenate([self.encoder.encode([w] * 6) for w in words])
        X = X + 0.01 * np.random.default_rng(0).standard_normal(X.shape).astype(np.float32)
        y = [lbl for lbl in labels for _ in range(6)]
        path = f"{self.model_dir}/bundle-{cls.__name__}"
        cls.train(X, y).save(path)
        return path

    def _boot(self, pipeline_cls, bundle_cls):
        bundle = self._bundle(bundle_cls)
        pipe = pipeline_cls(FakeBus(), {"model": self.model_dir, "model_path": bundle})
        self.assertTrue(pipe._ensure_model(background_ok=False))
        return pipe

    def test_hierarchical_bundle_uses_the_shared_embedding(self):
        pipe = self._boot(Model2VecHierarchicalIntentPipeline, HierarchicalIntentClassifier)
        self.assertIs(pipe.model, self.encoder)
        # the -low prototype stage runs on the same object
        self.assertTrue(pipe._low_prototype._ensure_model(background_ok=False))
        self.assertIs(pipe._low_prototype.model, pipe.model)
        self.assertEqual(shared_model_stats(), {"models": 1, "loads": 1})
        self.static_model_pipeline.from_pretrained.assert_not_called()

    def test_domain_bundle_uses_the_shared_embedding(self):
        pipe = self._boot(Model2VecDomainIntentPipeline, DomainIntentClassifier)
        self.assertIs(pipe.model, self.encoder)
        self.assertEqual(shared_model_stats(), {"models": 1, "loads": 1})

    def test_only_registered_labels_match(self):
        pipe = self._boot(Model2VecHierarchicalIntentPipeline, HierarchicalIntentClassifier)
        pipe.intents = {"media.skill:pause"}
        labels = [lbl for _, lbl, _, _ in pipe._match("play it", Message("x"))]
        self.assertEqual(labels, ["media.skill:pause"])
        pipe.intents = {"media.skill:play", "media.skill:pause"}
        top = next(iter(pipe._match("play it", Message("x"))))
        self.assertEqual(top[:2], ("media.skill", "media.skill:play"))

    def test_domain_threshold_config_overrides_the_bundle(self):
        bundle = self._bundle(HierarchicalIntentClassifier)
        pipe = Model2VecHierarchicalIntentPipeline(
            FakeBus(), {"model": self.model_dir, "model_path": bundle,
                        "domain_threshold": 0.9})
        self.assertEqual(pipe.classifier.domain_threshold, 0.9)

    def test_missing_bundle_path_is_refused(self):
        with self.assertRaises(FileNotFoundError):
            Model2VecHierarchicalIntentPipeline(FakeBus(), {"model": self.model_dir})


if __name__ == "__main__":
    unittest.main()
