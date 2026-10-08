"""Both plugins loaded through ``OVOSPipelineFactory`` share one model.

ovos-core loads every installed pipeline plugin with
``OVOSPipelineFactory.load_plugin(pipe_id, bus=...)``, which passes
``Configuration()["intents"][pipe_id]`` or an empty dict. The model here is
a small classifier saved to a temporary directory, so the real model2vec
load path runs with no network.
"""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from model2vec import StaticModel
from model2vec.inference import StaticModelPipeline
from model2vec.inference.mlp import Activation, Layer, MLPHead
from ovos_plugin_manager.pipeline import OVOSPipelineFactory
from ovos_utils.fakebus import FakeBus
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from ovos_m2v_pipeline import clear_shared_models, shared_model_stats

CLASSES = ["skill.a:one.intent", "skill.b:two.intent"]
WORDS = ["[UNK]", "[PAD]", "turn", "on", "the", "lights", "what", "time"]


def _save_tiny_classifier(path: str) -> None:
    vocab = {w: i for i, w in enumerate(WORDS)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((len(WORDS), 8)).astype(np.float32)
    embedding = StaticModel(vectors, tokenizer)
    layer = Layer(weight=rng.standard_normal((len(CLASSES), 8)).astype(np.float32),
                  bias=np.zeros(len(CLASSES), dtype=np.float32))
    head = MLPHead(layers=[layer], activation=Activation.SOFTMAX,
                   classes=np.asarray(CLASSES))
    StaticModelPipeline(embedding, head).save_pretrained(path)


class TestFactorySharedModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_dir = cls._tmp.name
        _save_tiny_classifier(cls.model_dir)
        cls.reference = StaticModelPipeline.from_pretrained(cls.model_dir)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        clear_shared_models()

    def _load_unloaded(self, intents: dict, *pipe_ids: str):
        config = {"lang": "en-US", "intents": intents}
        bus = FakeBus()
        with patch("ovos_plugin_manager.pipeline.Configuration", return_value=config), \
             patch("ovos_m2v_pipeline.Configuration", return_value=config):
            return [OVOSPipelineFactory.load_plugin(p, bus=bus) for p in pipe_ids]

    def _load(self, intents: dict, *pipe_ids: str):
        plugins = self._load_unloaded(intents, *pipe_ids)
        for plugin in plugins:
            self.assertTrue(plugin._ensure_model(background_ok=False))
            if getattr(plugin, "_low_prototype", None) is not None:
                # the classifier's own ``-low`` tier is a prototype stage
                self.assertTrue(plugin._low_prototype._ensure_model(background_ok=False))
        return plugins

    def _assert_one_model(self, clf, proto):
        self.assertEqual(shared_model_stats(), {"models": 1, "loads": 1})
        self.assertIsInstance(clf.model, StaticModelPipeline)
        self.assertIs(clf.model.model, proto.model)
        self.assertIs(clf._low_prototype.model, proto.model)
        # the classifier on the shared embedding scores exactly as a
        # pipeline model2vec loads on its own
        utterances = ["turn on the lights", "what time"]
        np.testing.assert_allclose(clf.model.predict_proba(utterances),
                                   self.reference.predict_proba(utterances))
        # and the classifier predicts with the prototype's embedding
        np.testing.assert_allclose(clf.model.predict_proba(utterances),
                                   clf.model.head.predict_proba(proto.model.encode(utterances)))
        self.assertEqual(list(clf.model.classes_), CLASSES)

    def _sections(self):
        return {"ovos-m2v-pipeline": {"model": self.model_dir},
                "ovos-m2v-prototype-pipeline": {"model": self.model_dir}}

    def test_classifier_first(self):
        with patch("model2vec.StaticModel.from_pretrained",
                   wraps=StaticModel.from_pretrained) as disk:
            clf, proto = self._load(self._sections(), "ovos-m2v-pipeline",
                                    "ovos-m2v-prototype-pipeline")
        self.assertEqual(disk.call_count, 1)
        self._assert_one_model(clf, proto)

    def test_prototype_first_reads_the_embedding_once(self):
        with patch("model2vec.StaticModel.from_pretrained",
                   wraps=StaticModel.from_pretrained) as disk:
            proto, clf = self._load(self._sections(), "ovos-m2v-prototype-pipeline",
                                    "ovos-m2v-pipeline")
        self.assertEqual(disk.call_count, 1)
        self._assert_one_model(clf, proto)

    def test_prototype_uses_the_classifier_model_by_default(self):
        """Only the classifier section names a model; one model serves both."""
        intents = {"ovos-m2v-pipeline": {"model": self.model_dir}}
        clf, proto = self._load(intents, "ovos-m2v-pipeline",
                                "ovos-m2v-prototype-pipeline")
        self.assertEqual(proto._model_path, self.model_dir)
        self._assert_one_model(clf, proto)

    def test_prototype_model_key_wins_over_the_classifier(self):
        intents = {"ovos-m2v-pipeline": {"model": self.model_dir},
                   "ovos-m2v-prototype-pipeline": {"model": "other/model"}}
        with patch("ovos_m2v_pipeline._load_labels_manifest", return_value={}):
            proto, = self._load_unloaded(intents, "ovos-m2v-prototype-pipeline")
        self.assertEqual(proto._model_path, "other/model")

    def test_prototype_reads_its_underscore_section(self):
        """Issue 100: with no dash-form section the factory passes ``{}``."""
        intents = {"ovos_m2v_prototype_pipeline": {"model": self.model_dir,
                                                   "prototype_k": 3}}
        proto, = self._load(intents, "ovos-m2v-prototype-pipeline")
        self.assertEqual(proto._model_path, self.model_dir)
        self.assertEqual(proto.config.get("prototype_k"), 3)
        self.assertEqual(proto.config.get("mode"), "prototype")
        self.assertNotIn("mode", intents["ovos_m2v_prototype_pipeline"])

    def test_dash_section_is_not_mutated(self):
        intents = self._sections()
        proto, = self._load(intents, "ovos-m2v-prototype-pipeline")
        self.assertEqual(proto.config.get("mode"), "prototype")
        self.assertEqual(intents["ovos-m2v-prototype-pipeline"], {"model": self.model_dir})


if __name__ == "__main__":
    unittest.main()
