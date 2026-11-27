"""The model2vec text embedder plugin.

The model is a small one saved to a temporary directory, so the real model2vec load path runs with no network.
"""
import tempfile
import unittest

import numpy as np
from model2vec import StaticModel
from ovos_plugin_manager.embeddings import find_text_embeddings_plugins
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from ovos_m2v_pipeline import clear_shared_models, load_shared_model, shared_model_stats
from ovos_m2v_pipeline.embeddings import Model2VecEmbeddings

WORDS = ["[UNK]", "[PAD]", "turn", "on", "the", "lights", "what", "time"]
TEXTS = ["turn on the lights", "what time", "the time", "lights"]


class TestModel2VecEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_dir = cls._tmp.name
        tokenizer = Tokenizer(WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        vectors = np.random.default_rng(0).standard_normal((len(WORDS), 8)).astype(np.float32)
        StaticModel(vectors, tokenizer).save_pretrained(cls.model_dir)
        # the expected vectors come from model2vec itself, not from the plugin
        cls.raw = StaticModel.from_pretrained(cls.model_dir).encode(TEXTS)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        clear_shared_models()

    def test_the_batch_is_one_normalised_row_per_text_in_order(self):
        rows = np.asarray(Model2VecEmbeddings({"model": self.model_dir}).get_embeddings_batch(TEXTS))
        expected = self.raw / np.linalg.norm(self.raw, axis=1, keepdims=True)
        np.testing.assert_allclose(rows, expected, rtol=1e-5, atol=1e-6)

    def test_one_text_gives_the_same_row_as_the_batch(self):
        embedder = Model2VecEmbeddings({"model": self.model_dir})
        batch = embedder.get_embeddings_batch(TEXTS)
        for text, row in zip(TEXTS, batch):
            np.testing.assert_allclose(embedder.get_embeddings(text), row, rtol=1e-5, atol=1e-6)

    def test_normalize_false_returns_the_model_output(self):
        rows = Model2VecEmbeddings({"model": self.model_dir, "normalize": False}).get_embeddings_batch(TEXTS)
        np.testing.assert_allclose(np.asarray(rows), self.raw, rtol=1e-5, atol=1e-6)

    def test_an_empty_batch_loads_nothing(self):
        self.assertEqual(Model2VecEmbeddings({"model": self.model_dir}).get_embeddings_batch([]), [])
        self.assertEqual(shared_model_stats()["loads"], 0)

    def test_the_embedder_uses_the_model_the_intent_pipeline_already_loaded(self):
        load_shared_model(self.model_dir, "prototype")
        Model2VecEmbeddings({"model": self.model_dir}).get_embeddings("lights")
        self.assertEqual(shared_model_stats(), {"models": 1, "loads": 1})

    def test_the_plugin_is_registered_as_a_text_embedder(self):
        self.assertIs(find_text_embeddings_plugins().get("ovos-m2v-embeddings-plugin"), Model2VecEmbeddings)


if __name__ == "__main__":
    unittest.main()
