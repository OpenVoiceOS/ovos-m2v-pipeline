"""Regression tests: prototype-mode matching must be partitioned by
language.

Before this fix, ``PrototypeIntentStore`` held every registered label's
prototypes in one shared cosine space regardless of which language they were
registered in, and ``match_high``/``match_medium``/``match_low`` accepted a
``lang`` argument but never passed it on to the store. A skill's Portuguese
samples could then out-score an English skill's own samples for an English
utterance, purely because the multilingual embedding model happens to place
that Portuguese phrase closer in vector space -- with no relation to the
utterance's actual language.
"""
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message

from ovos_m2v_pipeline.cache import compute_cache_key


def _make_encoder(vectors, dim=4):
    """Deterministic lookup encoder: each exact sentence string maps to a
    hand-picked vector, so cosine similarity between any two registered/
    queried strings is fully controlled by the test, not by real semantics."""
    def _encode(sentences, **kwargs):
        out = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            out[i] = vectors.get(s, np.zeros(dim, dtype=np.float32))
        return out
    return _encode


def _make_pipeline(cache_dir=None, extra_config=None):
    """Fresh prototype-mode pipeline with a mocked ``StaticModel`` and a
    disabled on-disk cache by default (tests that need the cache pass a
    ``cache_dir`` explicitly, mirroring ``test_prototype_cache.py``)."""
    config = {"model": "fake-embed-model", "mode": "prototype"}
    if cache_dir is not None:
        config["prototype_cache_dir"] = str(cache_dir)
    else:
        config["prototype_cache"] = False
    if extra_config:
        config.update(extra_config)

    mock_embed_model = MagicMock()
    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)
    pipeline.model = mock_embed_model
    return pipeline


def _register(pipeline, name, skill_id, samples, lang):
    pipeline.bus.emit(Message(
        "padatious:register_intent",
        {"name": name, "samples": samples, "lang": lang},
        {"skill_id": skill_id}))


class TestPrototypeLanguagePartition(unittest.TestCase):
    def test_cross_language_pollution_is_prevented(self):
        """An English utterance must never be answered by a skill whose
        prototypes were registered in Portuguese, even when the Portuguese
        sample happens to be the nearest neighbour in embedding space."""
        query = "hello there"
        vectors = {
            query: [1.0, 0.0, 0.0, 0.0],
            # en-US sample: close to the query, but not identical
            "hi there": [0.99, 0.14, 0.0, 0.0],
            # pt-PT sample: deliberately given the query's OWN vector, so it
            # is the single closest match in the shared embedding space --
            # the exact scenario that pollutes an unpartitioned store.
            "ola": [1.0, 0.0, 0.0, 0.0],
        }
        p = _make_pipeline()
        p.model.encode.side_effect = _make_encoder(vectors)

        _register(p, "skill_en:greet", "skill_en", ["hi there"], "en-US")
        _register(p, "skill_pt:cumprimento", "skill_pt", ["ola"], "pt-PT")

        candidates = list(p._match_prototype(query, message=None, lang="en-US"))
        labels = [label for _, label, _ in candidates]

        self.assertIn("skill_en:greet", labels)
        self.assertNotIn("skill_pt:cumprimento", labels,
                          "pt-PT prototype leaked into an en-US match despite "
                          "scoring higher in the shared embedding space")
        # the surviving (en-US) candidate must be the top-ranked one
        self.assertEqual(candidates[0][1], "skill_en:greet")

    def test_cross_language_pollution_fail_before_reproduction(self):
        """Same setup as above, but calling ``scores()`` the pre-fix way
        (no ``lang``) reproduces the pollution -- proving the crafted
        vectors really do make the pt-PT sample win when nothing filters
        by language."""
        query = "hello there"
        vectors = {
            query: [1.0, 0.0, 0.0, 0.0],
            "hi there": [0.99, 0.14, 0.0, 0.0],
            "ola": [1.0, 0.0, 0.0, 0.0],
        }
        p = _make_pipeline()
        p.model.encode.side_effect = _make_encoder(vectors)

        _register(p, "skill_en:greet", "skill_en", ["hi there"], "en-US")
        _register(p, "skill_pt:cumprimento", "skill_pt", ["ola"], "pt-PT")

        emb = p.model.encode([query])[0]
        scores = p.prototype_store.scores(emb)  # no lang -> unfiltered
        best = max(scores, key=scores.get)
        self.assertEqual(best, "skill_pt:cumprimento")

    def test_regional_fallback_matches_registered_dialect(self):
        """An en-GB utterance must still match prototypes registered under
        en-US when no en-GB partition exists (OVOS-INTENT-2 §2.2 dialect
        fallback), the same way locale/voice resolution falls back."""
        vectors = {
            "turn on the lights": [1.0, 0.0, 0.0, 0.0],
            "switch the lights on": [0.98, 0.2, 0.0, 0.0],
        }
        p = _make_pipeline()
        p.model.encode.side_effect = _make_encoder(vectors)

        _register(p, "skill_en:lights_on", "skill_en",
                  ["switch the lights on"], "en-US")

        candidates = list(p._match_prototype(
            "turn on the lights", message=None, lang="en-GB"))
        labels = [label for _, label, _ in candidates]
        self.assertIn("skill_en:lights_on", labels)

    def test_far_language_yields_no_match(self):
        """A pt-PT-only store must not answer an en-US utterance at all."""
        vectors = {
            "turn on the lights": [1.0, 0.0, 0.0, 0.0],
            "ligar as luzes": [0.95, 0.3, 0.0, 0.0],
        }
        p = _make_pipeline()
        p.model.encode.side_effect = _make_encoder(vectors)

        _register(p, "skill_pt:luzes", "skill_pt", ["ligar as luzes"], "pt-PT")

        candidates = list(p._match_prototype(
            "turn on the lights", message=None, lang="en-US"))
        self.assertEqual(candidates, [])

    def test_match_high_end_to_end_respects_language_partition(self):
        """Full ``match_high`` path (as the pipeline API is actually
        exercised in production, threading `lang` through from the caller)
        picks the correct-language skill, not the polluting one."""
        query = "hello there"
        vectors = {
            query: [1.0, 0.0, 0.0, 0.0],
            "hi there": [0.99, 0.14, 0.0, 0.0],
            "ola": [1.0, 0.0, 0.0, 0.0],
        }
        p = _make_pipeline()
        p.model.encode.side_effect = _make_encoder(vectors)

        _register(p, "skill_en:greet", "skill_en", ["hi there"], "en-US")
        _register(p, "skill_pt:cumprimento", "skill_pt", ["ola"], "pt-PT")

        match = p.match_high([query], "en-US", None)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, "skill_en:greet")
        self.assertEqual(match.skill_id, "skill_en")


class TestPrototypeCacheKeyIncludesLang(unittest.TestCase):
    def test_different_lang_yields_different_cache_key(self):
        key_en = compute_cache_key(
            "model-x", "0.9.0", {"k": None}, ["hello"], lang="en-US")
        key_pt = compute_cache_key(
            "model-x", "0.9.0", {"k": None}, ["hello"], lang="pt-PT")
        self.assertNotEqual(key_en, key_pt)

    def test_lang_partitioned_cache_entries_do_not_collide_on_disk(self):
        """Registering the same label under two languages must persist (and
        later hit) two independent cache entries, not overwrite one another."""
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(cache_dir=tmp)
            vectors = {"hi there": [1.0, 0.0, 0.0, 0.0],
                       "ola": [0.0, 1.0, 0.0, 0.0]}
            p1.model.encode.side_effect = _make_encoder(vectors)
            _register(p1, "skill_a:greet", "skill_a", ["hi there"], "en-US")
            _register(p1, "skill_a:greet", "skill_a", ["ola"], "pt-PT")
            # both registrations actually encoded (no bogus pre-existing hit)
            self.assertEqual(p1.model.encode.call_count, 2)

            # "restart": fresh pipeline, same cache dir, same inputs -- both
            # language partitions must hit their own cache entry
            p2 = _make_pipeline(cache_dir=tmp)
            p2.model.encode.side_effect = _make_encoder(vectors)
            _register(p2, "skill_a:greet", "skill_a", ["hi there"], "en-US")
            _register(p2, "skill_a:greet", "skill_a", ["ola"], "pt-PT")
            p2.model.encode.assert_not_called()

            candidates_en = list(p2._match_prototype("hi there", message=None, lang="en-US"))
            candidates_pt = list(p2._match_prototype("ola", message=None, lang="pt-PT"))
            self.assertEqual(candidates_en[0][1], "skill_a:greet")
            self.assertEqual(candidates_pt[0][1], "skill_a:greet")


if __name__ == "__main__":
    unittest.main()
