"""Unit tests for the boot-time prototype cache (``ovos_m2v_pipeline.cache``).

The cache exists so that an unchanged registration -- same model, same
templates, same registered entity values -- skips ``model.encode()`` on a
restart and loads its embeddings from disk instead. Each test here
simulates a "restart" by constructing a *fresh* pipeline instance against
the same on-disk cache directory, rather than re-registering against the
same live store (which would just hit the in-memory re-registration path,
not the cache).
"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage

from ovos_m2v_pipeline.cache import PrototypeCache, compute_cache_key


def _make_hash_encode(dim=16):
    """Deterministic bag-of-words hashing encoder: identical sentences
    produce identical *dim*-wide vectors, so a cache round-trip and a fresh
    encode of the same inputs are comparable."""
    def _encode(sentences, **kwargs):
        out = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, sent in enumerate(sentences):
            for tok in sent.lower().split():
                bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
                out[i, bucket] += 1.0
        return out
    return _encode


_hash_encode = _make_hash_encode(16)


def _make_pipeline(cache_dir, model="fake-embed-model", extra_config=None, dim=None):
    """Fresh prototype-mode pipeline instance, wired to *cache_dir*.

    Each call builds an independent ``Model2VecIntentPipeline`` with its own
    mocked encode call-count -- simulating a process restart against a
    persistent, shared on-disk cache. *dim* (when given) sets the mocked
    model's declared output dimension (``model2vec.StaticModel.dim`` in
    production) AND the width of the vectors its ``encode()`` produces, so
    a "model swapped in place at a different dimension" restart can be
    simulated realistically.
    """
    config = {"model": model, "mode": "prototype", "prototype_cache_dir": str(cache_dir)}
    if extra_config:
        config.update(extra_config)

    mock_embed_model = MagicMock()
    mock_embed_model.encode.side_effect = _make_hash_encode(dim) if dim else _hash_encode
    if dim:
        mock_embed_model.dim = dim

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


def _register(pipeline, samples, name="test_skill:demo", skill_id="test_skill"):
    pipeline.bus.emit(Message(
        "padatious:register_intent",
        {"name": name, "samples": samples, "lang": "en-US"},
        {"skill_id": skill_id}))


class TestPrototypeCacheHitMiss(unittest.TestCase):
    def test_cache_hit_skips_encode_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp)
            _register(p1, ["turn on the lights", "switch the lights on"])
            self.assertTrue(p1.model.encode.called)

            # simulate a restart: fresh pipeline, same cache dir, same inputs
            p2 = _make_pipeline(tmp)
            _register(p2, ["turn on the lights", "switch the lights on"])
            p2.model.encode.assert_not_called()
            self.assertEqual(len(p2.prototype_store), 2)

    def test_changed_samples_invalidate_and_reencode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp)
            _register(p1, ["turn on the lights"])

            p2 = _make_pipeline(tmp)
            _register(p2, ["switch off the lights"])
            p2.model.encode.assert_called_once()

    def test_changed_model_id_invalidates_and_reencodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp, model="model-a")
            _register(p1, ["turn on the lights"])

            p2 = _make_pipeline(tmp, model="model-b")
            _register(p2, ["turn on the lights"])
            p2.model.encode.assert_called_once()

    def test_changed_expansion_params_invalidate_and_reencode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp, extra_config={"prototype_k": None})
            _register(p1, ["turn on the lights"])

            p2 = _make_pipeline(tmp, extra_config={"prototype_k": 1})
            _register(p2, ["turn on the lights"])
            p2.model.encode.assert_called_once()

    def test_prototype_cache_disabled_always_reencodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp, extra_config={"prototype_cache": False})
            _register(p1, ["turn on the lights"])

            p2 = _make_pipeline(tmp, extra_config={"prototype_cache": False})
            _register(p2, ["turn on the lights"])
            p2.model.encode.assert_called_once()


class TestPrototypeCacheInvalidationOnRemoval(unittest.TestCase):
    def test_remove_skill_deletes_cache_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp)
            _register(p1, ["turn on the lights"])

            p1.bus.emit(Message("detach_skill", {"skill_id": "test_skill"}))

            # restart after removal: nothing on disk any more -> re-encode
            p2 = _make_pipeline(tmp)
            _register(p2, ["turn on the lights"])
            p2.model.encode.assert_called_once()

    def test_remove_intent_deletes_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp)
            _register(p1, ["turn on the lights"])

            p1.bus.emit(Message("detach_intent", {"intent_name": "test_skill:demo"}))

            p2 = _make_pipeline(tmp)
            _register(p2, ["turn on the lights"])
            p2.model.encode.assert_called_once()


class TestPrototypeCacheCorruption(unittest.TestCase):
    def test_corrupt_cache_entry_tolerated_and_reencoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp)
            _register(p1, ["turn on the lights"])

            cache_dir = Path(tmp)
            entries = list(cache_dir.glob("**/*.npz"))
            self.assertEqual(len(entries), 1)
            entries[0].write_bytes(b"not a valid npz file")

            p2 = _make_pipeline(tmp)
            with patch("ovos_m2v_pipeline.LOG.warning") as warn:
                _register(p2, ["turn on the lights"])
            p2.model.encode.assert_called_once()
            self.assertTrue(warn.called)
            self.assertEqual(len(p2.prototype_store), 1)


class TestPrototypeCacheKeyDeterminism(unittest.TestCase):
    def test_key_stable_across_entity_set_iteration_order(self):
        """Entity values are sourced from a Python ``set`` upstream, so their
        list order is unstable across runs; the key must not depend on it."""
        key_a = compute_cache_key(
            "model-x", "0.9.0", {"k": None}, ["play {media}"],
            entity_values={"media": ["jazz", "rock", "blues"]},
        )
        key_b = compute_cache_key(
            "model-x", "0.9.0", {"k": None}, ["play {media}"],
            entity_values={"media": ["blues", "jazz", "rock"]},
        )
        self.assertEqual(key_a, key_b)

    def test_key_stable_across_sample_line_order(self):
        key_a = compute_cache_key(
            "model-x", "0.9.0", {"k": None}, ["a", "b", "c"],
        )
        key_b = compute_cache_key(
            "model-x", "0.9.0", {"k": None}, ["c", "a", "b"],
        )
        self.assertEqual(key_a, key_b)

    def test_intent4_entity_registration_order_does_not_change_cache_key(self):
        """End-to-end: registering the same entity values via two INTENT-4
        ``ovos.entity.register`` calls whose ``samples`` list differs only in
        order must produce the same prototype-cache key for a template that
        references that entity."""
        with tempfile.TemporaryDirectory() as tmp:
            p1 = _make_pipeline(tmp)
            p1.bus.emit(Message(
                SpecMessage.ENTITY_REGISTER.value,
                {"entity_name": "media", "samples": ["jazz", "rock", "blues"],
                 "skill_id": "test_skill", "lang": "en-US"}))
            p1.bus.emit(Message(
                SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                {"skill_id": "test_skill", "intent_name": "play",
                 "samples": ["play {media}"], "lang": "en-US"}))
            self.assertTrue(p1.model.encode.called)

            p2 = _make_pipeline(tmp)
            p2.bus.emit(Message(
                SpecMessage.ENTITY_REGISTER.value,
                {"entity_name": "media", "samples": ["blues", "jazz", "rock"],
                 "skill_id": "test_skill", "lang": "en-US"}))
            p2.bus.emit(Message(
                SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                {"skill_id": "test_skill", "intent_name": "play",
                 "samples": ["play {media}"], "lang": "en-US"}))
            p2.model.encode.assert_not_called()


class TestPrototypeCacheDimensionMismatch(unittest.TestCase):
    """A cached entry whose embedding dimension disagrees with the
    currently-loaded model must never be accepted as a hit: merging it into
    the live store poisons _consolidate() (a ValueError there leaves BOTH
    ``_embeddings``/``_labels`` permanently nulled, per _consolidate()'s
    drop-before-copy sequencing) and every later scores() call, for every
    label in the store, not just the mismatched one."""

    def test_stale_dimension_entry_is_a_miss_and_store_stays_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            # a prior run of "same-model-id" cached a dim=16 entry
            p_old = _make_pipeline(tmp, model="same-model-id", dim=16)
            _register(p_old, ["turn on the lights"])
            self.assertTrue(p_old.model.encode.called)

            # the artifact behind "same-model-id" is retrained in place at
            # dim=8 -- model id/version/params/samples are all unchanged,
            # so the cache key matches exactly, but the vectors on disk are
            # the wrong shape for the now-loaded model
            p_new = _make_pipeline(tmp, model="same-model-id", dim=8)
            _register(p_new, ["turn on the lights"])
            # dimension mismatch -> treated as corrupt -> re-encoded, not
            # merged in as a dim=16 chunk
            p_new.model.encode.assert_called_once()

            # a second, ordinarily-cacheable label must ingest cleanly
            # alongside it (this is what a poisoned _consolidate() would
            # have nulled for the WHOLE store, not just the stale label)
            _register(p_new, ["completely different phrase"],
                      name="test_skill:other")

            embeddings = p_new.prototype_store.embeddings  # forces consolidate
            self.assertEqual(embeddings.shape[1], 8)
            self.assertEqual(len(p_new.prototype_store), 2)

            scores = p_new.prototype_store.scores(np.ones(8, dtype=np.float32))
            self.assertIn("test_skill:demo", scores)
            self.assertIn("test_skill:other", scores)

    def test_load_deletes_stale_dimension_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = PrototypeCache(Path(tmp))
            key = compute_cache_key(
                "same-model-id", "", {"k": None}, ["turn on the lights"])
            cache.save("test_skill:demo", key,
                       np.ones((1, 16), dtype=np.float32),
                       np.array(["test_skill:demo"], dtype=object))

            with patch("ovos_m2v_pipeline.cache.LOG.warning") as warn:
                hit = cache.load("test_skill:demo", key, expected_dim=8)
            self.assertIsNone(hit)
            self.assertTrue(warn.called)
            # the stale entry is gone, not just ignored
            self.assertIsNone(cache.load("test_skill:demo", key, expected_dim=8))
            self.assertEqual(list(Path(tmp).glob("**/*.npz")), [])


class TestPrototypeCacheSkillPrefixCollision(unittest.TestCase):
    """``remove_skill("skill")`` must never touch an unrelated skill whose
    id happens to literally start with "skill" (e.g. "skill_extra") --
    real OVOS skill ids are commonly namespaced this way."""

    def test_remove_skill_does_not_collide_with_prefixed_skill_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = PrototypeCache(Path(tmp))
            emb = np.zeros((1, 4), dtype=np.float32)
            labels = np.array(["x"], dtype=object)
            cache.save("skill:intent1", "keyA", emb, labels)
            cache.save("skill_extra:intent2", "keyB", emb, labels)

            self.assertIsNotNone(cache.load("skill:intent1", "keyA"))
            self.assertIsNotNone(cache.load("skill_extra:intent2", "keyB"))

            cache.remove_skill("skill")

            self.assertIsNone(cache.load("skill:intent1", "keyA"))
            self.assertIsNotNone(cache.load("skill_extra:intent2", "keyB"))

    def test_detach_skill_message_does_not_collide_with_prefixed_skill_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _make_pipeline(tmp)
            _register(p, ["turn on the lights"],
                      name="skill:demo", skill_id="skill")
            _register(p, ["play some music"],
                      name="skill_extra:demo", skill_id="skill_extra")
            self.assertEqual(p.model.encode.call_count, 2)

            p.bus.emit(Message("detach_skill", {"skill_id": "skill"}))

            # restart: "skill" must re-encode (its cache was removed), but
            # "skill_extra" must still hit its untouched cache entry
            p2 = _make_pipeline(tmp)
            p2.model.encode.reset_mock()
            _register(p2, ["turn on the lights"],
                      name="skill:demo", skill_id="skill")
            self.assertEqual(p2.model.encode.call_count, 1)
            _register(p2, ["play some music"],
                      name="skill_extra:demo", skill_id="skill_extra")
            self.assertEqual(p2.model.encode.call_count, 1)


if __name__ == "__main__":
    unittest.main()
