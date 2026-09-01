"""Regression tests for prototype-mode exact-sample recall and
store consistency under (re-)registration."""

import hashlib
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
from ovos_bus_client.message import Message

from tests.test_pipeline import _make_prototype_pipeline


def _hash_encode(sentences, **kwargs):
    """Deterministic bag-of-words hashing encoder.

    Identical sentences map to identical vectors (cosine 1.0); sentences
    sharing few words score low. Stands in for a real embedding model so the
    store's anchor bookkeeping can be tested exactly.
    """
    dim = 64
    out = np.zeros((len(sentences), dim), dtype=np.float32)
    for i, sent in enumerate(sentences):
        for tok in sent.lower().split():
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            out[i, bucket] += 1.0
    return out


class TestExactSampleHighTierMatch(unittest.TestCase):
    """A registered training sample must match at the high tier.

    Guards against per-label anchor subsampling dropping the very samples a
    skill registered: with hundreds of expanded templates per intent, a
    capped anchor set almost never contains a given sample, so an exact
    training utterance scores below ``conf_high`` and the pipeline silently
    declines a match its store should hit.
    """

    def _register_and_match(self, pipeline, samples, utterance):
        pipeline.model.encode.side_effect = _hash_encode
        pipeline.bus.emit(Message(
            "padatious:register_intent",
            {"name": "ovos-skill-date-time.openvoiceos:what.time.is.it.intent",
             "samples": samples, "lang": "en-US"},
            {"skill_id": "ovos-skill-date-time.openvoiceos"}))
        msg = Message("recognizer_loop:utterance",
                      {"utterances": [utterance], "lang": "en-US"})
        return pipeline.match_high([utterance], "en-US", msg)

    def test_exact_sample_matches_high_tier(self):
        pipeline = _make_prototype_pipeline()
        # A large intent file: many distinct templates, like the expanded
        # what.time.is.it.intent shipped by ovos-skill-date-time.
        samples = [f"placeholder sample number {i} variant" for i in range(400)]
        samples.append("what time is it")
        match = self._register_and_match(pipeline, samples, "what time is it")
        self.assertIsNotNone(match)
        self.assertEqual(
            match.match_type,
            "ovos-skill-date-time.openvoiceos:what.time.is.it")
        self.assertGreaterEqual(match.match_data["confidence"], 0.99)

    def test_every_sample_kept_by_default(self):
        pipeline = _make_prototype_pipeline()
        pipeline.model.encode.side_effect = _hash_encode
        samples = [f"sample number {i}" for i in range(50)]
        pipeline.bus.emit(Message(
            "padatious:register_intent",
            {"name": "skill:intent", "samples": samples, "lang": "en-US"},
            {"skill_id": "skill"}))
        self.assertEqual(len(pipeline.prototype_store), len(samples))

    def test_prototype_k_still_caps_anchor_count(self):
        pipeline = _make_prototype_pipeline(config={"prototype_k": 5})
        pipeline.model.encode.side_effect = _hash_encode
        samples = [f"sample number {i}" for i in range(50)]
        pipeline.bus.emit(Message(
            "padatious:register_intent",
            {"name": "skill:intent", "samples": samples, "lang": "en-US"},
            {"skill_id": "skill"}))
        self.assertEqual(len(pipeline.prototype_store), 5)


class TestReRegistration(unittest.TestCase):
    """Skills re-register their intents (e.g. on language reload — twice per
    skill in a normal boot). Re-registration is an implicit replacement and
    must never corrupt the store.

    Registration handlers run on an executor thread pool, so adds and
    removals also arrive concurrently; without internal locking the parallel
    embeddings/labels arrays tear apart and every later replacement fails
    with a boolean-index size mismatch.
    """

    def test_second_registration_replaces_intent(self):
        pipeline = _make_prototype_pipeline()
        pipeline.model.encode.side_effect = _hash_encode
        for _ in range(2):
            pipeline.bus.emit(Message(
                "padatious:register_intent",
                {"name": "skill-a:intent", "lang": "en-US",
                 "samples": [f"sample {i}" for i in range(10)]},
                {"skill_id": "skill-a"}))
            pipeline.bus.emit(Message(
                "padatious:register_intent",
                {"name": "skill-b:intent", "lang": "en-US",
                 "samples": [f"other sample {i}" for i in range(15)]},
                {"skill_id": "skill-b"}))
        store = pipeline.prototype_store
        self.assertEqual(len(store.labels), len(store.embeddings))
        self.assertEqual((store.labels == "skill-a:intent").sum(), 10)
        self.assertEqual((store.labels == "skill-b:intent").sum(), 15)

    def test_concurrent_registrations_keep_store_consistent(self):
        from ovos_m2v_pipeline import PrototypeIntentStore

        model = MagicMock()
        model.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()

        def register(worker):
            label = f"skill:{worker % 10}"
            for _ in range(10):
                store.add(model, label,
                          [f"sample {i} of {label}" for i in range(10)])

        with ThreadPoolExecutor(8) as pool:
            futures = [pool.submit(register, w) for w in range(40)]
            for f in futures:
                f.result()  # propagate any worker exception

        self.assertEqual(len(store.labels), len(store.embeddings))
        # exactly 10 anchors per surviving label (implicit replacement)
        for worker in range(10):
            self.assertEqual((store.labels == f"skill:{worker}").sum(), 10)


if __name__ == "__main__":
    unittest.main()
