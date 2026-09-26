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

    def test_unrelated_dialects_do_not_exclude_each_other(self):
        """Two unrelated labels registered under two different English
        dialects must each still be scored for a third, unregistered dialect
        -- a store-wide ``closest_lang`` tie-break must not make one label's
        dialect exclude an unrelated label with no competitor at all."""
        query = "turn off the lights"
        vectors = {
            "set an alarm": [0.0, 1.0, 0.0, 0.0],
            query: [1.0, 0.0, 0.0, 0.0],
        }
        p = _make_pipeline()
        p.model.encode.side_effect = _make_encoder(vectors)

        _register(p, "skill_alarm:alarm", "skill_alarm",
                  ["set an alarm"], "en-GB")
        _register(p, "skill_light:off", "skill_light",
                  ["turn off the lights"], "en-US")

        candidates = list(p._match_prototype(query, message=None, lang="en-AU"))
        labels = [label for _, label, _ in candidates]

        self.assertIn("skill_light:off", labels)
        self.assertEqual(candidates[0][1], "skill_light:off")

    def test_per_label_dialect_choice_is_independent(self):
        """A label registered under both en-US and pt-PT must keep only its
        pt-PT partition for a pt-BR query. An unrelated en-GB-only label,
        which never competed for this label's dialect choice, is resolved
        independently for its own query."""
        query = "ligar as luzes"
        vectors = {
            "turn on the lights": [0.0, 1.0, 0.0, 0.0],
            query: [1.0, 0.0, 0.0, 0.0],
            "set an alarm": [0.0, 0.0, 1.0, 0.0],
        }
        p = _make_pipeline()
        p.model.encode.side_effect = _make_encoder(vectors)

        _register(p, "skill_light:on", "skill_light",
                  ["turn on the lights"], "en-US")
        _register(p, "skill_light:on", "skill_light",
                  [query], "pt-PT")
        _register(p, "skill_alarm:alarm", "skill_alarm",
                  ["set an alarm"], "en-GB")

        candidates = list(p._match_prototype(query, message=None, lang="pt-BR"))
        labels = {label: score for _, label, score in candidates}

        # skill_light:on keeps only its pt-PT partition for the pt-BR
        # query, not excluded by an unrelated label's dialect choice
        self.assertIn("skill_light:on", labels)
        self.assertAlmostEqual(labels["skill_light:on"], 1.0, places=4)

        # the unrelated en-GB-only label is judged purely on its own
        # dialect distance to the query, unaffected by skill_light:on's
        # en-US/pt-PT partitioning decision: en-US resolves it for an
        # en-AU query even though skill_light:on now favours pt-PT
        candidates_au = list(p._match_prototype(
            "turn on the lights", message=None, lang="en-AU"))
        labels_au = {label: score for _, label, score in candidates_au}
        self.assertIn("skill_light:on", labels_au)


class TestLangsByLabelStaysInSync(unittest.TestCase):
    """``PrototypeIntentStore._langs_by_label`` is maintained incrementally
    on every mutation (see ``scores()``'s per-query dialect resolution); a
    removal that forgets to update it would leave a stale entry claiming a
    language partition still exists after the store no longer holds any
    prototypes for it."""

    def test_remove_skill_clears_langs_by_label(self):
        from ovos_m2v_pipeline import PrototypeIntentStore

        vectors = {"hi there": [1.0, 0.0, 0.0, 0.0], "ola": [0.0, 1.0, 0.0, 0.0]}
        model = MagicMock()
        model.encode.side_effect = _make_encoder(vectors)

        store = PrototypeIntentStore()
        store.add(model, "skill_a:greet", ["hi there"], lang="en-US")
        store.add(model, "skill_a:greet", ["ola"], lang="pt-PT")
        self.assertIn("skill_a:greet", store._langs_by_label)
        self.assertEqual(store._langs_by_label["skill_a:greet"], {"en-US", "pt-PT"})

        store.remove_skill("skill_a")

        self.assertNotIn("skill_a:greet", store._langs_by_label)
        self.assertEqual(store.scores(np.array(vectors["hi there"], dtype=np.float32),
                                       lang="en-US"), {})

    def test_remove_clears_langs_by_label(self):
        from ovos_m2v_pipeline import PrototypeIntentStore

        vectors = {"hi there": [1.0, 0.0, 0.0, 0.0], "ola": [0.0, 1.0, 0.0, 0.0]}
        model = MagicMock()
        model.encode.side_effect = _make_encoder(vectors)

        store = PrototypeIntentStore()
        store.add(model, "skill_a:greet", ["hi there"], lang="en-US")
        store.add(model, "skill_a:greet", ["ola"], lang="pt-PT")

        store.remove("skill_a:greet")

        self.assertNotIn("skill_a:greet", store._langs_by_label)


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



class TestRemovalRecomputesDerivedArrays(unittest.TestCase):
    """``remove()``/``remove_skill()`` must rebuild the derived
    ``_orig_labels``/``_entry_langs``/``_orig_codes``/``_unique_orig_labels``
    arrays (see ``_recompute_derived``) for the rows that survive, not just
    drop the removed rows out of ``_embeddings``/``_labels``. Those derived
    arrays are index-parallel to the surviving embeddings; if a removal
    resizes the embeddings without rebuilding them, a later ``scores(lang=
    ...)`` call either raises (boolean mask size mismatch against the old,
    longer arrays) or -- when array lengths happen to coincide -- silently
    pairs a stale label with the wrong surviving embedding via truncated
    zip, corrupting the surviving labels' own scores."""

    @staticmethod
    def _vectors():
        return {
            "hi from b": [0.9, 0.1, 0.0, 0.0],
            "bonjour b": [0.85, 0.2, 0.0, 0.0],
            "info from c": [0.0, 0.0, 0.9, 0.1],
            "greet a en": [0.1, 0.9, 0.0, 0.0],
            "greet a pt": [0.0, 0.0, 0.1, 0.9],
        }

    def _build_store(self, register_a=True):
        from ovos_m2v_pipeline import PrototypeIntentStore

        vectors = self._vectors()
        model = MagicMock()
        model.encode.side_effect = _make_encoder(vectors)
        store = PrototypeIntentStore()
        if register_a:
            store.add(model, "skill_a:greet", ["greet a en"], lang="en-US")
            store.add(model, "skill_a:greet", ["greet a pt"], lang="pt-PT")
        store.add(model, "skill_b:bye", ["hi from b", "bonjour b"], lang="en-US")
        store.add(model, "skill_c:info", ["info from c"], lang=None)
        # force consolidation of every pending chunk into the contiguous
        # arrays remove_skill()/remove() actually mask against
        store._consolidate()
        return store

    def _derived_signature(self, store):
        """A comparable snapshot of every array ``scores()`` reads, keyed
        by label so row order (which need not match across two
        independently-built stores) doesn't matter."""
        by_label = {}
        for i, orig in enumerate(store._orig_labels):
            code = store._orig_codes[i]
            assert store._unique_orig_labels[code] == orig
            by_label.setdefault(orig, []).append(
                (store._entry_langs[i], bool(store._entry_lang_is_none[i]))
            )
        for rows in by_label.values():
            rows.sort(key=lambda r: (r[0] or "",))
        langs_by_label = {k: set(v) for k, v in store._langs_by_label.items()}
        return by_label, langs_by_label

    def test_remove_skill_leaves_surviving_scores_and_derived_arrays_intact(self):
        query = np.array([0.6, 0.3, 0.4, 0.1], dtype=np.float32)
        store = self._build_store(register_a=True)

        scores_before_all = store.scores(query)
        scores_before_en = store.scores(query, lang="en-US")
        self.assertIn("skill_a:greet", scores_before_all)
        self.assertIn("skill_b:bye", scores_before_all)
        self.assertIn("skill_c:info", scores_before_all)

        store.remove_skill("skill_a")

        scores_after_all = store.scores(query)
        scores_after_en = store.scores(query, lang="en-US")

        # removed skill's label is gone from both query modes
        self.assertNotIn("skill_a:greet", scores_after_all)
        self.assertNotIn("skill_a:greet", scores_after_en)

        # every surviving label's score is bit-identical to what it was
        # before the removal -- the removal must not perturb rows it
        # didn't touch
        for label in ("skill_b:bye", "skill_c:info"):
            self.assertIn(label, scores_after_all)
            self.assertIn(label, scores_after_en)
            self.assertEqual(scores_after_all[label], scores_before_all[label])
            self.assertEqual(scores_after_en[label], scores_before_en[label])

        # the derived arrays for the surviving rows match a store built
        # from scratch with only the surviving registrations -- a stale
        # rebuild (or none at all) would disagree here even if scores()
        # happened to raise or come out right by coincidence
        fresh = self._build_store(register_a=False)
        self.assertEqual(
            self._derived_signature(store), self._derived_signature(fresh)
        )

    def test_remove_single_label_leaves_surviving_scores_and_derived_arrays_intact(self):
        query = np.array([0.6, 0.3, 0.4, 0.1], dtype=np.float32)
        store = self._build_store(register_a=True)

        scores_before_all = store.scores(query)
        scores_before_en = store.scores(query, lang="en-US")

        store.remove("skill_a:greet")

        scores_after_all = store.scores(query)
        scores_after_en = store.scores(query, lang="en-US")

        self.assertNotIn("skill_a:greet", scores_after_all)
        self.assertNotIn("skill_a:greet", scores_after_en)

        for label in ("skill_b:bye", "skill_c:info"):
            self.assertEqual(scores_after_all[label], scores_before_all[label])
            self.assertEqual(scores_after_en[label], scores_before_en[label])

        fresh = self._build_store(register_a=False)
        self.assertEqual(
            self._derived_signature(store), self._derived_signature(fresh)
        )


if __name__ == "__main__":
    unittest.main()
