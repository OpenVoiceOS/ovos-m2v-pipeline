"""An identical re-registration must not rebuild the prototype store.

Every skill registers each intent twice at boot, once on the legacy
``padatious:register_intent`` topic and once on OVOS-INTENT-4, with the same
samples. ``_add_anchors`` used to treat the second one like any change:
fold every pending chunk into the store, copy the whole store through a mask
to drop the label's rows, then re-derive the label metadata of every row. That
is one full-store pass per label, so the boot cost grew with labels x rows: on
a runtime serving eleven languages it was minutes of CPU, and with 46
languages the store never finished building.

A label that already holds exactly the rows it is being given now keeps them
and nothing else happens. A real change (other samples, another language, a
label removed in between) still replaces or adds rows as before.
"""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ovos_m2v_pipeline import PrototypeIntentStore
from ovos_m2v_pipeline.cache import PrototypeCache


class _Model:
    """Deterministic encoder: a sentence's vector is derived from its text."""

    dim = 8

    def __init__(self):
        self.calls = 0

    def encode(self, sentences, **kwargs):
        self.calls += 1
        out = np.zeros((len(sentences), self.dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            rng = np.random.default_rng(sum(map(ord, s)) + len(s))
            out[i] = rng.standard_normal(self.dim)
        return out


LABEL = "skill.test:time.current"
EN = ["what time is it", "tell me the time", "current time"]
FR = ["quelle heure est-il", "donne-moi l'heure"]


def _store(**kwargs):
    return PrototypeIntentStore(**kwargs)


class TestIdenticalReregistration(unittest.TestCase):
    def test_second_identical_registration_leaves_the_store_alone(self):
        model, store = _Model(), _store()
        store.add(model, "other.skill:hello", ["hello there", "hi"], lang="en-US")
        n1 = store.add(model, LABEL, EN, lang="en-US")
        emb_before = store.embeddings  # folds the pending chunks
        labels_before = store._labels
        with patch.object(store, "_consolidate", wraps=store._consolidate) as fold, \
                patch.object(store, "_recompute_derived",
                             wraps=store._recompute_derived) as derive:
            n2 = store.add(model, LABEL, EN, lang="en-US")
        self.assertEqual(n1, n2)
        fold.assert_not_called()
        derive.assert_not_called()
        # the very same arrays: nothing was copied
        self.assertIs(store._embeddings, emb_before)
        self.assertIs(store._labels, labels_before)
        self.assertEqual(store._pending, [])
        self.assertEqual(len(store), len(emb_before))

    def test_pending_identical_registration_is_not_duplicated(self):
        model, store = _Model(), _store()
        store.add(model, LABEL, EN, lang="en-US")
        store.add(model, LABEL, EN, lang="en-US")  # still pending, not folded
        self.assertEqual(len(store), len(EN))
        self.assertEqual(len(store.labels), len(EN))

    def test_changed_samples_still_replace_the_rows(self):
        model, store = _Model(), _store()
        store.add(model, LABEL, EN, lang="en-US")
        store.add(model, LABEL, EN[:1], lang="en-US")
        self.assertEqual(len(store), 1)
        np.testing.assert_allclose(
            store.embeddings[0],
            model.encode(EN[:1])[0] / np.linalg.norm(model.encode(EN[:1])[0]),
            rtol=1e-5)

    def test_same_label_in_another_language_is_added(self):
        model, store = _Model(), _store()
        store.add(model, LABEL, EN, lang="en-US")
        store.add(model, LABEL, FR, lang="fr-FR")
        store.add(model, LABEL, EN, lang="en-US")
        store.add(model, LABEL, FR, lang="fr-FR")
        self.assertEqual(len(store), len(EN) + len(FR))

    def test_removed_label_registers_again(self):
        model, store = _Model(), _store()
        store.add(model, LABEL, EN, lang="en-US")
        store.remove(LABEL)
        self.assertEqual(len(store), 0)
        self.assertEqual(store.add(model, LABEL, EN, lang="en-US"), len(EN))
        self.assertEqual(len(store), len(EN))

    def test_removed_skill_registers_again(self):
        model, store = _Model(), _store()
        store.add(model, LABEL, EN, lang="en-US")
        store.add(model, "other.skill:hello", ["hello there"], lang="en-US")
        store.remove_skill("skill.test")
        self.assertEqual(len(store), 1)
        store.add(model, LABEL, EN, lang="en-US")
        self.assertEqual(len(store), 1 + len(EN))

    def test_cache_hit_reregistration_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as d:
            model = _Model()
            store = _store(cache=PrototypeCache(d))
            store.add(model, LABEL, EN, lang="en-US", cache_key="k1")
            self.assertEqual(model.calls, 1)
            store.embeddings
            arrays = store._embeddings
            # the second registration is served from the on-disk cache and
            # carries the same rows, so the store is not touched
            store.add(model, LABEL, EN, lang="en-US", cache_key="k1")
            self.assertEqual(model.calls, 1)
            self.assertIs(store._embeddings, arrays)
            self.assertEqual(len(store), len(EN))

    def test_scores_unchanged_by_a_repeat(self):
        model = _Model()
        once, twice = _store(), _store()
        for s in (once, twice):
            s.add(model, LABEL, EN, lang="en-US")
            s.add(model, "other.skill:hello", ["hello there", "hi"], lang="en-US")
        twice.add(model, LABEL, EN, lang="en-US")
        q = model.encode(["what time is it"])[0]
        q = q / np.linalg.norm(q)
        self.assertEqual(once.scores(q, lang="en-US"), twice.scores(q, lang="en-US"))

    def _saved_and_loaded(self, model):
        store = _store()
        store.add(model, "other.skill:hello", ["hello there", "hi"], lang="en-US")
        store.add(model, LABEL, EN, lang="en-US")
        store.add(model, LABEL, FR, lang="fr-FR")
        with tempfile.TemporaryDirectory() as d:
            path = f"{d}/prototypes.npz"
            store.save(path)
            return store, PrototypeIntentStore.load(path)

    def test_restored_label_reregistered_unchanged_is_a_no_op(self):
        # a store restored from disk (load(), or a prebuilt artifact at
        # boot) knows its labels but had no digests for them, so the first
        # identical registration of each one used to rebuild the store
        model = _Model()
        original, loaded = self._saved_and_loaded(model)
        np.testing.assert_array_equal(loaded._embeddings, original.embeddings)
        emb_before = loaded._embeddings
        labels_before = loaded._labels
        snapshot = emb_before.copy()
        with patch.object(loaded, "_consolidate", wraps=loaded._consolidate) as fold, \
                patch.object(loaded, "_recompute_derived",
                             wraps=loaded._recompute_derived) as derive:
            self.assertEqual(loaded.add(model, LABEL, EN, lang="en-US"), len(EN))
            self.assertEqual(loaded.add(model, LABEL, FR, lang="fr-FR"), len(FR))
            loaded.add(model, "other.skill:hello", ["hello there", "hi"], lang="en-US")
        fold.assert_not_called()
        derive.assert_not_called()
        self.assertIs(loaded._embeddings, emb_before)
        self.assertIs(loaded._labels, labels_before)
        np.testing.assert_array_equal(loaded._embeddings, snapshot)
        self.assertEqual(loaded._pending, [])
        self.assertEqual(len(loaded), len(EN) + len(FR) + 2)

    def test_restored_label_with_new_samples_is_replaced(self):
        model = _Model()
        _original, loaded = self._saved_and_loaded(model)
        loaded.add(model, LABEL, EN[:1], lang="en-US")
        self.assertEqual(len(loaded), 1 + len(FR) + 2)
        q = model.encode(EN[1:2])[0]
        q = q / np.linalg.norm(q)
        # the retired phrasing no longer matches perfectly
        self.assertLess(loaded.scores(q, lang="en-US").get(LABEL, 0.0), 0.999)


if __name__ == "__main__":
    unittest.main()
