"""Regression tests for #102: prototype-mode ranking and store consolidation
must be deterministic across boots, independent of the order in which
concurrent bus registrations happen to land.

All tests drive the real `Model2VecIntentPipeline._match_prototype` /
`PrototypeIntentStore` code, not a re-implementation of the ranking rule, so
they exercise the exact sort/fold sites the bug lives in.
"""
import subprocess
import sys
import textwrap
import unittest

import numpy as np

from tests.test_pipeline import _make_prototype_pipeline


def _fake_encode(sentences, **kw):
    """Deterministic stand-in for a model2vec encoder: hashes each string
    into a small fixed-dim vector so no HF download or real model is needed.
    """
    dim = 4
    out = np.zeros((len(sentences), dim), dtype=np.float32)
    for i, s in enumerate(sentences):
        for j, ch in enumerate(s):
            out[i, j % dim] += ord(ch)
    return out


class FakeModel:
    dim = 4

    def encode(self, sentences, **kw):
        return _fake_encode(sentences, **kw)


def _prototype_pipeline():
    from ovos_m2v_pipeline import PrototypeIntentStore
    p = _make_prototype_pipeline(proto_store=PrototypeIntentStore())
    p.model.encode.side_effect = lambda sents, **kw: _fake_encode(sents, **kw)
    return p


class TestRankingTieBreak(unittest.TestCase):
    """Part (a): the probe's exact-tie case, driven through the real
    ``_match_prototype`` ranking sort.
    """

    def _ranked_winner(self, order):
        pipeline = _prototype_pipeline()
        model = FakeModel()
        # Both labels get anchors identical to the query below, forcing an
        # exact cosine tie regardless of registration order.
        for label in order:
            pipeline.prototype_store.add(model, label, ["tie sentence"])
        candidates = list(pipeline._match_prototype("tie sentence"))
        ranked = [(label, score) for _skill, label, score in candidates]
        return ranked

    def test_tie_break_is_registration_order_independent(self):
        ranked_ab = self._ranked_winner(["LabelA", "LabelB"])
        ranked_ba = self._ranked_winner(["LabelB", "LabelA"])

        # Scores themselves are untouched by the fix (bit-transparent).
        self.assertAlmostEqual(ranked_ab[0][1], ranked_ab[1][1], places=6)
        self.assertAlmostEqual(ranked_ba[0][1], ranked_ba[1][1], places=6)

        self.assertEqual(ranked_ab, ranked_ba)
        # Lexicographically smaller label wins the tie, both orders.
        self.assertEqual(ranked_ab[0][0], "LabelA")
        self.assertEqual(ranked_ba[0][0], "LabelA")


class TestConsolidationOrderInvariance(unittest.TestCase):
    """Part (b): registering a handful of intents in different orders must
    consolidate to bit-identical store arrays and produce identical
    ``_match_prototype`` rankings.
    """

    LABELS = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
    SENTENCES = {
        "Alpha": ["alpha one", "alpha two"],
        "Bravo": ["bravo one", "bravo two"],
        "Charlie": ["charlie one", "charlie two"],
        "Delta": ["delta one", "delta two"],
        "Echo": ["echo one", "echo two"],
    }
    QUERIES = ["alpha one", "bravo two", "charlie one", "random query text"]

    def _build(self, order):
        pipeline = _prototype_pipeline()
        model = FakeModel()
        for label in order:
            pipeline.prototype_store.add(model, label, self.SENTENCES[label])
        # Force consolidation so we can compare the store's internal arrays.
        _ = pipeline.prototype_store.embeddings
        return pipeline

    def _ranked(self, pipeline, query_text):
        candidates = list(pipeline._match_prototype(query_text))
        return [(label, score) for _skill, label, score in candidates]

    def test_order_invariant_ranking_and_arrays(self):
        import random
        orders = [
            sorted(self.LABELS),
            list(reversed(sorted(self.LABELS))),
        ]
        rnd = random.Random(1234)
        shuffled = list(self.LABELS)
        rnd.shuffle(shuffled)
        orders.append(shuffled)

        pipelines = [self._build(order) for order in orders]

        base_labels = pipelines[0].prototype_store.labels
        base_embeddings = pipelines[0].prototype_store.embeddings
        for pipeline in pipelines[1:]:
            np.testing.assert_array_equal(pipeline.prototype_store.labels, base_labels)
            np.testing.assert_allclose(pipeline.prototype_store.embeddings,
                                        base_embeddings, atol=0.0)

        base_rankings = [self._ranked(pipelines[0], q) for q in self.QUERIES]
        for pipeline in pipelines[1:]:
            rankings = [self._ranked(pipeline, q) for q in self.QUERIES]
            self.assertEqual(rankings, base_rankings)


_SUBPROCESS_SCRIPT = textwrap.dedent("""
    import sys
    sys.path.insert(0, {path!r})

    from tests.test_prototype_determinism import (
        FakeModel, _fake_encode, _prototype_pipeline,
        TestConsolidationOrderInvariance as _T,
    )

    order = {order!r}
    pipeline = _prototype_pipeline()
    model = FakeModel()
    for label in order:
        pipeline.prototype_store.add(model, label, _T.SENTENCES[label])
    _ = pipeline.prototype_store.embeddings  # force consolidation

    out = []
    for q in {queries!r}:
        candidates = list(pipeline._match_prototype(q))
        out.append([(label, score) for _skill, label, score in candidates])
    print("RESULT=" + repr(out))
""")


class TestSubprocessBootPairDeterminism(unittest.TestCase):
    """Part (c): two independent boots, different PYTHONHASHSEED AND
    different registration order, must print identical ranked lists.
    """

    def test_two_boots_different_seed_and_order_agree(self):
        import os

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        order_a = sorted(TestConsolidationOrderInvariance.SENTENCES.keys())
        order_b = list(reversed(order_a))
        queries = TestConsolidationOrderInvariance.QUERIES

        script_a = _SUBPROCESS_SCRIPT.format(path=repo_root, order=order_a, queries=queries)
        script_b = _SUBPROCESS_SCRIPT.format(path=repo_root, order=order_b, queries=queries)

        env_a = dict(os.environ, PYTHONHASHSEED="0")
        env_b = dict(os.environ, PYTHONHASHSEED="12345")

        proc_a = subprocess.run(
            [sys.executable, "-c", script_a],
            cwd=repo_root, env=env_a, capture_output=True, text=True, timeout=60)
        proc_b = subprocess.run(
            [sys.executable, "-c", script_b],
            cwd=repo_root, env=env_b, capture_output=True, text=True, timeout=60)

        self.assertEqual(proc_a.returncode, 0, proc_a.stderr)
        self.assertEqual(proc_b.returncode, 0, proc_b.stderr)

        def _extract(stdout):
            for line in stdout.splitlines():
                if line.startswith("RESULT="):
                    return eval(line[len("RESULT="):])
            raise AssertionError(f"no RESULT= line in subprocess output: {stdout!r}")

        out_a = _extract(proc_a.stdout)
        out_b = _extract(proc_b.stdout)

        self.assertEqual(out_a, out_b,
                          f"boot A (seed=0, order={order_a}) -> {out_a}\n"
                          f"boot B (seed=12345, order={order_b}) -> {out_b}")




class TestConsolidateMergedFold(unittest.TestCase):
    """A future rebase resolving a conflict in ``_consolidate`` could keep
    only one of #93's per-language derived-state rebuild or #102's
    label-sorted fold and still pass most tests -- both properties only
    show up together, on the same store, after a mixed-language
    registration.

    Two stores are built with the same labels/languages in different
    registration orders: ``skill_a:x`` under two languages (en-US, pt-PT),
    ``skill_b:y`` under one (en-US), ``skill_c:z`` never partitioned by
    language. One test asserts, on the same consolidated stores: the raw
    arrays fold identically regardless of order (#102), the derived
    per-language state matches a from-scratch expectation and a pt-PT query
    never lets ``skill_b:y``'s en-US-only rows compete (#93), and the two
    stores rank identically for two queries.
    """

    SENTENCES = {
        ("skill_a:x", "en-US"): ["alpha english one", "alpha english two"],
        ("skill_a:x", "pt-PT"): ["alpha portuguese one", "alpha portuguese two"],
        ("skill_b:y", "en-US"): ["bravo english one", "bravo english two"],
        ("skill_c:z", None): ["charlie plain one", "charlie plain two"],
    }
    REGISTRATIONS = list(SENTENCES.keys())

    def _build(self, order):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore()
        model = FakeModel()
        for label, lang in order:
            store.add(model, label, self.SENTENCES[(label, lang)], lang=lang)
        _ = store.embeddings  # force consolidation
        return store

    def test_merged_fold_and_per_language_rebuild(self):
        import random

        order_a = list(self.REGISTRATIONS)
        order_b = list(reversed(self.REGISTRATIONS))
        rnd = random.Random(99)
        order_c = list(self.REGISTRATIONS)
        rnd.shuffle(order_c)

        store1 = self._build(order_a)
        store2 = self._build(order_c)

        # (1) sorted fold: consolidated arrays are bit-identical across
        # registration orders, independent of which order finished first.
        np.testing.assert_array_equal(store1._labels, store2._labels)
        np.testing.assert_allclose(store1._embeddings, store2._embeddings,
                                    atol=0.0)

        # (2) per-language derived state, rebuilt from scratch on every
        # store mutation: _langs_by_label / _entry_langs must agree with an
        # independently computed expectation, and a pt-PT query must never
        # let skill_b:y's (en-US only) rows compete for it.
        expected_langs_by_label = {
            "skill_a:x": {"en-US", "pt-PT"},
            "skill_b:y": {"en-US"},
        }
        self.assertEqual(store1._langs_by_label, expected_langs_by_label)
        self.assertEqual(store2._langs_by_label, expected_langs_by_label)

        expected_entry_langs = set()
        for label, lang in self.REGISTRATIONS:
            for _ in self.SENTENCES[(label, lang)]:
                expected_entry_langs.add(lang)
        self.assertEqual(set(store1._entry_langs), expected_entry_langs)
        self.assertEqual(set(store2._entry_langs), expected_entry_langs)

        pt_query = _fake_encode(["alpha portuguese one"])[0]
        pt_scores_1 = store1.scores(pt_query, lang="pt-PT")
        pt_scores_2 = store2.scores(pt_query, lang="pt-PT")

        self.assertIn("skill_a:x", pt_scores_1)
        self.assertAlmostEqual(pt_scores_1["skill_a:x"], 1.0, places=5)
        self.assertNotIn("skill_b:y", pt_scores_1)
        self.assertIn("skill_c:z", pt_scores_1)
        self.assertEqual(pt_scores_1, pt_scores_2)

        # (3) ranked (label, score) lists agree across registration orders
        # for two distinct queries.
        for query_text in ["alpha portuguese one", "bravo english two"]:
            q = _fake_encode([query_text])[0]
            ranked_1 = sorted(store1.scores(q).items())
            ranked_2 = sorted(store2.scores(q).items())
            self.assertEqual(ranked_1, ranked_2)


if __name__ == "__main__":
    unittest.main()
