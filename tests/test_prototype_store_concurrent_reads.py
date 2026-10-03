"""Readers of ``PrototypeIntentStore`` must not see a fold half-way through.

``_consolidate()`` runs under the store's lock and, to bound its peak
memory, sets ``_labels``/``_embeddings`` to ``None`` while it copies the
pending chunks into the new arrays. A reader that skips the lock and lands
in that window gets ``len(None)``; in production that was the "prototype
store ready" log line raising ``TypeError`` from ``len(store)`` while a
skill's registration was still being folded in.

Each test holds a real ``_consolidate()`` open inside that window (a
pending chunk whose copy blocks until released), runs one reader from
another thread, then releases the fold. The reader must either wait for
the lock or read a consistent store; it must never raise.
"""
import tempfile
import threading
import unittest
from types import SimpleNamespace

import numpy as np

from ovos_m2v_pipeline import Model2VecIntentPipeline, PrototypeIntentStore
from ovos_m2v_pipeline.cache import PrototypeCache
from ovos_m2v_pipeline.prebuilt import export_store

DIM = 4
TIMEOUT = 5.0


class _BlockingChunk:
    """A pending chunk whose copy into the consolidated array blocks.

    ``_consolidate()`` copies each chunk with a slice assignment, which
    asks numpy for ``__array__``: by then ``_labels`` is already ``None``,
    so pausing here holds the store in the window under test.
    """

    def __init__(self, rows: np.ndarray):
        self._rows = rows
        self.shape = rows.shape
        self.entered = threading.Event()
        self.release = threading.Event()

    def __len__(self):
        return len(self._rows)

    def __array__(self, dtype=None, copy=None):
        self.entered.set()
        self.release.wait(TIMEOUT)
        return self._rows if dtype is None else self._rows.astype(dtype)


class _Model:
    """Declares no ``dim``, so ``add()`` probes the store for it."""

    def encode(self, sentences, **kwargs):
        return np.ones((len(sentences), DIM), dtype=np.float32)


def _store_with_blocking_fold():
    store = PrototypeIntentStore()
    store.add(_Model(), "skill:first", ["one", "two"])
    store.embeddings  # fold "skill:first" so the store has consolidated rows
    chunk = _BlockingChunk(np.ones((3, DIM), dtype=np.float32))
    store._add_anchors("skill:second", chunk)
    return store, chunk


class TestReadDuringConsolidate(unittest.TestCase):

    def _run_during_fold(self, store, chunk, reader):
        """Start a fold, run *reader* in another thread while the fold is
        inside its window, release the fold, return the reader's result."""
        folder = threading.Thread(target=lambda: store.embeddings)
        folder.start()
        self.assertTrue(chunk.entered.wait(TIMEOUT), "fold never started")

        outcome = {}

        def _read():
            try:
                outcome["value"] = reader()
            except Exception as exc:  # recorded, asserted on below
                outcome["error"] = exc

        t = threading.Thread(target=_read)
        t.start()
        # Without the lock the reader finishes (or fails) right here; with
        # it the reader is parked on the lock until the fold is released.
        t.join(0.2)
        chunk.release.set()
        t.join(TIMEOUT)
        folder.join(TIMEOUT)
        self.assertFalse(t.is_alive(), "reader deadlocked")
        self.assertFalse(folder.is_alive(), "fold deadlocked")
        self.assertNotIn("error", outcome, f"reader raised {outcome.get('error')!r}")
        return outcome["value"]

    def test_len_waits_for_the_fold(self):
        store, chunk = _store_with_blocking_fold()
        self.assertEqual(self._run_during_fold(store, chunk, lambda: len(store)), 5)

    def test_ready_log_during_a_fold(self):
        """The production traceback: the ready handler logs ``len(store)``."""
        store, chunk = _store_with_blocking_fold()
        pipeline = SimpleNamespace(prototype_store=store)
        self._run_during_fold(
            store, chunk,
            lambda: Model2VecIntentPipeline._handle_ready_prototype(pipeline, None))

    def test_add_dimension_probe_waits_for_the_fold(self):
        store, chunk = _store_with_blocking_fold()
        with tempfile.TemporaryDirectory() as tmp:
            store.cache = PrototypeCache(tmp)
            added = self._run_during_fold(
                store, chunk,
                lambda: store.add(_Model(), "skill:third", ["three"], cache_key="k"))
        self.assertEqual(added, 1)
        self.assertEqual(len(store), 6)

    def test_export_reads_one_consistent_store(self):
        """``export_store`` reads embeddings, then labels: a registration
        folded in between the two must not hand it the ``None`` labels."""
        chunk = _BlockingChunk(np.ones((3, DIM), dtype=np.float32))

        def _register_and_fold():
            store._add_anchors("skill:second", chunk)
            store.embeddings

        folder = threading.Thread(target=_register_and_fold)

        class _Store(PrototypeIntentStore):
            @property
            def embeddings(self):
                emb = super().embeddings
                if threading.current_thread() is not folder and not folder.ident:
                    # another thread registers right after the export's
                    # fold; holding the lock, the export keeps it waiting,
                    # so this wait times out and the export goes on
                    folder.start()
                    chunk.entered.wait(0.5)
                return emb

        store = _Store()
        store.add(_Model(), "skill:first", ["one", "two"])
        with tempfile.TemporaryDirectory() as tmp:
            try:
                export_store(store, tmp, model_id="m", model2vec_version="0")
            finally:
                chunk.release.set()
                folder.join(TIMEOUT)
            data = np.load(f"{tmp}/prototypes.npz", allow_pickle=False)
            self.assertEqual(len(data["embeddings"]), len(data["labels"]))
            self.assertEqual(list(data["labels"]), ["skill:first", "skill:first"])
        self.assertFalse(folder.is_alive(), "fold deadlocked")
        self.assertEqual(len(store), 5)


if __name__ == "__main__":
    unittest.main()
