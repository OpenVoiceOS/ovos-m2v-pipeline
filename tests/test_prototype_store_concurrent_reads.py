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

The release waits for an event, never a timer: either the reader is
blocked on the store's lock or it has already finished. A sleep could
expire before the reader thread ran at all, and the test would then pass
without exercising the interleaving it exists for.
"""
import tempfile
import threading
import time
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
        """Hold *rows* back until ``release`` is set."""
        self._rows = rows
        self.shape = rows.shape
        self.entered = threading.Event()
        self.release = threading.Event()

    def __len__(self):
        """The row count, which ``_consolidate()`` reads to size the new array."""
        return len(self._rows)

    def __array__(self, dtype=None, copy=None):
        """Signal ``entered``, then block until ``release`` (or the timeout)."""
        self.entered.set()
        self.release.wait(TIMEOUT)
        return self._rows if dtype is None else self._rows.astype(dtype)


class _WatchedLock:
    """The store's ``RLock``, recording each thread that found it held.

    A thread in ``contenders`` is parked on the lock (or was): that is the
    proof that it reached the lock while another thread held it.
    """

    def __init__(self):
        """Wrap a fresh ``RLock`` with no contenders recorded."""
        self._lock = threading.RLock()
        self.contenders = set()

    def acquire(self, blocking=True, timeout=-1):
        """Take the lock, recording the caller if it had to wait for it."""
        if self._lock.acquire(blocking=False):
            return True
        if not blocking:
            return False
        self.contenders.add(threading.get_ident())
        return self._lock.acquire(timeout=timeout)

    def release(self):
        """Release the wrapped lock."""
        self._lock.release()

    def __enter__(self):
        """Acquire the lock, as ``with store._lock:`` does."""
        self.acquire()
        return self

    def __exit__(self, *exc):
        """Release the lock on leaving the block."""
        self.release()


def _wait_until(predicate, message):
    """Poll *predicate* until it holds; fail with *message* after TIMEOUT."""
    deadline = time.monotonic() + TIMEOUT
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(message)
        time.sleep(0.001)


class _Model:
    """Declares no ``dim``, so ``add()`` probes the store for it."""

    def encode(self, sentences, **kwargs):
        """One row of ones per sentence, DIM wide."""
        return np.ones((len(sentences), DIM), dtype=np.float32)


def _store_with_blocking_fold():
    """A store with two consolidated rows and a pending chunk that blocks.

    Returns the store, its lock replaced by a ``_WatchedLock``, and the
    ``_BlockingChunk`` the next fold will stop on.
    """
    store = PrototypeIntentStore()
    store._lock = _WatchedLock()
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
            """Run the reader, keeping its value or the exception it raised."""
            try:
                outcome["value"] = reader()
            except Exception as exc:  # recorded, asserted on below
                outcome["error"] = exc

        t = threading.Thread(target=_read)
        t.start()
        # Without the lock the reader finishes (or fails) inside the
        # window; with it the reader is parked on the lock until the fold
        # is released. Either way, release only once one of them happened.
        _wait_until(
            lambda: t.ident in store._lock.contenders or not t.is_alive(),
            "reader neither reached the lock nor finished")
        chunk.release.set()
        t.join(TIMEOUT)
        folder.join(TIMEOUT)
        self.assertFalse(t.is_alive(), "reader deadlocked")
        self.assertFalse(folder.is_alive(), "fold deadlocked")
        self.assertNotIn("error", outcome, f"reader raised {outcome.get('error')!r}")
        return outcome["value"]

    def test_len_waits_for_the_fold(self):
        """``len(store)`` during a fold waits and counts all five rows."""
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
        """``add()`` probing the store for its dimension waits for the fold."""
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
            """Register a blocking chunk and fold it, from another thread."""
            store._add_anchors("skill:second", chunk)
            store.embeddings

        folder = threading.Thread(target=_register_and_fold)

        class _Store(PrototypeIntentStore):
            @property
            def embeddings(self):
                """Fold, then start the competing registration once."""
                emb = super().embeddings
                if threading.current_thread() is not folder and not folder.ident:
                    # another thread registers right after the export's
                    # fold. Holding the lock, the export keeps it parked
                    # in _add_anchors; without the lock it gets through and
                    # into its own fold. Go on once either has happened.
                    folder.start()
                    _wait_until(
                        lambda: folder.ident in self._lock.contenders
                        or chunk.entered.is_set(),
                        "registration neither reached the lock nor folded")
                return emb

        store = _Store()
        store._lock = _WatchedLock()
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
