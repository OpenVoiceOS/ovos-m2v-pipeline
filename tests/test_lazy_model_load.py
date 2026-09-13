"""Constructing Model2VecIntentPipeline must be cheap: ovos-core builds every
installed pipeline plugin at boot (ovos-core#903), so a heavy engine that
loads its model eagerly pays that cost on every boot even when unconfigured
or unused. The model loads lazily on first need instead, buffering
registrations that arrive before it exists.
"""
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus


def _fake_m2v_module(embed_model):
    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = embed_model
    return fake_m2v


class TestConstructorIsCheap(unittest.TestCase):
    def test_classifier_mode_constructor_does_not_load_model(self):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}):
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(), config={"model": "fake-model"})
            MockSMP.from_pretrained.assert_not_called()
        self.assertIsNone(pipeline.model)

    def test_prototype_mode_constructor_does_not_load_model(self):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        embed_model = MagicMock()
        fake_m2v = _fake_m2v_module(embed_model)
        with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
             patch("ovos_m2v_pipeline.Configuration", return_value={}), \
             patch.dict(sys.modules, {"model2vec": fake_m2v}):
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(),
                config={"model": "fake-embed-model", "mode": "prototype"})
            fake_m2v.StaticModel.from_pretrained.assert_not_called()
        self.assertIsNone(pipeline.model)

    def test_preload_model_loads_at_construction(self):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        mock_model = MagicMock()
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}), \
             patch.object(Model2VecIntentPipeline, "_resolve_model_revision",
                          lambda self, p: p):
            MockSMP.from_pretrained.return_value = mock_model
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(),
                config={"model": "fake-model", "preload_model": True})
            MockSMP.from_pretrained.assert_called_once()
        self.assertIs(pipeline.model, mock_model)


class TestBufferedRegistrations(unittest.TestCase):
    def _make_deferred_prototype_pipeline(self, embed_model):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        fake_m2v = _fake_m2v_module(embed_model)
        patcher = patch.dict(sys.modules, {"model2vec": fake_m2v})
        patcher.start()
        self.addCleanup(patcher.stop)
        revision_patcher = patch.object(
            Model2VecIntentPipeline, "_resolve_model_revision",
            lambda self, p: p)
        revision_patcher.start()
        self.addCleanup(revision_patcher.stop)
        with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
             patch("ovos_m2v_pipeline.Configuration", return_value={}):
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(),
                config={"model": "fake-embed-model", "mode": "prototype"})
        return pipeline

    def test_registration_before_load_buffers_and_encodes_once(self):
        embed_model = MagicMock()
        embed_model.encode.side_effect = \
            lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
        pipeline = self._make_deferred_prototype_pipeline(embed_model)

        pipeline._handle_register_padatious(Message(
            "padatious:register_intent",
            {"name": "skill.test:go.intent", "samples": ["go to work"]},
            context={"skill_id": "skill.test"}))

        # buffered: label already tracked, but nothing was encoded yet
        self.assertIn("skill.test:go", pipeline.intents)
        embed_model.encode.assert_not_called()
        self.assertEqual(len(pipeline._pending_additions), 1)
        self.assertEqual(len(pipeline.prototype_store), 0)

        self.assertTrue(pipeline._ensure_model(background_ok=False))

        embed_model.encode.assert_called_once()
        self.assertEqual(len(pipeline._pending_additions), 0)
        self.assertEqual(len(pipeline.prototype_store), 1)

        # a second call to _ensure_model must not re-load or re-encode
        self.assertTrue(pipeline._ensure_model(background_ok=False))
        embed_model.encode.assert_called_once()

    def test_detach_of_buffered_label_drops_it_without_encoding(self):
        embed_model = MagicMock()
        embed_model.encode.side_effect = \
            lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
        pipeline = self._make_deferred_prototype_pipeline(embed_model)

        pipeline._handle_register_padatious(Message(
            "padatious:register_intent",
            {"name": "skill.test:go.intent", "samples": ["go to work"]},
            context={"skill_id": "skill.test"}))
        self.assertEqual(len(pipeline._pending_additions), 1)

        pipeline._handle_detach_intent(Message(
            "detach_intent", {"intent_name": "skill.test:go.intent"}))

        self.assertEqual(len(pipeline._pending_additions), 0)
        self.assertNotIn("skill.test:go", pipeline.intents)

        self.assertTrue(pipeline._ensure_model(background_ok=False))
        embed_model.encode.assert_not_called()
        self.assertEqual(len(pipeline.prototype_store), 0)


class TestConcurrentFirstLoad(unittest.TestCase):
    def test_concurrent_callers_load_model_exactly_once(self):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        mock_model = MagicMock()
        load_calls = []

        def slow_from_pretrained(path):
            load_calls.append(path)
            time.sleep(0.05)
            return mock_model

        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}), \
             patch.object(Model2VecIntentPipeline, "_resolve_model_revision",
                          lambda self, p: p):
            MockSMP.from_pretrained.side_effect = slow_from_pretrained
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(), config={"model": "fake-model"})

            results = []

            def worker():
                results.append(pipeline._ensure_model(background_ok=False))

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        self.assertTrue(all(results))
        self.assertEqual(len(load_calls), 1)
        self.assertIs(pipeline.model, mock_model)


class TestSuccessPathClearsThreadAndModelAtomically(unittest.TestCase):
    """``_ensure_model`` checks ``self.model is not None`` before it ever
    looks at ``_model_load_thread``, so a successful ``_load_model_now``
    must make both visible under the same lock acquisition. If the thread
    handle were cleared in a separate locked block from the ``self.model``
    assignment, a concurrent caller landing in the gap between the two
    would see "no thread in flight, no model loaded" and spawn a second
    loader for a load that already succeeded."""

    def test_concurrent_caller_never_sees_thread_cleared_before_model_set(self):
        from ovos_m2v_pipeline import Model2VecIntentPipeline

        mock_model = MagicMock()
        load_calls = []

        def fake_from_pretrained(path):
            load_calls.append(path)
            return mock_model

        class WideningRLock:
            """Real lock, but on the specific release that would clear
            ``_model_load_thread`` while ``self.model`` is still ``None``,
            pauses just long enough for a second caller to observe that
            window if the two updates are not made atomically."""

            def __init__(self, pipeline):
                self._native = threading.RLock()
                self._pipeline = pipeline
                self._widened = False

            def acquire(self, *a, **k):
                return self._native.acquire(*a, **k)

            def release(self):
                if (not self._widened
                        and self._pipeline._model_load_thread is None
                        and self._pipeline.model is None):
                    self._widened = True
                    self._native.release()
                    time.sleep(0.2)
                    return
                self._native.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *a):
                self.release()

        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}), \
             patch.object(Model2VecIntentPipeline, "_resolve_model_revision",
                          lambda self, p: p):
            MockSMP.from_pretrained.side_effect = fake_from_pretrained
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(), config={"model": "fake-model"})
            pipeline._model_lock = WideningRLock(pipeline)

            results = []

            def caller():
                results.append(pipeline._ensure_model(background_ok=False))

            t1 = threading.Thread(target=caller)
            t1.start()
            # give t1 time to enter _load_model_now and reach the point
            # where the fake load has "completed" and is about to make it
            # visible
            time.sleep(0.05)
            t2 = threading.Thread(target=caller)
            t2.start()
            t1.join(timeout=5)
            t2.join(timeout=5)

        self.assertTrue(all(results))
        self.assertEqual(len(load_calls), 1)
        self.assertIs(pipeline.model, mock_model)


if __name__ == "__main__":
    unittest.main()


class TestLoadRetryBackoff(unittest.TestCase):
    """A load failure must not leave the plugin silently dead forever: a
    dev eager load at least raised loudly at boot, so the deferred path
    needs its own way back to life once the underlying cause (bad network,
    transient HF outage) clears."""

    def test_failed_load_retries_after_backoff_then_succeeds(self):
        from ovos_m2v_pipeline import Model2VecIntentPipeline

        mock_model = MagicMock()
        from_pretrained = MagicMock(
            side_effect=[RuntimeError("boom 1"), RuntimeError("boom 2"), mock_model])

        clock = [0.0]

        def fake_monotonic():
            return clock[0]

        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}), \
             patch("ovos_m2v_pipeline.time.monotonic", side_effect=fake_monotonic), \
             patch.object(Model2VecIntentPipeline, "_resolve_model_revision",
                          lambda self, p: p):
            MockSMP.from_pretrained = from_pretrained
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(), config={"model": "fake-model"})

            # first attempt: fails
            self.assertFalse(pipeline._ensure_model(background_ok=False))
            self.assertEqual(from_pretrained.call_count, 1)
            self.assertIsNone(pipeline.model)

            # immediately retrying (clock unchanged) must NOT re-invoke the
            # loader: still inside the first backoff window
            self.assertFalse(pipeline._ensure_model(background_ok=False))
            self.assertEqual(from_pretrained.call_count, 1)

            # advance past the first backoff (30s): second attempt fails too
            clock[0] += 31.0
            self.assertFalse(pipeline._ensure_model(background_ok=False))
            self.assertEqual(from_pretrained.call_count, 2)
            self.assertIsNone(pipeline.model)

            # backoff doubled to 60s: a retry before that elapses is a
            # fast no-op, no third call yet
            clock[0] += 31.0
            self.assertFalse(pipeline._ensure_model(background_ok=False))
            self.assertEqual(from_pretrained.call_count, 2)

            # advance past the doubled backoff: third attempt succeeds
            clock[0] += 30.0
            self.assertTrue(pipeline._ensure_model(background_ok=False))
            self.assertEqual(from_pretrained.call_count, 3)
            self.assertIs(pipeline.model, mock_model)

            # further calls reuse the now-loaded model, no more loader calls
            self.assertTrue(pipeline._ensure_model(background_ok=False))
            self.assertEqual(from_pretrained.call_count, 3)


class TestRevisionResolutionFailure(unittest.TestCase):
    """Resolving a pinned ``revision`` is a network round trip performed
    before ``from_pretrained`` is even called; a bad revision or a
    transient hub failure there must be treated like any other load
    failure (logged, counted, backed off, retried) rather than escaping
    the loader thread and leaving it permanently dead."""

    def test_resolution_failure_is_logged_and_retried(self):
        from ovos_m2v_pipeline import Model2VecIntentPipeline

        mock_model = MagicMock()
        resolve = MagicMock(
            side_effect=[RuntimeError("bad revision"), "fake-model"])

        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}), \
             patch.object(Model2VecIntentPipeline, "_resolve_model_revision", resolve), \
             patch("ovos_m2v_pipeline.LOG") as mock_log:
            MockSMP.from_pretrained.return_value = mock_model
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(),
                config={"model": "fake-model", "revision": "does-not-exist"})

            self.assertFalse(pipeline._ensure_model(background_ok=False))
            mock_log.exception.assert_called_once()

            self.assertIsNone(pipeline.model)
            self.assertIsNone(pipeline._model_load_thread)
            self.assertEqual(pipeline._model_load_failures, 1)
            self.assertIsNotNone(pipeline._model_load_retry_at)

            # cause cleared and backoff elapsed: a later `_ensure_model` call
            # must spawn a fresh attempt (proving the thread reference was
            # actually cleared) and that attempt must load the model.
            pipeline._model_load_retry_at = None
            self.assertTrue(pipeline._ensure_model(background_ok=False))
            self.assertIs(pipeline.model, mock_model)
            self.assertEqual(resolve.call_count, 2)
