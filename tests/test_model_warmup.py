"""The cold start belongs to the boot, not to the first user.

A user reported that after a restart the first intents are never matched
for minutes, with no sign that anything is wrong. The cause is where the
load starts: `self.model` was read for the first time by the first MATCH,
so the whole cold start -- resolve, download, load -- was paid inside the
first utterance, and every utterance until it finished was declined.

These tests assert three things: the load starts at construction, a match
inside the window declines quickly instead of blocking the bus thread, and
the same utterance matches once the load lands.
"""
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

INCREASE = "ovos-skill-volume.openvoiceos:increase_volume.intent"


def _message(utterance):
    return Message("recognizer_loop:utterance",
                   {"utterances": [utterance], "lang": "en-US"},
                   {"session": {"session_id": "default", "lang": "en-US"}})


class _SlowLoad:
    """A model load that takes a known time, on whatever thread calls it."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.started = threading.Event()
        self.thread_names = []

    def __call__(self, *args, **kwargs):
        self.thread_names.append(threading.current_thread().name)
        self.started.set()
        time.sleep(self.seconds)
        head = MagicMock()
        head.classes_ = np.array([INCREASE])
        head.predict_proba.return_value = np.array([[1.0]])
        return head


def _pipeline(loader, config=None):
    """A plugin whose only slow part is the model load.

    Two things are pinned out, because both are round trips this file is
    not measuring. `_resolve_model_revision` calls the Hub.
    `_initial_intent_sync` asks the bus for the adapt intents and waits for
    a reply, and a FakeBus answers nothing, so it costs its whole timeout.
    What is left is where the load starts.
    """
    import sys
    import ovos_m2v_pipeline as mod

    config = dict(config or {})
    config.setdefault("model", "fake-model")
    with patch("ovos_m2v_pipeline.load_shared_model", loader), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": MagicMock()}), \
         patch.object(mod.Model2VecIntentPipeline, "_resolve_model_revision",
                      lambda self, path: path), \
         patch.object(mod, "_resolve_model_id", lambda *a, **k: "fake-model"), \
         patch.object(mod.Model2VecIntentPipeline, "_initial_intent_sync",
                      lambda self: None):
        pipeline = mod.Model2VecIntentPipeline(bus=FakeBus(), config=config)
        pipeline._resolve_model_revision = lambda path: path
    return pipeline


class TestTheLoadStartsAtBoot(unittest.TestCase):

    def test_construction_starts_the_load_without_waiting_for_it(self):
        loader = _SlowLoad(2.0)
        start = time.monotonic()
        pipeline = _pipeline(loader)
        elapsed = time.monotonic() - start

        self.assertTrue(
            loader.started.wait(timeout=5),
            "the load never started: construction returned and nothing "
            "read the model, so the first utterance still pays for it")
        self.assertLess(
            elapsed, 1.0,
            "construction blocked for %.2fs; the eager load must return at "
            "once, only preload_model may block" % elapsed)
        self.assertFalse(pipeline.model_ready,
                         "the model cannot be ready while the load runs")
        # The classifier and its `-low` prototype stage each start one,
        # and `load_shared_model` shares the embedding between them, so
        # the second is cheap in real use. What matters here is that no
        # load ran on the calling thread.
        self.assertTrue(loader.thread_names)
        self.assertEqual(set(loader.thread_names), {"m2v-eager-model-load"},
                         "a load ran somewhere other than the eager thread: "
                         "%r" % (loader.thread_names,))

    def test_preload_still_blocks(self):
        # The other setting, unchanged: a deployment that prefers a slow
        # boot to a declined first utterance.
        loader = _SlowLoad(1.0)
        start = time.monotonic()
        pipeline = _pipeline(loader, {"preload_model": True})
        self.assertGreaterEqual(time.monotonic() - start, 1.0)
        self.assertTrue(pipeline.model_ready)

    def test_eager_load_can_be_turned_off(self):
        loader = _SlowLoad(0.1)
        _pipeline(loader, {"eager_model_load": False})
        self.assertFalse(
            loader.started.wait(timeout=0.5),
            "eager_model_load False must leave the load to the first match")


class TestAnUtteranceInsideTheWindow(unittest.TestCase):

    def test_it_declines_quickly_and_matches_once_ready(self):
        loader = _SlowLoad(3.0)
        pipeline = _pipeline(loader, {"model_load_budget": 0.2})
        pipeline.intents = {INCREASE}
        self.assertTrue(loader.started.wait(timeout=5))

        # One second into the window, which is where the user is.
        time.sleep(1.0)
        self.assertFalse(pipeline.model_ready)
        start = time.monotonic()
        match = pipeline.match_high(["turn it up"], "en-US",
                                    _message("turn it up"))
        waited = time.monotonic() - start

        self.assertIsNone(
            match,
            "a match answered before the model was loaded")
        self.assertLess(
            waited, 1.0,
            "the match held the bus thread for %.2fs; it must decline "
            "inside model_load_budget so the next pipeline entry answers "
            "this utterance" % waited)

        # And the same utterance matches once the load lands.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not pipeline.model_ready:
            time.sleep(0.05)
        self.assertTrue(pipeline.model_ready,
                        "the background load never finished")
        self.assertIsNotNone(
            pipeline.match_high(["turn it up"], "en-US",
                                _message("turn it up")),
            "the utterance that was declined during warm-up does not match "
            "after it, so the load bought nothing")


if __name__ == "__main__":
    unittest.main()
