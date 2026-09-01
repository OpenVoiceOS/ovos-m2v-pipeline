"""End-to-end tests for Model2VecIntentPipeline using ovoscope.

`ovoscope.pipeline.PipelineHarness` has an upstream `_SinkSkill(bus=None)`
bug, so we drive a `MiniCroft` directly and capture the intent-dispatch
Message emitted by `IntentService._emit_match_message` when our pipeline
returns a match (the same signal ovoscope itself uses).

`StaticModelPipeline.from_pretrained` is patched at class level so the
plugin loads without downloading any real model.
"""
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ovoscope = pytest.importorskip("ovoscope", reason="ovoscope not installed; skipping E2E tests")

from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session  # noqa: E402
from ovos_config.config import Configuration  # noqa: E402
from ovoscope import get_minicroft  # noqa: E402

from ovos_m2v_pipeline import Model2VecIntentPipeline  # noqa: E402

PIPELINE_ID = "ovos-m2v-pipeline"
CONFIG_KEY = "ovos_m2v_pipeline"


class _E2EBase(unittest.TestCase):
    """Shared setup: spin up MiniCroft with our pipeline + mocked model."""

    extra_config: dict | None = None

    @classmethod
    def setUpClass(cls):
        cls.mock_model = MagicMock()
        cls.mock_model.classes_ = np.array([])
        cls.mock_model.predict_proba.return_value = np.array([[]])

        cls._patch = patch(
            "ovos_m2v_pipeline.StaticModelPipeline.from_pretrained",
            return_value=cls.mock_model,
        )
        cls._patch.start()

        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig_intents_cfg = intents_cfg.get(CONFIG_KEY)
        plugin_cfg = {"model": "fake-model", "renormalize": False}
        if cls.extra_config:
            plugin_cfg.update(cls.extra_config)
        intents_cfg[CONFIG_KEY] = plugin_cfg

        cls.mc = get_minicroft(
            skill_ids=[],
            lang="en-US",
            default_pipeline=[PIPELINE_ID],
            max_wait=60,
        )
        cls.pipeline: Model2VecIntentPipeline = (
            cls.mc.intents.pipeline_plugins[PIPELINE_ID]
        )
        cls.pipeline.model = cls.mock_model

    @classmethod
    def tearDownClass(cls):
        try:
            cls.mc.stop()
        finally:
            cfg = Configuration()
            intents_cfg = cfg.get("intents", {})
            if cls._orig_intents_cfg is None:
                intents_cfg.pop(CONFIG_KEY, None)
            else:
                intents_cfg[CONFIG_KEY] = cls._orig_intents_cfg
            cls._patch.stop()

    def setUp(self):
        self.pipeline.intents = []
        self.pipeline.ignore_labels = list(
            (self.extra_config or {}).get("ignore_intents", []) or []
        )
        self.mock_model.classes_ = np.array([])
        self.mock_model.predict_proba.return_value = np.array([[]])

    def _set_probs(self, labels: list[str], probs: list[float]):
        self.mock_model.classes_ = np.array(labels)
        self.mock_model.predict_proba.return_value = np.array([probs])

    def _utterance_msg(self, utterance: str,
                       session_pipeline: list[str] | None = None) -> Message:
        ctx = {}
        if session_pipeline is not None:
            sess = Session(session_id="ovoscope-test",
                           pipeline=session_pipeline)
            ctx["session"] = sess.serialize()
        return Message(
            "recognizer_loop:utterance",
            data={"utterances": [utterance], "lang": "en-US"},
            context=ctx,
        )

    def _send_and_capture(self, utterance: str, expected_types: list[str],
                          timeout: float = 5.0,
                          session_pipeline: list[str] | None = None) -> Message | None:
        got: list[Message] = []
        done = threading.Event()
        failed = threading.Event()

        def _capture_match(msg):
            got.append(msg)
            done.set()

        def _capture_fail(msg):
            failed.set()
            done.set()

        for t in expected_types:
            self.mc.bus.on(t, _capture_match)
        self.mc.bus.on("complete_intent_failure", _capture_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance, session_pipeline))
            done.wait(timeout=timeout)
        finally:
            for t in expected_types:
                self.mc.bus.remove(t, _capture_match)
            self.mc.bus.remove("complete_intent_failure", _capture_fail)
        if failed.is_set() and not got:
            return None
        return got[0] if got else None

    def _expect_no_match(self, utterance: str, timeout: float = 2.0,
                         session_pipeline: list[str] | None = None):
        failed = threading.Event()

        def _on_fail(_msg):
            failed.set()

        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance, session_pipeline))
            failed.wait(timeout=timeout)
        finally:
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        self.assertTrue(
            failed.is_set(),
            f"Expected no match for {utterance!r}, but got no intent_failure.",
        )


class TestRegisteredIntentMatch(_E2EBase):
    def test_high_confidence_dispatches_intent(self):
        self.pipeline.intents = ["skill_a:my.intent"]
        self._set_probs(["skill_a:my.intent"], [0.9])
        msg = self._send_and_capture(
            "turn on the lights",
            expected_types=["skill_a:my.intent"],
        )
        self.assertIsNotNone(msg, "expected match message on bus")
        self.assertEqual(msg.msg_type, "skill_a:my.intent")
        self.assertAlmostEqual(msg.data.get("confidence"), 0.9, places=5)
        self.assertEqual(msg.data.get("utterance"), "turn on the lights")

    def test_below_all_thresholds_no_match(self):
        self.pipeline.intents = ["skill_a:my.intent"]
        self._set_probs(["skill_a:my.intent"], [0.05])
        self._expect_no_match("turn on the lights")

    def test_unregistered_intent_no_match(self):
        self.pipeline.intents = []
        self._set_probs(["skill_z:other.intent"], [0.99])
        self._expect_no_match("anything goes here")


class TestSpecialLabelRouting(_E2EBase):
    """Special labels (ocp/common_query/stop) are gated by the caller's
    session.pipeline."""

    def test_ocp_special_label_routed(self):
        self.pipeline.intents = []
        self._set_probs(["ocp:play"], [0.95])
        msg = self._send_and_capture(
            "play some music",
            expected_types=["ovos.common_play.play_search"],
            session_pipeline=["ovos-ocp-pipeline-plugin-high",
                              "ovos-m2v-pipeline"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "ovos.common_play.play_search")

    def test_common_query_special_label_routed(self):
        self.pipeline.intents = []
        self._set_probs(["common_query:common_query"], [0.8])
        msg = self._send_and_capture(
            "what is the capital of france",
            expected_types=["common_query.question"],
            session_pipeline=["ovos-common-query-pipeline-plugin",
                              "ovos-m2v-pipeline"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "common_query.question")

    def test_stop_special_label_routed(self):
        self.pipeline.intents = []
        self._set_probs(["stop:stop"], [0.85])
        msg = self._send_and_capture(
            "stop", expected_types=["mycroft.stop"],
            session_pipeline=["ovos-stop-pipeline-plugin-high",
                              "ovos-m2v-pipeline"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "mycroft.stop")

    def test_ocp_filtered_when_pipeline_absent(self):
        self.pipeline.intents = []
        self._set_probs(["ocp:play"], [0.99])
        self._expect_no_match(
            "play some music",
            session_pipeline=["ovos-m2v-pipeline"],
        )

    def test_stop_filtered_when_pipeline_absent(self):
        self.pipeline.intents = []
        self._set_probs(["stop:stop"], [0.99])
        self._expect_no_match(
            "stop",
            session_pipeline=["ovos-m2v-pipeline"],
        )


class TestMediumConfidence(_E2EBase):
    def test_medium_threshold_window(self):
        # prob 0.55: below conf_high (0.7) but above conf_medium (0.5).
        self.pipeline.intents = ["skill_a:my.intent"]
        self._set_probs(["skill_a:my.intent"], [0.55])
        msg = self._send_and_capture(
            "ambient command",
            expected_types=["skill_a:my.intent"],
        )
        self.assertIsNotNone(msg)
        self.assertAlmostEqual(msg.data.get("confidence"), 0.55, places=5)


class TestIgnoreIntents(_E2EBase):
    extra_config = {"ignore_intents": ["skill_a:my.intent"]}

    def test_ignored_label_not_matched(self):
        self.pipeline.intents = []
        self._set_probs(["skill_a:my.intent"], [0.99])
        self._expect_no_match("turn on the lights")


if __name__ == "__main__":
    unittest.main()
