"""OVOS-INTENT-4 *consumer* end-to-end tests for the Model2Vec pipeline.

``tests/test_ovoscope_prototype_e2e.py`` proves the prototype pipeline matches
intents registered via the legacy ``padatious:register_intent`` event. This
suite proves m2v *consumes the INTENT-4 spec registration topics*
(``ovos-intent-4.md``) and then matches.

m2v is a **template** engine: it consumes ``ovos.intent.register.template``
(§6) and not ``ovos.intent.register.keyword`` (§11). It runs in *prototype*
mode here (the only mode that ingests fresh samples e2e); ``model2vec.StaticModel``
is patched at the ``sys.modules`` level with a deterministic, linearly-separable
mock encoder so no model download is needed and cosine scoring is exact.

Each test emits the spec registration on the wire, sends a matching utterance,
and asserts the intent dispatches ``<skill_id>:<intent_name>`` — proving
spec-topic consumption.

``ovos.intent.enable``/``ovos.intent.disable`` are session-scoped (§8.5):
disable suppresses the intent only for the caller's session, and enable
lifts that suppression without touching the underlying registration, so a
disabled-then-enabled intent matches again immediately.
"""
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ovoscope = pytest.importorskip(
    "ovoscope", reason="ovoscope not installed; skipping E2E tests"
)

from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session  # noqa: E402
from ovos_config.config import Configuration  # noqa: E402
from ovos_spec_tools import SpecMessage  # noqa: E402
from ovoscope import get_minicroft  # noqa: E402

from ovos_m2v_pipeline import Model2VecPrototypePipeline  # noqa: E402

PIPELINE_ID = "ovos-m2v-prototype-pipeline"
CONFIG_KEY = "ovos_m2v_prototype_pipeline"

REGISTER_TEMPLATE = str(SpecMessage.INTENT_REGISTER_TEMPLATE)
REGISTER_KEYWORD = str(SpecMessage.INTENT_REGISTER_KEYWORD)
INTENT_DEREGISTER = str(SpecMessage.INTENT_DEREGISTER)
SKILL_DEREGISTER = str(SpecMessage.SKILL_DEREGISTER)
INTENT_DISABLE = str(SpecMessage.INTENT_DISABLE)
INTENT_ENABLE = str(SpecMessage.INTENT_ENABLE)

SKILL_ID = "intent4_m2v.skill"
ADMIN_SKILL_ID = "admin.skill"

# Deterministic orthogonal directions for the mock encoder (cosine 0 between
# distinct keys, 1.0 for identical inputs).
_DIRECTIONS = {
    "lights": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "music":  np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
}
_NOISE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _fake_encode(sentences, **kwargs):
    out = []
    for s in sentences:
        sl = s.lower()
        vec = _NOISE.copy()
        for key, direction in _DIRECTIONS.items():
            if key in sl:
                vec = direction.copy()
                break
        out.append(vec)
    return np.stack(out)


class TestIntent4Consume(unittest.TestCase):
    """OVOS-INTENT-4 consumer assertions for m2v prototype mode."""

    @classmethod
    def setUpClass(cls):
        cls._mock_model = MagicMock()
        cls._mock_model.encode.side_effect = _fake_encode
        fake_m2v = MagicMock()
        fake_m2v.StaticModel.from_pretrained.return_value = cls._mock_model
        cls._sys_modules_patch = patch.dict(sys.modules, {"model2vec": fake_m2v})
        cls._sys_modules_patch.start()

        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig = intents_cfg.get(CONFIG_KEY)
        intents_cfg[CONFIG_KEY] = {"model": "fake-model", "prototype_k": 5}

        cls.mc = get_minicroft(skill_ids=[], lang="en-US",
                               default_pipeline=[PIPELINE_ID], max_wait=60)
        cls.pipeline: Model2VecPrototypePipeline = (
            cls.mc.intents.pipeline_plugins[PIPELINE_ID]
        )
        cls.pipeline.model = cls._mock_model

    @classmethod
    def tearDownClass(cls):
        try:
            cls.mc.stop()
        finally:
            cfg = Configuration()
            intents_cfg = cfg.get("intents", {})
            if cls._orig is None:
                intents_cfg.pop(CONFIG_KEY, None)
            else:
                intents_cfg[CONFIG_KEY] = cls._orig
            cls._sys_modules_patch.stop()

    def setUp(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        self.pipeline.prototype_store = PrototypeIntentStore()
        self.pipeline.intents = set()
        self.pipeline.ignore_labels = []

    # -- helpers --------------------------------------------------------

    def _register_template(self, intent_name, samples, lang="en-US"):
        self.mc.bus.emit(Message(REGISTER_TEMPLATE, {
            "skill_id": SKILL_ID, "intent_name": intent_name,
            "lang": lang, "samples": samples,
        }, {"skill_id": SKILL_ID}))

    def _emit(self, topic, intent_name=None, session_id=None, **extra):
        data = {"skill_id": SKILL_ID, "lang": "en-US"}
        if intent_name is not None:
            data["intent_name"] = intent_name
        data.update(extra)
        context = {"skill_id": SKILL_ID}
        if session_id is not None:
            context["session"] = Session(session_id=session_id).serialize()
        self.mc.bus.emit(Message(topic, data, context))

    def _utterance(self, utterance, session_id=None):
        context = {}
        if session_id is not None:
            context["session"] = Session(session_id=session_id).serialize()
        return Message("recognizer_loop:utterance",
                       {"utterances": [utterance], "lang": "en-US"}, context)

    def _send_and_capture(self, utterance, expected_types, timeout=5.0,
                          session_id=None):
        got, done, failed = [], threading.Event(), threading.Event()

        def _on_match(msg):
            got.append(msg)
            done.set()

        def _on_fail(_msg):
            failed.set()
            done.set()

        for t in expected_types:
            self.mc.bus.on(t, _on_match)
        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance(utterance, session_id=session_id))
            done.wait(timeout=timeout)
        finally:
            for t in expected_types:
                self.mc.bus.remove(t, _on_match)
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        if failed.is_set() and not got:
            return None
        return got[0] if got else None

    def _expect_no_match(self, utterance, timeout=2.0, session_id=None):
        failed = threading.Event()

        def _on_fail(_msg):
            failed.set()

        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance(utterance, session_id=session_id))
            failed.wait(timeout=timeout)
        finally:
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        self.assertTrue(failed.is_set(),
                        f"Expected no match for {utterance!r}.")

    # -- §6 spec template registration is matchable ---------------------

    def test_spec_template_registration_is_matchable(self):
        self._register_template("lights", ["turn on the lights", "lights on"])
        self.assertIn(f"{SKILL_ID}:lights", self.pipeline.intents)
        msg = self._send_and_capture("lights on now",
                                     expected_types=[f"{SKILL_ID}:lights"])
        self.assertIsNotNone(msg, "expected match from spec registration")
        self.assertEqual(msg.msg_type, f"{SKILL_ID}:lights")

    def test_spec_template_builds_prototypes(self):
        self._register_template("music", ["play some music", "start the music"])
        self.assertIn(f"{SKILL_ID}:music",
                      list(self.pipeline.prototype_store.labels))

    # -- back-compat: legacy registration still matches -----------------

    def test_legacy_registration_still_matches(self):
        self.mc.bus.emit(Message("padatious:register_intent", {
            "name": f"{SKILL_ID}:lights",
            "samples": ["turn on the lights", "lights on"],
        }, {"skill_id": SKILL_ID}))
        msg = self._send_and_capture("lights on now",
                                     expected_types=[f"{SKILL_ID}:lights"])
        self.assertIsNotNone(msg, "legacy registration must still match")

    # -- §8.2 / §8.4 deregistration -------------------------------------

    def test_spec_deregister_removes_intent(self):
        self._register_template("lights", ["turn on the lights", "lights on"])
        self.assertIsNotNone(
            self._send_and_capture("lights on now",
                                   expected_types=[f"{SKILL_ID}:lights"]),
            "sanity: should match before deregister",
        )
        self._emit(INTENT_DEREGISTER, "lights")
        self._expect_no_match("lights on now", timeout=3.0)

    def test_spec_skill_deregister_removes_intent(self):
        self._register_template("lights", ["turn on the lights", "lights on"])
        self._emit(SKILL_DEREGISTER)
        self._expect_no_match("lights on now", timeout=3.0)

    # -- §8.5 disable / enable ------------------------------------------

    def test_spec_disable_suppresses_intent(self):
        """Disable suppresses the intent for the caller's session (§8.5)."""
        self._register_template("lights", ["turn on the lights", "lights on"])
        self._emit(INTENT_DISABLE, "lights")
        self._expect_no_match("lights on now", timeout=3.0)

    def test_spec_disable_is_cross_skill_by_design(self):
        """§8.5 exemption: the payload skill_id names the TARGET intent's
        skill, context skill_id names the source requesting the control
        action, and the two MAY differ. An admin skill disabling another
        skill's intent must suppress it."""
        self._register_template("lights", ["turn on the lights", "lights on"])
        self.mc.bus.emit(Message(
            INTENT_DISABLE,
            {"skill_id": SKILL_ID, "intent_name": "lights", "lang": "en-US"},
            {"skill_id": ADMIN_SKILL_ID}))
        self._expect_no_match("lights on now", timeout=3.0)

    def test_spec_enable_rearms_intent(self):
        self._register_template("lights", ["turn on the lights", "lights on"])
        self._emit(INTENT_DISABLE, "lights")
        self._emit(INTENT_ENABLE, "lights")
        msg = self._send_and_capture("lights on now",
                                     expected_types=[f"{SKILL_ID}:lights"])
        self.assertIsNotNone(msg, "intent should match again after enable")

    def test_spec_disable_is_session_scoped(self):
        """§8.5: disable affects only registrations under the session_id read
        from the disabling message's ``context.session.session_id`` -- a
        disable carried by session A must not suppress the intent for
        session B."""
        self._register_template("lights", ["turn on the lights", "lights on"])
        self._emit(INTENT_DISABLE, "lights", session_id="session-a")
        self._expect_no_match("lights on now", session_id="session-a")
        msg = self._send_and_capture(
            "lights on now", expected_types=[f"{SKILL_ID}:lights"],
            session_id="session-b")
        self.assertIsNotNone(
            msg, "session B must still match; disable was scoped to session A")

    def test_spec_enable_rearms_intent_in_disabling_session(self):
        """§8.5: enable re-arms the intent in the session it was disabled in,
        without requiring re-registration, in prototype mode too."""
        self._register_template("lights", ["turn on the lights", "lights on"])
        self._emit(INTENT_DISABLE, "lights", session_id="session-a")
        self._expect_no_match("lights on now", session_id="session-a")
        self._emit(INTENT_ENABLE, "lights", session_id="session-a")
        msg = self._send_and_capture(
            "lights on now", expected_types=[f"{SKILL_ID}:lights"],
            session_id="session-a")
        self.assertIsNotNone(
            msg, "intent should match again in session A after enable")

    # -- §11 negative: template engine ignores the keyword topic --------

    def test_keyword_topic_does_not_match_on_template_engine(self):
        self.mc.bus.emit(Message(REGISTER_KEYWORD, {
            "skill_id": SKILL_ID, "intent_name": "lights",
            "lang": "en-US",
            "required": [{"name": "TurnOn", "samples": ["on"]},
                         {"name": "Light", "samples": ["lights"]}],
            "optional": [], "one_of": [], "excluded": [],
        }, {"skill_id": SKILL_ID}))
        self.assertNotIn(f"{SKILL_ID}:lights", self.pipeline.intents)
        self._expect_no_match("lights on now", timeout=3.0)


if __name__ == "__main__":
    unittest.main()
