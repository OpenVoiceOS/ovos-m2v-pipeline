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

DIVERGENCE (real finding, ``test_spec_enable_rearms_intent`` is xfail): in
prototype mode ``ovos.intent.disable`` drops the label's prototypes, and
``ovos.intent.enable`` only restores *label tracking* — it cannot re-embed the
samples, so a disabled-then-enabled intent does NOT match again (re-arming
requires re-registration). This departs from INTENT-4 §8.5's "re-arm a
previously disabled intent".
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

    def _emit(self, topic, intent_name=None, **extra):
        data = {"skill_id": SKILL_ID, "lang": "en-US"}
        if intent_name is not None:
            data["intent_name"] = intent_name
        data.update(extra)
        self.mc.bus.emit(Message(topic, data, {"skill_id": SKILL_ID}))

    def _utterance(self, utterance):
        return Message("recognizer_loop:utterance",
                       {"utterances": [utterance], "lang": "en-US"}, {})

    def _send_and_capture(self, utterance, expected_types, timeout=5.0):
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
            self.mc.bus.emit(self._utterance(utterance))
            done.wait(timeout=timeout)
        finally:
            for t in expected_types:
                self.mc.bus.remove(t, _on_match)
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        if failed.is_set() and not got:
            return None
        return got[0] if got else None

    def _expect_no_match(self, utterance, timeout=2.0):
        failed = threading.Event()

        def _on_fail(_msg):
            failed.set()

        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance(utterance))
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
        """Disable drops the label (and its prototypes) from the match-eligible
        set, suppressing the intent (§8.5)."""
        self._register_template("lights", ["turn on the lights", "lights on"])
        self._emit(INTENT_DISABLE, "lights")
        self._expect_no_match("lights on now", timeout=3.0)

    @pytest.mark.xfail(
        strict=False,
        reason="INTENT-4 §8.5: 'ovos.intent.enable re-arms a previously disabled "
               "intent' — in m2v prototype mode disable drops the prototypes and "
               "enable only restores label tracking (cannot re-embed samples), so "
               "the intent does not match again; re-arming requires re-registration.",
    )
    def test_spec_enable_rearms_intent(self):
        self._register_template("lights", ["turn on the lights", "lights on"])
        self._emit(INTENT_DISABLE, "lights")
        self._emit(INTENT_ENABLE, "lights")
        msg = self._send_and_capture("lights on now",
                                     expected_types=[f"{SKILL_ID}:lights"])
        self.assertIsNotNone(msg, "intent should match again after enable")

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
