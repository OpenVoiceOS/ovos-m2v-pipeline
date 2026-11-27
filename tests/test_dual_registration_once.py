"""ovos-workshop sends every intent twice, on ``padatious:register_intent`` and
on OVOS-INTENT-4's ``ovos.intent.register.template``, with the same samples.
Both copies used to be expanded, keyed, read from the prototype cache and
compared; on an 18-skill, 11-language runtime that was ~4 s of a boot's CPU for
an identical second copy."""
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from ovos_bus_client.message import Message

from tests.test_intent4 import _make_prototype_pipeline

SKILL = "lights.skill"
LABEL = f"{SKILL}:lights"
SAMPLES = ["turn on the lights", "lights (on|please)"]


def legacy(samples=SAMPLES, lang="en-US"):
    return Message("padatious:register_intent",
                   {"name": f"{LABEL}.intent", "samples": list(samples), "lang": lang},
                   {"skill_id": SKILL})


def spec(samples=SAMPLES, lang="en-US", **extra):
    return Message("ovos.intent.register.template",
                   {"skill_id": SKILL, "intent_name": "lights", "lang": lang,
                    "samples": list(samples), **extra},
                   {"skill_id": SKILL})


class TestDualRegistrationAppliedOnce(unittest.TestCase):
    def setUp(self):
        self.p = _make_prototype_pipeline()
        self.p.model.encode.side_effect = (
            lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32))
        store = self.p.prototype_store
        adds = patch.object(store, "add", wraps=store.add)
        self.add = adds.start()
        self.addCleanup(adds.stop)

    def encodes(self):
        """Registrations that reached the store (encode or cache read)."""
        return self.add.call_count

    def test_second_contract_copy_is_not_encoded_again(self):
        self.p._handle_register_padatious(legacy())
        first = self.encodes()
        self.assertGreater(first, 0)
        before = len(self.p.prototype_store)
        self.p._handle_intent4_register_template(spec(requires_context=["room"]))
        self.assertEqual(self.encodes(), first)
        self.assertEqual(len(self.p.prototype_store), before)
        self.assertIn(LABEL, self.p.intents)
        # the copy's own metadata is still applied
        self.assertEqual(self.p._context_gates[LABEL][0], ["room"])

    def test_other_language_is_its_own_registration(self):
        self.p._handle_register_padatious(legacy())
        first = self.encodes()
        self.p._handle_intent4_register_template(spec(lang="fr-FR"))
        self.assertGreater(self.encodes(), first)

    def test_changed_samples_are_applied(self):
        self.p._handle_register_padatious(legacy())
        first = self.encodes()
        self.p._handle_intent4_register_template(spec(["switch the lights on"]))
        self.assertGreater(self.encodes(), first)

    def test_detach_then_identical_registration_applies_again(self):
        self.p._handle_register_padatious(legacy())
        self.p._handle_detach_intent(Message("detach_intent",
                                             {"intent_name": f"{LABEL}.intent"}))
        self.assertEqual(len(self.p.prototype_store), 0)
        self.p._handle_intent4_register_template(spec())
        self.assertGreater(len(self.p.prototype_store), 0)

    def test_store_emptied_elsewhere_is_refilled(self):
        self.p._handle_register_padatious(legacy())
        self.p.prototype_store.remove(LABEL)
        self.p._handle_intent4_register_template(spec())
        self.assertGreater(len(self.p.prototype_store), 0)

    def test_rejected_copy_does_not_shadow_a_valid_one(self):
        self.p._handle_intent4_register_template(spec(["   "]))
        self.p._handle_register_padatious(legacy())
        self.assertGreater(len(self.p.prototype_store), 0)


class TestDeferredModelLoad(unittest.TestCase):
    """With ``preload_model: false`` both copies arrive before the model has
    loaded and are buffered for the deferred encode."""

    def test_second_copy_is_not_buffered_again(self):
        p = _make_prototype_pipeline()
        p.model = None
        p._handle_register_padatious(legacy())
        p._handle_intent4_register_template(spec(requires_context=["room"]))
        self.assertEqual(len(p._pending_additions), 1)
        self.assertEqual(p._context_gates[LABEL][0], ["room"])


class TestCopyOfARegistrationBeingApplied(unittest.TestCase):
    def test_copy_waits_and_writes_when_the_first_write_fails(self):
        p = _make_prototype_pipeline()
        p.model.encode.side_effect = (
            lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32))
        copy_waiting = threading.Event()
        real_add = p._add_prototypes
        real_wait = p._applied_lock.wait_for
        calls = []

        def wait_for(predicate, timeout=None):
            if not predicate():
                copy_waiting.set()
            return real_wait(predicate, timeout)

        def add_prototypes(*a, **kw):
            calls.append(a[0])
            if len(calls) == 1:
                self.assertTrue(copy_waiting.wait(10))
                raise MemoryError("transient")
            return real_add(*a, **kw)

        # a copy that only finished by timing out would not join in time
        with patch("ovos_m2v_pipeline._REGISTRATION_WAIT_S", 60.0), \
                patch.object(p._applied_lock, "wait_for", side_effect=wait_for), \
                patch.object(p, "_add_prototypes", side_effect=add_prototypes):
            a = threading.Thread(target=p._handle_register_padatious, args=(legacy(),))
            b = threading.Thread(target=p._handle_intent4_register_template,
                                 args=(spec(),))
            a.start()
            while not calls:
                self.assertTrue(a.is_alive())
                a.join(0.01)
            b.start()
            a.join(10)
            b.join(10)
            self.assertFalse(a.is_alive() or b.is_alive())
        self.assertEqual(len(calls), 2)
        self.assertGreater(len(p.prototype_store), 0)


class TestRejectedRegistration(unittest.TestCase):
    def test_legacy_copy_with_no_valid_template_is_not_recorded(self):
        p = _make_prototype_pipeline()
        p.model = None
        bad = ["(lights|on"]
        p._handle_register_padatious(legacy(bad))
        self.assertEqual(p._applied_registrations, {})
        self.assertEqual(p._registrations_in_flight, {})
        p._handle_intent4_register_template(spec(bad))
        self.assertNotIn(LABEL, p.intents)


class TestFileRegistration(unittest.TestCase):
    def test_copy_of_earlier_templates_after_a_file_write_is_applied(self):
        p = _make_prototype_pipeline()
        p.model.encode.side_effect = (
            lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32))
        p._handle_intent4_register_template(spec())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "lights.intent")
            with open(path, "w") as f:
                f.write("open the garage door\n")
            p._handle_register_padatious(Message(
                "padatious:register_intent",
                {"name": f"{LABEL}.intent", "file_name": path, "lang": "en-US"},
                {"skill_id": SKILL}))
        store = p.prototype_store
        with patch.object(store, "add", wraps=store.add) as add:
            p._handle_intent4_register_template(spec())
        self.assertEqual(add.call_count, 1)


class TestOverlappingRegistrationsOfOneLabel(unittest.TestCase):
    """Two registrations of one (label, lang) with different templates whose
    store writes land in another order than their claims, so the store holds
    A's rows while B was claimed last."""

    def setUp(self):
        self.p = _make_prototype_pipeline()
        self.p.model.encode.side_effect = (
            lambda sents, **kw: np.eye(len(sents), 4, dtype=np.float32))
        self.a = ["turn on the lights", "lights on"]
        self.b = ["switch the light on"]
        self.ta = self.p._claim_registration(LABEL, "en-US", self.a, SKILL)
        self.tb = self.p._claim_registration(LABEL, "en-US", self.b, SKILL)
        store = self.p.prototype_store
        store.add(self.p.model, LABEL, self.b, lang="en-US")
        store.add(self.p.model, LABEL, self.a, lang="en-US")

    def claim_b(self):
        return self.p._claim_registration(LABEL, "en-US", self.b, SKILL)

    def test_b_is_not_recorded_when_a_settles_first(self):
        self.p._settle_registration(self.ta)
        self.p._settle_registration(self.tb)
        self.assertIsNotNone(self.claim_b())

    def test_b_is_not_recorded_while_a_is_in_flight(self):
        self.p._settle_registration(self.tb)
        self.assertIsNotNone(self.claim_b())
