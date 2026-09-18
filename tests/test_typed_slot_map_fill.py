"""OVOS-INTENT-1 §5.6: the prototype plugin fills a declared typed slot
from the typed-slot map.

"An engine MAY use the map to constrain where ``{type:name}`` matches —
preferring or requiring a span the map lists for that type — and MAY
report the corresponding normalized value alongside the match. [...]
``Match.slots[name]`` remains the surface string in every case
(OVOS-PIPELINE-1 §4.3)".

m2v never extracts a slot value from the utterance. The map is the input
that lets it fill a declared ``{number:b}``: one entry of that type is
taken; of several, the one nearest the placeholder's position in the
template; none, the slot stays unfilled.
"""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage

from test_intent4 import _make_prototype_pipeline

SKILL = "light.skill"
LABEL = f"{SKILL}:brightness"
UTT = "set the brightness to twenty five please"
MAP = {"number": [{"span": [22, 33], "surface": "twenty five", "value": 25}]}


class TestTypedSlotMapFill(unittest.TestCase):

    def _register(self, p, samples, slot_types=None):
        data = {"skill_id": SKILL, "intent_name": "brightness", "lang": "en-US",
                "samples": samples}
        if slot_types is not None:
            data["slot_types"] = slot_types
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value, data=data,
            context={"skill_id": SKILL}))

    def _match(self, p, utterance=UTT, typed_slots=MAP, intent_context=None):
        p.model.encode.side_effect = None
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        sess = MagicMock()
        sess.intent_context = intent_context or {}
        sess.blacklisted_intents = []
        sess.blacklisted_skills = []
        data = {"utterances": [utterance], "lang": "en-US"}
        if typed_slots is not None:
            data["typed_slots"] = typed_slots
        with patch("ovos_m2v_pipeline.SessionManager.get", return_value=sess):
            return p.match_high([utterance], "en-US",
                                Message("recognizer_loop:utterance", data))

    def test_declared_types_are_stored_from_the_samples(self):
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {number:b}"])
        self.assertEqual(p._intent_slot_types.get(LABEL), {"b": "number"})

    def test_declared_types_are_stored_from_the_payload(self):
        # ovos-workshop strips the prefix and sends slot_types beside the
        # bare samples
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {b}"], slot_types={"b": "number"})
        self.assertEqual(p._intent_slot_types.get(LABEL), {"b": "number"})

    def test_one_entry_fills_the_slot_with_its_surface(self):
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {number:b}"])
        match = self._match(p)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_data.get("b"), "twenty five")

    def test_no_map_leaves_the_slot_unfilled(self):
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {number:b}"])
        match = self._match(p, typed_slots=None)
        self.assertIsNotNone(match)
        self.assertNotIn("b", match.match_data)

    def test_an_entry_of_another_type_does_not_fill(self):
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {number:b}"])
        match = self._match(p, typed_slots={
            "duration": [{"span": [22, 33], "surface": "twenty five", "value": 25.0}]})
        self.assertNotIn("b", match.match_data)

    def test_an_entry_whose_span_does_not_hold_is_ignored(self):
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {number:b}"])
        match = self._match(p, typed_slots={
            "number": [{"span": [0, 3], "surface": "twenty five", "value": 25}]})
        self.assertNotIn("b", match.match_data)

    def test_several_entries_take_the_one_nearest_the_template_position(self):
        # {number:a} sits early in the template, {number:b} late
        p = _make_prototype_pipeline()
        self._register(p, ["change {number:a} to {number:b}"])
        utt = "change 5 to 25"
        typed = {"number": [{"span": [7, 8], "surface": "5", "value": 5},
                            {"span": [12, 14], "surface": "25", "value": 25}]}
        match = self._match(p, utterance=utt, typed_slots=typed)
        self.assertEqual(match.match_data.get("a"), "5")
        self.assertEqual(match.match_data.get("b"), "25")

    def test_context_value_wins_over_the_map(self):
        # a live context entry is the deliberate value; the map only fills
        # what is still empty
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {number:b}"])
        match = self._match(p, intent_context={"b": {"value": "ten"}})
        self.assertEqual(match.match_data.get("b"), "ten")

    def test_a_malformed_map_is_ignored(self):
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {number:b}"])
        match = self._match(p, typed_slots={"number": "not a list"})
        self.assertIsNotNone(match)
        self.assertNotIn("b", match.match_data)

    def test_types_cleared_on_deregister(self):
        p = _make_prototype_pipeline()
        self._register(p, ["set the brightness to {number:b}"])
        p._handle_intent4_deregister_intent(Message(
            SpecMessage.INTENT_DEREGISTER.value,
            data={"skill_id": SKILL, "intent_name": "brightness", "lang": "en-US"},
            context={"skill_id": SKILL}))
        self.assertNotIn(LABEL, p._intent_slot_types)
