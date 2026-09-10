"""An entity belongs to one language, and a registration must name it.

OVOS-INTENT-4 §8.1: entities are "keyed on the quadruple ``(session_id,
skill_id, entity_name, lang)``". §3.2 marks ``lang`` required. A registration
that omits it is malformed: deriving the value from the session or from the
closest registered language would invent the field the producer left out.
"""
import unittest
from unittest import mock

from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus
from ovos_spec_tools import SpecMessage

from ovos_m2v_pipeline import Model2VecPrototypePipeline


def _pipeline():
    with mock.patch.object(Model2VecPrototypePipeline, "_ensure_model",
                           lambda self, **kw: None):
        return Model2VecPrototypePipeline(FakeBus(), {})


def _entity(pipe, values, lang="en-US", name="genre", skill_id="a.skill"):
    data = {"skill_id": skill_id, "entity_name": name, "samples": values}
    if lang is not None:
        data["lang"] = lang
    pipe._handle_intent4_register_entity(
        Message(SpecMessage.ENTITY_REGISTER.value, data))


class EntityLanguageKeyTest(unittest.TestCase):
    def setUp(self):
        self.pipe = _pipeline()

    def _values(self, lang="en-US", name="genre", skill_id="a.skill"):
        return (self.pipe.entities.get(skill_id, {})
                .get(lang, {}).get(name))

    def test_two_languages_of_one_name_coexist(self):
        _entity(self.pipe, ["jazz", "rock"], lang="en-US")
        _entity(self.pipe, ["fado"], lang="pt-PT")
        self.assertEqual(sorted(self._values("en-US")), ["jazz", "rock"])
        self.assertEqual(self._values("pt-PT"), ["fado"])

    def test_a_registration_without_lang_is_rejected(self):
        _entity(self.pipe, ["jazz"], lang=None)
        self.assertEqual(self.pipe.entities, {})

    def test_lang_is_compared_case_insensitively(self):
        _entity(self.pipe, ["jazz"], lang="EN-us")
        self.assertEqual(self._values("en-US"), ["jazz"])

    def test_a_deregister_naming_a_lang_leaves_the_others(self):
        _entity(self.pipe, ["jazz"], lang="en-US")
        _entity(self.pipe, ["fado"], lang="pt-PT")
        self.pipe._handle_intent4_deregister_entity(Message(
            SpecMessage.ENTITY_DEREGISTER.value,
            {"skill_id": "a.skill", "entity_name": "genre", "lang": "en-US"}))
        self.assertIsNone(self._values("en-US"))
        self.assertEqual(self._values("pt-PT"), ["fado"])

    def test_a_deregister_without_lang_removes_every_language(self):
        _entity(self.pipe, ["jazz"], lang="en-US")
        _entity(self.pipe, ["fado"], lang="pt-PT")
        self.pipe._handle_intent4_deregister_entity(Message(
            SpecMessage.ENTITY_DEREGISTER.value,
            {"skill_id": "a.skill", "entity_name": "genre"}))
        self.assertIsNone(self._values("en-US"))
        self.assertIsNone(self._values("pt-PT"))


if __name__ == "__main__":
    unittest.main()
