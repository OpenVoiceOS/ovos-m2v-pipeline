"""A slot with no entity values puts no placeholder text into a prototype.

OVOS-INTENT-4 §7: "An entity is an **optional value-set hint** for a
template-intent slot (INTENT-3 §5.2, INTENT-1 §5.4, INTENT-2 §4.3).
Registering an entity is **never** a precondition for an intent that
references the slot name; a slot with no entity still fills normally."

A free slot is open text, so ``say {sentence}`` must be embedded as ``say``
and not as the string ``"say {sentence}"``, which no speaker says.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage
from ovos_utils.fakebus import FakeBus

from ovos_m2v_pipeline import MAX_ENTITY_EXPANSIONS
from ovos_m2v_pipeline.cache import PrototypeCache, compute_cache_key
from ovos_m2v_pipeline.slots import drop_free_slots, expand_entities
from tests.test_intent4 import _make_prototype_pipeline

SKILL = "parrot.skill"


def _recording(config=None):
    """A mocked prototype pipeline that records every sentence it encodes."""
    p = _make_prototype_pipeline({"prototype_cache": False, **(config or {})})
    encoded = []
    p.model.encode.side_effect = lambda sents, **kw: (
        encoded.extend(sents), np.eye(len(sents), 4, dtype=np.float32))[1]
    return p, encoded


def _template(p, samples, intent="speak", lang="en-US"):
    p._handle_intent4_register_template(Message(
        SpecMessage.INTENT_REGISTER_TEMPLATE.value,
        data={"skill_id": SKILL, "intent_name": intent, "lang": lang,
              "samples": samples},
        context={"skill_id": SKILL}))


def _padatious(p, samples, intent="speak", lang="en-US"):
    p._handle_register_padatious(Message(
        "padatious:register_intent",
        data={"name": f"{SKILL}:{intent}", "lang": lang, "samples": samples},
        context={"skill_id": SKILL}))


class TestDropFreeSlots(unittest.TestCase):

    def test_placeholder_removed_and_whitespace_collapsed(self):
        self.assertEqual(
            drop_free_slots(["say {sentence}", "repeat {sentence} after me",
                             "{sentence} رو تکرار کن"]),
            ["say", "repeat after me", "رو تکرار کن"])

    def test_a_bare_slot_leaves_nothing(self):
        self.assertEqual(drop_free_slots(["{sentence}", " {a} {b} "]), [])

    def test_the_corpus_fill_keeps_the_placeholder(self):
        # train/build_dataset.py drops a row by the placeholder still in it
        self.assertEqual(expand_entities(["say {sentence}"], {"x": ["y"]}),
                         ["say {sentence}"])


class TestFreeSlotPrototypeText(unittest.TestCase):

    def test_template_topic_embeds_the_frame(self):
        p, encoded = _recording()
        _template(p, ["say {sentence}", "repeat {sentence} after me"])
        self.assertEqual(encoded, ["say", "repeat after me"])
        self.assertIn(f"{SKILL}:speak", p.intents)

    def test_padatious_topic_embeds_the_frame(self):
        p, encoded = _recording()
        _padatious(p, ["say {sentence}", "say good morning"])
        self.assertEqual(encoded, ["say", "say good morning"])
        self.assertIn(f"{SKILL}:speak", p.intents)

    def test_an_entity_slot_still_fills_beside_a_free_slot(self):
        p, encoded = _recording()
        p._handle_intent4_register_entity(Message(
            SpecMessage.ENTITY_REGISTER.value,
            data={"skill_id": SKILL, "entity_name": "engine", "lang": "en-US",
                  "samples": ["spotify", "youtube"]},
            context={"skill_id": SKILL}))
        _template(p, ["play {song} on {engine}"], intent="play")
        self.assertEqual(sorted(encoded), ["play on spotify", "play on youtube"])

    def test_a_bare_slot_sample_is_left_out_on_the_template_topic(self):
        # "[please] {sentence}" yields the samples "please {sentence}" and
        # "{sentence}"; the second has no word to embed
        p, encoded = _recording()
        _template(p, ["[please] {sentence}"])
        self.assertEqual(encoded, ["please"])

    def test_a_bare_slot_sample_is_left_out_on_the_padatious_topic(self):
        p, encoded = _recording()
        _padatious(p, ["[please] {sentence}"])
        self.assertEqual(encoded, ["please"])

    def test_a_slot_only_template_stays_malformed(self):
        # OVOS-INTENT-1 §3.6: a template MUST carry at least one literal word
        p, encoded = _recording()
        with patch("ovos_m2v_pipeline.LOG.warning"):
            _template(p, ["{sentence}"])
        self.assertEqual(encoded, [])
        self.assertNotIn(f"{SKILL}:speak", p.intents)


class TestCacheWrittenWithThePlaceholder(unittest.TestCase):
    """An on-disk entry encoded from ``"say {sentence}"`` must not be served
    for the same templates once the placeholder is dropped."""

    def test_an_entry_under_the_placeholder_key_is_re_encoded(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            p, encoded = _recording({"prototype_cache": True,
                                     "prototype_cache_dir": cache_dir})
            samples = ["say {sentence}"]
            placeholder_key = compute_cache_key(
                p._model_id, p._model2vec_version,
                {"k": p._prototype_k,
                 "strategy": p._prototype_strategy.value,
                 "max_expansions": MAX_ENTITY_EXPANSIONS},
                samples, {}, lang="en-US")
            PrototypeCache(Path(cache_dir)).save(
                f"{SKILL}:speak", placeholder_key,
                np.ones((1, 4), dtype=np.float32),
                np.array([f"{SKILL}:speak"]), lang="en-US")
            _template(p, samples)
        self.assertEqual(encoded, ["say"])


class TestPrebuiltExportFromASkill(unittest.TestCase):

    def test_export_embeds_the_frame(self):
        from ovos_m2v_pipeline.cli import _build_from_skill_dir
        encoded = []
        model = MagicMock()
        model.encode.side_effect = lambda sents, **kw: (
            encoded.extend(sents), np.eye(len(sents), 4, dtype=np.float32))[1]
        with tempfile.TemporaryDirectory() as skill_dir:
            locale = Path(skill_dir, "locale", "en-US")
            locale.mkdir(parents=True)
            (locale / "speak.intent").write_text("say {sentence}\nsay good morning\n")
            with patch("model2vec.StaticModel.from_pretrained", return_value=model):
                _build_from_skill_dir(Path(skill_dir), SKILL, "some/model", None,
                                      None, "max_over_all")
        self.assertEqual(sorted(encoded), ["say", "say good morning"])


class TestRealModelScoresAFreeSlotRequest(unittest.TestCase):
    """The default model, not a mock: a request to repeat a phrase reaches
    the low tier against the skill's free-slot templates alone. The requests
    are lowercase and carry no punctuation, so the score does not depend on
    how the query is normalized before it is embedded."""

    TEMPLATES = {
        "en-US": ["say {sentence}", "say after me {sentence}",
                  "repeat {sentence} after me"],
        "de-DE": ["sage {sentence}", "sag bitte {sentence}",
                  "sag mir nach {sentence}"],
        "fa-IR": ["بگو {sentence}", "به تقلید از من بگو {sentence}",
                  "{sentence} رو تکرار کن"],
    }
    REQUESTS = {
        "en-US": "say i love you after me",
        "de-DE": "sag mir nach ich liebe dich",
        "fa-IR": "به تقلید از من بگو دوستت دارم",
    }

    @classmethod
    def setUpClass(cls):
        from ovos_m2v_pipeline import Model2VecPrototypePipeline
        with patch("ovos_m2v_pipeline.Configuration", return_value={}):
            cls.p = Model2VecPrototypePipeline(bus=FakeBus(),
                                               config={"prototype_cache": False})
        assert cls.p._ensure_model(background_ok=False)
        for lang, samples in cls.TEMPLATES.items():
            _template(cls.p, samples, lang=lang)

    def test_each_request_matches_at_the_low_tier(self):
        for lang, utterance in self.REQUESTS.items():
            with self.subTest(lang=lang):
                message = Message("recognizer_loop:utterance",
                                  {"utterances": [utterance], "lang": lang})
                match = self.p.match_low([utterance], lang, message)
                self.assertIsNotNone(match, "no intent reached the low tier")
                self.assertEqual(match.match_type, f"{SKILL}:speak")
                self.assertGreaterEqual(match.match_data["confidence"],
                                        self.p._min_conf("conf_low"))


if __name__ == "__main__":
    unittest.main()
