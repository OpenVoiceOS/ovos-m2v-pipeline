"""The utterance and the prototypes are scored in the OVOS-INTENT-1 §2
normalized form."""

import hashlib
import unittest

import numpy as np
from ovos_bus_client.message import Message

from tests.test_pipeline import _make_prototype_pipeline


def _cased_encode(sentences, **kwargs):
    """Bag-of-words hashing encoder that keeps case and punctuation, as a
    cased embedding vocabulary does: ``What`` and ``what`` are different
    tokens, and so are ``it?`` and ``it``."""
    dim = 64
    out = np.zeros((len(sentences), dim), dtype=np.float32)
    for i, sent in enumerate(sentences):
        for tok in sent.split():
            out[i, int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim] += 1.0
    return out


class TestQueryNormalized(unittest.TestCase):

    def _match(self, utterance):
        pipeline = _make_prototype_pipeline()
        pipeline.model.encode.side_effect = _cased_encode
        pipeline.bus.emit(Message(
            "padatious:register_intent",
            {"name": "skill:time.intent", "samples": ["what time is it", "tell me the time"],
             "lang": "en-US"},
            {"skill_id": "skill"}))
        msg = Message("recognizer_loop:utterance", {"utterances": [utterance], "lang": "en-US"})
        return pipeline.match_high([utterance], "en-US", msg)

    def test_capitalised_utterance_scores_as_its_lowercase_form(self):
        match = self._match("What Time Is It")
        self.assertEqual(match.match_type if match else None, "skill:time")
        self.assertGreaterEqual(match.match_data["confidence"], 0.99)

    def test_punctuated_utterance_scores_without_its_punctuation(self):
        match = self._match("what time is it?")
        self.assertEqual(match.match_type if match else None, "skill:time")
        self.assertGreaterEqual(match.match_data["confidence"], 0.99)

    def test_match_reports_the_utterance_as_spoken(self):
        match = self._match("What Time Is It")
        self.assertEqual(match.utterance, "What Time Is It")


class TestPrototypeNormalized(unittest.TestCase):
    """A template line with capitals or punctuation embeds as its normalized
    form, on both registration topics."""

    SAMPLES = ["What Time Is It?", "Tell me the time, please"]

    def _pipeline(self, message):
        pipeline = _make_prototype_pipeline()
        pipeline.model.encode.side_effect = _cased_encode
        pipeline.bus.emit(message)
        return pipeline

    def _match(self, pipeline, utterance):
        msg = Message("recognizer_loop:utterance", {"utterances": [utterance], "lang": "en-US"})
        return pipeline.match_high([utterance], "en-US", msg)

    def _assert_matches(self, pipeline, label):
        for utterance in ("what time is it", "tell me the time please"):
            with self.subTest(utterance=utterance):
                match = self._match(pipeline, utterance)
                self.assertEqual(match.match_type if match else None, label)
                self.assertGreaterEqual(match.match_data["confidence"], 0.99)

    def test_legacy_topic(self):
        pipeline = self._pipeline(Message(
            "padatious:register_intent",
            {"name": "skill:time.intent", "samples": self.SAMPLES, "lang": "en-US"},
            {"skill_id": "skill"}))
        self._assert_matches(pipeline, "skill:time")

    def test_template_topic(self):
        pipeline = self._pipeline(Message(
            "ovos.intent.register.template",
            {"skill_id": "skill", "intent_name": "time", "lang": "en-US",
             "samples": self.SAMPLES},
            {"skill_id": "skill"}))
        self._assert_matches(pipeline, "skill:time")


if __name__ == "__main__":
    unittest.main()
