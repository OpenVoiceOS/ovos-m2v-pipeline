"""``mask_free_text_slots``: a template's free-text slot is not words.

A slot with no registered entity keeps its placeholder after
``_expand_entities``, so ``speak {sentence}`` is embedded as the two words
"speak sentence". Where the slot IS most of the utterance, the prototype
then sits far from anything a user says: measured on ten skills and 140
en-US gold rows, 48 of 71 such rows matched, and every miss was on a label
whose slot dominates (parrot `speak`, icanhazdadjokes `search_joke`,
application-launcher `launch`). The audit is
`knowledge/wiki/audits/m2v-training/prototype-free-text-slots.md`.
"""
import time
import unittest
from unittest import mock

import numpy as np

from ovos_bus_client.message import Message

from tests.test_exact_sample_match import _hash_encode
from tests.test_pipeline import _make_prototype_pipeline

SKILL = "ovos-skill-parrot.openvoiceos"
SPEAK = f"{SKILL}:speak"
STOP = f"{SKILL}:stop_parrot"
JOKES = "ovos-skill-icanhazdadjokes.openvoiceos"
SEARCH_JOKE = f"{JOKES}:search_joke"
JOKE = f"{JOKES}:joke"


def _pipeline(**config):
    """A prototype pipeline whose deferred load returns the same mock the
    test patched, so every embedded text is recorded on it."""
    config.setdefault("conf_low", 0.5)
    config.setdefault("conf_medium", 0.5)
    config.setdefault("conf_high", 0.5)
    pipeline = _make_prototype_pipeline(config=config)
    pipeline.model.encode.side_effect = _hash_encode
    return pipeline


def _register(pipeline, label, samples):
    """Register and wait for the handler thread to store the prototypes."""
    before = len(pipeline.model.encode.call_args_list)
    pipeline.bus.emit(Message("padatious:register_intent",
                              {"name": f"{label}.intent", "samples": samples,
                               "lang": "en-US"},
                              {"skill_id": label.split(":", 1)[0]}))
    deadline = time.monotonic() + 5
    while (len(pipeline.model.encode.call_args_list) == before
           and time.monotonic() < deadline):
        time.sleep(0.01)


def _prototype_of(pipeline, text):
    """Is *text* one of the prototypes the store holds?

    Read from the store, not from the mock, so the assertion holds whoever
    did the encoding: `_hash_encode` is deterministic, so the row of a
    stored text is its normalised hash vector and nothing else is.
    """
    rows = pipeline.prototype_store.embeddings
    if not len(rows):
        return False
    want = _hash_encode([text])[0]
    want = want / np.linalg.norm(want)
    return any(np.allclose(row, want, atol=1e-6) for row in rows)


def _match(pipeline, utterance):
    message = Message("recognizer_loop:utterance",
                      {"utterances": [utterance], "lang": "en-US"})
    return pipeline.match_low([utterance], "en-US", message)


class TestMaskFreeTextSlots(unittest.TestCase):

    def test_the_key_is_off_by_default(self):
        pipeline = _pipeline()
        self.assertFalse(pipeline._mask_free_text_slots)
        _register(pipeline, SPEAK, ["speak {sentence}"])
        self.assertTrue(_prototype_of(pipeline, "speak {sentence}"))
        self.assertFalse(_prototype_of(pipeline, "speak"))

    def test_the_slot_is_dropped_when_the_key_is_on(self):
        pipeline = _pipeline(mask_free_text_slots=True)
        _register(pipeline, SPEAK, ["speak {sentence}", "say {sentence} after me"])
        self.assertTrue(_prototype_of(pipeline, "speak"))
        self.assertTrue(_prototype_of(pipeline, "say after me"))
        self.assertFalse(_prototype_of(pipeline, "speak {sentence}"))

    def test_a_slot_with_an_entity_behind_it_is_untouched(self):
        """`_expand_entities` fills it, so there is no placeholder left to
        mask and the values stay in the prototype."""
        pipeline = _pipeline(mask_free_text_slots=True)
        pipeline.bus.emit(Message("ovos.entity.register",
                                  {"skill_id": SKILL, "entity_name": "colour",
                                   "samples": ["red", "blue"]},
                                  {"skill_id": SKILL}))
        _register(pipeline, SPEAK, ["paint it {colour}"])
        self.assertFalse(_prototype_of(pipeline, "paint it"))
        self.assertTrue(_prototype_of(pipeline, "paint it red"))

    def test_a_template_with_no_slot_is_untouched(self):
        pipeline = _pipeline(mask_free_text_slots=True)
        _register(pipeline, STOP, ["stop parrot", "stop repeating"])
        self.assertTrue(_prototype_of(pipeline, "stop parrot"))
        self.assertTrue(_prototype_of(pipeline, "stop repeating"))

    def test_a_masked_line_that_is_another_label_s_own_line_stays_a_template(self):
        """The exclusion. `tell me a joke about {query}` masks to `tell me a
        joke about`, which is what the plain `joke` label declares, so the
        template keeps its placeholder instead of taking that label's rows."""
        pipeline = _pipeline(mask_free_text_slots=True)
        _register(pipeline, JOKE, ["tell me a joke about"])
        _register(pipeline, SEARCH_JOKE, ["tell me a joke about {query}"])
        self.assertTrue(_prototype_of(pipeline, "tell me a joke about {query}"),
                        "the template kept its placeholder")
        self.assertTrue(_prototype_of(pipeline, "tell me a joke about"),
                        "the plain label's own line is still there")

    def test_the_same_label_may_mask_onto_its_own_line(self):
        pipeline = _pipeline(mask_free_text_slots=True)
        _register(pipeline, SEARCH_JOKE,
                  ["tell me a joke about", "tell me a joke about {query}"])
        self.assertFalse(_prototype_of(pipeline, "tell me a joke about {query}"))

    def test_the_plain_line_wins_when_it_registers_last(self):
        """Registration order decides nothing: the label that declares the
        line as its own takes the utterance, whichever came first."""
        pipeline = _pipeline(mask_free_text_slots=True)
        _register(pipeline, SEARCH_JOKE, ["tell me a joke about {query}"])
        _register(pipeline, JOKE, ["tell me a joke about"])
        match = _match(pipeline, "tell me a joke about")
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, JOKE)

    def test_an_ignored_label_does_not_win_on_its_line(self):
        pipeline = _pipeline(mask_free_text_slots=True, ignore_intents=[JOKE])
        _register(pipeline, SEARCH_JOKE, ["tell me a joke about {query}"])
        _register(pipeline, JOKE, ["tell me a joke about"])
        match = _match(pipeline, "tell me a joke about")
        self.assertTrue(match is None or match.match_type != JOKE)


class TestParrotSpeakFailsBefore(unittest.TestCase):
    """The case the mechanism was built for, both ways round."""

    def _parrot(self, **config):
        pipeline = _pipeline(**config)
        _register(pipeline, SPEAK, ["speak {sentence}"])
        _register(pipeline, STOP, ["stop parrot", "stop repeating"])
        return pipeline

    def test_the_slot_template_misses_with_the_key_off(self):
        pipeline = self._parrot()
        self.assertTrue(_prototype_of(pipeline, "speak {sentence}"),
                        "the placeholder is in the store")
        self.assertIsNone(_match(pipeline, "speak hello world"))

    def test_the_masked_template_matches_with_the_key_on(self):
        pipeline = self._parrot(mask_free_text_slots=True)
        match = _match(pipeline, "speak hello world")
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, SPEAK)

    def test_a_slot_free_row_still_matches_with_the_key_on(self):
        pipeline = self._parrot(mask_free_text_slots=True)
        match = _match(pipeline, "stop parrot")
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, STOP)


if __name__ == "__main__":
    unittest.main()


class TestTheKeyAndThePrototypeCache(unittest.TestCase):
    """The key decides what is embedded, so it belongs in the cache key.
    It belongs there only when it is on: adding a field unconditionally
    changes every key written before this release, and every prebuilt
    artifact and cache entry then misses for nothing."""

    def test_the_key_off_hashes_what_it_always_hashed(self):
        off = _pipeline()
        params = []
        with mock.patch("ovos_m2v_pipeline.compute_cache_key",
                        side_effect=lambda *a, **k: params.append(a[2]) or "x"):
            off._prototype_cache_key(["speak {sentence}"])
        self.assertNotIn("mask_free_text_slots", params[0])

    def test_the_key_on_hashes_itself_in(self):
        on = _pipeline(mask_free_text_slots=True)
        params = []
        with mock.patch("ovos_m2v_pipeline.compute_cache_key",
                        side_effect=lambda *a, **k: params.append(a[2]) or "x"):
            on._prototype_cache_key(["speak {sentence}"])
        self.assertTrue(params[0]["mask_free_text_slots"])

    def test_the_two_settings_never_share_a_cache_entry(self):
        off = _pipeline()
        on = _pipeline(mask_free_text_slots=True)
        samples = ["speak {sentence}"]
        self.assertNotEqual(off._prototype_cache_key(samples),
                            on._prototype_cache_key(samples))
