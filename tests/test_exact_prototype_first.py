"""T-3659: an exact template line of a label the frozen head cannot emit
belongs to the prototype stage, whatever the stage order is.

The live cell on ovoscope#211 put the classifier before the prototype stage
at every tier. Five of ovos-skill-volume's own template lines then routed to
`increase_volume` at 1.00, because the head cannot emit the dotted label
that declares them (`volume.max.boost`, `volume.unmute`) and answered with
the nearest label it can emit. An exact line is the skill author's declared
truth, so the head yields it.
"""

import unittest

import numpy as np
from ovos_bus_client.message import Message
from ovos_bus_client.session import Session

from tests.test_pipeline import _make_pipeline

SKILL = "ovos-skill-volume.openvoiceos"
BOOST = f"{SKILL}:volume.max.boost"
UNMUTE = f"{SKILL}:volume.unmute"
INCREASE = f"{SKILL}:increase_volume"
#: the dual layout of M2V_DUAL_PIPELINE: this head, then the prototype stage
DUAL = ["ovos-m2v-pipeline-high", "ovos-m2v-prototype-pipeline-high",
        "ovos-m2v-pipeline-medium", "ovos-m2v-prototype-pipeline-medium"]


def _message(utterance, pipeline=DUAL):
    session = Session(session_id="t3659", pipeline=list(pipeline))
    return Message("recognizer_loop:utterance",
                   {"utterances": [utterance], "lang": "en-US"},
                   {"session": session.serialize()})


def _head(config=None, trained=(INCREASE,), answer=INCREASE):
    """A classifier-mode pipeline whose head knows *trained* only.

    `answer` is what the head predicts for every utterance, at 1.00: the
    live cell's `increase_volume@1.00` on `crank the volume up`.
    """
    pipeline = _make_pipeline(config=config, intents=list(trained) + [BOOST, UNMUTE])
    pipeline.model.classes_ = np.array(list(trained))
    probs = np.array([[1.0 if label == answer else 0.0 for label in trained]])
    pipeline.model.predict_proba.return_value = probs
    return pipeline


def _register(pipeline, label, samples):
    pipeline.bus.emit(Message("padatious:register_intent",
                              {"name": f"{label}.intent", "samples": samples,
                               "lang": "en-US"},
                              {"skill_id": SKILL}))


class TestExactPrototypeFirst(unittest.TestCase):

    def test_the_head_answers_an_exact_prototype_owned_line_without_the_fix(self):
        """The live fail, reproduced: `exact_prototype_first: False` is the
        behaviour harness-b measured on ovoscope#211."""
        pipeline = _head(config={"exact_prototype_first": False})
        _register(pipeline, BOOST, ["crank [the] volume [up]"])
        match = pipeline.match_high(["crank the volume up"], "en-US",
                                    _message("crank the volume up"))
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, INCREASE)
        self.assertEqual(match.match_data["confidence"], 1.0)

    def test_an_exact_prototype_owned_line_is_yielded_at_the_high_tier(self):
        pipeline = _head()
        _register(pipeline, BOOST, ["crank [the] volume [up]"])
        for utterance in ("crank the volume up", "crank volume up",
                          "crank the volume", "crank volume"):
            with self.subTest(utterance=utterance):
                self.assertIsNone(pipeline.match_high([utterance], "en-US",
                                                      _message(utterance)))

    def test_the_line_is_yielded_at_the_medium_tier_too(self):
        """The `low_tier: prototype` layout answers below medium, so the
        head must yield there as well."""
        pipeline = _head()
        _register(pipeline, UNMUTE, ["turn volume back on"])
        self.assertIsNone(pipeline.match_medium(["turn volume back on"],
                                                "en-US",
                                                _message("turn volume back on")))

    def test_case_and_punctuation_do_not_decide(self):
        pipeline = _head()
        _register(pipeline, BOOST, ["crank [the] volume [up]"])
        self.assertIsNone(pipeline.match_high(["Crank the volume, up!"],
                                              "en-US",
                                              _message("Crank the volume, up!")))

    def test_an_exact_line_of_a_label_the_head_can_emit_still_matches(self):
        """The five lines ovoscope#211 gains: their label is trained, so the
        head keeps them."""
        pipeline = _head()
        _register(pipeline, INCREASE,
                  ["turn up [the] volume", "turn [the] volume up", "crank it up"])
        for utterance in ("turn up the volume", "turn the volume up",
                          "crank it up"):
            with self.subTest(utterance=utterance):
                match = pipeline.match_high([utterance], "en-US",
                                            _message(utterance))
                self.assertIsNotNone(match)
                self.assertEqual(match.match_type, INCREASE)

    def test_an_utterance_that_is_no_exact_line_still_gets_the_head(self):
        pipeline = _head()
        _register(pipeline, BOOST, ["crank [the] volume [up]"])
        match = pipeline.match_high(["make the volume really loud"], "en-US",
                                    _message("make the volume really loud"))
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, INCREASE)

    def test_a_line_two_labels_declare_stays_with_the_head(self):
        """One of the two owners is trained, so the line is ambiguous and
        the head is allowed to answer it."""
        pipeline = _head()
        _register(pipeline, BOOST, ["turn it way up"])
        _register(pipeline, INCREASE, ["turn it way up"])
        match = pipeline.match_high(["turn it way up"], "en-US",
                                    _message("turn it way up"))
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, INCREASE)

    def test_no_prototype_stage_means_no_yield(self):
        """With nothing behind it, yielding would only lose the utterance."""
        pipeline = _head()
        _register(pipeline, BOOST, ["crank [the] volume [up]"])
        lonely = ["ovos-m2v-pipeline-high", "ovos-m2v-pipeline-medium"]
        match = pipeline.match_high(["crank the volume up"], "en-US",
                                    _message("crank the volume up", lonely))
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, INCREASE)

    def test_a_line_with_a_slot_is_no_exact_line(self):
        pipeline = _head()
        _register(pipeline, BOOST, ["set the volume to {level}"])
        match = pipeline.match_high(["set the volume to eleven"], "en-US",
                                    _message("set the volume to eleven"))
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, INCREASE)

    def test_a_detached_intent_takes_its_lines_with_it(self):
        pipeline = _head()
        _register(pipeline, BOOST, ["crank [the] volume [up]"])
        self.assertIsNone(pipeline.match_high(["crank the volume up"], "en-US",
                                              _message("crank the volume up")))
        pipeline.bus.emit(Message("detach_intent",
                                  {"intent_name": f"{BOOST}.intent"},
                                  {"skill_id": SKILL}))
        match = pipeline.match_high(["crank the volume up"], "en-US",
                                    _message("crank the volume up"))
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, INCREASE)

    def test_a_detached_skill_takes_its_lines_with_it(self):
        pipeline = _head()
        _register(pipeline, BOOST, ["crank [the] volume [up]"])
        self.assertTrue(pipeline._exact_lines)
        pipeline.bus.emit(Message("detach_skill", {"skill_id": SKILL},
                                  {"skill_id": SKILL}))
        # the manifest sync that rides the same topic empties `intents`
        # here, so the store itself is what this asserts
        self.assertEqual(pipeline._exact_lines, {})


if __name__ == "__main__":
    unittest.main()
