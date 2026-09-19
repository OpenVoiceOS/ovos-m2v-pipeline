"""T-3659: an exact template line of a label the frozen head cannot emit
belongs to the prototype stage, whatever the stage order is.

The live cell on ovoscope#211 put the classifier before the prototype stage
at every tier. Five of ovos-skill-volume's own template lines then routed to
`increase_volume` at 1.00, because the head cannot emit the dotted label
that declares them (`volume.max.boost`, `volume.unmute`) and answered with
the nearest label it can emit. An exact line is the skill author's declared
truth, so the head yields it.
"""

import time
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


#: the default ovos-config list carries all three m2v tiers
THREE_TIERS = ["ovos-stop-pipeline-plugin-high", "ovos-m2v-pipeline-high",
               "ovos-adapt-pipeline-plugin-high", "ovos-m2v-pipeline-medium",
               "ovos-adapt-pipeline-plugin-medium", "ovos-m2v-pipeline-low"]
#: the same list with the -low entry left out
NO_LOW = [s for s in THREE_TIERS if not s.endswith("m2v-pipeline-low")]
#: the dual layout: this head plus the standalone prototype plugin
STANDALONE = ["ovos-m2v-pipeline-high", "ovos-m2v-prototype-pipeline-high",
              "ovos-m2v-pipeline-medium", "ovos-m2v-prototype-pipeline-medium"]


def _unpinned_head(config=None, trained=(INCREASE,)):
    """A head whose `low_tier` is whatever the plugin defaults to.

    `tests.test_pipeline._make_pipeline` pins `low_tier: classifier`, which
    is what hid this defect: with the default (`prototype`) the instance
    owns a `_low_prototype` object whether or not the caller's session ever
    calls `-low`.
    """
    import sys
    from unittest.mock import MagicMock, patch

    from ovos_utils.fakebus import FakeBus

    config = dict(config or {})
    config.setdefault("model", "fake-model")
    head = MagicMock()
    head.classes_ = np.array(list(trained))
    head.predict_proba.return_value = np.array([[1.0] * len(trained)])
    fake_m2v = MagicMock()
    with patch("ovos_m2v_pipeline.StaticModelPipeline") as smp, \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        smp.from_pretrained.return_value = head
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)
    pipeline.model = head
    pipeline.intents = set(trained) | {BOOST, UNMUTE}
    return pipeline


class TestTheYieldNeedsTheStageInTheSession(unittest.TestCase):
    """The stage must be in the CALLER'S session pipeline.

    Owning a `_low_prototype` object proves nothing: with `low_tier`
    prototype (the default) every classifier instance owns one, and a
    session that lists `-high` and `-medium` alone never calls it. Yielding
    there leaves the utterance unanswered by this plugin.
    """

    def setUp(self):
        self.pipeline = _unpinned_head()
        self.assertIsNotNone(self.pipeline._low_prototype,
                             "the default low_tier owns a prototype stage")
        _register(self.pipeline, BOOST, ["crank [the] volume [up]"])

    def _match(self, session_pipeline):
        utterance = "crank the volume up"
        return self.pipeline.match_high([utterance], "en-US",
                                        _message(utterance, session_pipeline))

    def test_all_three_tiers_yield(self):
        self.assertIsNone(self._match(THREE_TIERS))

    def test_high_and_medium_only_keep_the_head_s_answer(self):
        """No -low entry, so nothing runs behind this one."""
        match = self._match(NO_LOW)
        self.assertIsNotNone(match, "yielding here answers nobody")
        self.assertEqual(match.match_type, INCREASE)

    def test_the_standalone_prototype_layout_yields(self):
        self.assertIsNone(self._match(STANDALONE))

    def test_medium_follows_the_same_rule(self):
        utterance = "crank the volume up"
        self.assertIsNone(self.pipeline.match_medium(
            [utterance], "en-US", _message(utterance, THREE_TIERS)))
        match = self.pipeline.match_medium(
            [utterance], "en-US", _message(utterance, NO_LOW))
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, INCREASE)


class TestExactLinesUnderThreads(unittest.TestCase):
    """Six threads: two remembering, two forgetting, two reading.

    A detach during a registration wave is ordinary skill-reload traffic,
    and an unguarded dict raises "dictionary changed size during iteration"
    within 0.1 s of it.
    """

    def _drive(self, pipeline, seconds=1.0):
        """Run the six threads for *seconds* and return what they raised."""
        import threading

        labels = [f"{SKILL}:label{i}" for i in range(40)]
        for label in labels:
            _register(pipeline, label, [f"line {label} one", f"line {label} two"])
        errors = []
        stop = threading.Event()

        def remember(offset):
            i = offset
            while not stop.is_set():
                label = labels[i % len(labels)]
                pipeline._remember_exact_lines(
                    label, [f"line {label} one", f"line {label} three"])
                i += 1

        def forget(offset):
            i = offset
            while not stop.is_set():
                pipeline._forget_exact_lines(labels[i % len(labels)])
                i += 1

        def read(offset):
            i = offset
            while not stop.is_set():
                label = labels[i % len(labels)]
                pipeline._exact_prototype_owner(f"line {label} one",
                                                _message("x"))
                i += 1

        def guarded(fn, offset):
            try:
                fn(offset)
            except Exception as exc:  # the race this test exists for
                errors.append(f"{type(exc).__name__}: {exc}")
                stop.set()

        threads = [threading.Thread(target=guarded, args=(fn, n))
                   for n, fn in enumerate([remember, remember, forget,
                                           forget, read, read])]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and not errors:
            time.sleep(0.01)
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        return errors

    def test_the_same_traffic_races_on_the_code_this_replaces(self):
        """Fail-before, against the code as it was.

        The defect was two things at once: no lock, and a forget that
        iterated the live map (`for k, labels in lines.items()`). This
        control puts both back on one instance and runs the same six
        threads. Removing only the lock is not the control: the snapshot
        this commit adds hides the race on its own.
        """
        import contextlib

        pipeline = _head()
        pipeline._exact_lines_lock = contextlib.nullcontext()

        def forget_as_it_was(label, _lines=pipeline._exact_lines):
            for key in [k for k, labels in _lines.items() if label in labels]:
                _lines[key].discard(label)
                if not _lines[key]:
                    _lines.pop(key, None)

        pipeline._forget_exact_lines = forget_as_it_was
        errors = self._drive(pipeline, seconds=5.0)
        self.assertTrue(errors, "the unguarded map did not race in 5s")
        self.assertTrue(any("changed size during iteration" in e
                            for e in errors), errors)

    def test_no_thread_raises(self):
        import threading

        pipeline = _head()
        labels = [f"{SKILL}:label{i}" for i in range(40)]
        for label in labels:
            _register(pipeline, label, [f"line {label} one", f"line {label} two"])
        errors = []
        stop = threading.Event()

        def remember(offset):
            i = offset
            while not stop.is_set():
                label = labels[i % len(labels)]
                pipeline._remember_exact_lines(
                    label, [f"line {label} one", f"line {label} three"])
                i += 1

        def forget(offset):
            i = offset
            while not stop.is_set():
                pipeline._forget_exact_lines(labels[i % len(labels)])
                i += 1

        def read(offset):
            i = offset
            while not stop.is_set():
                label = labels[i % len(labels)]
                pipeline._exact_prototype_owner(f"line {label} one",
                                                _message("x"))
                i += 1

        def guarded(fn, offset):
            try:
                fn(offset)
            except Exception as exc:  # the race this test exists for
                errors.append(f"{type(exc).__name__}: {exc}")
                stop.set()

        threads = [threading.Thread(target=guarded, args=(fn, n))
                   for n, fn in enumerate([remember, remember, forget,
                                           forget, read, read])]
        for thread in threads:
            thread.start()
        time.sleep(1.0)
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(errors, [])
