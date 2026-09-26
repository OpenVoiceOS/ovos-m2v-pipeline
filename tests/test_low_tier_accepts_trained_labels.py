"""What the ``-low`` prototype stage accepts, and when it is reached.

Review finding C1 on #251: the stage denies only ``ignore_intents``, so a
label the model was trained on is in the ``-low`` store as well as in the
head. That is the intent, and this file pins it: the stage holds the
trained label, and the head answers first at ``conf_high`` and
``conf_medium``, so the stage only ever sees an utterance both tiers
declined.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_utils.fakebus import FakeBus

from ovos_m2v_pipeline import (Model2VecIntentPipeline, Model2VecPrototypePipeline,
                               clear_shared_models)
from tests.test_exact_sample_match import _hash_encode

SKILL = "ovos-skill-alerts.openvoiceos"
TRAINED = f"{SKILL}:AddListSubitems"
SAMPLES = ["add milk to my shopping list", "put milk on the shopping list"]


class TestLowTierAcceptsTrainedLabels(unittest.TestCase):

    def setUp(self):
        clear_shared_models()

    def _classifier(self, config=None, confidence=0.95):
        """A head trained on TRAINED that answers it at *confidence*."""
        config = dict(config or {})
        config.setdefault("model", "fake-model")
        head = MagicMock(name="pipeline")
        embedding = MagicMock(name="embedding")
        embedding.encode.side_effect = _hash_encode
        head.model = embedding
        head.classes_ = np.array([TRAINED])
        head.predict_proba.return_value = np.array([[confidence]])
        fake_m2v = MagicMock()
        fake_m2v.StaticModel.from_pretrained.return_value = embedding
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as smp, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}), \
             patch.dict(sys.modules, {"model2vec": fake_m2v}):
            smp.from_pretrained.return_value = head
            pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)
        pipeline.model = head
        pipeline.intents = {TRAINED}
        low = pipeline._low_prototype
        low.model = embedding
        low._flush_pending_additions()
        return pipeline

    def _register(self, pipeline):
        pipeline.bus.emit(Message("padatious:register_intent",
                                  {"name": f"{TRAINED}.intent",
                                   "samples": SAMPLES, "lang": "en-US"},
                                  {"skill_id": SKILL}))
        pipeline._low_prototype._flush_pending_additions()

    def _message(self, utterance):
        return Message("recognizer_loop:utterance",
                       {"utterances": [utterance], "lang": "en-US"})

    def test_the_low_stage_is_prototype_mode_on_the_head_s_model(self):
        pipeline = self._classifier()
        self.assertIsInstance(pipeline._low_prototype,
                              Model2VecPrototypePipeline)
        self.assertIs(pipeline._low_prototype.model, pipeline.model.model)

    def test_a_trained_label_lands_in_the_low_store(self):
        """Only `ignore_intents` is denied to the stage. A trained class is
        registered like any other label."""
        pipeline = self._classifier()
        self._register(pipeline)
        self.assertIn(TRAINED,
                      list(pipeline._low_prototype.prototype_store.unique_labels))

    def test_an_ignored_label_does_not_land_in_the_low_store(self):
        pipeline = self._classifier({"ignore_intents": [TRAINED]})
        self.assertIn(TRAINED, pipeline._low_prototype.ignore_labels)
        self._register(pipeline)
        self.assertEqual(
            list(pipeline._low_prototype.prototype_store.unique_labels), [])

    def test_the_head_answers_a_trained_label_before_the_stage_is_reached(self):
        """The order is the reason the T-1621 warning does not apply at the
        `-low` position: for a trained label the head answers at
        `conf_high`, and `match_low` is never called."""
        pipeline = self._classifier(confidence=0.95)
        self._register(pipeline)
        pipeline._low_prototype.match_low = MagicMock(
            side_effect=AssertionError("the -low stage must not be reached"))
        match = pipeline.match_high(["add milk to my shopping list"], "en-US",
                                    self._message("add milk to my shopping list"))
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, TRAINED)
        pipeline._low_prototype.match_low.assert_not_called()

    def test_the_stage_answers_only_what_high_and_medium_declined(self):
        pipeline = self._classifier(confidence=0.2)
        self._register(pipeline)
        utterance = "add milk to my shopping list"
        message = self._message(utterance)
        self.assertIsNone(pipeline.match_high([utterance], "en-US", message))
        self.assertIsNone(pipeline.match_medium([utterance], "en-US", message))
        match = pipeline.match_low([utterance], "en-US", message)
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, TRAINED)
        self.assertGreaterEqual(match.match_data["confidence"], 0.99)


class TestLowPrototypeModelIsDiscarded(unittest.TestCase):
    """Review finding C2: the discard must not be silent."""

    def setUp(self):
        clear_shared_models()

    def _classifier(self, low_prototype):
        head = MagicMock(name="pipeline")
        head.model = MagicMock(name="embedding")
        head.classes_ = np.array([])
        fake_m2v = MagicMock()
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as smp, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}), \
             patch.dict(sys.modules, {"model2vec": fake_m2v}), \
             patch("ovos_m2v_pipeline.LOG") as log:
            smp.from_pretrained.return_value = head
            pipeline = Model2VecIntentPipeline(
                bus=FakeBus(), config={"model": "head-model",
                                       "low_prototype": low_prototype})
        return pipeline, log

    def test_a_model_key_inside_low_prototype_is_named_in_a_warning(self):
        pipeline, log = self._classifier({"model": "other/model"})
        self.assertEqual(pipeline._low_prototype._model_path_config(),
                         pipeline._model_path_config())
        warnings = [str(call.args[0]) for call in log.warning.call_args_list]
        named = [w for w in warnings if "other/model" in w]
        self.assertEqual(len(named), 1, warnings)
        self.assertIn("discarded", named[0])

    def test_the_same_model_is_no_warning(self):
        pipeline, log = self._classifier({"model": "head-model"})
        warnings = [str(call.args[0]) for call in log.warning.call_args_list]
        self.assertEqual([w for w in warnings if "discarded" in w], [])

    def test_no_model_key_is_no_warning(self):
        pipeline, log = self._classifier({"conf_low": 0.9})
        warnings = [str(call.args[0]) for call in log.warning.call_args_list]
        self.assertEqual([w for w in warnings if "discarded" in w], [])


if __name__ == "__main__":
    unittest.main()
