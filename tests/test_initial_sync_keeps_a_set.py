"""``_initial_intent_sync`` must leave ``self.intents`` a set.

When a skill registered before this pipeline was built, the startup
manifest pull is non-empty and ``self.intents`` was replaced by a list.
The OVOS-INTENT-4 template handler then called ``self.intents.add(label)``
inside the bus handler and raised ``AttributeError``.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage


def _classifier_with_seeded_manifest(adapt, padatious):
    """Classifier pipeline whose startup manifest pull returns the given
    intent names, as it does when skills loaded before the plugin."""
    config = {"model": "fake-model"}
    mock_model = MagicMock()
    mock_model.classes_ = np.array([])
    mock_model.predict_proba.return_value = np.array([[]])
    with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch("ovos_m2v_pipeline.Model2VecIntentPipeline._get_adapt_intents",
               return_value=list(adapt)), \
         patch("ovos_m2v_pipeline.Model2VecIntentPipeline._get_padatious_intents",
               return_value=list(padatious)):
        MockSMP.from_pretrained.return_value = mock_model
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)
    pipeline.model = mock_model
    return pipeline


class TestInitialSyncKeepsASet(unittest.TestCase):

    def test_seeded_intents_are_a_set(self):
        p = _classifier_with_seeded_manifest(["a.skill:one"], ["b.skill:two", "a.skill:one"])
        self.assertIsInstance(p.intents, set)
        self.assertEqual(p.intents, {"a.skill:one", "b.skill:two"})

    def test_template_registration_after_a_seeded_start_does_not_raise(self):
        p = _classifier_with_seeded_manifest(["a.skill:one"], [])
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": "c.skill", "intent_name": "three", "lang": "en-US",
                  "samples": ["three please"]},
            context={"skill_id": "c.skill"}))
        self.assertIn("c.skill:three", p.intents)
        self.assertIn("a.skill:one", p.intents)

    def test_empty_manifests_keep_the_empty_set(self):
        p = _classifier_with_seeded_manifest([], [])
        self.assertEqual(p.intents, set())
