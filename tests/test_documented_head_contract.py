"""Pins the documented behaviour of the classifier's own ``match_high``,
independent of padacioso.

The plugin's docs (docs/ovos_pipeline.md, "Place Padacioso Before the
Classifier") say a frozen classifier answers only with labels it was
trained on, and recommend running ``ovos-padacioso-pipeline-plugin-high``
ahead of the classifier so an exact template line of an untrained skill is
claimed before the classifier ever sees it. That ordering is what protects
the classifier from ever being asked about labels outside its training set;
it is enforced by the deployment's ``pipeline`` list, not by this plugin.

This test does not exercise padacioso at all. It pins the other half of the
contract: for a label the classifier WAS trained on, ``match_high`` answers
on its own, with no padacioso stage involved.
"""

import unittest

from ovos_bus_client.message import Message

from tests.test_pipeline import _make_pipeline, _setup_model


class TestClassifierAnswersTrainedLabelAlone(unittest.TestCase):
    """match_high answers a trained label without any earlier pipeline stage."""

    def test_match_high_answers_trained_label(self):
        pipeline = _make_pipeline(
            intents=["ovos-skill-volume.openvoiceos:increase_volume"],
            renormalize=False)
        _setup_model(
            pipeline,
            ["ovos-skill-volume.openvoiceos:increase_volume"],
            [1.00])
        msg = Message("recognizer_loop:utterance")

        match = pipeline.match_high(["crank the volume up"], "en-US", msg)

        self.assertIsNotNone(match)
        self.assertEqual(
            match.match_type,
            "ovos-skill-volume.openvoiceos:increase_volume")
        self.assertAlmostEqual(match.match_data["confidence"], 1.00)


if __name__ == "__main__":
    unittest.main()
