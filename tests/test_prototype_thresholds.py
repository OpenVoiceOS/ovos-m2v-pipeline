"""T-1621: prototype mode scores a cosine, not a softmax probability, so its
confidence tiers must not default to the classifier's 0.7 / 0.5 / 0.15.

The store holds one label. The query sits at a known cosine from that
label's only prototype, so each tier's claim is decided by the default
threshold alone. Before the fix ``match_medium`` claimed at cosine 0.6 and
``match_low`` at 0.3 (0.6 >= 0.5, 0.3 >= 0.15); a store that holds no
prototype for the utterance's real label always has *some* nearest label,
so those claims were misroutes.
"""
import math
import unittest

import numpy as np

from ovos_bus_client.message import Message

from tests.test_pipeline import _make_pipeline, _make_prototype_pipeline

LABEL = "skill-a.test:intent_a"
ANCHOR = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _query_at(cosine: float) -> np.ndarray:
    """A unit vector at exactly *cosine* from ``ANCHOR``."""
    return np.array([cosine, math.sqrt(1.0 - cosine ** 2), 0.0, 0.0],
                    dtype=np.float32)


def _pipeline(config=None, cosine=0.6):
    from ovos_m2v_pipeline import PrototypeIntentStore
    p = _make_prototype_pipeline(config=config, proto_store=PrototypeIntentStore())
    # one anchor for LABEL; every later encode() is the query
    p.model.encode.side_effect = [ANCHOR.reshape(1, -1)]
    p.prototype_store.add(p.model, LABEL, ["anchor sentence"])
    p.model.encode.side_effect = None
    p.model.encode.return_value = _query_at(cosine).reshape(1, -1)
    p._ensure_model = lambda *a, **k: True
    return p


def _msg():
    return Message("recognizer_loop:utterance", {"utterances": ["q"], "lang": "en-US"})


class TestPrototypeDefaults(unittest.TestCase):

    def test_defaults_per_mode(self):
        from ovos_m2v_pipeline import DEFAULT_CONF
        self.assertEqual(DEFAULT_CONF["classifier"],
                         {"conf_high": 0.7, "conf_medium": 0.5, "conf_low": 0.15})
        self.assertEqual(DEFAULT_CONF["prototype"],
                         {"conf_high": 0.85, "conf_medium": 0.7, "conf_low": 0.65})
        clf = _make_pipeline()
        self.assertEqual(clf._min_conf("conf_high"), 0.7)
        self.assertEqual(clf._min_conf("conf_medium"), 0.5)
        self.assertEqual(clf._min_conf("conf_low"), 0.15)
        proto = _make_prototype_pipeline()
        self.assertEqual(proto._min_conf("conf_high"), 0.85)
        self.assertEqual(proto._min_conf("conf_medium"), 0.7)
        self.assertEqual(proto._min_conf("conf_low"), 0.65)

    def test_medium_does_not_claim_at_cosine_0_6(self):
        # fail-before: 0.6 >= the classifier's 0.5 default claimed this
        p = _pipeline(cosine=0.6)
        self.assertIsNone(p.match_medium(["q"], "en-US", _msg()))
        self.assertIsNone(p.match_high(["q"], "en-US", _msg()))
        self.assertIsNone(p.match_low(["q"], "en-US", _msg()))

    def test_low_claims_at_cosine_0_68(self):
        p = _pipeline(cosine=0.68)
        self.assertIsNone(p.match_medium(["q"], "en-US", _msg()))
        match = p.match_low(["q"], "en-US", _msg())
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, LABEL)

    def test_low_does_not_claim_at_cosine_0_3(self):
        # fail-before: 0.3 >= the classifier's 0.15 default claimed this
        p = _pipeline(cosine=0.3)
        self.assertIsNone(p.match_low(["q"], "en-US", _msg()))

    def test_high_claims_at_cosine_0_9(self):
        p = _pipeline(cosine=0.9)
        match = p.match_high(["q"], "en-US", _msg())
        self.assertIsNotNone(match)
        self.assertEqual(match.match_type, LABEL)
        self.assertAlmostEqual(match.match_data["confidence"], 0.9, places=4)

    def test_configured_threshold_still_wins(self):
        # an explicit conf_medium keeps the old behaviour for a deployment
        # that tuned it
        p = _pipeline(config={"conf_medium": 0.5}, cosine=0.6)
        self.assertIsNotNone(p.match_medium(["q"], "en-US", _msg()))


if __name__ == "__main__":
    unittest.main()
