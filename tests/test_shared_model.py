"""One embedding model per process, and the ``-low`` tier is prototype mode.

The mocked tests prove the cache contract in both load orders. The live
test boots the three pipeline ids a default deployment can carry
(``ovos-m2v-pipeline`` in classifier mode with its ``-low`` prototype
stage, and ``ovos-m2v-prototype-pipeline``) on the published model and
asserts one load, one object, with RSS read before and after each boot.
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_utils.fakebus import FakeBus

from ovos_m2v_pipeline import (DEFAULT_MULTILINGUAL, Model2VecIntentPipeline,
                               Model2VecPrototypePipeline,
                               clear_shared_models, load_shared_model,
                               shared_model_stats)


def _fake_pipeline():
    embedding = MagicMock(name="embedding")
    pipeline = MagicMock(name="pipeline")
    pipeline.model = embedding
    return pipeline, embedding


class TestLoadSharedModel(unittest.TestCase):

    def setUp(self):
        clear_shared_models()

    def test_classifier_first_shares_its_embedding(self):
        pipeline, embedding = _fake_pipeline()
        fake_m2v = MagicMock()
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as smp, \
             patch.dict(sys.modules, {"model2vec": fake_m2v}):
            smp.from_pretrained.return_value = pipeline
            self.assertIs(load_shared_model("/m", "classifier"), pipeline)
            self.assertIs(load_shared_model("/m", "prototype"), embedding)
            self.assertIs(load_shared_model("/m", "classifier"), pipeline)
            smp.from_pretrained.assert_called_once_with("/m")
            fake_m2v.StaticModel.from_pretrained.assert_not_called()
        self.assertEqual(shared_model_stats(), {"models": 1, "loads": 1})

    def test_prototype_first_repoints_the_classifier(self):
        pipeline, copy = _fake_pipeline()
        fake_m2v = MagicMock()
        first = MagicMock(name="first-embedding")
        fake_m2v.StaticModel.from_pretrained.return_value = first
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as smp, \
             patch.dict(sys.modules, {"model2vec": fake_m2v}):
            smp.from_pretrained.return_value = pipeline
            self.assertIs(load_shared_model("/m", "prototype"), first)
            got = load_shared_model("/m", "classifier")
            self.assertIs(got, pipeline)
            # the copy the pipeline loaded is dropped for the one in memory
            self.assertIs(pipeline.model, first)
            self.assertIs(load_shared_model("/m", "prototype"), first)
        self.assertEqual(shared_model_stats(), {"models": 1, "loads": 1})

    def test_another_path_is_another_model(self):
        p1, _ = _fake_pipeline()
        p2, _ = _fake_pipeline()
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as smp:
            smp.from_pretrained.side_effect = [p1, p2]
            self.assertIs(load_shared_model("/a", "classifier"), p1)
            self.assertIs(load_shared_model("/b", "classifier"), p2)
        self.assertEqual(shared_model_stats(), {"models": 2, "loads": 2})

    def test_a_failed_load_caches_nothing(self):
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as smp:
            smp.from_pretrained.side_effect = OSError("no such model")
            with self.assertRaises(OSError):
                load_shared_model("/m", "classifier")
        self.assertEqual(shared_model_stats(), {"models": 0, "loads": 0})


class TestLowTier(unittest.TestCase):
    """``low_tier`` on the classifier plugin."""

    def _classifier(self, config):
        config = dict(config)
        config.setdefault("model", "fake-model")
        pipeline, _ = _fake_pipeline()
        pipeline.classes_ = np.array([])
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as smp, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}):
            smp.from_pretrained.return_value = pipeline
            return Model2VecIntentPipeline(bus=FakeBus(), config=config)

    def test_default_low_tier_is_a_prototype_stage_on_the_same_model(self):
        p = self._classifier({"ignore_intents": ["skill:x"]})
        low = p._low_prototype
        self.assertIsInstance(low, Model2VecPrototypePipeline)
        self.assertEqual(low._model_path, p._model_path)
        self.assertIn("skill:x", low.ignore_labels)
        self.assertIs(low.bus, p.bus)

    def test_match_low_is_answered_by_the_prototype_stage(self):
        p = self._classifier({})
        sentinel = object()
        p._low_prototype.match_low = MagicMock(return_value=sentinel)
        msg = MagicMock()
        self.assertIs(p.match_low(["hello"], "en", msg), sentinel)
        p._low_prototype.match_low.assert_called_once_with(["hello"], "en", msg)

    def test_low_tier_classifier_keeps_the_head(self):
        p = self._classifier({"low_tier": "classifier"})
        self.assertIsNone(p._low_prototype)

    def test_low_prototype_keys_reach_the_stage(self):
        p = self._classifier({"low_prototype": {"conf_low": 0.9,
                                                "ignore_intents": ["a:b"]},
                              "ignore_intents": ["c:d"]})
        low = p._low_prototype
        self.assertEqual(low.config.get("conf_low"), 0.9)
        self.assertEqual(sorted(low.ignore_labels), ["a:b", "c:d"])

    def test_an_unknown_low_tier_is_refused(self):
        with self.assertRaises(ValueError):
            self._classifier({"low_tier": "keyword"})


def _rss_mb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


@unittest.skipUnless(os.environ.get("OVOSCOPE_LIVE") == "1",
                     "live model test; set OVOSCOPE_LIVE=1 to enable")
class TestThreeIdsOneModel(unittest.TestCase):
    """Three pipeline ids, the published model, one load, one object."""

    def test_three_ids_one_model(self):
        clear_shared_models()
        bus = FakeBus()
        rows = [("boot", _rss_mb())]
        with patch("ovos_m2v_pipeline.Configuration", return_value={}):
            clf = Model2VecIntentPipeline(bus, {"model": DEFAULT_MULTILINGUAL})
            proto = Model2VecPrototypePipeline(bus, {"model": DEFAULT_MULTILINGUAL})
        self.assertTrue(clf._ensure_model(background_ok=False))
        rows.append(("ovos-m2v-pipeline (-high, -medium)", _rss_mb()))
        self.assertTrue(clf._low_prototype._ensure_model(background_ok=False))
        rows.append(("ovos-m2v-pipeline-low prototype stage", _rss_mb()))
        self.assertTrue(proto._ensure_model(background_ok=False))
        rows.append(("ovos-m2v-prototype-pipeline", _rss_mb()))
        stats = shared_model_stats()
        self.assertEqual(stats["loads"], 1, stats)
        self.assertEqual(stats["models"], 1, stats)
        self.assertIs(clf.model.model, clf._low_prototype.model)
        self.assertIs(clf.model.model, proto.model)
        report = "\n".join(f"{name}: {mb:.1f} MB" for name, mb in rows)
        print("\nRSS\n" + report)
        path = os.environ.get("M2V_RSS_REPORT")
        if path:
            with open(path, "w") as fh:
                fh.write(report + "\n")
        # the second and third consumers add no embedding matrix: the two
        # extra boots together stay under a tenth of the first load
        first = rows[1][1] - rows[0][1]
        extra = rows[3][1] - rows[1][1]
        self.assertLess(extra, max(first * 0.1, 5.0), report)
