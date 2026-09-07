"""Unit tests for the m2v x nebulento fusion pipeline
(``ovos_m2v_pipeline.fusion``).
"""
import hashlib
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def _hash_encode(sentences, **kwargs):
    """Deterministic bag-of-words hashing encoder (same trick used by
    ``tests/test_prototype_cache.py``): identical sentences produce
    identical vectors so tests don't depend on real model2vec weights."""
    dim = 16
    out = np.zeros((len(sentences), dim), dtype=np.float32)
    for i, sent in enumerate(sentences):
        for tok in sent.lower().split():
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            out[i, bucket] += 1.0
    return out


def _make_pipeline(extra_config=None):
    """Fresh fusion-pipeline instance with a mocked m2v embedding model and
    disk prototype caching disabled (fresh per-process cache dir, and
    ``prototype_cache: False`` to skip the on-disk cache entirely)."""
    config = {
        "model": "fake-embed-model",
        "prototype_cache": False,
        "prototype_cache_dir": tempfile.mkdtemp(),
    }
    if extra_config:
        config.update(extra_config)

    mock_embed_model = MagicMock()
    mock_embed_model.encode.side_effect = _hash_encode

    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch("ovos_m2v_pipeline.fusion.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        from ovos_m2v_pipeline.fusion import Model2VecNebulentoFusionPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecNebulentoFusionPipeline(bus=FakeBus(), config=config)
    pipeline.model = mock_embed_model
    return pipeline


class TestNoisyOrArithmetic(unittest.TestCase):
    """Test 5: the noisy-OR formula in isolation."""

    def test_both_signals(self):
        from ovos_m2v_pipeline.fusion import Model2VecNebulentoFusionPipeline as P
        conf = P.fuse(0.55, 0.6)
        self.assertAlmostEqual(conf, 1 - (0.45 * 0.4))
        self.assertGreaterEqual(conf, 0.55)
        self.assertGreaterEqual(conf, 0.6)

    def test_agreement_beats_either_component_always(self):
        from ovos_m2v_pipeline.fusion import Model2VecNebulentoFusionPipeline as P
        for a in (0.1, 0.3, 0.5, 0.7, 0.9):
            for b in (0.1, 0.3, 0.5, 0.7, 0.9):
                conf = P.fuse(a, b)
                self.assertGreaterEqual(conf, a)
                self.assertGreaterEqual(conf, b)
                self.assertLessEqual(conf, 1.0)

    def test_single_signal_degrades_to_component(self):
        from ovos_m2v_pipeline.fusion import Model2VecNebulentoFusionPipeline as P
        self.assertEqual(P.fuse(0.45, None), 0.45)
        self.assertEqual(P.fuse(None, 0.6), 0.6)

    def test_hand_worked_threshold_derivation_cases(self):
        # Same cases documented in the fusion.py module docstring.
        from ovos_m2v_pipeline.fusion import Model2VecNebulentoFusionPipeline as P
        self.assertAlmostEqual(P.fuse(0.55, 0.60), 0.82, places=6)
        self.assertAlmostEqual(P.fuse(0.70, 0.70), 0.91, places=6)
        self.assertAlmostEqual(P.fuse(0.50, 0.50), 0.75, places=6)
        self.assertAlmostEqual(P.fuse(0.30, 0.30), 0.51, places=6)


class TestFusionMatching(unittest.TestCase):
    def test_agreement_boosts_above_either_component_and_clears_medium(self):
        """Test 1: m2v alone ~0.55, nebulento alone ~0.6 on the SAME label
        fuses to >= either component and clears the medium tier (0.65)."""
        pipeline = _make_pipeline()
        pipeline.prototype_store.scores = MagicMock(
            return_value={"skill:demo": 0.55})
        pipeline.nebulento.registered_intents = {"skill:demo": ["turn on the lights"]}
        pipeline.nebulento.calc_intents = MagicMock(
            return_value=iter([{"name": "skill:demo", "conf": 0.6,
                                 "entities": {}}]))
        candidates = list(pipeline._match_prototype("turn on the lights"))
        self.assertEqual(len(candidates), 1)
        _, label, conf = candidates[0]
        self.assertEqual(label, "skill:demo")
        self.assertGreaterEqual(conf, 0.55)
        self.assertGreaterEqual(conf, 0.6)
        self.assertGreaterEqual(conf, pipeline.config.get("conf_medium", 0.65))

    def test_disagreement_gives_no_boost(self):
        """Test 2: nebulento aligning only a DIFFERENT label contributes
        nothing to the m2v candidate's fused score."""
        pipeline = _make_pipeline()
        pipeline.prototype_store.scores = MagicMock(
            return_value={"skill:demo": 0.55})
        pipeline.nebulento.registered_intents = {"skill:demo": ["turn on the lights"],
                                                   "skill:other": ["do something else"]}
        pipeline.nebulento.calc_intents = MagicMock(
            return_value=iter([{"name": "skill:other", "conf": 0.9,
                                 "entities": {}}]))
        candidates = list(pipeline._match_prototype("turn on the lights"))
        self.assertEqual(len(candidates), 1)
        _, label, conf = candidates[0]
        self.assertEqual(label, "skill:demo")
        self.assertAlmostEqual(conf, 0.55)

    def test_slot_attachment_on_winning_label(self):
        """Test 3: nebulento's utterance-extracted slot values attach to the
        winning candidate's match_data."""
        pipeline = _make_pipeline()
        pipeline.prototype_store.scores = MagicMock(
            return_value={"skill:demo": 0.55})
        pipeline.nebulento.registered_intents = {"skill:demo": ["play {song}"]}
        pipeline.nebulento.calc_intents = MagicMock(
            return_value=iter([{"name": "skill:demo", "conf": 0.6,
                                 "entities": {"song": ["jazz"]}}]))
        matches = list(pipeline._match("play some jazz"))
        self.assertEqual(len(matches), 1)
        _, label, conf, slots = matches[0]
        self.assertEqual(label, "skill:demo")
        self.assertEqual(slots.get("song"), ["jazz"])


class TestNebulentoOptionalDependency(unittest.TestCase):
    def test_missing_nebulento_raises_clear_import_error(self):
        """Test 4a: construction without nebulento installed fails loudly
        and unambiguously."""
        with patch.dict(sys.modules, {"nebulento": None}):
            with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
                 patch("ovos_m2v_pipeline.Configuration", return_value={}), \
                 patch("ovos_m2v_pipeline.fusion.Configuration", return_value={}):
                # re-import fusion.py fresh so the `import nebulento` inside
                # __init__ actually re-runs against the patched sys.modules
                import importlib
                import ovos_m2v_pipeline.fusion as fusion_mod
                importlib.reload(fusion_mod)
                from ovos_utils.fakebus import FakeBus
                with self.assertRaises(ImportError) as ctx:
                    fusion_mod.Model2VecNebulentoFusionPipeline(
                        bus=FakeBus(),
                        config={"model": "fake-embed-model",
                                "prototype_cache": False})
                self.assertIn("nebulento", str(ctx.exception))
                self.assertIn("fusion", str(ctx.exception))
        # restore a clean import for any test running after this one
        import importlib
        import ovos_m2v_pipeline.fusion as fusion_mod
        importlib.reload(fusion_mod)

    def test_package_import_unaffected_by_missing_nebulento(self):
        """Test 4b: importing ovos_m2v_pipeline itself never needs
        nebulento -- the rest of the package works whether or not the
        'fusion' extra is installed."""
        with patch.dict(sys.modules, {"nebulento": None}):
            import importlib
            import ovos_m2v_pipeline as pkg
            importlib.reload(pkg)
            self.assertTrue(hasattr(pkg, "Model2VecIntentPipeline"))
            self.assertTrue(hasattr(pkg, "Model2VecPrototypePipeline"))


if __name__ == "__main__":
    unittest.main()
