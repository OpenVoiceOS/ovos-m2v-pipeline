"""The numpy forward path must answer what model2vec answers.

Step 2 of `knowledge/wiki/plans/m2v-lean-inference.md`. The parity bar the
plan sets is per-locale label agreement at 100 percent on the whole
evaluation set, never an aggregate, and a logit tolerance of 1e-3 (the
reference embedding is float16, so 1e-5 asks the re-implementation to
reproduce float16 rounding error exactly).

The whole 16-locale run belongs in the measurement record, not in a unit
test that every contributor pays for. What runs here is the same comparison
on one locale's worth of real utterances, plus the wiring: the config key,
the refusal of an unknown runtime, the separate cache entries, the named
error for a checkpoint the lean path cannot read, and the one claim the
whole step rests on -- that the forward path runs with `model2vec` absent.

These tests do not mock the model. A mocked parity test proves nothing:
every number they compare comes out of the same two implementations a user
gets.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ovos_m2v_pipeline import (DEFAULT_MULTILINGUAL, DEFAULT_RUNTIME, RUNTIMES,
                               clear_shared_models, load_shared_model,
                               shared_model_stats)
from ovos_m2v_pipeline.lean import (LeanStaticModelPipeline,
                                    LeanUnsupportedCheckpoint)

#: real utterances, one per domain, in four of the shipped locales. The
#: evaluation set itself is pinned at `decontaminated` in the measurement
#: run; these are literals so the test needs no dataset download.
UTTERANCES = [
    "turn on the kitchen light",
    "what time is it",
    "play some jazz",
    "set an alarm for seven",
    "remove the dentist appointment",
    "apaga la luz del salon",
    "que hora es",
    "pon musica",
    "allume la lumiere de la cuisine",
    "quelle heure est-il",
    "mach das licht im wohnzimmer aus",
    "wie spat ist es",
]


def _model_path():
    """The default checkpoint on disk."""
    from huggingface_hub import snapshot_download
    return snapshot_download(DEFAULT_MULTILINGUAL)


class TestLeanParity(unittest.TestCase):
    """The lean path against model2vec, on the real default checkpoint."""

    @classmethod
    def setUpClass(cls):
        from model2vec.inference import StaticModelPipeline
        cls.path = _model_path()
        cls.ref = StaticModelPipeline.from_pretrained(cls.path)
        cls.lean = LeanStaticModelPipeline.from_pretrained(cls.path)

    def test_the_class_lists_are_the_same_list(self):
        self.assertEqual(list(self.ref.classes_), list(self.lean.classes_))

    def test_every_utterance_gets_the_same_label(self):
        ref = self.ref.predict(list(UTTERANCES), use_multiprocessing=False)
        lean = self.lean.predict(list(UTTERANCES))
        # reported per utterance, because an aggregate can hold while one
        # utterance's tokenisation drifts
        for text, want, got in zip(UTTERANCES, ref, lean):
            self.assertEqual(want, got, f"label differs for {text!r}")

    def test_the_logits_are_within_the_plan_tolerance(self):
        ref = self.ref.head._logits(
            self.ref.model.encode(list(UTTERANCES), use_multiprocessing=False))
        lean = self.lean.head.logits(self.lean.model.encode(list(UTTERANCES)))
        self.assertEqual(ref.shape, lean.shape)
        self.assertLess(float(np.abs(ref - lean).max()), 1e-3)

    def test_the_embeddings_are_the_same_vectors(self):
        ref = self.ref.model.encode(list(UTTERANCES), use_multiprocessing=False)
        lean = self.lean.model.encode(list(UTTERANCES))
        self.assertLess(float(np.abs(ref - lean).max()), 1e-3)
        self.assertEqual(self.ref.model.dim, self.lean.model.dim)

    def test_an_empty_utterance_answers_a_zero_vector_in_both(self):
        # the reference's float64 zero row and its 1e-32 norm guard; a lean
        # path that skipped either would divide by zero here
        ref = self.ref.model.encode([""], use_multiprocessing=False)
        lean = self.lean.model.encode([""])
        self.assertFalse(np.isnan(lean).any())
        self.assertLess(float(np.abs(ref - lean).max()), 1e-3)

    def test_normalisation_is_what_the_agreement_rests_on(self):
        """The control: the comparison moves when the lean path is wrong.

        Without this, a parity test that compared an object with itself
        would report the same pass.
        """
        ref = self.ref.predict(list(UTTERANCES), use_multiprocessing=False)
        self.lean.model.normalize = False
        try:
            broken = self.lean.predict(list(UTTERANCES))
        finally:
            self.lean.model.normalize = True
        self.assertTrue((ref != broken).any(),
                        "dropping L2 normalisation changed no label, so the "
                        "comparison does not measure the forward path")


class TestTheRuntimeIsSelectable(unittest.TestCase):

    def setUp(self):
        clear_shared_models()

    def tearDown(self):
        clear_shared_models()

    def test_the_default_runtime_is_still_model2vec(self):
        self.assertEqual(DEFAULT_RUNTIME, "model2vec")
        self.assertEqual(RUNTIMES, ("model2vec", "lean"))

    def test_the_lean_runtime_loads_the_lean_pipeline(self):
        path = _model_path()
        got = load_shared_model(path, "classifier", "lean")
        self.assertIsInstance(got, LeanStaticModelPipeline)

    def test_the_two_runtimes_are_separate_cache_entries(self):
        path = _model_path()
        lean = load_shared_model(path, "classifier", "lean")
        ref = load_shared_model(path, "classifier")
        self.assertIsNot(lean, ref)
        self.assertIs(load_shared_model(path, "classifier", "lean"), lean)
        self.assertEqual(shared_model_stats()["models"], 2)

    def test_the_lean_prototype_mode_answers_the_embedding_half(self):
        path = _model_path()
        got = load_shared_model(path, "prototype", "lean")
        self.assertEqual(got.dim, 128)
        self.assertTrue(hasattr(got, "encode"))

    def test_an_unknown_runtime_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            load_shared_model("/m", "classifier", "onnx")
        self.assertIn("onnx", str(caught.exception))

    def test_the_plugin_refuses_an_unknown_runtime_at_construction(self):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        with self.assertRaises(ValueError):
            Model2VecIntentPipeline(bus=FakeBus(),
                                    config={"runtime": "tensorflow"})


class TestACheckpointTheLeanPathCannotRead(unittest.TestCase):

    def test_a_folder_with_no_head_names_the_way_out(self):
        with tempfile.TemporaryDirectory() as folder:
            src = Path(_model_path())
            for name in ("model.safetensors", "tokenizer.json", "config.json"):
                os.symlink(src / name, Path(folder) / name)
            with self.assertRaises(LeanUnsupportedCheckpoint) as caught:
                LeanStaticModelPipeline.from_pretrained(folder)
            message = str(caught.exception)
            self.assertIn("head.safetensors", message)
            self.assertIn("model2vec", message)

    def test_a_folder_with_no_embeddings_names_the_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(LeanUnsupportedCheckpoint) as caught:
                LeanStaticModelPipeline.from_pretrained(folder)
            self.assertIn("model.safetensors", str(caught.exception))


class TestTheForwardPathRunsWithoutModel2Vec(unittest.TestCase):
    """The claim the whole step rests on, checked in a fresh process.

    `ovos_m2v_pipeline/__init__.py` still imports model2vec at module
    import; step 3 of the plan is what removes that. So the module is loaded
    from its file here, which is exactly the import the package will do once
    model2vec moves to the `train` extra.
    """

    def test_a_prediction_runs_with_model2vec_blocked(self):
        lean_py = Path(__file__).resolve().parent.parent / \
            "ovos_m2v_pipeline" / "lean.py"
        script = (
            "import importlib.util, sys\n"
            "class Block:\n"
            # find_spec, not find_module: find_module is gone in 3.12 and a
            # blocker written that way silently stops blocking, which reads
            # as a pass. The self-check two lines down catches that anyway.
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'model2vec' or name.startswith('model2vec.'):\n"
            "            raise ImportError('model2vec is not installed')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "try:\n"
            "    import model2vec\n"
            "    raise SystemExit('the blocker did not block model2vec')\n"
            "except ImportError:\n"
            "    pass\n"
            f"spec = importlib.util.spec_from_file_location('lean', {str(lean_py)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "from huggingface_hub import snapshot_download\n"
            f"path = snapshot_download({DEFAULT_MULTILINGUAL!r})\n"
            "pipe = mod.LeanStaticModelPipeline.from_pretrained(path)\n"
            "label = pipe.predict(['turn on the kitchen light'])[0]\n"
            "import json\n"
            "print(json.dumps({'label': str(label),\n"
            "                  'model2vec': 'model2vec' in sys.modules,\n"
            "                  'sklearn': 'sklearn' in sys.modules}))\n"
        )
        out = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        got = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertFalse(got["model2vec"])
        self.assertFalse(got["sklearn"])
        self.assertTrue(got["label"])


if __name__ == "__main__":
    unittest.main()
