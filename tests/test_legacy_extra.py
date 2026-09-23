"""scikit-learn and skops belong to the LEGACY checkpoint format.

A current checkpoint ships `head.safetensors`, which model2vec reads as plain
numpy. A legacy one ships `pipeline.skops`, which it reads through skops and
scikit-learn. Those two, with the scipy and joblib they pull, are 151 MB of
site-packages measured on python 3.11, and every model this plugin ships uses
the current format. They are an extra, not a runtime dependency.

The failure a user meets without the extra has to say so. model2vec's own
message is `Converting a legacy pipeline requires 'scikit-learn' and
'skops'.`, which names neither this package, nor the checkpoint, nor the way
out.
"""
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from ovos_m2v_pipeline import (LegacyCheckpointExtraMissing,
                               _LEGACY_EXTRA, _load_classifier_pipeline)

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
#: exactly what model2vec's own guard raises when its legacy reader is
#: reached without the extra. Copied from a measured run, not invented.
MODEL2VEC_GUARD = ImportError(
    "Converting a legacy pipeline requires `scikit-learn` and `skops`.")


def _pyproject():
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)


class TestThePackagingKeepsTheRuntimeLean(unittest.TestCase):

    def test_the_runtime_does_not_require_the_legacy_readers(self):
        runtime = " ".join(_pyproject()["project"]["dependencies"])
        self.assertNotIn("scikit-learn", runtime)
        self.assertNotIn("skops", runtime)

    def test_the_legacy_extra_carries_them(self):
        extras = _pyproject()["project"]["optional-dependencies"]
        self.assertIn("legacy", extras)
        legacy = " ".join(extras["legacy"])
        self.assertIn("scikit-learn", legacy)
        self.assertIn("skops", legacy)

    def test_the_test_extra_still_has_scikit_learn(self):
        # control: the training and test paths really do need it, so this
        # change must not take it away from them
        extras = _pyproject()["project"]["optional-dependencies"]
        self.assertIn("scikit-learn", " ".join(extras["test"]))


class TestTheLegacyFailureNamesTheExtra(unittest.TestCase):

    def test_model2vec_s_own_guard_is_translated(self):
        """The shape with no `name`, which is what model2vec raises."""
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as pipeline:
            pipeline.from_pretrained.side_effect = MODEL2VEC_GUARD
            with self.assertRaises(LegacyCheckpointExtraMissing) as caught:
                _load_classifier_pipeline("some/legacy-model")
        message = str(caught.exception)
        self.assertIn("some/legacy-model", message)
        self.assertIn("pipeline.skops", message)
        self.assertIn(_LEGACY_EXTRA, message)
        self.assertIn("convert_legacy_pipeline", message)

    def test_a_bare_missing_module_is_translated_too(self):
        """The shape WITH a `name`, which a plain import raises."""
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as pipeline:
            pipeline.from_pretrained.side_effect = ModuleNotFoundError(
                "No module named 'skops'", name="skops")
            with self.assertRaises(LegacyCheckpointExtraMissing) as caught:
                _load_classifier_pipeline("some/legacy-model")
        self.assertIn(_LEGACY_EXTRA, str(caught.exception))

    def test_an_unrelated_import_error_is_not_translated(self):
        """Control: a broken install must not read as a checkpoint problem."""
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as pipeline:
            pipeline.from_pretrained.side_effect = ModuleNotFoundError(
                "No module named 'numpy'", name="numpy")
            with self.assertRaises(ModuleNotFoundError) as caught:
                _load_classifier_pipeline("some/model")
        self.assertNotIsInstance(caught.exception,
                                 LegacyCheckpointExtraMissing)

    def test_a_working_load_is_returned_unchanged(self):
        """Control: the wrapper is a translation, not a gate."""
        with patch("ovos_m2v_pipeline.StaticModelPipeline") as pipeline:
            pipeline.from_pretrained.return_value = "the pipeline"
            self.assertEqual(_load_classifier_pipeline("ok/model"),
                             "the pipeline")


if __name__ == "__main__":
    unittest.main()
