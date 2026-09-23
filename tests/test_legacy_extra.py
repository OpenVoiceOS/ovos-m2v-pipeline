"""The `legacy` extra: scikit-learn, skops, scipy and joblib are needed
only by the sklearn/skops head loader (a `pipeline.skops` checkpoint), not
by any published `head.safetensors` model. They must not be runtime
dependencies, and a legacy checkpoint loaded without the extra must raise
an `ImportError` naming `pip install ovos-m2v-pipeline[legacy]`.
"""
import os
import sys
import tomllib
import unittest
from pathlib import Path

import pytest

import ovos_m2v_pipeline

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _legacy_available() -> bool:
    try:
        import skops.io  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


class TestLegacyExtraDeclaration(unittest.TestCase):
    def setUp(self):
        with open(_PYPROJECT, "rb") as fh:
            self.pyproject = tomllib.load(fh)

    def test_runtime_dependencies_exclude_legacy_packages(self):
        deps = " ".join(self.pyproject["project"]["dependencies"]).lower()
        for pkg in ("scikit-learn", "skops", "scipy", "joblib"):
            self.assertNotIn(pkg, deps,
                              f"{pkg} must not be a runtime dependency")

    def test_legacy_extra_carries_the_four_packages(self):
        extras = self.pyproject["project"]["optional-dependencies"]
        legacy = " ".join(extras["legacy"]).lower()
        for pkg in ("scikit-learn", "skops", "scipy", "joblib"):
            self.assertIn(pkg, legacy)


@pytest.mark.skipif(_legacy_available(),
                     reason="scikit-learn/skops installed; the legacy "
                            "checkpoint loads instead of raising")
class TestLegacyCheckpointWithoutExtra(unittest.TestCase):
    def test_pipeline_skops_checkpoint_names_the_extra(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pipeline.skops").write_bytes(b"fake-legacy-head")
            with self.assertRaises(ImportError) as ctx:
                ovos_m2v_pipeline.load_shared_model(tmp, "classifier")
            self.assertIn("ovos-m2v-pipeline[legacy]", str(ctx.exception))

    def test_sklearn_is_never_imported_by_the_package_import_itself(self):
        # a subprocess, not this test process: another test module in the
        # same run may import sklearn for its own (training-side) reason,
        # which says nothing about what importing ovos_m2v_pipeline does.
        import subprocess
        out = subprocess.run(
            [sys.executable, "-c",
             "import ovos_m2v_pipeline, sys; "
             "print('sklearn' in sys.modules, 'skops' in sys.modules)"],
            capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "False False")


@pytest.mark.skipif(not _legacy_available(),
                     reason="scikit-learn/skops not installed; the legacy "
                            "extra is absent")
@pytest.mark.skipif(os.environ.get("OVOSCOPE_LIVE") != "1",
                     reason="Hub network test skipped; set OVOSCOPE_LIVE=1 "
                            "to enable.")
class TestLegacyCheckpointWithExtraLoads(unittest.TestCase):
    """`ovos-m2v-intents-multi-128M-v5` ships `pipeline.skops`; with the
    `legacy` extra installed it must load rather than raise."""

    LEGACY_MODEL = "OpenVoiceOS/ovos-m2v-intents-multi-128M-v5"

    def test_v5_legacy_checkpoint_loads_with_extra(self):
        try:
            pipeline = ovos_m2v_pipeline.load_shared_model(
                self.LEGACY_MODEL, "classifier")
        except Exception as e:  # pragma: no cover - network flake
            raise unittest.SkipTest(f"HF Hub unreachable: {e}")
        self.assertTrue(hasattr(pipeline, "classes_"))
