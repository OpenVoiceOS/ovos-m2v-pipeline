"""The two published repos load their ``int8/`` copy unless the config says
otherwise: ``subfolder`` set (``""`` for the root), or a ``revision`` pinned.
Any other model loads from its root as before."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ovos_utils.fakebus import FakeBus

EN = "OpenVoiceOS/ovos-m2v-intents-en"
MULTI = "OpenVoiceOS/ovos-m2v-intents-multilingual"


def _subfolder(config, lang="en-US"):
    from ovos_m2v_pipeline import Model2VecIntentPipeline
    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline._load_labels_manifest", return_value={}) as manifest, \
         patch("ovos_m2v_pipeline.Configuration", return_value={"lang": lang}):
        pipe = Model2VecIntentPipeline(bus=FakeBus(), config=config)
    assert manifest.call_args_list[0].args[1] == pipe._subfolder
    # the -low prototype stage runs on the same folder as the classifier
    assert pipe._low_prototype._subfolder == pipe._subfolder
    return pipe._subfolder


class TestInt8Default(unittest.TestCase):

    def test_unconfigured_install_loads_multilingual_int8(self):
        self.assertEqual(_subfolder({}, "pt-PT"), "int8")

    def test_english_model_through_models_loads_int8(self):
        self.assertEqual(_subfolder({"models": {"en": EN}}), "int8")

    def test_explicit_model_of_ours_loads_int8(self):
        self.assertEqual(_subfolder({"model": MULTI}), "int8")

    def test_empty_subfolder_loads_the_root(self):
        self.assertIsNone(_subfolder({"subfolder": ""}))
        self.assertIsNone(_subfolder({"models": {"en": EN}, "subfolder": ""}))

    def test_explicit_subfolder_wins(self):
        self.assertEqual(_subfolder({"subfolder": "fp16"}), "fp16")

    def test_pinned_revision_loads_the_root(self):
        self.assertIsNone(_subfolder({"model": MULTI, "revision": "145a1bd94acb2620d9aa92c82567cf2bbb55f0a2"}))

    def test_other_repo_loads_the_root(self):
        self.assertIsNone(_subfolder({"model": "someone/their-model"}))

    def test_local_path_loads_the_root(self):
        self.assertIsNone(_subfolder({"model": "/opt/models/my-model"}))

    def test_null_revision_is_unset(self):
        self.assertEqual(_subfolder({"revision": None}), "int8")

    def test_low_prototype_subfolder_follows_the_classifier(self):
        self.assertEqual(_subfolder({"low_prototype": {"subfolder": "fp16"}}), "int8")
        self.assertIsNone(_subfolder({"subfolder": "", "low_prototype": {"subfolder": "int8"}}))


SHA = "0123456789abcdef0123456789abcdef01234567"


class TestCacheWithoutInt8(unittest.TestCase):
    """A device that cached the model before its repo had ``int8/`` and then
    cannot reach the Hub loads the cached root, as it did before the default."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        repo = Path(self.tmp.name) / "models--OpenVoiceOS--ovos-m2v-intents-multilingual"
        snap = repo / "snapshots" / SHA
        snap.mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(SHA)
        for f in ("config.json", "model.safetensors", "head.safetensors", "tokenizer.json", "modules.json"):
            (snap / f).write_text("{}")
        (snap / "labels.json").write_text(json.dumps({"foo": "skill.foo:bar"}))
        self.snapshot = str(snap)
        self.cache = patch("huggingface_hub.constants.HF_HUB_CACHE", self.tmp.name)
        self.cache.start()

    def tearDown(self):
        self.cache.stop()
        self.tmp.cleanup()

    def _pipe(self, config):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
             patch("ovos_m2v_pipeline.Configuration", return_value={"lang": "pt-PT"}):
            return Model2VecIntentPipeline(bus=FakeBus(), config=config)

    def test_offline_loads_the_cached_root(self):
        pipe = self._pipe({})
        with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            path = pipe._resolve_model_revision(MULTI)
        self.assertEqual(path, self.snapshot)
        self.assertIsNone(pipe._subfolder)

    def test_unreachable_hub_loads_the_cached_root(self):
        pipe = self._pipe({})
        with patch("huggingface_hub.HfApi.model_info", side_effect=ConnectionError):
            path = pipe._resolve_model_revision(MULTI)
        self.assertEqual(path, self.snapshot)

    def test_offline_incomplete_snapshot_error_loads_the_cached_root(self):
        """huggingface_hub 2.x raises for a cached snapshot missing the requested files."""
        import huggingface_hub
        from huggingface_hub.errors import IncompleteSnapshotError
        real = huggingface_hub.snapshot_download

        def strict(*args, allow_patterns=None, **kwargs):
            if any(p.startswith("int8/") for p in allow_patterns or []):
                raise IncompleteSnapshotError("int8/ files missing", snapshot_path="")
            return real(*args, allow_patterns=allow_patterns, **kwargs)

        pipe = self._pipe({})
        with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True), \
             patch("huggingface_hub.snapshot_download", strict):
            path = pipe._resolve_model_revision(MULTI)
        self.assertEqual(path, self.snapshot)
        self.assertIsNone(pipe._subfolder)

    def test_explicit_int8_is_not_replaced(self):
        pipe = self._pipe({"subfolder": "int8"})
        with patch("huggingface_hub.constants.HF_HUB_OFFLINE", True):
            path = pipe._resolve_model_revision(MULTI)
        self.assertEqual(path, f"{self.snapshot}/int8")

    def test_root_labels_are_read_until_int8_is_cached(self):
        pipe = self._pipe({})
        self.assertEqual(pipe.label_map.get("foo"), "skill.foo:bar")


if __name__ == "__main__":
    unittest.main()
