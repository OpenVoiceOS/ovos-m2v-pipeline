"""A Hub model repo carries several models: the default one at the root and
smaller copies in subfolders (``fp16/``, ``int8/``, ``onnx/``). Every model
download fetches only the files of one folder, the ``subfolder`` config key
picks which, and everything read from the model (weights, ``labels.json``,
the shared in-memory entry, the prototype disk cache) follows that folder.

Expected file sets are written out by hand and checked through
huggingface_hub's own ``filter_repo_objects``, the function
``snapshot_download`` applies to ``allow_patterns``.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from huggingface_hub.utils import filter_repo_objects
from ovos_utils.fakebus import FakeBus

REPO = "OpenVoiceOS/fake-repo"
SNAPSHOT = "/cache/fake-repo/snapshots/abc123"

REPO_FILES = [
    ".gitattributes", "config.json", "model.safetensors", "head.safetensors",
    "tokenizer.json", "modules.json", "labels.json", "README.md",
    "fp16/config.json", "fp16/model.safetensors", "fp16/head.safetensors",
    "fp16/tokenizer.json", "fp16/modules.json", "fp16/labels.json",
    "fp16/README.md",
    "int8/config.json", "int8/model.safetensors", "int8/head.safetensors",
    "int8/tokenizer.json", "int8/modules.json", "int8/labels.json",
    "int8/README.md",
    "onnx/model.onnx", "onnx/tokenizer.json", "onnx/config.json",
]
ROOT_FILES = {"config.json", "model.safetensors", "head.safetensors",
              "tokenizer.json", "modules.json", "labels.json", "README.md"}
INT8_FILES = {"int8/config.json", "int8/model.safetensors",
              "int8/head.safetensors", "int8/tokenizer.json",
              "int8/modules.json", "int8/labels.json", "int8/README.md"}


def _make_pipeline(config):
    from ovos_m2v_pipeline import Model2VecIntentPipeline
    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}):
        return Model2VecIntentPipeline(bus=FakeBus(), config=config)


def _selected(call):
    return set(filter_repo_objects(
        REPO_FILES, allow_patterns=call.kwargs["allow_patterns"]))


class _AllCallPaths:
    """Runs every ``snapshot_download`` call path of model resolution and
    returns the calls together with the path each path resolved to."""

    config: dict

    def _online(self):
        pipeline = _make_pipeline(self.config)
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None), \
             patch("huggingface_hub.snapshot_download",
                   return_value=SNAPSHOT) as dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            result = pipeline._resolve_model_revision(REPO)
        return dl.call_args_list, result

    def _unreachable(self):
        pipeline = _make_pipeline(self.config)
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None), \
             patch("huggingface_hub.snapshot_download",
                   side_effect=[requests.exceptions.ConnectionError("no net"),
                                SNAPSHOT]) as dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False), \
             patch("ovos_m2v_pipeline.LOG"):
            result = pipeline._resolve_model_revision(REPO)
        return dl.call_args_list, result

    def _offline(self):
        pipeline = _make_pipeline(self.config)
        with patch("huggingface_hub.snapshot_download",
                   return_value=SNAPSHOT) as dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", True), \
             patch("ovos_m2v_pipeline.LOG"):
            result = pipeline._resolve_model_revision(REPO)
        return dl.call_args_list, result

    def _cached_current(self):
        pipeline = _make_pipeline(self.config)
        with patch("huggingface_hub.snapshot_download",
                   return_value=SNAPSHOT) as dl, \
             patch("huggingface_hub.HfApi") as api, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            api.return_value.model_info.return_value = MagicMock(
                sha="abc123", siblings=None)
            result = pipeline._resolve_model_revision(REPO)
        return dl.call_args_list, result

    def all_paths(self):
        return {"online": self._online(), "unreachable": self._unreachable(),
                "offline": self._offline(),
                "cached_current": self._cached_current()}


class TestNoSubfolderDownloadsRootOnly(_AllCallPaths, unittest.TestCase):
    config = {"model": REPO}

    def test_every_call_path_fetches_root_files_only(self):
        for name, (calls, result) in self.all_paths().items():
            with self.subTest(path=name):
                self.assertTrue(calls)
                for call in calls:
                    self.assertIn("allow_patterns", call.kwargs)
                    self.assertEqual(_selected(call), ROOT_FILES)
                self.assertEqual(result, SNAPSHOT)


class TestSubfolderDownloadsThatFolderOnly(_AllCallPaths, unittest.TestCase):
    config = {"model": REPO, "subfolder": "int8"}

    def test_every_call_path_fetches_subfolder_files_only(self):
        for name, (calls, result) in self.all_paths().items():
            with self.subTest(path=name):
                self.assertTrue(calls)
                for call in calls:
                    self.assertIn("allow_patterns", call.kwargs)
                    self.assertEqual(_selected(call), INT8_FILES)
                self.assertEqual(Path(result), Path(SNAPSHOT) / "int8")

    def test_local_directory_loads_its_subfolder(self):
        pipeline = _make_pipeline(self.config)
        with tempfile.TemporaryDirectory() as tmp, \
             patch("huggingface_hub.snapshot_download") as dl:
            result = pipeline._resolve_model_revision(tmp)
        dl.assert_not_called()
        self.assertEqual(Path(result), Path(tmp) / "int8")


class TestSlashesAroundSubfolderAreIgnored(unittest.TestCase):
    """``"/int8/"`` names the same folder as ``"int8"`` everywhere."""

    config = {"model": REPO, "subfolder": "/int8/"}

    def test_download_and_path_name_the_folder(self):
        pipeline = _make_pipeline(self.config)
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None), \
             patch("huggingface_hub.snapshot_download",
                   return_value=SNAPSHOT) as dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            result = pipeline._resolve_model_revision(REPO)
        self.assertEqual(_selected(dl.call_args), INT8_FILES)
        self.assertEqual(result, f"{SNAPSHOT}/int8")

    def test_local_directory_path_stays_inside_it(self):
        pipeline = _make_pipeline(self.config)
        with tempfile.TemporaryDirectory() as tmp:
            result = pipeline._resolve_model_revision(tmp)
        self.assertEqual(result, f"{tmp}/int8")

    def test_labels_and_cache_id_use_the_bare_name(self):
        with patch("huggingface_hub.hf_hub_download",
                   side_effect=OSError("not cached")) as hf:
            pipeline = _make_pipeline({**self.config, "mode": "prototype"})
        for call in hf.call_args_list:
            self.assertEqual(call.kwargs.get("subfolder"), "int8")
        self.assertEqual(pipeline._model_id, f"{REPO}/int8")

    def test_empty_subfolder_is_the_root(self):
        pipeline = _make_pipeline({"model": REPO, "subfolder": "/"})
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None), \
             patch("huggingface_hub.snapshot_download",
                   return_value=SNAPSHOT) as dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            result = pipeline._resolve_model_revision(REPO)
        self.assertEqual(_selected(dl.call_args), ROOT_FILES)
        self.assertEqual(result, SNAPSHOT)


class TestSentenceTransformersLayout(unittest.TestCase):
    """model2vec also loads the sentence-transformers layouts, including
    the one nesting the embedding under ``0_StaticEmbedding/`` that
    ``sentence-transformers/static-retrieval-mrl-en-v1`` uses."""

    FILES = [".gitattributes", "README.md", "modules.json",
             "config_sentence_transformers.json",
             "0_StaticEmbedding/model.safetensors",
             "0_StaticEmbedding/tokenizer.json"]

    def test_nested_layout_is_downloaded_whole(self):
        pipeline = _make_pipeline({"model": REPO, "mode": "prototype"})
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None), \
             patch("huggingface_hub.snapshot_download",
                   return_value=SNAPSHOT) as dl, \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False):
            pipeline._resolve_model_revision(REPO)
        selected = set(filter_repo_objects(
            self.FILES, allow_patterns=dl.call_args.kwargs["allow_patterns"]))
        self.assertEqual(selected, set(self.FILES) - {".gitattributes"})

    def test_cache_without_the_embedding_is_not_current(self):
        root = tempfile.mkdtemp(prefix="m2v-st-snapshot-")
        snapshot = Path(root) / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        for name in ("README.md", "modules.json"):
            (snapshot / name).write_text("x")
        pipeline = _make_pipeline({"model": REPO, "mode": "prototype"})
        siblings = [MagicMock(rfilename=n) for n in self.FILES]
        with patch("huggingface_hub.snapshot_download",
                   return_value=str(snapshot)), \
             patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value = MagicMock(
                sha="abc123", siblings=siblings)
            self.assertIsNone(pipeline._cached_snapshot_if_current(REPO))


class TestCompleteSnapshotIgnoresOtherFolders(unittest.TestCase):
    """A cached snapshot holding every file of the chosen folder is current
    although the other folders of the repo were never downloaded."""

    def _probe(self, config, present):
        root = tempfile.mkdtemp(prefix="m2v-subfolder-snapshot-")
        snapshot = Path(root) / "snapshots" / "abc123"
        for name in present:
            (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
            (snapshot / name).write_text("x")
        pipeline = _make_pipeline(config)
        siblings = [MagicMock(rfilename=n) for n in REPO_FILES]
        with patch("huggingface_hub.snapshot_download",
                   return_value=str(snapshot)), \
             patch("huggingface_hub.HfApi") as api:
            api.return_value.model_info.return_value = MagicMock(
                sha="abc123", siblings=siblings)
            return pipeline._cached_snapshot_if_current(REPO), str(snapshot)

    def test_root_files_alone_are_current(self):
        result, snapshot = self._probe({"model": REPO}, ROOT_FILES)
        self.assertEqual(result, snapshot)

    def test_subfolder_files_alone_are_current(self):
        result, snapshot = self._probe(
            {"model": REPO, "subfolder": "int8"}, INT8_FILES)
        self.assertEqual(result, snapshot)

    def test_root_files_do_not_complete_a_subfolder(self):
        result, _ = self._probe({"model": REPO, "subfolder": "int8"},
                                ROOT_FILES)
        self.assertIsNone(result)


class TestModelLoadsFromSubfolder(unittest.TestCase):
    def setUp(self):
        from ovos_m2v_pipeline import clear_shared_models
        clear_shared_models()
        self.addCleanup(clear_shared_models)

    def _load(self, config, loader):
        pipeline = _make_pipeline(config)
        with patch.object(pipeline, "_cached_snapshot_if_current",
                          return_value=None), \
             patch("huggingface_hub.snapshot_download",
                   return_value=SNAPSHOT), \
             patch("huggingface_hub.constants.HF_HUB_OFFLINE", False), \
             patch("ovos_m2v_pipeline.StaticModelPipeline") as smp:
            smp.from_pretrained.side_effect = loader
            pipeline._load_model_now()
        return pipeline

    def test_from_pretrained_receives_subfolder_path(self):
        seen = []
        self._load({"model": REPO, "subfolder": "int8"},
                   lambda path: seen.append(path) or MagicMock())
        self.assertEqual([Path(p) for p in seen], [Path(SNAPSHOT) / "int8"])

    def test_shared_model_entry_differs_by_subfolder(self):
        from ovos_m2v_pipeline import shared_model_stats
        seen = []
        root = self._load({"model": REPO},
                          lambda path: seen.append(path) or MagicMock())
        int8 = self._load({"model": REPO, "subfolder": "int8"},
                          lambda path: seen.append(path) or MagicMock())
        self.assertEqual(shared_model_stats(), {"models": 2, "loads": 2})
        self.assertIsNot(root.model, int8.model)
        self.assertEqual({Path(p) for p in seen},
                         {Path(SNAPSHOT), Path(SNAPSHOT) / "int8"})


class TestLabelsFromSubfolder(unittest.TestCase):
    def test_local_directory_reads_subfolder_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "labels.json").write_text(
                json.dumps({"root_label": "root.skill:intent"}))
            (Path(tmp) / "int8").mkdir()
            (Path(tmp) / "int8" / "labels.json").write_text(
                json.dumps({"int8_label": "int8.skill:intent"}))
            pipeline = _make_pipeline({"model": tmp, "subfolder": "int8"})
        self.assertEqual(pipeline.label_map.get("int8_label"),
                         "int8.skill:intent")
        self.assertNotIn("root_label", pipeline.label_map)

    def test_hub_cache_lookup_names_the_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "labels.json"
            manifest.write_text(json.dumps({"int8_label": "int8.skill:intent"}))
            with patch("huggingface_hub.hf_hub_download",
                       return_value=str(manifest)) as hf:
                pipeline = _make_pipeline({"model": REPO, "subfolder": "int8"})
        # the classifier and its -low prototype stage each read the manifest
        self.assertTrue(hf.call_args_list)
        for call in hf.call_args_list:
            self.assertEqual(call.kwargs.get("subfolder"), "int8")
            self.assertTrue(call.kwargs.get("local_files_only"))
        self.assertEqual(pipeline.label_map.get("int8_label"),
                         "int8.skill:intent")


class TestPrototypeCacheKeyDiffersBySubfolder(unittest.TestCase):
    def test_prototype_disk_cache_model_id_names_the_subfolder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_pipeline({"model": REPO, "mode": "prototype",
                                   "prototype_cache_dir": tmp})
            int8 = _make_pipeline({"model": REPO, "mode": "prototype",
                                   "subfolder": "int8",
                                   "prototype_cache_dir": tmp})
        self.assertNotEqual(root._model_id, int8._model_id)


if __name__ == "__main__":
    unittest.main()
