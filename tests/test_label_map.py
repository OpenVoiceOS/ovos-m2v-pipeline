"""Tests for the configurable ``label_map`` / ``valid_labels`` layering.

Covers: default behaviour is byte-identical to the old hardcoded special-label
table when no config is given, a user `label_map` override, `valid_labels`
allow-list filtering, `labels.json` model-manifest layering (including
tolerance of a missing/corrupt manifest), and classifier-mode coverage of the
mapping being applied to a mocked `StaticModelPipeline` label head.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from tests.test_pipeline import (_make_pipeline, _make_prototype_pipeline,
                                  _setup_model as _setup_classifier)


class TestDefaultLabelMapUnchanged(unittest.TestCase):
    """With no `label_map` / `valid_labels` config, behaviour must match the
    old hardcoded OCP/stop/query table exactly."""

    def test_ocp_default_remap_through_match(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_classifier(p, ["ocp:play"], [0.95])
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play.play_search")

    def test_regular_label_default_split(self):
        p = _make_pipeline(intents=["skill_a:my.intent"], renormalize=False)
        _setup_classifier(p, ["skill_a:my.intent"], [0.9])
        skill_id, label = p._apply_special_label_map("skill_a:my.intent")
        self.assertEqual(skill_id, "skill_a")
        self.assertEqual(label, "skill_a:my.intent")


class TestUserLabelMapOverride(unittest.TestCase):
    def test_user_label_map_overrides_default_ocp(self):
        p = _make_pipeline(config={"model": "fake",
                                    "label_map": {"ocp:play": "custom.skill:custom.play"}},
                            intents=[], renormalize=False)
        _setup_classifier(p, ["ocp:play"], [0.95])
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "custom.skill")
        self.assertEqual(label, "custom.skill:custom.play")

    def test_user_label_map_new_label(self):
        p = _make_pipeline(config={"model": "fake",
                                    "label_map": {"weird:model_label": "my_skill:my.intent"}})
        skill_id, label = p._apply_special_label_map("weird:model_label")
        self.assertEqual(skill_id, "my_skill")
        self.assertEqual(label, "my_skill:my.intent")

    def test_missing_colon_target_used_as_is_with_warning(self):
        p = _make_pipeline(config={"model": "fake",
                                    "label_map": {"weird:model_label": "no_colon_target"}})
        with patch("ovos_m2v_pipeline.LOG.warning") as mock_warn:
            skill_id, label = p._apply_special_label_map("weird:model_label")
        self.assertEqual(skill_id, "no_colon_target")
        self.assertEqual(label, "no_colon_target")
        self.assertTrue(any("skill_id:intent" in c.args[0] for c in mock_warn.call_args_list))


class TestValidLabels(unittest.TestCase):
    def test_valid_labels_filters_non_listed(self):
        p = _make_pipeline(config={"model": "fake",
                                    "valid_labels": ["skill_a:allowed.intent"]},
                            intents=["skill_a:allowed.intent", "skill_b:blocked.intent"],
                            renormalize=False)
        _setup_classifier(p, ["skill_a:allowed.intent", "skill_b:blocked.intent"], [0.9, 0.8])
        results = list(p._match("test"))
        labels = [r[1] for r in results]
        self.assertEqual(labels, ["skill_a:allowed.intent"])

    def test_valid_labels_unset_does_not_filter(self):
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=False)
        _setup_classifier(p, ["skill_a:a.intent", "skill_b:b.intent"], [0.9, 0.8])
        results = list(p._match("test"))
        self.assertEqual(len(results), 2)

    def test_ignore_intents_still_works_post_mapping(self):
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill_a:a.intent"]},
                            intents=["skill_a:a.intent", "skill_b:b.intent"],
                            renormalize=False)
        _setup_classifier(p, ["skill_a:a.intent", "skill_b:b.intent"], [0.9, 0.8])
        results = list(p._match("test"))
        labels = [r[1] for r in results]
        self.assertEqual(labels, ["skill_b:b.intent"])


class TestValidLabelsCheckedBeforeSpecialMap(unittest.TestCase):
    """`labels.json` manifests built from the raw training-label vocabulary
    (as `train/build_dataset.py` writes and docs/README prescribe) list
    labels like `ocp:play`/`stop:stop`/`common_query:common_query` verbatim.
    `valid_labels` must therefore be checked against the raw label, before
    `_apply_special_label_map` rewrites it to its canonical bus topic -
    otherwise a manifest built exactly as documented discards every OCP-play,
    stop and common-query match."""

    def _manifest_pipeline(self, valid_labels):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "labels.json"), "w") as f:
                json.dump({"valid_labels": valid_labels}, f)
            p = _make_pipeline(config={"model": d}, intents=[], renormalize=False)
        return p

    def test_raw_special_label_in_manifest_is_matched_and_mapped(self):
        p = self._manifest_pipeline(["ocp:play", "stop:stop"])
        _setup_classifier(p, ["ocp:play"], [0.95])
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, _, _ = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play.play_search")

    def test_label_absent_from_manifest_is_still_dropped(self):
        p = self._manifest_pipeline(["stop:stop"])
        _setup_classifier(p, ["ocp:play"], [0.95])
        results = list(p._match("play some music"))
        self.assertEqual(results, [])


class TestValidLabelsNotAppliedInPrototypeMode(unittest.TestCase):
    """A model manifest's `valid_labels` describes the frozen classifier
    training vocabulary. Prototype labels are registered at runtime and are
    legitimately absent from that manifest, so the allow-list must not gate
    prototype-mode candidates."""

    def _manifest_prototype_pipeline(self, valid_labels, store):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "labels.json"), "w") as f:
                json.dump({"valid_labels": valid_labels}, f)
            p = _make_prototype_pipeline(config={"model": d}, intents=[],
                                          proto_store=store)
        return p

    def test_runtime_registered_label_absent_from_manifest_still_matches(self):
        from ovos_m2v_pipeline import PrototypeIntentStore
        store = PrototypeIntentStore(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            np.array(["skill_a:runtime.intent"]))
        # manifest only lists the classifier's frozen training labels -
        # the runtime-registered prototype label is not among them.
        p = self._manifest_prototype_pipeline(["skill_a:trained.intent"], store)
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("some utterance"))
        self.assertEqual(len(results), 1)
        skill_id, label, score, _ = results[0]
        self.assertEqual(label, "skill_a:runtime.intent")
        self.assertAlmostEqual(score, 1.0, places=4)


class TestModelManifestLayering(unittest.TestCase):
    """`labels.json` next to a local model directory forms the middle layer:
    defaults < manifest < user config."""

    def test_manifest_label_map_applies(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = {"custom:label": "manifest_skill:manifest.intent"}
            with open(os.path.join(d, "labels.json"), "w") as f:
                json.dump(manifest, f)
            p = _make_pipeline(config={"model": d})
            skill_id, label = p._apply_special_label_map("custom:label")
            self.assertEqual(skill_id, "manifest_skill")
            self.assertEqual(label, "manifest_skill:manifest.intent")

    def test_manifest_valid_labels_applies(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = {"valid_labels": ["skill_a:a.intent"]}
            with open(os.path.join(d, "labels.json"), "w") as f:
                json.dump(manifest, f)
            p = _make_pipeline(config={"model": d})
            self.assertEqual(p.valid_labels, ["skill_a:a.intent"])

    def test_user_config_overrides_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = {"custom:label": "manifest_skill:manifest.intent"}
            with open(os.path.join(d, "labels.json"), "w") as f:
                json.dump(manifest, f)
            p = _make_pipeline(config={"model": d,
                                        "label_map": {"custom:label": "user_skill:user.intent"}})
            skill_id, label = p._apply_special_label_map("custom:label")
            self.assertEqual(skill_id, "user_skill")
            self.assertEqual(label, "user_skill:user.intent")

    def test_missing_manifest_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = _make_pipeline(config={"model": d})
            # falls back to built-in default
            skill_id, label = p._apply_special_label_map("ocp:play")
            self.assertEqual(skill_id, "ovos.common_play")
            self.assertEqual(label, "ovos.common_play.play_search")

    def test_corrupt_manifest_is_ignored_with_warning(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "labels.json"), "w") as f:
                f.write("{not valid json")
            with patch("ovos_m2v_pipeline.LOG.warning") as mock_warn:
                p = _make_pipeline(config={"model": d})
            self.assertTrue(any("labels.json" in c.args[0] for c in mock_warn.call_args_list))
            # falls back to built-in default, plugin does not crash
            skill_id, label = p._apply_special_label_map("ocp:play")
            self.assertEqual(skill_id, "ovos.common_play")

    def test_manifest_not_an_object_is_ignored_with_warning(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "labels.json"), "w") as f:
                json.dump(["not", "a", "dict"], f)
            with patch("ovos_m2v_pipeline.LOG.warning") as mock_warn:
                p = _make_pipeline(config={"model": d})
            self.assertTrue(any("labels.json" in c.args[0] for c in mock_warn.call_args_list))
            skill_id, label = p._apply_special_label_map("ocp:play")
            self.assertEqual(skill_id, "ovos.common_play")


class TestClassifierModeLabelHead(unittest.TestCase):
    """Classifier mode wraps a frozen `StaticModelPipeline` label head; the
    label map must apply to whatever labels that head was trained with."""

    def test_label_map_applies_to_static_model_pipeline_head(self):
        mock_model = MagicMock()
        mock_model.classes_ = np.array(["trained:ocp_intent", "trained:other"])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])

        with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
             patch("ovos_m2v_pipeline.Configuration", return_value={}):
            MockSMP.from_pretrained.return_value = mock_model
            from ovos_m2v_pipeline import Model2VecIntentPipeline
            from ovos_utils.fakebus import FakeBus
            config = {"model": "fake-model", "renormalize": False,
                      "label_map": {"trained:ocp_intent": "ovos.common_play:ovos.common_play.play_search"}}
            p = Model2VecIntentPipeline(bus=FakeBus(), config=config)
        p.model = mock_model
        p.intents = {"trained:ocp_intent"}

        results = list(p._match("play some jazz"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob, _ = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play:ovos.common_play.play_search")
        self.assertAlmostEqual(prob, 0.9)


class TestManifestNeverBlocksNetwork(unittest.TestCase):
    """`_load_labels_manifest` must never reach out to the network from the
    plugin constructor for a hub-id model: only the local HF cache."""

    def test_offline_hub_id_construction_is_fast_and_empty(self):
        # `_initial_intent_sync` (bus roundtrip, unrelated to labels.json)
        # has its own timeout-bounded cost; isolate the manifest lookup by
        # patching it out so this test measures only what this PR touches.
        import time
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            with patch("ovos_m2v_pipeline.Model2VecIntentPipeline._initial_intent_sync"):
                start = time.monotonic()
                p = _make_pipeline(config={"model": "definitely-not-a/real-repo-m2v-test"})
                elapsed = time.monotonic() - start
        finally:
            os.environ.pop("HF_HUB_OFFLINE", None)
        self.assertLess(elapsed, 1.0,
                         "manifest lookup for an uncached hub id must not block")
        self.assertEqual(p.label_map.get("custom:label"), None)

    def test_manifest_lookup_alone_is_fast_and_empty(self):
        import time
        from ovos_m2v_pipeline import _load_labels_manifest
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            start = time.monotonic()
            result = _load_labels_manifest("definitely-not-a/real-repo-m2v-test")
            elapsed = time.monotonic() - start
        finally:
            os.environ.pop("HF_HUB_OFFLINE", None)
        self.assertLess(elapsed, 1.0)
        self.assertEqual(result, {})

    def test_hub_id_local_cache_hit_applies_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            cached_path = os.path.join(d, "labels.json")
            with open(cached_path, "w") as f:
                json.dump({"custom:label": "cached_skill:cached.intent"}, f)

            with patch("huggingface_hub.hf_hub_download",
                       return_value=cached_path) as mock_dl:
                p = _make_pipeline(config={"model": "some-org/some-repo"})

            # local_files_only=True: only the local cache is consulted, no
            # network call is made by this lookup.
            _, kwargs = mock_dl.call_args
            self.assertTrue(kwargs.get("local_files_only"))
            skill_id, label = p._apply_special_label_map("custom:label")
            self.assertEqual(skill_id, "cached_skill")
            self.assertEqual(label, "cached_skill:cached.intent")

    def test_hub_id_cache_miss_uses_local_files_only(self):
        import huggingface_hub
        with patch("huggingface_hub.hf_hub_download",
                   side_effect=huggingface_hub.errors.LocalEntryNotFoundError(
                       "not cached")) as mock_dl:
            p = _make_pipeline(config={"model": "some-org/some-repo"})
        _, kwargs = mock_dl.call_args
        self.assertTrue(kwargs.get("local_files_only"))
        # falls back to built-in defaults, does not crash
        skill_id, label = p._apply_special_label_map("ocp:play")
        self.assertEqual(skill_id, "ovos.common_play")


class TestLabelMapWarningDeduped(unittest.TestCase):
    def test_colon_less_warning_logs_once_per_label(self):
        p = _make_pipeline(config={"model": "fake",
                                    "label_map": {"weird:model_label": "no_colon_target"}})
        with patch("ovos_m2v_pipeline.LOG.warning") as mock_warn:
            p._apply_special_label_map("weird:model_label")
            p._apply_special_label_map("weird:model_label")
            p._apply_special_label_map("weird:model_label")
        self.assertEqual(mock_warn.call_count, 1)


if __name__ == "__main__":
    unittest.main()
