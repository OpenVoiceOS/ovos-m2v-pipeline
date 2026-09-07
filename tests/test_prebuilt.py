"""Tests for exporting/loading prebuilt prototype-mode centroids.

Covers the Raspberry Pi workflow this module exists for: prebuild the
centroids once (desktop), ship the artifact, and boot a device against it
without calling the encoder for any label the artifact already covers.
"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message

from ovos_m2v_pipeline import PrototypeIntentStore
from ovos_m2v_pipeline.cache import compute_cache_key
from ovos_m2v_pipeline.prebuilt import export_store, load_prebuilt_store


def _make_hash_encode(dim=16):
    def _encode(sentences, **kwargs):
        out = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, sent in enumerate(sentences):
            for tok in sent.lower().split():
                bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
                out[i, bucket] += 1.0
        return out
    return _encode


_hash_encode = _make_hash_encode(16)


def _make_pipeline(extra_config=None, prebuilt_dir=None):
    # `prototype_cache` disabled by default: these tests exercise the
    # PREBUILT-artifact skip-the-encoder path, not the on-disk
    # PrototypeCache (see test_prototype_cache.py for that), and the
    # default cache dir is a real, persistent, shared XDG path -- reusing
    # it here would let one test's cache entry silently satisfy another
    # test's "encoder must be called" assertion.
    config = {"model": "fake-embed-model", "mode": "prototype",
              "prototype_cache": False}
    if prebuilt_dir is not None:
        config["prebuilt_prototypes"] = str(prebuilt_dir)
    if extra_config:
        config.update(extra_config)

    mock_embed_model = MagicMock()
    mock_embed_model.encode.side_effect = _hash_encode
    mock_embed_model.dim = 16

    fake_m2v = MagicMock()
    fake_m2v.__version__ = "0.9.0"
    fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)
    pipeline.model = mock_embed_model
    # Real boot validates a loaded prebuilt store's dimension against the
    # model the moment it becomes available (`_load_model_now`); these tests
    # set `model` directly rather than going through that deferred load, so
    # replicate the one step it would otherwise have run.
    pipeline._validate_prebuilt_dim()
    return pipeline


def _register(pipeline, samples, name="test_skill:demo", skill_id="test_skill"):
    pipeline.bus.emit(Message(
        "padatious:register_intent",
        {"name": name, "samples": samples, "lang": "en-US"},
        {"skill_id": skill_id}))


class TestExportImportRoundTrip(unittest.TestCase):
    def test_round_trip_identical_matrices_and_labels(self):
        model = MagicMock()
        model.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()
        store.add(model, "weather:current", ["what's the weather", "how is the weather"])
        store.add(model, "alarm:set", ["set an alarm"])

        with tempfile.TemporaryDirectory() as tmp:
            out = export_store(store, tmp, model_id="fake-model",
                                model2vec_version="0.9.0")
            self.assertTrue((Path(out) / "manifest.json").exists())
            self.assertTrue((Path(out) / "prototypes.npz").exists())

            loaded, cache_keys, reason = load_prebuilt_store(
                out, model_id="fake-model", model2vec_version="0.9.0")

        self.assertEqual(reason, "ok")
        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(
            np.sort(loaded.unique_labels), np.sort(store.unique_labels))
        # every original embedding is present in the reloaded store (order
        # across labels is not guaranteed, so match rows by nearest neighbor
        # rather than exact float32 equality -- the constructor re-derives
        # the L2 norm on load, which is idempotent up to float rounding)
        orig = store.embeddings
        loaded_emb = loaded.embeddings
        self.assertEqual(orig.shape, loaded_emb.shape)
        similarity = orig @ loaded_emb.T
        best_match = similarity.argmax(axis=1)
        for i, j in enumerate(best_match):
            np.testing.assert_allclose(orig[i], loaded_emb[j], atol=1e-6)


class TestPrebuiltSkipsEncoder(unittest.TestCase):
    def test_boot_with_prebuilt_skips_encoder_for_matching_label(self):
        # Pass 1: build + export a prebuilt artifact for one label.
        p0 = _make_pipeline()
        _register(p0, ["turn on the lights", "switch the lights on"])
        with tempfile.TemporaryDirectory() as tmp:
            p0.prototype_store.export(
                tmp, model_id=p0._model_id,
                model2vec_version=p0._model2vec_version,
                cache_keys=p0._prebuilt_cache_keys or {
                    "test_skill:demo": p0._prototype_cache_key(
                        ["turn on the lights", "switch the lights on"],
                        lang="en-US")
                },
            )

            # Pass 2: fresh boot, prebuilt configured -> encoder untouched.
            p1 = _make_pipeline(prebuilt_dir=tmp)
            _register(p1, ["turn on the lights", "switch the lights on"])
            p1.model.encode.assert_not_called()
            self.assertEqual(len(p1.prototype_store), 2)

    def test_without_prebuilt_encoder_is_called(self):
        p = _make_pipeline()
        _register(p, ["turn on the lights", "switch the lights on"])
        self.assertTrue(p.model.encode.called)
        self.assertEqual(len(p.prototype_store), 2)

    def test_mismatched_model_id_falls_back_to_encoding(self):
        p0 = _make_pipeline()
        _register(p0, ["turn on the lights"])
        with tempfile.TemporaryDirectory() as tmp:
            p0.prototype_store.export(
                tmp, model_id="a-different-model",
                model2vec_version=p0._model2vec_version,
                cache_keys={"test_skill:demo": p0._prototype_cache_key(
                    ["turn on the lights"], lang="en-US")},
            )
            p1 = _make_pipeline(prebuilt_dir=tmp)
            self.assertEqual(p1._prebuilt_cache_keys, {})
            _register(p1, ["turn on the lights"])
            p1.model.encode.assert_called_once()


class TestLoadPrebuiltVerification(unittest.TestCase):
    def test_manifest_model_id_mismatch_reports_reason_and_returns_none(self):
        model = MagicMock()
        model.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()
        store.add(model, "weather:current", ["what's the weather"])
        with tempfile.TemporaryDirectory() as tmp:
            export_store(store, tmp, model_id="model-a", model2vec_version="0.9.0")
            loaded, keys, reason = load_prebuilt_store(
                tmp, model_id="model-b", model2vec_version="0.9.0")
        self.assertIsNone(loaded)
        self.assertEqual(keys, {})
        self.assertIn("model id mismatch", reason)

    def test_manifest_model2vec_version_mismatch_returns_none(self):
        model = MagicMock()
        model.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()
        store.add(model, "weather:current", ["what's the weather"])
        with tempfile.TemporaryDirectory() as tmp:
            export_store(store, tmp, model_id="model-a", model2vec_version="0.9.0")
            loaded, keys, reason = load_prebuilt_store(
                tmp, model_id="model-a", model2vec_version="0.9.9")
        self.assertIsNone(loaded)
        self.assertIn("model2vec version mismatch", reason)

    def test_missing_artifact_files_reports_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            loaded, keys, reason = load_prebuilt_store(
                tmp, model_id="model-a", model2vec_version="0.9.0")
        self.assertIsNone(loaded)
        self.assertIn("missing", reason)


class TestPrebuiltFromHFRepoId(unittest.TestCase):
    def test_hf_repo_id_resolves_via_snapshot_download(self):
        model = MagicMock()
        model.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()
        store.add(model, "weather:current", ["what's the weather"])
        with tempfile.TemporaryDirectory() as tmp:
            export_store(store, tmp, model_id="model-a", model2vec_version="0.9.0")

            fake_hub = MagicMock()
            fake_hub.snapshot_download.return_value = tmp
            fake_hub.errors = MagicMock()
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                loaded, keys, reason = load_prebuilt_store(
                    "SomeOrg/prebuilt-prototypes",
                    model_id="model-a", model2vec_version="0.9.0")
            fake_hub.snapshot_download.assert_called_once_with(
                "SomeOrg/prebuilt-prototypes", repo_type="dataset")
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(loaded)


class TestCLIExportFromSkillDir(unittest.TestCase):
    def test_export_from_skill_dir_produces_expected_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skill-demo"
            locale_dir = skill_dir / "locale" / "en-us"
            locale_dir.mkdir(parents=True)
            (locale_dir / "hello.intent").write_text(
                "hello\nhi there\n", encoding="utf-8")
            (locale_dir / "bye.intent").write_text(
                "goodbye\nsee you\n", encoding="utf-8")
            out_dir = Path(tmp) / "artifact"

            fake_m2v = MagicMock()
            fake_m2v.__version__ = "0.9.0"
            mock_embed_model = MagicMock()
            mock_embed_model.encode.side_effect = _hash_encode
            fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

            from ovos_m2v_pipeline.cli import main
            with patch.dict(sys.modules, {"model2vec": fake_m2v}):
                rc = main([
                    "export", "--out", str(out_dir),
                    "--model", "fake-model",
                    "--skill-dir", str(skill_dir),
                    "--skill-id", "skill-demo",
                ])
            self.assertEqual(rc, 0)

            import json
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(
                sorted(manifest["labels"]),
                ["skill-demo:bye", "skill-demo:hello"],
            )
            self.assertEqual(manifest["model_id"], "fake-model")

    def test_nested_locale_layout_is_discovered(self):
        # Every real skill nests templates under a resource subdirectory
        # (`locale/<lang>/intents/`, mirroring `ovos_workshop.resource_files`)
        # rather than the flat `locale/<lang>/*.intent` this exporter used to
        # assume.
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skill-demo"
            locale_dir = skill_dir / "locale" / "en-us" / "intents"
            locale_dir.mkdir(parents=True)
            (locale_dir / "foo.intent").write_text("do the foo\n", encoding="utf-8")
            out_dir = Path(tmp) / "artifact"

            fake_m2v = MagicMock()
            fake_m2v.__version__ = "0.9.0"
            mock_embed_model = MagicMock()
            mock_embed_model.encode.side_effect = _hash_encode
            fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

            from ovos_m2v_pipeline.cli import main
            with patch.dict(sys.modules, {"model2vec": fake_m2v}):
                rc = main([
                    "export", "--out", str(out_dir),
                    "--model", "fake-model",
                    "--skill-dir", str(skill_dir),
                    "--skill-id", "skill-demo",
                ])
            self.assertEqual(rc, 0)

            import json
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["labels"], ["skill-demo:foo"])

    def test_flat_locale_layout_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skill-demo"
            locale_dir = skill_dir / "locale" / "en-us"
            locale_dir.mkdir(parents=True)
            (locale_dir / "foo.intent").write_text("do the foo\n", encoding="utf-8")
            out_dir = Path(tmp) / "artifact"

            fake_m2v = MagicMock()
            fake_m2v.__version__ = "0.9.0"
            mock_embed_model = MagicMock()
            mock_embed_model.encode.side_effect = _hash_encode
            fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

            from ovos_m2v_pipeline.cli import main
            with patch.dict(sys.modules, {"model2vec": fake_m2v}):
                rc = main([
                    "export", "--out", str(out_dir),
                    "--model", "fake-model",
                    "--skill-dir", str(skill_dir),
                    "--skill-id", "skill-demo",
                ])
            self.assertEqual(rc, 0)

            import json
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["labels"], ["skill-demo:foo"])

    def test_zero_labels_found_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "skill-demo"
            (skill_dir / "locale" / "en-us").mkdir(parents=True)
            out_dir = Path(tmp) / "artifact"

            fake_m2v = MagicMock()
            fake_m2v.__version__ = "0.9.0"
            mock_embed_model = MagicMock()
            mock_embed_model.encode.side_effect = _hash_encode
            fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

            from ovos_m2v_pipeline.cli import main
            with patch.dict(sys.modules, {"model2vec": fake_m2v}):
                rc = main([
                    "export", "--out", str(out_dir),
                    "--model", "fake-model",
                    "--skill-dir", str(skill_dir),
                    "--skill-id", "skill-demo",
                ])
            self.assertNotEqual(rc, 0)


class TestPrebuiltDimensionMismatch(unittest.TestCase):
    def test_dimension_mismatch_discards_store_and_falls_back_to_encoding(self):
        # A model retrained in place under the same id/model2vec version can
        # still change its output dimension; the manifest/npz are internally
        # consistent (4-d, 4-d) so `load_prebuilt_store`'s own check passes,
        # but the real running model now produces 8-d embeddings.
        model_4d = MagicMock()
        model_4d.encode.side_effect = _make_hash_encode(4)
        store = PrototypeIntentStore()
        store.add(model_4d, "skillA:foo", ["do the foo"])
        with tempfile.TemporaryDirectory() as tmp:
            export_store(store, tmp, model_id="fake/model", model2vec_version="0.9.0")

            p1 = _make_pipeline(prebuilt_dir=tmp)  # model.dim == 16 in _make_pipeline
        # The mismatched artifact must have been discarded at validation time.
        self.assertEqual(p1._prebuilt_cache_keys, {})
        self.assertEqual(len(p1.prototype_store), 0)

        # Every registration re-encodes through the (16-d) live model instead
        # of reusing the discarded, wrong-dimension centroids.
        _register(p1, ["do the foo"])
        p1.model.encode.assert_called()

        # A match call against the fallback store must not raise -- the bad
        # dimension can never reach `scores()`.
        query = np.zeros(16, dtype=np.float32)
        scores = p1.prototype_store.scores(query)
        self.assertEqual(len(scores), len(p1.prototype_store.unique_labels))

    def test_registration_before_model_load_waits_for_validation(self):
        # A registration whose cache key matches the prebuilt manifest must
        # not take the skip-the-encoder fast path before `_prebuilt_validated`
        # is set -- i.e. before the real model dimension has been checked.
        model_16d = MagicMock()
        model_16d.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()
        store.add(model_16d, "test_skill:demo",
                  ["turn on the lights", "switch the lights on"], lang="en-US")
        with tempfile.TemporaryDirectory() as tmp:
            samples = ["turn on the lights", "switch the lights on"]
            from ovos_m2v_pipeline import MAX_ENTITY_EXPANSIONS
            cache_key = compute_cache_key(
                "fake-embed-model", "0.9.0",
                {"k": None, "strategy": "max_over_all",
                 "max_expansions": MAX_ENTITY_EXPANSIONS},
                samples, lang="en-US",
            )
            export_store(store, tmp, model_id="fake-embed-model",
                         model2vec_version="0.9.0",
                         cache_keys={"test_skill:demo": cache_key})

            config = {"model": "fake-embed-model", "mode": "prototype",
                      "prototype_cache": False,
                      "prebuilt_prototypes": tmp}
            mock_embed_model = MagicMock()
            mock_embed_model.encode.side_effect = _hash_encode
            mock_embed_model.dim = 16
            fake_m2v = MagicMock()
            fake_m2v.__version__ = "0.9.0"
            fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model
            with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
                 patch("ovos_m2v_pipeline.Configuration", return_value={}), \
                 patch.dict(sys.modules, {"model2vec": fake_m2v}):
                from ovos_m2v_pipeline import Model2VecIntentPipeline
                from ovos_utils.fakebus import FakeBus
                pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)

            # Model not loaded yet: `_prebuilt_validated` is False, so a
            # registration whose cache key matches the manifest must still
            # buffer rather than take the skip-the-encoder fast path.
            self.assertFalse(pipeline._prebuilt_validated)
            _register(pipeline, samples)
            self.assertEqual(len(pipeline._pending_additions), 1)
            mock_embed_model.encode.assert_not_called()

            # Model becomes available (mirrors `_load_model_now`): validation
            # confirms the dimension matches, then buffered registrations
            # flush by encoding through the now-known-good store (a buffered
            # registration is always encoded once flushed -- the prebuilt
            # fast path only ever applies to registrations that arrive
            # *after* validation, per `_handle_register_padatious`).
            with pipeline._model_lock:
                pipeline.model = mock_embed_model
                pipeline._validate_prebuilt_dim()
                pipeline._flush_pending_additions()
            self.assertTrue(pipeline._prebuilt_validated)
            self.assertEqual(pipeline._pending_additions, [])
            mock_embed_model.encode.assert_called()

            # A registration arriving after validation, with a matching
            # cache key, now takes the fast path and skips the encoder.
            mock_embed_model.encode.reset_mock()
            _register(pipeline, samples)
            mock_embed_model.encode.assert_not_called()


class TestPrebuiltIsNotAnAllowList(unittest.TestCase):
    """A prebuilt artifact is a cache for the registration encode step, not
    a second allow-list: only labels a loaded skill actually registers may
    ever become matchable, exactly as without a prebuilt artifact at all."""

    def test_unregistered_label_never_matches(self):
        # Artifact carries a label no skill in this process ever registers
        # (e.g. built for a skill that isn't installed here, or dropped
        # since). It must not be matchable, must not appear in `intents`,
        # and must not occupy any row of the live store.
        model = MagicMock()
        model.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()
        store.add(model, "ghost-skill:someone", ["nuke the reactor"])
        with tempfile.TemporaryDirectory() as tmp:
            from ovos_m2v_pipeline import MAX_ENTITY_EXPANSIONS
            export_store(store, tmp, model_id="fake-embed-model",
                         model2vec_version="0.9.0",
                         cache_keys={"ghost-skill:someone": compute_cache_key(
                             "fake-embed-model", "0.9.0",
                             {"k": None, "strategy": "max_over_all",
                              "max_expansions": MAX_ENTITY_EXPANSIONS},
                             ["nuke the reactor"], lang="en-US")})

            p1 = _make_pipeline(prebuilt_dir=tmp)

        self.assertEqual(p1.intents, set())
        self.assertEqual(len(p1.prototype_store), 0)
        candidates = list(p1._match_prototype("nuke the reactor"))
        self.assertEqual(candidates, [])

    def test_stale_label_only_registered_sibling_is_matchable(self):
        # Artifact carries two labels; only one has a live registration
        # (the other simulates a skill that renamed/dropped its intent
        # after the artifact was built -- no detach event will ever fire
        # for a label the runtime never registered).
        model = MagicMock()
        model.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()
        live_samples = ["turn on the lights", "switch the lights on"]
        stale_samples = ["nuke the reactor"]
        store.add(model, "test_skill:demo", live_samples, lang="en-US")
        store.add(model, "ghost-skill:stale", stale_samples, lang="en-US")
        with tempfile.TemporaryDirectory() as tmp:
            from ovos_m2v_pipeline import MAX_ENTITY_EXPANSIONS
            cache_keys = {
                "test_skill:demo": compute_cache_key(
                    "fake-embed-model", "0.9.0",
                    {"k": None, "strategy": "max_over_all",
                     "max_expansions": MAX_ENTITY_EXPANSIONS},
                    live_samples, lang="en-US"),
                "ghost-skill:stale": compute_cache_key(
                    "fake-embed-model", "0.9.0",
                    {"k": None, "strategy": "max_over_all",
                     "max_expansions": MAX_ENTITY_EXPANSIONS},
                    stale_samples, lang="en-US"),
            }
            export_store(store, tmp, model_id="fake-embed-model",
                         model2vec_version="0.9.0", cache_keys=cache_keys)

            p1 = _make_pipeline(prebuilt_dir=tmp)
            _register(p1, live_samples)
            p1.model.encode.assert_not_called()  # fast path for the registered label

        self.assertEqual(p1.intents, {"test_skill:demo"})
        np.testing.assert_array_equal(
            np.sort(p1.prototype_store.unique_labels), ["test_skill:demo"])
        candidates = dict(
            (label, score) for _, label, score in
            p1._match_prototype("nuke the reactor"))
        self.assertNotIn("stale", candidates)
        self.assertEqual(len(p1.prototype_store), len(live_samples))

    def test_detach_after_prebuilt_fed_registration_removes_label(self):
        model = MagicMock()
        model.encode.side_effect = _hash_encode
        store = PrototypeIntentStore()
        samples = ["turn on the lights", "switch the lights on"]
        store.add(model, "test_skill:demo", samples, lang="en-US")
        with tempfile.TemporaryDirectory() as tmp:
            from ovos_m2v_pipeline import MAX_ENTITY_EXPANSIONS
            cache_key = compute_cache_key(
                "fake-embed-model", "0.9.0",
                {"k": None, "strategy": "max_over_all",
                 "max_expansions": MAX_ENTITY_EXPANSIONS},
                samples, lang="en-US")
            export_store(store, tmp, model_id="fake-embed-model",
                         model2vec_version="0.9.0",
                         cache_keys={"test_skill:demo": cache_key})

            p1 = _make_pipeline(prebuilt_dir=tmp)
            _register(p1, samples)
            self.assertEqual(p1.intents, {"test_skill:demo"})
            self.assertEqual(len(p1.prototype_store), len(samples))

            p1.bus.emit(Message(
                "detach_intent", {"intent_name": "test_skill:demo"}, {}))

        self.assertEqual(p1.intents, set())
        self.assertEqual(len(p1.prototype_store), 0)


if __name__ == "__main__":
    unittest.main()
