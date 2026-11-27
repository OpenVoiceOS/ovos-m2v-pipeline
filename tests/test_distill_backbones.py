"""Backbone manifest handling in train/distill.py."""
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

import distill

MANIFEST = {
    "version": 1,
    "backbones": [
        {"id": "mul-a", "model": "org/mul-a", "revision": "aaa",
         "languages": ["mul"]},
        {"id": "gl-a", "model": "org/gl-a", "revision": "bbb",
         "languages": ["gl"]},
        {"id": "pt-a", "model": "org/pt-a", "revision": "ccc",
         "languages": ["pt"]},
    ],
    "selected": {},
}


class TestSelection(unittest.TestCase):
    def test_a_language_gets_its_own_backbones_and_the_multilingual_ones(self):
        picked = [b["id"] for b in distill.selected_backbones(MANIFEST, "gl")]
        self.assertEqual(picked, ["mul-a", "gl-a"])

    def test_the_multilingual_baseline_can_be_dropped(self):
        picked = [b["id"] for b in
                  distill.selected_backbones(MANIFEST, "gl", multilingual=False)]
        self.assertEqual(picked, ["gl-a"])

    def test_a_locale_takes_its_language_and_its_own_backbones(self):
        manifest = {"backbones": [
            {"id": "pt", "model": "o/pt", "revision": "a", "languages": ["pt"]},
            {"id": "ptpt", "model": "o/ptpt", "revision": "b", "languages": ["pt-PT"]},
            {"id": "ptbr", "model": "o/ptbr", "revision": "c", "languages": ["pt-BR"]},
        ]}
        self.assertEqual(
            [b["id"] for b in distill.selected_backbones(manifest, "pt-PT")],
            ["pt", "ptpt"])
        self.assertEqual(
            [b["id"] for b in distill.selected_backbones(manifest, "pt")],
            ["pt"])

    def test_no_lang_selects_every_backbone(self):
        picked = [b["id"] for b in distill.selected_backbones(MANIFEST, None)]
        self.assertEqual(picked, ["mul-a", "gl-a", "pt-a"])

    def test_a_language_with_no_backbone_at_all_is_an_error(self):
        manifest = {"backbones": [b for b in MANIFEST["backbones"]
                                  if "mul" not in b["languages"]]}
        with self.assertRaises(SystemExit):
            distill.selected_backbones(manifest, "de")


class TestManifestIds(unittest.TestCase):
    """Each backbone builds into ``<out>/<id>``, so an id names one directory."""

    def test_every_backbone_id_in_the_shipped_manifest_is_unique(self):
        manifest = yaml.safe_load(distill.MANIFEST.read_text())
        ids = [b["id"] for b in manifest["backbones"]]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        self.assertEqual(duplicates, [])

    def test_a_manifest_that_repeats_an_id_is_refused_when_it_loads(self):
        text = ("version: 1\nbackbones:\n"
                "  - id: bert-de\n    model: one/bert-de\n    revision: a\n"
                "    languages: [de]\n"
                "  - id: bert-de\n    model: two/bert-de\n    revision: b\n"
                "    languages: [de]\n")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "backbones.yaml"
            path.write_text(text)
            with self.assertRaises(SystemExit) as raised:
                distill.load_manifest(path)
        self.assertIn("bert-de", str(raised.exception))


class TestResume(unittest.TestCase):
    def _record(self, target: Path, model: str, revision: str):
        target.mkdir(parents=True)
        (target / "distill.json").write_text(
            json.dumps({"model": model, "revision": revision}))

    def test_a_build_at_the_same_revision_is_skipped(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "gl-a"
            self._record(target, "org/gl-a", "bbb")
            self.assertTrue(distill.already_built(target, MANIFEST["backbones"][1]))

    def test_a_build_at_another_revision_is_rebuilt(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "gl-a"
            self._record(target, "org/gl-a", "older-sha")
            self.assertFalse(distill.already_built(target, MANIFEST["backbones"][1]))

    def test_a_directory_without_a_record_is_rebuilt(self):
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "gl-a"
            target.mkdir()
            self.assertFalse(distill.already_built(target, MANIFEST["backbones"][1]))


class TestCustomCode(unittest.TestCase):
    def test_a_backbone_shipping_code_is_skipped_without_the_flag(self):
        entry = {"id": "risky", "model": "org/risky", "revision": "aaa",
                 "languages": ["cy"], "custom_code": True}
        with TemporaryDirectory() as tmp:
            self.assertIsNone(distill.distil_one(entry, Path(tmp)))
            self.assertFalse((Path(tmp) / "risky").exists())


class TestVocabCeiling(unittest.TestCase):
    def test_a_backbone_over_the_ceiling_is_skipped_before_it_loads(self):
        loaded = []
        entry = {"id": "huge", "model": "org/huge", "revision": "aaa",
                 "languages": ["mul"]}

        class Tokenizer:
            @staticmethod
            def get_vocab():
                return {str(i): i for i in range(500)}

        class Auto:
            @staticmethod
            def from_pretrained(*a, **k):
                loaded.append(a)
                return Tokenizer()

        import types
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = Auto
        transformers.AutoModel = Auto
        sys.modules["transformers"] = transformers
        try:
            with TemporaryDirectory() as tmp:
                self.assertIsNone(
                    distill.distil_one(entry, Path(tmp), max_vocab=100))
                self.assertFalse((Path(tmp) / "huge").exists())
            # the tokenizer is read, the encoder never is
            self.assertEqual(len(loaded), 1)
        finally:
            del sys.modules["transformers"]


class TestPinning(unittest.TestCase):
    def test_only_empty_revisions_are_resolved_and_comments_survive(self):
        text = ("# keep me\nversion: 1\nbackbones:\n"
                "  - id: gl-a\n    model: org/gl-a\n    revision:\n"
                "    languages: [gl]\n"
                "  - id: pt-a\n    model: org/pt-a\n    revision: kept\n"
                "    languages: [pt]\n")
        with TemporaryDirectory() as tmp, mock.patch.object(
                distill, "resolve_revision", lambda model: "resolved-sha"):
            path = Path(tmp) / "backbones.yaml"
            path.write_text(text)
            self.assertEqual(distill.pin_revisions(path), 1)
            written = path.read_text()
            self.assertIn("# keep me", written)
            parsed = yaml.safe_load(written)
            self.assertEqual(parsed["backbones"][0]["revision"], "resolved-sha")
            self.assertEqual(parsed["backbones"][1]["revision"], "kept")

    def test_an_empty_string_or_null_revision_is_pinned(self):
        for empty in ('""', "''", "null", "~"):
            text = ("version: 1\nbackbones:\n"
                    f"  - id: gl-a\n    model: org/gl-a\n    revision: {empty}\n"
                    "    languages: [gl]\n")
            with TemporaryDirectory() as tmp, mock.patch.object(
                    distill, "resolve_revision", lambda model: "resolved-sha"):
                path = Path(tmp) / "backbones.yaml"
                path.write_text(text)
                self.assertEqual(distill.pin_revisions(path), 1, empty)
                parsed = yaml.safe_load(path.read_text())
                self.assertEqual(parsed["backbones"][0]["revision"],
                                 "resolved-sha", empty)

    def test_an_entry_the_rewrite_cannot_find_is_not_counted(self):
        # `revision` before `model`: the entry is empty, but the in-place
        # rewrite has no line to change, so nothing was pinned
        text = ("version: 1\nbackbones:\n"
                "  - id: gl-a\n    revision:\n    model: org/gl-a\n"
                "    languages: [gl]\n")
        with TemporaryDirectory() as tmp, mock.patch.object(
                distill, "resolve_revision", lambda model: "resolved-sha"):
            path = Path(tmp) / "backbones.yaml"
            path.write_text(text)
            self.assertEqual(distill.pin_revisions(path), 0)
            self.assertEqual(path.read_text(), text)


class TestPinningLeavesTheModuleUnpatched(unittest.TestCase):
    def test_resolve_revision_is_restored_after_each_pinning_test(self):
        original = distill.resolve_revision
        try:
            for name in unittest.defaultTestLoader.getTestCaseNames(TestPinning):
                result = unittest.TestResult()
                TestPinning(name).run(result)
                self.assertTrue(result.wasSuccessful(), name)
                self.assertIs(distill.resolve_revision, original, name)
        finally:
            distill.resolve_revision = original


if __name__ == "__main__":
    unittest.main()
