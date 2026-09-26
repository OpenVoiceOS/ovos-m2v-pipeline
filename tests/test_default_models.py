"""Per-language default model resolution.

The plugin no longer falls back to the deprecated
``Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2``
default: an unconfigured install now resolves to
`OpenVoiceOS/ovos-m2v-intents-multilingual` for every language.
`OpenVoiceOS/ovos-m2v-intents-en` is a smaller, English-only alternative
with comparable held-out accuracy; it is never wired in as a built-in
default (purely to keep the built-in per-language table small) and is
only used when a deployment opts into it via `config["models"]`.
"""
import os
import unittest

import pytest

from ovos_m2v_pipeline import (
    DEFAULT_MODELS,
    DEFAULT_MULTILINGUAL,
    _resolve_model_id,
)

EN_MODEL = "OpenVoiceOS/ovos-m2v-intents-en"


class TestResolveModelId(unittest.TestCase):
    def test_en_us_resolves_to_multilingual_model_by_default(self):
        self.assertEqual(_resolve_model_id({}, "en-US"), DEFAULT_MULTILINGUAL)

    def test_pt_br_resolves_to_multilingual_model(self):
        self.assertEqual(_resolve_model_id({}, "pt-BR"),
                          DEFAULT_MULTILINGUAL)

    def test_unknown_language_resolves_to_multilingual_model(self):
        self.assertEqual(_resolve_model_id({}, "xx-XX"), DEFAULT_MULTILINGUAL)

    def test_explicit_model_config_wins_over_everything(self):
        config = {
            "model": "explicit/override",
            "models": {"en": "should-be-ignored"},
        }
        self.assertEqual(_resolve_model_id(config, "en-US"), "explicit/override")

    def test_models_map_can_opt_into_the_en_model(self):
        config = {"models": {"en": EN_MODEL}}
        self.assertEqual(_resolve_model_id(config, "en-US"), EN_MODEL)

    def test_models_map_full_locale_beats_builtin_default(self):
        config = {"models": {"en-us": "custom/en-us-model"}}
        self.assertEqual(_resolve_model_id(config, "en-US"), "custom/en-us-model")

    def test_models_map_primary_subtag_beats_builtin_default(self):
        config = {"models": {"pt": "custom/pt-model"}}
        self.assertEqual(_resolve_model_id(config, "pt-BR"), "custom/pt-model")

    def test_models_map_full_locale_beats_primary_subtag_entry(self):
        config = {"models": {"pt": "custom/pt-model",
                              "pt-br": "custom/pt-br-model"}}
        self.assertEqual(_resolve_model_id(config, "pt-BR"), "custom/pt-br-model")

    def test_no_fallback_to_deprecated_jarbas_model(self):
        for lang in ("en-US", "pt-BR", "xx-XX"):
            resolved = _resolve_model_id({}, lang)
            self.assertNotIn("Jarbas", resolved)

    def test_default_models_is_empty_multilingual_covers_every_language(self):
        self.assertEqual(DEFAULT_MODELS, {})


@pytest.mark.skipif(os.environ.get("OVOSCOPE_LIVE") != "1",
                     reason="Hub network test skipped; set OVOSCOPE_LIVE=1 to enable.")
class TestDefaultModelsLoadFromHub(unittest.TestCase):
    """The multilingual default and the opt-in English model must both
    actually exist on the Hub and ship a ``labels.json`` whose raw (pre
    `label_map`) ``valid_labels`` vocabulary covers the three built-in
    special labels.
    """

    SPECIAL_LABELS = ("ocp:play", "common_query:common_query", "stop:stop")

    @classmethod
    def setUpClass(cls):
        import huggingface_hub
        cls.manifests = {}
        for repo in (EN_MODEL, DEFAULT_MULTILINGUAL):
            try:
                path = huggingface_hub.hf_hub_download(repo, "labels.json")
            except Exception as e:  # pragma: no cover - network flake
                raise unittest.SkipTest(
                    f"HF Hub unreachable, cannot fetch '{repo}/labels.json': {e}")
            import json
            cls.manifests[repo] = json.loads(open(path, encoding="utf-8").read())

    def test_en_model_labels_manifest_covers_special_labels(self):
        valid = self.manifests[EN_MODEL]["valid_labels"]
        for label in self.SPECIAL_LABELS:
            self.assertIn(label, valid)

    def test_multilingual_model_labels_manifest_covers_special_labels(self):
        valid = self.manifests[DEFAULT_MULTILINGUAL]["valid_labels"]
        for label in self.SPECIAL_LABELS:
            self.assertIn(label, valid)


if __name__ == "__main__":
    unittest.main()
