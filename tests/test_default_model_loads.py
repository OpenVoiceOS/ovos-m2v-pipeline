"""The default model must actually load.

`model2vec` asks for `tokenizers>=0.20` with no ceiling. Under
`--prerelease=allow`, which is how every alpha of this plugin is installed,
the resolver took `tokenizers` 1.0.0rc2. That release ships without the
`fancy-regex` backend, so any model whose tokenizer uses a regex `Split`
pre-tokenizer raises on load:

    ValueError: no system-regex backend compiled: enable the `fancy-regex`
    feature to use a regex pattern in a `Split` pre-tokenizer or a
    `Replace` normalizer

The default model is one of them, so the plugin could not load any model at
all and no unit test noticed: every one of them mocks the load.

This test does not mock it. It never skips: a skip here is the same silence
that let the defect ship.
"""
import unittest

from ovos_m2v_pipeline import DEFAULT_MULTILINGUAL, load_shared_model


class TestTheDefaultModelLoads(unittest.TestCase):

    def test_tokenizers_is_below_the_release_that_cannot_load_it(self):
        # The cheap half, so a resolver that drifts back is named by the
        # test rather than by a 70 MB download timing out.
        from importlib.metadata import version
        from packaging.version import Version
        installed = Version(version("tokenizers"))
        # `.release` and not the Version itself: under PEP 440
        # `Version("1.0.0rc2") < Version("1.0.0")` is True, so comparing the
        # versions would pass on exactly the release this guards against.
        # The specifier `tokenizers<1.0.0` does exclude the release
        # candidate, which is why the pin works; the test has to say the
        # same thing a different way.
        self.assertLess(
            installed.release, (1, 0, 0),
            "tokenizers %s has no fancy-regex backend, so the default "
            "model cannot load; the ceiling in pyproject.toml is gone or "
            "was overridden" % installed)

    def test_the_default_model_loads_for_real(self):
        from huggingface_hub import snapshot_download

        path = snapshot_download(DEFAULT_MULTILINGUAL)
        model = load_shared_model(path, "classifier")
        self.assertIsNotNone(model)
        classes = getattr(model, "classes_", None)
        self.assertIsNotNone(
            classes, "the pipeline loaded without a classifier head")
        self.assertGreater(len(classes), 0)


if __name__ == "__main__":
    unittest.main()
