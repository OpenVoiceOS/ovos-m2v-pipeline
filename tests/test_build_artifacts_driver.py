"""The two published artifacts must describe the same label set.

The driver runs both producers against one corpus and publishes only when
their label sets agree, so the tests drive it with stub producers: what is
under test is the invariant and the refusal, not the fitting or the encoding.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[1] / "train" / "build_artifacts.py"


def _stub_tree(tmp_path, classifier_labels, prototype_labels):
    """A fake train.py and ovos_m2v_pipeline.cli writing the given labels."""
    root = tmp_path / "stub"
    (root / "train").mkdir(parents=True)
    (root / "ovos_m2v_pipeline").mkdir(parents=True)
    (root / "train" / "build_artifacts.py").write_text(DRIVER.read_text(),
                                                        encoding="utf-8")
    (root / "train" / "train.py").write_text(f'''
import json, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "labels.json").write_text(json.dumps({{"valid_labels": {classifier_labels!r}}}))
(out / "model.safetensors").write_bytes(b"weights")
''', encoding="utf-8")
    (root / "ovos_m2v_pipeline" / "__init__.py").write_text("", encoding="utf-8")
    (root / "ovos_m2v_pipeline" / "cli.py").write_text(f'''
import json, sys
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "manifest.json").write_text(json.dumps({{"labels": {prototype_labels!r}}}))
''', encoding="utf-8")
    return root


def _run(root, out, dataset):
    return subprocess.run(
        [sys.executable, str(root / "train" / "build_artifacts.py"),
         "--dataset", str(dataset), "--out", str(out)],
        capture_output=True, text=True, cwd=str(root))


def test_matching_label_sets_are_published(tmp_path):
    labels = ["music.skill:play", "time.skill:current"]
    root = _stub_tree(tmp_path, labels, labels)
    out = tmp_path / "artifacts"
    result = _run(root, out, tmp_path / "dataset")
    assert result.returncode == 0, result.stderr
    assert (out / "classifier" / "labels.json").is_file()
    assert (out / "prototypes" / "manifest.json").is_file()
    assert not (tmp_path / "artifacts.staging").exists()


def test_drifted_label_sets_publish_nothing(tmp_path):
    root = _stub_tree(tmp_path,
                      ["music.skill:play", "time.skill:current"],
                      ["music.skill:play", "weather.skill:forecast"])
    out = tmp_path / "artifacts"
    result = _run(root, out, tmp_path / "dataset")
    assert result.returncode == 1
    assert not out.exists()


def test_the_failure_names_the_labels_that_differ(tmp_path):
    root = _stub_tree(tmp_path,
                      ["music.skill:play", "time.skill:current"],
                      ["music.skill:play", "weather.skill:forecast"])
    result = _run(root, tmp_path / "artifacts", tmp_path / "dataset")
    assert "time.skill:current" in result.stderr
    assert "weather.skill:forecast" in result.stderr
    assert "music.skill:play" not in result.stderr.split("only in", 1)[-1].split("\n")[0]


def test_a_failing_producer_publishes_nothing(tmp_path):
    root = _stub_tree(tmp_path, ["a:b"], ["a:b"])
    (root / "train" / "train.py").write_text("import sys; sys.exit(3)",
                                              encoding="utf-8")
    out = tmp_path / "artifacts"
    result = _run(root, out, tmp_path / "dataset")
    assert result.returncode == 3
    assert not out.exists()


# --- the two fixes from the review of #155 -------------------------------


def _dataset_with_manifest(tmp_path, payload=None):
    """A dataset directory carrying a manifest.json, as build_dataset writes."""
    dataset = tmp_path / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "manifest.json").write_text(
        json.dumps(payload if payload is not None else {"rows_final": 12}),
        encoding="utf-8")
    return dataset


def test_a_drift_refusal_leaves_no_staging_directory(tmp_path):
    root = _stub_tree(tmp_path, ["a.skill:one"], ["a.skill:two"])
    out = tmp_path / "artifacts"
    result = _run(root, out, _dataset_with_manifest(tmp_path))
    assert result.returncode == 1
    assert not out.exists()
    assert not (tmp_path / "artifacts.staging").exists(), (
        "a refused build left its staging tree on disk, holding a full "
        "classifier and a full prototype artifact for a real corpus"
    )


def test_a_failing_producer_leaves_no_staging_directory(tmp_path):
    root = _stub_tree(tmp_path, ["a.skill:one"], ["a.skill:one"])
    (root / "train" / "train.py").write_text("import sys\nsys.exit(3)\n",
                                             encoding="utf-8")
    out = tmp_path / "artifacts"
    result = _run(root, out, _dataset_with_manifest(tmp_path))
    assert result.returncode == 3
    assert not (tmp_path / "artifacts.staging").exists()


def test_both_sides_carry_the_same_build_identity(tmp_path):
    labels = ["a.skill:one"]
    root = _stub_tree(tmp_path, labels, labels)
    out = tmp_path / "artifacts"
    result = _run(root, out, _dataset_with_manifest(tmp_path))
    assert result.returncode == 0, result.stderr
    left = json.loads((out / "classifier" / "build.json").read_text())
    right = json.loads((out / "prototypes" / "build.json").read_text())
    assert left == right
    assert left["dataset_manifest_sha256"], "the identity must not be empty"


def test_two_different_corpora_get_different_identities(tmp_path):
    """The control: the stamp has to follow the corpus, not be a constant.

    Without this, writing a fixed string into both files would pass the test
    above and pair artifacts that nothing actually pairs.
    """
    labels = ["a.skill:one"]
    ids = []
    for rows in (12, 13):
        root = _stub_tree(tmp_path / f"r{rows}", labels, labels)
        out = tmp_path / f"artifacts{rows}"
        dataset = _dataset_with_manifest(tmp_path / f"d{rows}",
                                         {"rows_final": rows})
        result = _run(root, out, dataset)
        assert result.returncode == 0, result.stderr
        ids.append(json.loads(
            (out / "classifier" / "build.json").read_text())["dataset_manifest_sha256"])
    assert ids[0] != ids[1]


# --- the three fixes from reviewer-skills' second round (T-6024) ---------


def test_a_producer_that_exits_0_and_writes_nothing_leaves_no_staging(tmp_path):
    """The exit the earlier try/except did not cover.

    A producer that exits 0 and writes no labels.json raised straight out of
    main(), past every cleanup, and the staging tree stayed on disk holding
    whatever the other producer had written.
    """
    root = _stub_tree(tmp_path, ["a.skill:one"], ["a.skill:one"])
    (root / "train" / "train.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[sys.argv.index('--out') + 1]).mkdir(parents=True, "
        "exist_ok=True)\n", encoding="utf-8")
    out = tmp_path / "artifacts"
    result = _run(root, out, _dataset_with_manifest(tmp_path))
    assert result.returncode == 1
    assert not out.exists()
    assert not (tmp_path / "artifacts.staging").exists(), (
        "a producer that exited 0 and wrote no labels left the staging tree")
    assert "labels could not be read" in result.stderr


def test_a_producer_that_writes_unreadable_labels_leaves_no_staging(tmp_path):
    """The same exit by a different route: the file is there and is not JSON."""
    root = _stub_tree(tmp_path, ["a.skill:one"], ["a.skill:one"])
    (root / "train" / "train.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'labels.json').write_text('not json')\n", encoding="utf-8")
    out = tmp_path / "artifacts"
    result = _run(root, out, _dataset_with_manifest(tmp_path))
    assert result.returncode == 1
    assert not (tmp_path / "artifacts.staging").exists()


def test_the_identity_survives_a_reformatted_manifest(tmp_path):
    """Two spellings of one manifest are one corpus.

    The identity used to hash the raw bytes, so a writer that changed its
    indent or its key order renamed the corpus and two builds of one dataset
    stopped looking like one dataset.
    """
    payload = {"rows_final": 12, "revisions": {"b": "2", "a": "1"}}
    root = _stub_tree(tmp_path, ["a.skill:one"], ["a.skill:one"])

    first = tmp_path / "compact"
    first.mkdir()
    (first / "manifest.json").write_text(json.dumps(payload, separators=(",", ":")),
                                         encoding="utf-8")
    out_a = tmp_path / "art-a"
    assert _run(root, out_a, first).returncode == 0

    second = tmp_path / "indented"
    second.mkdir()
    (second / "manifest.json").write_text(
        json.dumps(payload, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    out_b = tmp_path / "art-b"
    assert _run(root, out_b, second).returncode == 0

    id_a = json.loads((out_a / "classifier" / "build.json").read_text())
    id_b = json.loads((out_b / "classifier" / "build.json").read_text())
    assert id_a["dataset_manifest_sha256"] == id_b["dataset_manifest_sha256"], (
        "the same manifest written two ways produced two corpus identities")
    assert id_a["dataset_manifest_sha256"], "the identity must not be empty"


def test_a_different_manifest_is_a_different_identity(tmp_path):
    """The control for the test above: canonicalising must not flatten
    everything to one id."""
    root = _stub_tree(tmp_path, ["a.skill:one"], ["a.skill:one"])
    out_a = tmp_path / "art-a"
    out_b = tmp_path / "art-b"
    one = _dataset_with_manifest(tmp_path / "one", {"rows_final": 12})
    two = _dataset_with_manifest(tmp_path / "two", {"rows_final": 13})
    assert _run(root, out_a, one).returncode == 0
    assert _run(root, out_b, two).returncode == 0
    assert (json.loads((out_a / "classifier" / "build.json").read_text())
            ["dataset_manifest_sha256"]
            != json.loads((out_b / "classifier" / "build.json").read_text())
            ["dataset_manifest_sha256"])


def test_a_manifest_that_is_not_json_takes_the_empty_identity(tmp_path):
    """An id that cannot be reproduced pairs nothing, so there is none."""
    root = _stub_tree(tmp_path, ["a.skill:one"], ["a.skill:one"])
    dataset = tmp_path / "broken"
    dataset.mkdir()
    (dataset / "manifest.json").write_text("{not json", encoding="utf-8")
    out = tmp_path / "artifacts"
    assert _run(root, out, dataset).returncode == 0
    stamp = json.loads((out / "classifier" / "build.json").read_text())
    assert stamp["dataset_manifest_sha256"] == ""


def test_both_sides_name_the_backbone_and_the_classifier(tmp_path):
    """A reader of one published half can tell what it was built from."""
    import hashlib
    labels = ["a.skill:one"]
    root = _stub_tree(tmp_path, labels, labels)
    out = tmp_path / "artifacts"
    assert _run(root, out, _dataset_with_manifest(tmp_path)).returncode == 0
    stamp = json.loads((out / "classifier" / "build.json").read_text())
    assert stamp["backbone"] == "minishlab/M2V_multilingual_output"
    assert stamp["model_id"] == str(out.resolve() / "classifier")
    assert stamp["classifier_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert json.loads((out / "prototypes" / "build.json").read_text()) == stamp


def test_overrides_are_recorded_as_given(tmp_path):
    """The control: the stamp reports what ran, it does not echo a default."""
    labels = ["a.skill:one"]
    root = _stub_tree(tmp_path, labels, labels)
    out = tmp_path / "artifacts"
    result = subprocess.run(
        [sys.executable, str(root / "train" / "build_artifacts.py"),
         "--dataset", str(_dataset_with_manifest(tmp_path)),
         "--out", str(out),
         "--classifier-base", "some/other-backbone",
         "--model-id", "some/published-classifier"],
        capture_output=True, text=True, cwd=str(root))
    assert result.returncode == 0, result.stderr
    stamp = json.loads((out / "classifier" / "build.json").read_text())
    assert stamp["backbone"] == "some/other-backbone"
    assert stamp["model_id"] == "some/published-classifier"


def test_the_exporter_reads_the_staged_classifier(tmp_path):
    """The exporter is handed the classifier the same run trained, never a
    separate backbone, and records the id the runtime loads it under."""
    labels = ["a.skill:one"]
    root = _stub_tree(tmp_path, labels, labels)
    cli = root / "ovos_m2v_pipeline" / "cli.py"
    cli.write_text(cli.read_text() + '''
(out / "argv.json").write_text(json.dumps(sys.argv[1:]))
''', encoding="utf-8")
    out = tmp_path / "artifacts"
    assert _run(root, out, _dataset_with_manifest(tmp_path)).returncode == 0
    argv = json.loads((out / "prototypes" / "argv.json").read_text())
    assert argv[argv.index("--model") + 1] == str(
        tmp_path / "artifacts.staging" / "classifier")
    assert argv[argv.index("--model-id") + 1] == str(out.resolve() / "classifier")


def test_a_classifier_without_weights_publishes_nothing(tmp_path):
    root = _stub_tree(tmp_path, ["a.skill:one"], ["a.skill:one"])
    train = root / "train" / "train.py"
    train.write_text(train.read_text().replace(
        '(out / "model.safetensors").write_bytes(b"weights")\n', ""),
        encoding="utf-8")
    out = tmp_path / "artifacts"
    result = _run(root, out, _dataset_with_manifest(tmp_path))
    assert result.returncode == 1
    assert not out.exists()
    assert not (tmp_path / "artifacts.staging").exists()


def test_a_failed_publish_keeps_the_previous_artifact(tmp_path, monkeypatch):
    import importlib.util
    labels = ["a.skill:one"]
    root = _stub_tree(tmp_path, labels, labels)
    out = tmp_path / "artifacts"
    out.mkdir()
    (out / "sentinel.txt").write_text("previous", encoding="utf-8")
    dataset = _dataset_with_manifest(tmp_path)

    spec = importlib.util.spec_from_file_location(
        "stub_build_artifacts", root / "train" / "build_artifacts.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    real_rename = Path.rename

    def failing_rename(self, target):
        if self.name == "artifacts.staging":
            raise OSError(28, "No space left on device")
        return real_rename(self, target)

    monkeypatch.chdir(root)
    monkeypatch.setattr(Path, "rename", failing_rename)
    try:
        driver.main(["--dataset", str(dataset), "--out", str(out)])
    except OSError as err:
        assert err.errno == 28
    else:
        raise AssertionError("the publish rename was expected to fail")

    assert (out / "sentinel.txt").read_text(encoding="utf-8") == "previous"
    assert not (out / "classifier").exists()
    assert not (tmp_path / "artifacts.staging").exists()
    assert not (tmp_path / "artifacts.previous").exists()


def test_a_successful_publish_replaces_the_previous_artifact(tmp_path):
    labels = ["a.skill:one"]
    root = _stub_tree(tmp_path, labels, labels)
    out = tmp_path / "artifacts"
    out.mkdir()
    (out / "sentinel.txt").write_text("previous", encoding="utf-8")
    result = _run(root, out, _dataset_with_manifest(tmp_path))
    assert result.returncode == 0, result.stderr
    assert not (out / "sentinel.txt").exists()
    assert (out / "classifier" / "labels.json").is_file()
    assert not (tmp_path / "artifacts.previous").exists()


# --- the prototypes live in the classifier's own embedding space ----------


_TUNED_TRAIN = '''
import json, sys
from pathlib import Path

import numpy as np
from model2vec import StaticModel
from model2vec.inference.model import (Activation, Layer, MLPHead,
                                       StaticModelPipeline)

args = sys.argv[1:]
base = StaticModel.from_pretrained(args[args.index("--base-model") + 1])
out = Path(args[args.index("--out") + 1])
# what a trainable fit does to the backbone: the published embedding is no
# longer the one it started from
shift = np.random.default_rng(1).normal(size=base.embedding.shape)
tuned = StaticModel(vectors=(base.embedding + shift).astype(np.float32),
                    tokenizer=base.tokenizer, normalize=False)
labels = ["music.skill:play", "time.skill:current"]
head = MLPHead([Layer(np.ones((2, tuned.dim), np.float32),
                      np.zeros(2, np.float32))],
               Activation.SOFTMAX, np.array(labels))
StaticModelPipeline(tuned, head).save_pretrained(str(out))
(out / "labels.json").write_text(json.dumps({"valid_labels": labels}))
'''

_SENTENCES = {"music.skill:play": "play some music",
              "time.skill:current": "what time is it"}


def _tiny_backbone(path):
    """A word-level StaticModel built offline, standing in for the hub one."""
    import numpy as np
    from model2vec import StaticModel
    from tokenizers import Tokenizer, models, pre_tokenizers

    words = ["[UNK]", "[PAD]", "play", "some", "music", "what", "time",
             "is", "it"]
    tokenizer = Tokenizer(models.WordLevel(
        {w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    vectors = np.random.default_rng(0).normal(size=(len(words), 8))
    StaticModel(vectors=vectors.astype(np.float32), tokenizer=tokenizer,
                normalize=False).save_pretrained(str(path))
    return path


def _tiny_corpus(path):
    import pandas as pd

    path.mkdir(parents=True)
    rows = pd.DataFrame(
        [{"lang": "en-US", "label": label, "utterance": sentence,
          "source": "test", "family": "test"}
         for label, sentence in _SENTENCES.items()])
    rows.to_parquet(path / "train.parquet")
    rows.to_parquet(path / "test.parquet")
    (path / "manifest.json").write_text(json.dumps({"rows_final": 2}),
                                        encoding="utf-8")
    return path


def _unit(vector):
    import numpy as np
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_the_prototypes_are_built_with_the_classifiers_embedding(tmp_path):
    """The runtime keeps one model in memory, the classifier, and embeds
    every query with it. Prototypes built on the base backbone sit in another
    embedding space as soon as the fit tunes the embedding, and their
    artifact names a model the runtime never loads."""
    import hashlib

    import numpy as np
    from model2vec import StaticModel

    base = _tiny_backbone(tmp_path / "base")
    root = tmp_path / "tree"
    (root / "train").mkdir(parents=True)
    driver = DRIVER.read_text(encoding="utf-8")
    default = 'DEFAULT_BACKBONE = "minishlab/M2V_multilingual_output"'
    assert default in driver
    (root / "train" / "build_artifacts.py").write_text(
        driver.replace(default, f"DEFAULT_BACKBONE = {str(base)!r}"),
        encoding="utf-8")
    (root / "train" / "train.py").write_text(_TUNED_TRAIN, encoding="utf-8")

    out = tmp_path / "artifacts"
    result = subprocess.run(
        [sys.executable, str(root / "train" / "build_artifacts.py"),
         "--dataset", str(_tiny_corpus(tmp_path / "dataset")),
         "--out", str(out)],
        capture_output=True, text=True, cwd=str(root),
        env={**os.environ, "HF_HUB_OFFLINE": "1",
             "PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    assert result.returncode == 0, result.stderr

    classifier = StaticModel.from_pretrained(str(out / "classifier"))
    backbone = StaticModel.from_pretrained(str(base))
    stored = np.load(out / "prototypes" / "prototypes.npz")
    by_label = dict(zip((str(l).split("\0")[0] for l in stored["labels"]),
                        stored["embeddings"]))
    for label, sentence in _SENTENCES.items():
        expected = _unit(classifier.encode([sentence])[0])
        # the control: the fit really did move this sentence's vector
        assert not np.allclose(expected, _unit(backbone.encode([sentence])[0]),
                               atol=1e-3)
        np.testing.assert_allclose(by_label[label], expected, atol=1e-5)

    manifest = json.loads((out / "prototypes" / "manifest.json").read_text())
    assert manifest["model_id"] == str(out.resolve() / "classifier")
    stamp = json.loads((out / "prototypes" / "build.json").read_text())
    weights = (out / "classifier" / "model.safetensors").read_bytes()
    assert stamp["classifier_sha256"] == hashlib.sha256(weights).hexdigest()
    assert json.loads((out / "classifier" / "build.json").read_text()) == stamp
