"""The two published artifacts must describe the same label set.

The driver runs both producers against one corpus and publishes only when
their label sets agree, so the tests drive it with stub producers: what is
under test is the invariant and the refusal, not the fitting or the encoding.
"""
import json
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


def test_both_sides_default_to_one_backbone_and_say_which(tmp_path):
    """The classifier and the prototypes are published as a pair, so a pair
    built on two embedding spaces is a pair in name only."""
    labels = ["a.skill:one"]
    root = _stub_tree(tmp_path, labels, labels)
    out = tmp_path / "artifacts"
    assert _run(root, out, _dataset_with_manifest(tmp_path)).returncode == 0
    stamp = json.loads((out / "classifier" / "build.json").read_text())
    backbones = stamp["backbones"]
    assert backbones["classifier"] == backbones["prototypes"], (
        "the two sides defaulted to two different backbones")
    assert backbones["classifier"], "build.json must name the backbone"
    assert json.loads((out / "prototypes" / "build.json").read_text()) == stamp


def test_an_overridden_backbone_is_recorded_as_given(tmp_path):
    """The control: the stamp reports what ran, it does not echo the default."""
    labels = ["a.skill:one"]
    root = _stub_tree(tmp_path, labels, labels)
    out = tmp_path / "artifacts"
    result = subprocess.run(
        [sys.executable, str(root / "train" / "build_artifacts.py"),
         "--dataset", str(_dataset_with_manifest(tmp_path)),
         "--out", str(out),
         "--classifier-base", "some/other-backbone"],
        capture_output=True, text=True, cwd=str(root))
    assert result.returncode == 0, result.stderr
    stamp = json.loads((out / "classifier" / "build.json").read_text())
    assert stamp["backbones"]["classifier"] == "some/other-backbone"
    assert stamp["backbones"]["prototypes"] != "some/other-backbone"


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
