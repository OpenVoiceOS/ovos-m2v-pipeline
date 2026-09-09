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
