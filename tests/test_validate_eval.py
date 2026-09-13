"""The shipped evaluation set must never share an utterance with the corpus.

A synthetic corpus stands in for the real build: the labels come from the
shipped file itself, so the test checks the validator's overlap, coverage and
duplicate rules rather than the contents of any particular corpus build.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "train" / "validate_eval.py"
EVAL = ROOT / "train" / "eval" / "en-US.jsonl"


def eval_rows() -> list:
    with EVAL.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_corpus(path: Path, utterances: list, labels: list) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"utterance": utterances, "label": [labels[0]] * len(utterances)})
    frame.to_parquet(path / "train.parquet")
    frame.iloc[:0].to_parquet(path / "test.parquet")
    (path / "labels.json").write_text(json.dumps({"valid_labels": labels}))
    return path


def run(corpus: Path):
    return subprocess.run([sys.executable, str(VALIDATOR), "--corpus", str(corpus)],
                          capture_output=True, text=True)


@pytest.fixture
def labels():
    return sorted({row["label"] for row in eval_rows()})


def test_shipped_file_passes_against_a_clean_corpus(tmp_path, labels):
    corpus = write_corpus(tmp_path / "clean", ["nothing like an eval line"], labels)
    result = run(corpus)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_one_shared_utterance_fails(tmp_path, labels):
    leaked = eval_rows()[0]["utterance"]
    corpus = write_corpus(tmp_path / "leaky", [leaked], labels)
    result = run(corpus)
    assert result.returncode == 1
    assert "also appears in the corpus" in result.stderr
    assert leaked in result.stderr


def test_missing_label_fails(tmp_path, labels):
    corpus = write_corpus(tmp_path / "extra", ["unrelated"], labels + ["made-up:label"])
    result = run(corpus)
    assert result.returncode == 1
    assert "no rows for label made-up:label" in result.stderr
