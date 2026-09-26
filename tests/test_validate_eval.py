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


def run(corpus: Path, eval_path: Path = None):
    command = [sys.executable, str(VALIDATOR), "--corpus", str(corpus)]
    if eval_path is not None:
        command += ["--eval", str(eval_path)]
    return subprocess.run(command, capture_output=True, text=True)


def assert_rejected(result, *expected):
    """Non-zero exit, no traceback, and every *expected* fragment in stderr.

    The status is asserted as non-zero rather than as exactly 1. A validator
    run loads pyarrow, and on Python 3.10 the interpreter sometimes aborts in
    native teardown after the script has finished and printed its report:
    "terminate called without an active exception", exit -6. The report is
    complete and the status is still non-zero, so a gate that reads the status
    still fails. Pinning the number would make these tests fail for a reason
    that is not the behaviour under test.
    """
    assert result.returncode != 0, result.stdout
    assert "Traceback" not in result.stderr, result.stderr
    for fragment in expected:
        assert fragment in result.stderr, result.stderr


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


# The tests above build their label set from the shipped file, so the
# unknown-label, thin-label and repeat rules are satisfied by construction and
# cannot fail. The tests below write their own evaluation file instead, so each
# rule is driven from a corpus the shipped file does not decide.


def write_eval(path: Path, rows: list) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def eval_row(label: str, utterance: str, kind: str = "paraphrase") -> dict:
    return {"label": label, "utterance": utterance, "kind": kind}


def enough_rows(label: str, prefix: str) -> list:
    """`MIN_ROWS` distinct rows, so a thin-label error cannot fire by accident."""
    return [eval_row(label, f"{prefix} number {n}") for n in range(8)]


def test_an_eval_label_the_corpus_does_not_have_fails(tmp_path):
    rows = enough_rows("known:label", "a known utterance")
    rows += enough_rows("absent:label", "an utterance whose label is absent")
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], ["known:label"])
    result = run(corpus, write_eval(tmp_path / "eval.jsonl", rows))
    assert_rejected(result, "unknown label 'absent:label'",
                    "an utterance whose label is absent number 0")


def test_a_label_under_the_row_floor_fails(tmp_path):
    rows = enough_rows("fat:label", "a well covered utterance")
    rows += [eval_row("thin:label", f"a thin utterance {n}") for n in range(3)]
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], ["fat:label", "thin:label"])
    result = run(corpus, write_eval(tmp_path / "eval.jsonl", rows))
    assert_rejected(result, "thin:label has 3 rows, fewer than 8")


def test_a_repeated_utterance_fails(tmp_path):
    rows = enough_rows("first:label", "a plain utterance")
    rows += enough_rows("second:label", "another plain utterance")
    rows.append(eval_row("second:label", "a plain utterance number 0"))
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], ["first:label", "second:label"])
    result = run(corpus, write_eval(tmp_path / "eval.jsonl", rows))
    assert_rejected(result, "'a plain utterance number 0' is repeated")


# A malformed row is the contributor's own typo, so it must be named the way
# every other problem is named, not raised as a traceback out of the reader.


def test_a_row_missing_a_key_is_named_not_raised(tmp_path):
    rows = enough_rows("known:label", "a known utterance")
    path = write_eval(tmp_path / "eval.jsonl", rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"label": "known:label", "utterance": "no kind here"}) + "\n")
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], ["known:label"])
    result = run(corpus, path)
    assert_rejected(result, "eval.jsonl", "line 9", "'kind'")


def test_a_line_that_is_not_json_is_named_not_raised(tmp_path):
    rows = enough_rows("known:label", "a known utterance")
    path = write_eval(tmp_path / "eval.jsonl", rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all}\n")
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], ["known:label"])
    result = run(corpus, path)
    assert_rejected(result, "eval.jsonl", "line 9")


def test_a_malformed_row_does_not_hide_the_other_problems(tmp_path):
    """The reader must keep going, or one typo masks every real error."""
    rows = enough_rows("known:label", "a known utterance")
    rows += [eval_row("absent:label", f"an absent label utterance {n}") for n in range(8)]
    path = write_eval(tmp_path / "eval.jsonl", rows)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json at all}\n")
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], ["known:label"])
    result = run(corpus, path)
    assert_rejected(result, "line 17", "unknown label 'absent:label'")


# The label shapes the real corpus uses. The toy labels above ("known:label")
# cannot express the defect these guard against: an eval set written with the
# handler's PascalCase class name instead of the `.intent` file's snake_case
# name passes every toy assertion and names an intent the corpus does not
# have. 555 rows shipped that way in #129 and the class was invisible here.
REAL_LABEL = "ovos-skill-alerts.openvoiceos:add_list_subitems"
PASCAL_LABEL = "ovos-skill-alerts.openvoiceos:AddListSubitems"
DOTTED_LABEL = "ovos-skill-volume.openvoiceos:volume.unmute"
HYPHEN_LABEL = "skill-ovos-randomness.openvoiceos:flip-a-coin"
UNDERSCORE_LABEL = "skill-ovos-randomness.openvoiceos:flip_a_coin"


def test_a_pascal_case_label_is_rejected_against_a_snake_case_corpus(tmp_path):
    rows = enough_rows(PASCAL_LABEL, "put milk on the shopping list")
    path = write_eval(tmp_path / "eval.jsonl", rows)
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], [REAL_LABEL])
    result = run(corpus, path)
    assert_rejected(result, f"unknown label '{PASCAL_LABEL}'")


def test_the_same_label_in_the_corpus_spelling_is_accepted(tmp_path):
    """The control: the only difference is the spelling of the intent part."""
    rows = enough_rows(REAL_LABEL, "put milk on the shopping list")
    path = write_eval(tmp_path / "eval.jsonl", rows)
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], [REAL_LABEL])
    result = run(corpus, path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_dotted_intent_name_is_a_real_shape(tmp_path):
    """`volume.unmute` is how the corpus spells it, dots and all.

    A validator that assumed snake_case everywhere would reject a correct
    label, which is the opposite failure and just as wrong.
    """
    rows = enough_rows(DOTTED_LABEL, "turn the sound back on")
    path = write_eval(tmp_path / "eval.jsonl", rows)
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], [DOTTED_LABEL])
    result = run(corpus, path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_hyphenated_label_is_rejected_when_the_corpus_uses_underscores(tmp_path):
    rows = enough_rows(HYPHEN_LABEL, "flip a coin for me")
    path = write_eval(tmp_path / "eval.jsonl", rows)
    corpus = write_corpus(tmp_path / "corpus", ["unrelated"], [UNDERSCORE_LABEL])
    result = run(corpus, path)
    assert_rejected(result, f"unknown label '{HYPHEN_LABEL}'")
