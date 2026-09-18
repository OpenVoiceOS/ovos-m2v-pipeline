"""The census must fail the build on a gold label that cannot be trained.

A test label with no train rows is never answerable: the model has no class
for it. The census exists to make that visible and to stop a publish, so the
test that matters is the exit status, not the text.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import census_gold_labels as census  # noqa: E402


def write(dataset: Path, name: str, rows):
    with (dataset / name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build(tmp_path: Path, train_labels, test_labels) -> Path:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write(dataset, "train.jsonl",
          [{"label": label, "utterance": "u", "lang": "en-US"} for label in train_labels])
    write(dataset, "test.jsonl",
          [{"label": label, "utterance": "u", "lang": "en-US"} for label in test_labels])
    return dataset


def test_every_test_label_trainable_exits_zero(tmp_path):
    dataset = build(tmp_path, ["a:one", "a:two"], ["a:one"])
    assert census.main(["--dataset", str(dataset)]) == 0


def test_a_test_label_with_no_train_rows_exits_one(tmp_path):
    dataset = build(tmp_path, ["a:one"], ["a:one", "a:two"])
    assert census.main(["--dataset", str(dataset)]) == 1


def test_the_spelling_is_what_is_compared(tmp_path):
    """The census compares labels literally, which is the point of it.

    `count_to_N` and `count_to_n` are the same intent to a reader and two
    classes to a model.
    """
    dataset = build(tmp_path, ["a:count_to_n"], ["a:count_to_N"])
    assert census.main(["--dataset", str(dataset)]) == 1


def test_an_empty_test_set_exits_zero(tmp_path):
    dataset = build(tmp_path, ["a:one"], [])
    assert census.main(["--dataset", str(dataset)]) == 0


def test_a_row_without_a_label_is_refused(tmp_path):
    dataset = build(tmp_path, ["a:one"], [])
    with (dataset / "test.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"utterance": "u"}) + "\n")
    with pytest.raises(SystemExit):
        census.main(["--dataset", str(dataset)])


def test_a_blank_line_is_not_a_row(tmp_path):
    dataset = build(tmp_path, ["a:one"], ["a:one"])
    with (dataset / "test.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert census.main(["--dataset", str(dataset)]) == 0
