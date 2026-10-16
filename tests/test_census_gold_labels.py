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


def build_by_lang(tmp_path: Path, train, test) -> Path:
    """``train`` and ``test`` are (lang, label) pairs."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write(dataset, "train.jsonl",
          [{"label": label, "utterance": "u", "lang": lang} for lang, label in train])
    write(dataset, "test.jsonl",
          [{"label": label, "utterance": "u", "lang": lang} for lang, label in test])
    return dataset


def test_a_label_trained_only_in_another_locale_is_reported_and_not_refused(tmp_path, capsys):
    """The flat census passes (the label has train rows) while the locale
    has none: v6.1's shape. Without --per-locale the exit stays 0 and the
    pair is printed; with it the exit is 1."""
    dataset = build_by_lang(tmp_path, [("en-US", "a:one"), ("fr-FR", "a:two")],
                            [("fr-FR", "a:one")])
    assert census.main(["--dataset", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "every test label has train rows" in out
    assert "1 test (lang, label) pairs have no train row in the same locale" in out
    assert "fr-FR    a:one" in out
    assert census.main(["--dataset", str(dataset), "--per-locale"]) == 1


def test_per_locale_passes_when_every_locale_trains_its_own_labels(tmp_path, capsys):
    """The positive control for the second gate."""
    dataset = build_by_lang(tmp_path, [("en-US", "a:one"), ("fr-FR", "a:one")],
                            [("fr-FR", "a:one"), ("en-US", "a:one")])
    assert census.main(["--dataset", str(dataset), "--per-locale"]) == 0
    assert "every test (lang, label) has a train row in the same locale" in capsys.readouterr().out


def test_per_locale_reads_the_lang_field_and_refuses_a_row_without_it(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write(dataset, "train.jsonl", [{"label": "a:one", "utterance": "u", "lang": "en-US"}])
    write(dataset, "test.jsonl", [{"label": "a:one", "utterance": "u"}])
    with pytest.raises(SystemExit, match="row has no 'lang'"):
        census.main(["--dataset", str(dataset)])


RENAMES = {"a:OldName": "a:new_name"}


def test_train_under_the_old_name_and_gold_under_the_new_is_not_a_gap(tmp_path):
    """reviewer-b's case on #247: a corpus built before a rename with gold
    written after it. The runtime serves the old class through
    RENAMED_LABELS, so neither census may report it."""
    dataset = build_by_lang(tmp_path, [("en-US", "a:OldName")], [("en-US", "a:new_name")])
    _, _, missing = census.census(dataset, renames=RENAMES)
    assert missing == {}
    assert census.census_per_locale(dataset / "train.jsonl", dataset / "test.jsonl",
                                    renames=RENAMES) == {}


def test_train_and_gold_both_under_the_old_name_is_not_a_gap_either(tmp_path):
    """The reverse: the bridge landed while the skill's rename PR is still
    open (laugh#131 on v6.1). Both sides under the old name are one class."""
    dataset = build_by_lang(tmp_path, [("en-US", "a:OldName")], [("en-US", "a:OldName")])
    _, _, missing = census.census(dataset, renames=RENAMES)
    assert missing == {}


def test_without_the_table_the_renamed_pair_is_a_gap():
    """The control: the fold is what closes it, not the data."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dataset = build_by_lang(Path(tmp), [("en-US", "a:OldName")], [("en-US", "a:new_name")])
        _, _, missing = census.census(dataset)
        assert missing == {"a:new_name": 1}


def test_the_table_is_read_from_renames_py_by_path(tmp_path):
    """The real file loads without the package's runtime imports, and an
    absent file is an empty table, not an error."""
    real = census.renamed_labels()
    assert isinstance(real, dict) and all(":" in k and ":" in v for k, v in real.items())
    assert census.renamed_labels(tmp_path / "missing.py") == {}
