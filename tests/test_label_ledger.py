"""A label leaves the corpus only when the ledger says why.

The floor this replaces was one number. A number cannot tell a label that
left on purpose from a label whose resource was deleted by mistake, and the
only repair it offers is to lower itself. Worse, it is a net: a label lost on
the same day another is gained moves the total by nothing, so the loss never
reaches the number at all. These tests hold the two properties that replace
it -- the floor is derived from the listed changes and never typed, and an
unlisted loss fails the build whatever the total says.
"""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402

LEDGER = Path(__file__).resolve().parents[1] / "train" / "labels.yaml"


def write(tmp_path, doc):
    path = tmp_path / "labels.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def small(**over):
    doc = {
        "version": 1,
        "baseline": {"corpus": "test", "labels_trained": 3,
                     "labels": ["s:a", "s:b", "s:c"]},
        "removed": [{"label": "s:b", "reason": "folded into s:a",
                     "merged_into": "s:a"}],
        "added": [{"label": "s:d", "reason": "a new intent"}],
    }
    doc.update(over)
    return doc


def test_the_floor_is_derived_from_the_listed_changes(tmp_path):
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    assert ledger["min_labels"] == 3 - 1 + 1


def test_a_listed_loss_is_accepted(tmp_path):
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    out = bfs.check_label_ledger(ledger, {"s:a", "s:c", "s:d"})
    assert out["labels_lost_and_unlisted"] == []
    assert out["labels_lost_against_baseline"] == ["s:b"]


def test_an_unlisted_loss_is_reported(tmp_path):
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    out = bfs.check_label_ledger(ledger, {"s:a", "s:b", "s:d"})
    assert out["labels_lost_and_unlisted"] == ["s:c"]


def test_an_unlisted_loss_is_reported_even_when_the_total_holds(tmp_path):
    """The failure a net count cannot see: one lost, one gained, total flat."""
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    trained = {"s:a", "s:b", "s:d"}
    assert len(trained) == ledger["min_labels"]
    out = bfs.check_label_ledger(ledger, trained)
    assert out["labels_lost_and_unlisted"] == ["s:c"]


def test_a_removal_whose_survivor_is_missing_is_reported(tmp_path):
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    out = bfs.check_label_ledger(ledger, {"s:c", "s:d"})
    assert out["merge_targets_missing_from_the_corpus"] == ["s:a"]


def test_a_removal_not_yet_applied_is_reported_and_is_not_a_loss(tmp_path):
    """The ledger may lead the pins, so this is a note and not a failure."""
    ledger = bfs.load_label_ledger(write(tmp_path, small()))
    out = bfs.check_label_ledger(ledger, {"s:a", "s:b", "s:c", "s:d"})
    assert out["listed_removals_not_yet_applied"] == ["s:b"]
    assert out["labels_lost_and_unlisted"] == []


def test_a_removal_needs_a_reason(tmp_path):
    doc = small(removed=[{"label": "s:b", "reason": "  ", "merged_into": None}])
    with pytest.raises(SystemExit):
        bfs.load_label_ledger(write(tmp_path, doc))


def test_a_removal_needs_to_say_where_the_phrasings_went(tmp_path):
    doc = small(removed=[{"label": "s:b", "reason": "gone"}])
    with pytest.raises(SystemExit):
        bfs.load_label_ledger(write(tmp_path, doc))


def test_a_removal_of_a_label_the_baseline_never_had_is_refused(tmp_path):
    doc = small(removed=[{"label": "s:z", "reason": "gone",
                          "merged_into": None}])
    with pytest.raises(SystemExit):
        bfs.load_label_ledger(write(tmp_path, doc))


def test_a_baseline_whose_count_disagrees_with_its_list_is_refused(tmp_path):
    doc = small()
    doc["baseline"]["labels_trained"] = 4
    with pytest.raises(SystemExit):
        bfs.load_label_ledger(write(tmp_path, doc))


def test_the_shipped_ledger_loads_and_derives_its_floor():
    ledger = bfs.load_label_ledger(LEDGER)
    assert ledger["baseline_count"] == 232
    assert ledger["min_labels"] == (
        232 - len(ledger["removed"]) + len(ledger["added"]))


def test_every_shipped_removal_names_a_label_the_baseline_carried():
    ledger = bfs.load_label_ledger(LEDGER)
    assert set(ledger["removed"]) <= ledger["baseline_labels"]
