"""Every label big enough to evaluate must get test rows.

A proportional split rounds a small label's share down to nothing, and
stratifying on groups rather than rows rounds it away entirely -- a label
with no test rows is never measured again while still occupying probability
mass in the head.
"""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset  # noqa: E402


def corpus(spec):
    """A frame of `(label, n_groups, rows_per_group)` triples."""
    rows = []
    for label, n_groups, per in spec:
        for t in range(n_groups):
            for r in range(per):
                rows.append({"lang": "en-US", "label": label,
                             "template": f"{label}:{t}",
                             "utterance": f"{label} {t} {r}"})
    return pd.DataFrame(rows)


def split(df, seed=42):
    return build_dataset.template_split(df, "label", 0.2, seed)


def test_the_floor_is_two_rows_from_ten_and_one_below_it():
    assert build_dataset.per_label_test_floor(4) == 0
    assert build_dataset.per_label_test_floor(5) == 1
    assert build_dataset.per_label_test_floor(9) == 1
    assert build_dataset.per_label_test_floor(10) == 2
    assert build_dataset.per_label_test_floor(61478) == 2


def test_a_tiny_label_gets_its_floor_instead_of_no_test_rows():
    # a two-group label is what the proportional split rounds away: 20% of
    # two groups is zero, whatever the rows behind them.
    df = corpus([("big:one", 60, 40), ("tiny:six", 2, 3), ("tiny:twelve", 2, 6)])
    train, test = split(df)
    counts = test["label"].value_counts().to_dict()
    assert counts.get("tiny:six", 0) >= 1
    assert counts.get("tiny:twelve", 0) >= 2
    # and the promotion never empties a label's train side
    assert set(train["label"]) == set(df["label"])


def test_a_single_group_label_keeps_all_its_rows_in_train():
    # its only group cannot be split without straddling, and moving it
    # whole would leave the label untrained.
    df = corpus([("big:one", 60, 40), ("one:group", 1, 50)])
    train, test = split(df)
    assert "one:group" not in set(test["label"])
    assert (train["label"] == "one:group").sum() == 50


def test_no_group_straddles_the_split():
    df = corpus([("big:one", 60, 40), ("tiny:six", 2, 3), ("tiny:twelve", 2, 6)])
    train, test = split(df)
    # a template's rows all move together; two templates that expand into
    # the same utterance are also one group, absent from this corpus.
    assert not set(train["template"]) & set(test["template"])


def test_the_overall_ratio_survives_the_floor():
    df = corpus([("big:one", 300, 40)] + [(f"tiny:{i}", 2, 3) for i in range(20)])
    _train, test = split(df)
    assert 0.15 <= len(test) / len(df) <= 0.25


def test_the_split_is_deterministic():
    df = corpus([("big:one", 60, 40), ("tiny:six", 2, 3)])
    a, _ = split(df)
    b, _ = split(df)
    assert list(a["utterance"]) == list(b["utterance"])


def sized_corpus(spec):
    """A frame of `(label, [rows per group])` pairs."""
    rows = []
    for label, sizes in spec:
        for t, per in enumerate(sizes):
            for r in range(per):
                rows.append({"lang": "en-US", "label": label,
                             "template": f"{label}:{t}",
                             "utterance": f"{label} {t} {r}"})
    return pd.DataFrame(rows)


def test_the_floor_moves_the_smallest_groups_first():
    # groups of 1, 2 and 30 rows: at seed 0 the ratio puts none of them in
    # test, so the floor moves the 1-row and 2-row groups. A largest-first
    # order moves the 30-row group instead and shifts the ratio for nothing.
    df = sized_corpus([("big:one", [40] * 60), ("thin:mixed", [1, 2, 30])])
    _train, test = split(df, seed=0)
    moved = set(test.loc[test["label"] == "thin:mixed", "template"])
    assert moved == {"thin:mixed:0", "thin:mixed:1"}


def test_every_label_under_its_floor_is_reported_with_its_group_sizes():
    # a two-group label whose only movable group holds one row stays under
    # its floor of two. It still has a test row, so labels_without_test_rows
    # misses it; the report must not.
    df = sized_corpus([("big:one", [40] * 60), ("two:uneven", [30, 1])])
    report = {}
    _train, test = build_dataset.template_split(df, "label", 0.2, 0, report=report)
    assert (test["label"] == "two:uneven").sum() == 1
    below = report["labels_below_test_floor"]
    assert below["names"] == {
        "two:uneven": {"floor": 2, "test_rows": 1, "group_rows": [30, 1]}}
    # positive control: a label that meets its floor is not listed
    assert below["labels"] == 1 and "big:one" not in below["names"]


def test_a_group_that_carries_two_labels_counts_for_both():
    # with --allow-ambiguous one utterance can carry two labels, and the
    # shared utterance ties a template of each label into one group. The
    # group counts toward both labels: the report must match the real test
    # rows, and moving it for one label must not empty the other's train side.
    shapes = [([12, 12, 12], [12, 12], 1), ([12, 1, 12], [0, 30], 2),
              ([12, 12], [0, 1, 30], 2)]
    for a_sizes, b_sizes, shared_b in shapes:
        rows = [{"lang": "en-US", "label": "big:one", "template": f"big:{t}",
                 "utterance": f"big {t} {r}"} for t in range(60) for r in range(40)]
        for label, sizes in (("amb:a", a_sizes), ("amb:b", b_sizes)):
            rows += [{"lang": "en-US", "label": label, "template": f"{label}:{t}",
                      "utterance": f"{label} {t} {r}"}
                     for t, per in enumerate(sizes) for r in range(per)]
        rows.append({"lang": "en-US", "label": "amb:a", "template": "amb:a:0",
                     "utterance": "the same sentence"})
        rows += [{"lang": "en-US", "label": "amb:b", "template": "amb:b:0",
                  "utterance": "the same sentence" if r == 0 else f"amb:b shared {r}"}
                 for r in range(shared_b)]
        df = pd.DataFrame(rows)
        for seed in range(10):
            report = {}
            train, test = build_dataset.template_split(df, "label", 0.2, seed,
                                                       report=report)
            names = report["labels_below_test_floor"]["names"]
            case = (a_sizes, b_sizes, seed)
            for label, n in df["label"].value_counts().items():
                actual = int((test["label"] == label).sum())
                floor = build_dataset.per_label_test_floor(int(n))
                assert (actual < floor) == (label in names), (case, label, names)
                if label in names:
                    assert names[label]["test_rows"] == actual, (case, label)
                assert (train["label"] == label).any(), (case, label)
