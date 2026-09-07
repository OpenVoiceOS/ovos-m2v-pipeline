"""Every label big enough to evaluate must get test rows.

A proportional split rounds a small label's share down to nothing, and
stratifying on templates rather than rows rounds it away entirely -- a label
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
    """A frame of `(label, n_templates, rows_per_template)` triples."""
    rows = []
    for label, n_templates, per in spec:
        for t in range(n_templates):
            for r in range(per):
                rows.append({"lang": "en-US", "label": label,
                             "template": f"{label}:{t}",
                             "utterance": f"{label} {t} {r}"})
    return pd.DataFrame(rows)


def split(df, seed=42):
    return build_dataset.template_split(df, "label", 0.2, seed)


def test_the_floor_is_two_rows_from_ten_and_one_below_it():
    assert build_dataset.min_test_rows(4) == 0
    assert build_dataset.min_test_rows(5) == 1
    assert build_dataset.min_test_rows(9) == 1
    assert build_dataset.min_test_rows(10) == 2
    assert build_dataset.min_test_rows(61478) == 2


def test_a_tiny_label_gets_its_floor_instead_of_no_test_rows():
    # a two-template label is what the proportional split rounds away: 20% of
    # two groups is zero, whatever the rows behind them.
    df = corpus([("big:one", 60, 40), ("tiny:six", 2, 3), ("tiny:twelve", 2, 6)])
    train, test = split(df)
    counts = test["label"].value_counts().to_dict()
    assert counts.get("tiny:six", 0) >= 1
    assert counts.get("tiny:twelve", 0) >= 2
    # and the promotion never empties a label's train side
    assert set(train["label"]) == set(df["label"])


def test_a_single_template_label_keeps_all_its_rows_in_train():
    # its only template cannot be split without straddling, and moving it
    # whole would leave the label untrained.
    df = corpus([("big:one", 60, 40), ("one:template", 1, 50)])
    train, test = split(df)
    assert "one:template" not in set(test["label"])
    assert (train["label"] == "one:template").sum() == 50


def test_no_template_straddles_the_split():
    df = corpus([("big:one", 60, 40), ("tiny:six", 2, 3), ("tiny:twelve", 2, 6)])
    train, test = split(df)
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
