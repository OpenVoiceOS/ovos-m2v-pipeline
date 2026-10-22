"""A published accuracy figure (e.g. `train/model_comparison.md`) is only
honest when it comes from a TEMPLATE-level split. Every row expanded from
the same source template shares that template's wording, so a row-level
split -- stratifying on individual rows the way `sklearn.model_selection.
train_test_split(all_X, all_y, ...)` does -- puts near-identical expansions
of the same template on both sides, and the model is scored on phrasings it
has already seen. `train/build_dataset.py`'s `template_split` groups by
``(lang, template)`` before splitting so a template's expansions land on one
side as a unit.

This test fires the alarm this markdown correction is about: it builds a
corpus of templates that each expand into several near-identical rows and
asserts the split `train.py` reads its train/test rows from never leaks a
template across both sides. Swap `template_split` for a bare row-level
`train_test_split` and this test fails.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset  # noqa: E402


def _corpus():
    """Every label is one template, expanded into several rows that only
    differ by a trailing digit -- the shape a `(alt|alt)` `.intent` line
    produces. A row-level split can place `label-0 row 3` in train and
    `label-0 row 7` in test even though both came from the same template.
    """
    import pandas as pd

    rows = []
    for label_n in range(6):
        label = f"skill:intent{label_n}"
        for template_n in range(5):
            template = f"template-{label_n}-{template_n}"
            for row_n in range(10):
                rows.append({"lang": "en-US", "template": template,
                            "label": label, "utterance": f"{template} row {row_n}"})
    return pd.DataFrame(rows)


def test_template_split_never_leaks_a_template_across_sides():
    df = _corpus()
    train, test = build_dataset.template_split(df, "label", test_size=0.34, seed=0)
    train_templates = set(train["template"])
    test_templates = set(test["template"])
    leaked = train_templates & test_templates
    assert not leaked, (
        f"templates {leaked} were split across train and test -- a row-level "
        "split scores the model on wording it has already seen and cannot "
        "back a published accuracy figure")
    assert not train.empty and not test.empty


def test_row_level_split_would_leak_the_same_corpus():
    """The failure mode this markdown correction documents, reproduced
    directly: a bare row-level split -- what `train/model_comparison.md`'s
    0.9916 figure came from -- puts expansions of the same template on both
    sides of the same corpus `template_split` above keeps clean. This is not
    a defect to fix; it is what makes the template-level assertion above a
    real regression guard rather than a tautology.
    """
    from sklearn.model_selection import train_test_split

    df = _corpus()
    train_rows, test_rows = train_test_split(
        df, test_size=0.34, random_state=0, stratify=df["label"])
    leaked = set(train_rows["template"]) & set(test_rows["template"])
    assert leaked, (
        "expected the row-level split to leak templates across sides on this "
        "corpus -- if it stopped leaking, this test no longer demonstrates "
        "the gap the template-level split closes")
