"""No utterance may appear on both sides of the split.

Grouping by template catches a template's own expansions straddling the
split. It cannot catch two different templates filling their slots into the
same sentence, nor two regional variants of one language carrying the same
sentence once the corpus groups them, and both put a training row into the
test set where it scores as a held-out success.

Each test asserts three things: both sides of `template_split`'s output are
non-empty (an empty side makes the no-overlap check pass vacuously), the two
sides do not share the property under test, and a row-level split of the
same frame WOULD exhibit that property — proving the fixture is thin enough
to break, so the pass means the guarantee held rather than that the data was
too small to fail.
"""
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))

from build_dataset import template_split  # noqa: E402


def frame(rows):
    return pd.DataFrame(rows, columns=["lang", "template", "utterance", "label"])


def test_two_templates_producing_one_sentence_do_not_straddle():
    """`play {track}` and `play {song}` fill into the same sentence."""
    rows = [("en", "play {track}", "play tosca", "skill:play"),
            ("en", "play {song}", "play tosca", "skill:play"),
            ("en", "play {track}", "play carmen", "skill:play"),
            ("en", "play {song}", "play aida", "skill:play"),
            ("en", "stop it", "stop it", "skill:stop"),
            ("en", "halt", "halt", "skill:stop")]
    df = frame(rows)
    train, test = template_split(df, "label", test_size=0.5, seed=42)
    assert len(train) > 0 and len(test) > 0
    assert not set(train["utterance"]) & set(test["utterance"])

    # rows 0 and 1 both carry "play tosca"; a row-level split of the same
    # frame that puts them on opposite sides would leak, so the fixture is
    # capable of exhibiting the defect the grouped split guards against.
    row_train, row_test = df.iloc[[0, 2, 4]], df.iloc[[1, 3, 5]]
    assert len(row_train) > 0 and len(row_test) > 0
    assert set(row_train["utterance"]) & set(row_test["utterance"])


def test_collapsed_variants_of_one_language_do_not_straddle():
    """pt-PT and pt-BR both say "tocar musica"; once they are both `pt` the
    two rows are the same sentence in one language."""
    rows = [("pt", "tocar musica", "tocar musica", "skill:play"),
            ("pt", "toca musica", "tocar musica", "skill:play"),
            ("pt", "tocar samba", "tocar samba", "skill:play"),
            ("pt", "parar", "parar", "skill:stop"),
            ("pt", "pare", "pare", "skill:stop")]
    df = frame(rows)
    train, test = template_split(df, "label", test_size=0.5, seed=42)
    assert len(train) > 0 and len(test) > 0
    assert not set(train["utterance"]) & set(test["utterance"])

    # rows 0 and 1 both carry "tocar musica"; splitting them onto opposite
    # sides by row rather than by group would leak.
    row_train, row_test = df.iloc[[0, 2]], df.iloc[[1, 3, 4]]
    assert len(row_train) > 0 and len(row_test) > 0
    assert set(row_train["utterance"]) & set(row_test["utterance"])


def test_the_same_sentence_in_two_languages_does_not_straddle():
    """"cool track" is attested in en and nl under one label; a text
    classifier scoring it in test was fitted on the identical string."""
    rows = [("en", "cool track", "cool track", "ocp:like_song"),
            ("nl", "cool track", "cool track", "ocp:like_song"),
            ("en", "nice song", "nice song", "ocp:like_song"),
            ("nl", "mooi liedje", "mooi liedje", "ocp:like_song"),
            ("en", "stop", "stop", "ocp:stop"),
            ("nl", "stoppen", "stoppen", "ocp:stop")]
    df = frame(rows)
    train, test = template_split(df, "label", test_size=0.5, seed=42)
    assert len(train) > 0 and len(test) > 0
    assert not set(train["utterance"]) & set(test["utterance"])

    # rows 0 and 1 both carry "cool track" across en and nl; splitting them
    # onto opposite sides by row rather than by group would leak.
    row_train, row_test = df.iloc[[0, 2, 4]], df.iloc[[1, 3, 5]]
    assert len(row_train) > 0 and len(row_test) > 0
    assert set(row_train["utterance"]) & set(row_test["utterance"])


def test_a_templates_own_expansions_still_do_not_straddle():
    """The property the split already had must survive the new grouping."""
    rows = [("en", "set alarm for {time}", f"set alarm for {n}", "skill:alarm")
            for n in range(6)]
    rows += [("en", "wake me at {time}", f"wake me at {n}", "skill:alarm")
             for n in range(6)]
    rows += [("en", "cancel it", "cancel it", "skill:cancel"),
             ("en", "forget it", "forget it", "skill:cancel")]
    df = frame(rows)
    train, test = template_split(df, "label", test_size=0.5, seed=42)
    assert len(train) > 0 and len(test) > 0
    for side in (train, test):
        templates = set(side["template"])
        other = test if side is train else train
        assert not templates & set(other["template"])

    # a row-level split of the same frame would leak: alternating rows
    # scatters "set alarm for {time}"'s own six expansions across both
    # sides, so the same template appears on both — the failure the
    # template grouping exists to prevent.
    row_train = df.iloc[list(range(0, 12, 2))]
    row_test = df.iloc[list(range(1, 12, 2))]
    assert len(row_train) > 0 and len(row_test) > 0
    assert set(row_train["template"]) & set(row_test["template"])
