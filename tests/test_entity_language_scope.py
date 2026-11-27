"""A value written for one language may not fill another language's template.

The corpus reads every locale of every skill at once. A running pipeline holds
one language at a time, so pooling those values produces sentences no runtime
could ever emit and no speaker would say: an Italian template filled with an
English value reads as neither language and teaches the model that Italian
speakers use the English word.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset  # noqa: E402


def test_the_locale_directory_decides_the_language():
    lang_of = build_dataset.entity_lang_of
    assert lang_of("locale/pt-PT/entity/kind.entity") == "pt-PT"
    assert lang_of("skill/locale/en-US/kind.entity") == "en-US"
    assert lang_of("locale/kab/entity/kind.entity") == "kab"


def test_a_directory_that_is_not_a_language_is_not_read_as_one():
    assert build_dataset.entity_lang_of("locale/vocabulary/kind.entity") == ""


def test_a_value_does_not_cross_languages():
    entities = {"en-US": {"kind": ["voice memo"]},
                "it-IT": {"kind": ["memo vocale"]},
                "": {"shared": ["fallback"]}}
    for_lang = build_dataset.entities_for_lang

    assert for_lang(entities, "it-IT")["kind"] == ["memo vocale"]
    assert for_lang(entities, "en-US")["kind"] == ["voice memo"]


def test_a_language_with_no_values_of_its_own_borrows_none():
    entities = {"en-US": {"kind": ["voice memo"]}, "": {"shared": ["fallback"]}}
    da = build_dataset.entities_for_lang(entities, "da-DK")
    assert "kind" not in da
    assert da["shared"] == ["fallback"]


def test_a_locale_less_file_is_the_fallback_not_an_override():
    """The positive control for the test above: the same helper must still
    return a value when one is attested, or 'borrows none' would pass on a
    helper that returns nothing at all."""
    entities = {"da-DK": {"kind": ["talebesked"]}, "": {"kind": ["generic"]}}
    assert build_dataset.entities_for_lang(entities, "da-DK")["kind"] == ["talebesked"]


def test_the_fill_path_itself_keeps_a_language_to_its_own_values():
    """The call site, not only the helpers it is built from.

    `fill_templates_by_language` is what `main` runs over the frame. The five
    tests above exercise `entity_lang_of` and `entities_for_lang` and all pass
    with the call site returned to the pooled behaviour, so they cannot tell
    whether the wiring still scopes anything. This drives a two-language frame
    through the real path and asserts the Italian row did not take the English
    value.
    """
    pd = pytest.importorskip("pandas")
    entities = {"en-US": {"kind": ["voice memo"]},
                "it-IT": {"kind": ["memo vocale"]}}
    frame = pd.DataFrame({"lang": ["en-US", "it-IT"],
                          "utterance": ["save a {kind}", "salva un {kind}"]})

    filled = build_dataset.fill_templates_by_language(frame, entities)
    by_lang = dict(zip(filled["lang"], filled["utterance"]))

    assert by_lang["it-IT"] == "salva un memo vocale"
    assert by_lang["en-US"] == "save a voice memo"
    assert "voice memo" not in by_lang["it-IT"]
    assert "memo vocale" not in by_lang["en-US"]


def test_the_fill_path_drops_nothing_and_borrows_nothing_when_a_language_has_no_value():
    """A language with no own value keeps the placeholder, it does not borrow.

    The caller drops such a row. The fill path must leave the literal in place
    rather than reach into another language, which is the whole defect.
    """
    pd = pytest.importorskip("pandas")
    entities = {"en-US": {"kind": ["voice memo"]}}
    frame = pd.DataFrame({"lang": ["it-IT"], "utterance": ["salva un {kind}"]})

    filled = build_dataset.fill_templates_by_language(frame, entities)

    assert list(filled["utterance"]) == ["salva un {kind}"]


def test_a_template_count_is_not_the_row_count_it_explodes_into():
    """Why the ledger reports the two numbers separately.

    The ledger line used to print every row as a "template row" and compare it
    against the frame after `explode`, so it read as the fill increasing the
    row count. The two quantities are different by construction, and this pins
    that: one template with two values becomes two rows, so a count of
    templates can never be read as the count the frame grew from.
    """
    pd = pytest.importorskip("pandas")
    entities = {"en-US": {"kind": ["voice memo", "note"]}}
    frame = pd.DataFrame({"lang": ["en-US", "en-US"],
                          "utterance": ["save a {kind}", "no slot here"]})

    n_slot_templates = int(
        frame["utterance"].str.contains("{", regex=False).sum())
    filled = build_dataset.fill_templates_by_language(frame, entities)

    assert n_slot_templates == 1
    assert len(frame) == 2
    assert len(filled) == 3
    assert n_slot_templates != len(filled)
