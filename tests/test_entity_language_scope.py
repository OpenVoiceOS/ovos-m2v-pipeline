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
    lang_of = build_dataset._ENTITY_LOCALE_RE
    assert lang_of.match("locale/pt-PT/entity/kind.entity").group(1) == "pt-PT"
    assert lang_of.match("skill/locale/en-US/kind.entity").group(1) == "en-US"
    assert lang_of.match("locale/kab/entity/kind.entity").group(1) == "kab"


def test_a_directory_that_is_not_a_language_is_not_read_as_one():
    assert build_dataset._ENTITY_LOCALE_RE.match("locale/vocabulary/kind.entity") is None


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
