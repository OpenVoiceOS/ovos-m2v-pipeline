"""No train row may carry a typed-slot placeholder.

This is the test the whole change exists for. A row that reads
"what time will it be in {number:offset} minutes" teaches the classifier a
brace, and the runtime never says one.
"""
import collections
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("ovos_spec_tools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_from_skills as bfs  # noqa: E402
import typed_slots as ts  # noqa: E402

PLACEHOLDER = re.compile(r"\{[a-z_]+:[a-z_]+\}")


def sentences(template, lang="en-US", hints=None, keywords=None):
    return bfs.fill(template, hints or {}, keywords or {},
                    collections.Counter(), lang)


def test_a_typed_slot_never_survives_into_a_row():
    rows = sentences("what time will it be in {number:offset} minutes")
    assert rows, "the template produced no row at all"
    for row in rows:
        assert not PLACEHOLDER.search(row), row


def _word_surfaces(lang):
    """A language's own number words, with the language-neutral digits left out."""
    return [v for v in ts.values_for("number", lang) if not v.isdigit()]


def test_the_word_form_is_this_language_and_not_english():
    """A German row may not say "forty two".

    Only the WORD surfaces carry a language. The digits are shared by every
    language on purpose, so asserting over every surface would assert that
    German rows contain no digits, which is not the property that matters.
    """
    german = sentences("wie spät ist es in {number:offset} minuten", lang="de-DE")
    english = sentences("what time will it be in {number:offset} minutes")
    assert german and english
    english_words = _word_surfaces("en-US")
    assert english_words, "no en-US word surface, so this test proves nothing"
    assert not any(value in " ".join(german)
                   for value in english_words), german


def test_the_digit_form_is_shared_by_every_language():
    """The positive control for the test above.

    It proves the narrowing is real rather than a hole: the digits DO appear
    in both languages, so the test above passes because the word forms differ
    and not because no surface reaches a German row at all.
    """
    german = sentences("wie spät ist es in {number:offset} minuten", lang="de-DE")
    english = sentences("what time will it be in {number:offset} minutes")
    digits = [v for v in ts.values_for("number", "en-US") if v.isdigit()]
    assert digits, "no digit surface at all, so the ruling is not implemented"
    for value in digits:
        assert value in " ".join(english), value
        assert value in " ".join(german), value


def test_both_numeric_surfaces_reach_a_row():
    """The ruling itself: an ASR front end emits digits, a person says words.

    A corpus with only one of the two leaves the classifier blind to the
    other.
    """
    joined = " ".join(sentences("what time will it be in {number:offset} minutes"))
    values = ts.values_for("number", "en-US")
    assert any(v.isdigit() and v in joined for v in values), joined
    assert any(not v.isdigit() and v in joined for v in values), joined


def test_an_untyped_slot_still_behaves_as_before():
    """The control: this change must not touch the plain-slot path.

    INTENT-1 5.4 keeps a slot with no value set, and the corpus keeps the
    sentence with the placeholder literal, which is what the runtime does.
    """
    rows = sentences("play {query}")
    assert rows == ["play {query}"]


def test_a_typed_slot_with_no_generator_for_this_language_drops_the_sentence():
    """Silence beats a row in the wrong language."""
    stats = collections.Counter()
    rows = bfs.fill("zzz {timezone:zone} zzz", {}, {}, stats, "en-US")
    assert rows == []
    assert stats["sentence_dropped_no_typed_values_for_this_language"] == 1


def test_two_typed_slots_in_one_template_both_fill():
    rows = sentences("remind me in {duration:wait} about {number:count} things")
    assert rows
    for row in rows:
        assert not PLACEHOLDER.search(row), row


def test_a_typed_and_an_untyped_slot_together():
    """The typed one fills, the untyped one stays, as each rule says."""
    rows = sentences("set {query} in {number:offset} minutes")
    assert rows
    for row in rows:
        assert "{query}" in row
        assert not PLACEHOLDER.search(row), row


def test_the_row_count_is_capped():
    rows = sentences("a {number:one} b {number:two} c {number:three}")
    assert 0 < len(rows) <= bfs.EXPANSION_CAP
