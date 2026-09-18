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


def test_the_value_is_this_language_and_not_english():
    german = sentences("wie spät ist es in {number:offset} minuten", lang="de-DE")
    english = sentences("what time will it be in {number:offset} minutes")
    assert german and english
    assert not any(value in " ".join(german)
                   for value in ts.values_for("number", "en-US")), german


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
