"""A label whose phrasings cannot be expanded is not a label without phrasings.

The floor report lists labels that got no test rows. Two very different things
land there: a label the skills express in two ways, which wants a
contribution, and a label whose every phrasing ends in a slot the pinned refs
register no values for, which wants an entity file or a change here. Reporting
them together turns the second into a request nobody upstream can act on.
"""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset  # noqa: E402


def dropped(rows):
    return pd.DataFrame(rows, columns=["lang", "label", "utterance"])


def test_the_slot_and_the_language_travel_with_the_label():
    found = build_dataset.unfillable_slots(dropped([
        ("en-US", "common_reading:ReadContentByType", "read me my {content_type}"),
        ("en-US", "common_reading:ReadContentByType", "tell me today's {content_type}"),
        ("de-DE", "common_reading:ReadContentByType", "lies mir meine {content_type}"),
    ]))
    slots, langs = found["common_reading:ReadContentByType"]
    assert slots == {"content_type"}
    assert langs == {"en-US", "de-DE"}


def test_several_slots_in_one_phrasing_are_all_named():
    found = build_dataset.unfillable_slots(dropped([
        ("en-US", "skill:intent", "play {track} by {artist}"),
    ]))
    slots, _ = found["skill:intent"]
    assert slots == {"track", "artist"}


def test_an_unfillable_label_leaves_the_thin_list_and_states_why():
    no_test = {"common_reading:ReadContentByType", "skill:genuinely_thin"}
    unfillable = {"common_reading:ReadContentByType": ({"content_type"}, {"en-US"})}
    limited = build_dataset.limited_by_unfilled_slots(no_test, unfillable)

    assert set(limited) == {"common_reading:ReadContentByType"}
    assert sorted(no_test - set(limited)) == ["skill:genuinely_thin"]

    entry = limited["common_reading:ReadContentByType"]
    assert entry["slots"] == ["content_type"]
    assert entry["langs"] == ["en-US"]
    assert entry["reason"] == ("every phrasing in en-US ends in {content_type}, "
                               "which the pinned refs register no values for")


def test_a_label_that_kept_its_rows_is_not_reported_as_limited():
    # the slot cost this label some phrasings, but it still reached the split
    # with enough rows to be scored, so it is not on the no-test list at all.
    limited = build_dataset.limited_by_unfilled_slots(
        set(), {"skill:intent": ({"content_type"}, {"en-US"})})
    assert limited == {}
