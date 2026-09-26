"""ovos-skill-alerts#216 merged four intents into two, and all four need a pair.

The rename waves in `renames.py` are one-to-one: a resource file takes a
compliant base name and the label follows it. #216 is not that. It removed
`DeleteListEntries.intent`, `DeleteTodoEntries.intent`,
`QueryListEntries.intent` and `QueryTodoEntries.intent`, and added
`delete_list_entries.intent` and `query_list_entries.intent`: the todo kind
stopped being an intent of its own, so two names collapse into one.

Every published model was trained before that merge and emits all four old
labels. The builder's own note records the arithmetic: 235 labels at the old
pins, 233 at the new, "two labels leave" being the todo pair. A label no skill
registers can never match, so each of the four needs a bridge or those
utterances stop routing the moment the merged skill ships.

The many-to-one shape is the part a reader has to see: a rename table maps a
key to a value, and nothing stops two keys sharing one value. A test written for
a rename wave would assume a bijection and pass while two of these were missing.
"""
import re

from ovos_m2v_pipeline.renames import RENAMED_LABELS

SKILL = "ovos-skill-alerts.openvoiceos"

MERGED = {
    "DeleteListEntries": "delete_list_entries",
    "DeleteTodoEntries": "delete_list_entries",
    "QueryListEntries": "query_list_entries",
    "QueryTodoEntries": "query_list_entries",
}

COMPLIANT = re.compile(r"^[a-z0-9_]+$")


def test_the_table_is_read():
    """The control: an empty table would make every case below vacuous."""
    assert len(RENAMED_LABELS) > 20


def test_every_merged_source_has_a_pair():
    missing = sorted(f"{SKILL}:{old}" for old in MERGED
                     if f"{SKILL}:{old}" not in RENAMED_LABELS)
    assert not missing, (
        "ovos-skill-alerts#216 merged these intents away and a published model "
        f"still emits them, with no bridge: {', '.join(missing)}")


def test_each_pair_points_at_the_surviving_intent():
    for old, new in MERGED.items():
        assert RENAMED_LABELS[f"{SKILL}:{old}"] == f"{SKILL}:{new}", (
            f"{old} must bridge to {new}, the intent that absorbed it")


def test_the_targets_are_compliant_base_names():
    for new in set(MERGED.values()):
        assert COMPLIANT.match(new), new


def test_two_keys_share_one_target_on_purpose():
    """The shape that distinguishes a merge from a rename.

    If a later edit ever makes this a bijection again, one of the four sources
    has lost its bridge and this says so before the model does.
    """
    targets = [RENAMED_LABELS[f"{SKILL}:{old}"] for old in MERGED]
    assert len(set(targets)) == 2, targets
    assert targets.count(f"{SKILL}:delete_list_entries") == 2
    assert targets.count(f"{SKILL}:query_list_entries") == 2


def test_no_source_is_also_a_target():
    """A label on both sides would map to itself through another key, which is
    how a half-applied rename hides: the model's label is 'bridged' to a name
    nothing registers either."""
    for old in MERGED:
        assert f"{SKILL}:{old}" not in RENAMED_LABELS.values()


def test_the_surviving_names_are_not_themselves_bridged_away():
    """The bridge target has to be what the skill registers TODAY.

    If a later rename moves delete_list_entries again, its own pair belongs in
    the table and these four must follow it, or they point at a dead name.
    """
    for new in set(MERGED.values()):
        assert f"{SKILL}:{new}" not in RENAMED_LABELS, (
            f"{new} is itself a rename source now; these four pairs must be "
            "re-pointed at whatever it became")
