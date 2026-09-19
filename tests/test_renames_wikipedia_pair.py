"""The wikipedia rename must be bridged, and only once.

`ovos-skill-wikipedia` renames `WikiMore.intent` to `wiki_more.intent` and
removes the byte-identical duplicate copies that already shipped beside it.
A model trained before the rename emits the old label, so the pair keeps
that intent routing until a model trained on the new name is published.

The duplicate is why this one is worth a test of its own: the compliant name
already existed as a second file, so a reader who checks only "does
wiki_more exist" concludes nothing is needed.
"""
import re

from ovos_m2v_pipeline.renames import RENAMED_LABELS

SKILL = "ovos-skill-wikipedia.openvoiceos"


def test_the_old_label_has_a_pair():
    assert RENAMED_LABELS[f"{SKILL}:WikiMore"] == f"{SKILL}:wiki_more"


def test_the_new_label_is_not_itself_a_source():
    """The control: a target must not be re-mapped by another entry."""
    assert f"{SKILL}:wiki_more" not in RENAMED_LABELS


def test_the_dialogs_this_skill_renamed_get_no_pair():
    """Dialog names are not labels a model emits."""
    for dialog in ("nothing.more", "thats all", "no entry found"):
        assert f"{SKILL}:{dialog}" not in RENAMED_LABELS, dialog


def test_every_target_stays_spec_compliant():
    for target in RENAMED_LABELS.values():
        assert re.match(r"^[a-z0-9_]+$", target.partition(":")[2]), target
