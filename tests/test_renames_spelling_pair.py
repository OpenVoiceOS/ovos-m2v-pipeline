"""The spelling rename must be bridged under the id the skill registers.

`ovos-skill-spelling` registers `skill-ovos-spelling.openvoiceos`, not
`ovos-skill-spelling.openvoiceos`: the entry point in its pyproject says so,
and the repository name is not the id. A pair filed under the repository
spelling would bridge nothing.

The skill renames `Spell.intent` to `spell.intent`, so a model trained
before the rename emits `skill-ovos-spelling.openvoiceos:Spell`.
"""
import re

from ovos_m2v_pipeline.renames import RENAMED_LABELS

REGISTERED = "skill-ovos-spelling.openvoiceos"


def test_the_pair_is_filed_under_the_registered_id():
    assert RENAMED_LABELS[f"{REGISTERED}:Spell"] == f"{REGISTERED}:spell"


def test_the_repository_spelling_is_not_used_as_an_id():
    """The control: a pair under the repo name would look right and do nothing."""
    assert "ovos-skill-spelling.openvoiceos:Spell" not in RENAMED_LABELS


def test_the_blacklist_rename_gets_no_pair():
    """A .blacklist name is not a label a model emits."""
    assert f"{REGISTERED}:spell.blacklist" not in RENAMED_LABELS
    assert f"{REGISTERED}:Spell.blacklist" not in RENAMED_LABELS


def test_every_target_stays_spec_compliant():
    for target in RENAMED_LABELS.values():
        assert re.match(r"^[a-z0-9_]+$", target.partition(":")[2]), target
