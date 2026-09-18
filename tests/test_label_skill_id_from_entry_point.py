"""A corpus label must carry the skill id the entry point declares.

`ovos-skill-easter-eggs` is the repository name. The entry point declares
`skill-easter-eggs.openvoiceos`, and that is the id ovos-workshop computes
and the id the skill registers on the bus. A corpus row filed under the
repository name therefore trains a label no skill answers to, and the model
that learns it can never route.

`reduce_skill_id` resolves a corpus spelling to a registered id by folding
both to a token multiset. The repository name carries a vendor token the
entry point does not, so the two folds differed and the resolution failed
silently, leaving the row on the wrong label.
"""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ovos_spec_tools")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
import build_dataset as bd  # noqa: E402

REPO_SPELLING = "ovos-skill-easter-eggs.openvoiceos"
REGISTERED = "skill-easter-eggs.openvoiceos"


def _registry(*skill_ids):
    by_fold = {}
    for skill_id in skill_ids:
        by_fold.setdefault(bd.fold_skill(skill_id), []).append(skill_id)
    return by_fold


def test_repo_spelling_resolves_to_the_registered_id():
    assert bd.reduce_skill_id(REPO_SPELLING, _registry(REGISTERED)) == REGISTERED


def test_the_word_order_case_still_resolves():
    """The case fold_skill was written for keeps working."""
    assert bd.reduce_skill_id(
        "ovos-skill-wallpapers.openvoiceos",
        _registry("skill-ovos-wallpapers.openvoiceos"),
    ) == "skill-ovos-wallpapers.openvoiceos"


def test_an_id_that_is_registered_is_returned_unchanged():
    assert bd.reduce_skill_id(REGISTERED, _registry(REGISTERED)) == REGISTERED


def test_an_unknown_skill_still_resolves_to_nothing():
    """The control: folding must not turn every id into a match."""
    assert bd.reduce_skill_id(
        "ovos-skill-not-a-real-skill.openvoiceos", _registry(REGISTERED)) is None


def test_two_registered_ids_that_fold_together_resolve_to_neither():
    """Ambiguity is refused, not guessed.

    A wrong label is worse than an unresolved one: the row would train a
    skill that never claims the utterance.
    """
    by_fold = _registry("skill-camera.openvoiceos", "ovos-skill-camera.openvoiceos")
    assert bd.reduce_skill_id("camera.openvoiceos", by_fold) is None


def test_the_entry_point_is_where_the_id_comes_from():
    """The id is read off the repository, never built from its name."""
    repo = Path("/home/agent/AgentWorkspaces/ovos/skills/ovos-skill-easter-eggs")
    if not repo.is_dir():
        pytest.skip("easter-eggs clone not present")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "origin/dev"],
        capture_output=True, text=True)
    if head.returncode:
        pytest.skip("origin/dev not fetched")
    assert bd.skill_id_from_repo(repo, "origin/dev") == REGISTERED
