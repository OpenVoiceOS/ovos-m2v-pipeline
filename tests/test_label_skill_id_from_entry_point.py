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
import os
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


def _git_repo(path, files):
    """A one-commit repository holding *files*, so skill_id_from_repo can
    read them back with `git show HEAD:<path>` the way it reads a clone."""
    path.mkdir()
    for name, text in files.items():
        (path / name).write_text(text)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x", "HOME": str(path), "PATH": os.environ["PATH"]}
    for cmd in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "fixture"]):
        subprocess.run(["git", "-C", str(path)] + cmd, check=True, env=env,
                       capture_output=True, text=True)
    return path


# The setup.py of ovos-skill-easter-eggs at dev, the lines the id comes from:
# the repository name is "ovos-skill-easter-eggs", the entry point key is
# "skill-easter-eggs.openvoiceos", built from module constants, never from
# the directory name.
EASTER_EGGS_SETUP = """\
from setuptools import setup

SKILL_NAME = "skill-easter-eggs"
SKILL_PKG = SKILL_NAME.replace("-", "_")
PLUGIN_ENTRY_POINT = f"{SKILL_NAME}.openvoiceos={SKILL_PKG}:EasterEggsSkill"

setup(
    name=f"{SKILL_NAME}",
    url=f"https://github.com/OpenVoiceOS/{SKILL_NAME}",
    entry_points={"ovos.plugin.skill": PLUGIN_ENTRY_POINT},
)
"""

EASTER_EGGS_PYPROJECT = """\
[project]
name = "skill-easter-eggs"

[project.entry-points."ovos.plugin.skill"]
"skill-easter-eggs.openvoiceos" = "skill_easter_eggs:EasterEggsSkill"
"""


def test_the_entry_point_is_where_the_id_comes_from(tmp_path):
    """The id is read off the repository's setup.py, never built from its
    directory name: the directory is the repository spelling."""
    repo = _git_repo(tmp_path / "ovos-skill-easter-eggs", {"setup.py": EASTER_EGGS_SETUP})
    assert bd.skill_id_from_repo(repo, "HEAD") == REGISTERED


def test_a_pyproject_entry_point_gives_the_same_id(tmp_path):
    repo = _git_repo(tmp_path / "ovos-skill-easter-eggs", {"pyproject.toml": EASTER_EGGS_PYPROJECT})
    assert bd.skill_id_from_repo(repo, "HEAD") == REGISTERED


def test_a_repo_that_declares_nothing_is_not_named_after_its_directory(tmp_path):
    """The control: with no entry point the reader refuses rather than
    inventing an id from the directory name."""
    repo = _git_repo(tmp_path / "ovos-skill-easter-eggs", {"README.md": "no packaging\n"})
    with pytest.raises(SystemExit):
        bd.skill_id_from_repo(repo, "HEAD")
