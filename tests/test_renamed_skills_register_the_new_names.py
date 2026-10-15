"""A skill with a bridge in renames.py must register the new names at its head.

A pair in `RENAMED_LABELS` says the skill registers the NEW name now. The
builder reads every skill at its branch head, so the check reads the same
head, from the workspace clone's remote-tracking ref, with the builder's own
`registered_intents`. A head that registers every bridge target and no
source is right. A head that registers every source and no target is a
rename that has not merged yet: the bridge is inert and that is allowed.
A head that registers some of each is a half-done rename, which trains both
labels, and is refused. It is skipped for a skill whose clone is absent, and
the positive control proves the reader sees the old names at a pre-rename
sha.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import yaml

from ovos_m2v_pipeline.renames import RENAMED_LABELS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "train" / "sources.yaml"

sys.path.insert(0, str(ROOT / "train"))
_spec = importlib.util.spec_from_file_location("build_dataset", ROOT / "train" / "build_dataset.py")
bd = importlib.util.module_from_spec(_spec)
sys.modules["build_dataset"] = bd
_spec.loader.exec_module(bd)

#: Pipeline ids are not skill repositories.
PIPELINE_IDS = {"ocp", "common_query", "stop", "persona", "common_reading"}

#: Skill id -> repository path under the workspace, for the ids whose
#: spelling is not the repository name.
REPO_FOR_ID = {
    "skill-ovos-spelling.openvoiceos": "ovos/skills/ovos-skill-spelling",
    "skill-ovos-wallpapers.openvoiceos": "ovos/skills/ovos-skill-wallpapers",
}


def _cfg():
    return yaml.safe_load(SOURCES.read_text(encoding="utf-8"))


def _workspace():
    """The clones the builder reads: `M2V_WORKSPACE`, else sources.yaml's."""
    return Path(os.environ.get("M2V_WORKSPACE") or _cfg()["workspace"]).expanduser()


def _repo_path(skill_id: str, refs: dict) -> str:
    if skill_id in REPO_FOR_ID:
        return REPO_FOR_ID[skill_id]
    name = skill_id.rsplit(".", 1)[0]
    matches = [p for p in refs if Path(p).name == name]
    assert len(matches) == 1, f"{skill_id}: {matches or 'no'} skill_refs entry"
    return matches[0]


def _by_skill() -> dict:
    out = {}
    for old, new in RENAMED_LABELS.items():
        skill_id, _, old_name = old.partition(":")
        if skill_id in PIPELINE_IDS:
            continue
        out.setdefault(skill_id, []).append((old_name, new.partition(":")[2]))
    return out


def _verdict(skill_id, pairs, registered):
    """None when the head is consistent with the bridge, else the refusal."""
    # Several old names can fold into one new name, so the sets are compared,
    # not their sizes. Before a fold merges, its target may already be a
    # label of its own, so an unmerged head is told by its old names alone.
    wanted_new = {new for _, new in pairs}
    wanted_old = {old for old, _ in pairs}
    news = wanted_new & registered
    olds = wanted_old & registered
    if (news == wanted_new and not olds) or olds == wanted_old:
        return None
    return (f"{skill_id}: the head registers new {sorted(news)} and old "
            f"{sorted(olds)} of {len(pairs)} bridged names")


@pytest.mark.parametrize("skill_id", sorted(_by_skill()))
def test_the_head_registers_the_new_names_and_not_the_old(skill_id):
    sys.path.insert(0, str(ROOT / "train"))
    import source_refs
    branches = source_refs.skill_entries(_cfg())
    path = _repo_path(skill_id, branches)
    repo = _workspace() / path
    if not repo.is_dir():
        pytest.skip(f"{path} is not cloned here")
    ref = f"origin/{branches[path]}"
    registered = bd.registered_intents(repo, ref)
    assert registered, f"{path}@{ref} registers nothing"
    refused = _verdict(skill_id, _by_skill()[skill_id], registered)
    assert refused is None, refused


def test_positive_control_a_pre_rename_sha_shows_the_old_names():
    """The reader sees the old names where they were: hello-world before #140."""
    repo = _workspace() / "ovos/skills/adult/ovos-skill-hello-world"
    if not repo.is_dir():
        pytest.skip("ovos-skill-hello-world is not cloned here")
    before = bd.registered_intents(repo, "a71f4e826f3a6d8c837deb41b1801db8ca86e063")
    assert "HelloWorldIntent" in before and "hello_world_intent" not in before


def test_negative_control_a_half_done_rename_is_refused():
    pairs = [("Laugh", "laugh"), ("RandomLaugh", "random_laugh")]
    assert _verdict("x", pairs, {"laugh", "RandomLaugh"}) is not None


def test_a_rename_that_has_not_merged_is_allowed():
    pairs = [("Laugh", "laugh"), ("RandomLaugh", "random_laugh")]
    assert _verdict("x", pairs, {"Laugh", "RandomLaugh"}) is None
    assert _verdict("x", pairs, {"laugh", "random_laugh"}) is None


def test_a_fold_of_several_names_into_one_is_a_completed_rename():
    pairs = [("create_alarm_alt", "create_alarm"), ("CreateAlarm", "create_alarm")]
    assert _verdict("x", pairs, {"create_alarm"}) is None
