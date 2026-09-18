"""A skill with a bridge in renames.py must be pinned at or after the rename.

A pair in `RENAMED_LABELS` says the skill registers the NEW name now. If
`train/sources.yaml` still pins that skill before the rename, the builder
reads the old resource files, trains the old label, and the bridge maps a
label the corpus itself still carries: the model keeps emitting the old
name and nothing moved. That is how ovos-skill-hello-world#140 was missed:
the bridge was absent and the pin predated the commit, and no test read
the two together.

The check reads each repository at its pin with the builder's own
`registered_intents`, from the local workspace clone the builder reads
from. A pin that registers every bridge target and no source is right.
A pin that registers every source and no target is allowed only while
the repository's `origin/dev` registers the sources too, which is a
rename that has not merged yet. Anything else is refused. It is skipped for
a skill whose clone is absent, and the positive control proves the reader
sees the old names at a pre-rename sha.
"""
import importlib.util
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

#: Pipeline ids are not skill repositories and have no pin.
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
    return Path(_cfg()["workspace"]).expanduser()


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


@pytest.mark.parametrize("skill_id", sorted(_by_skill()))
def test_the_pin_registers_the_new_names_and_not_the_old(skill_id):
    refs = _cfg()["skill_refs"]["refs"]
    path = _repo_path(skill_id, refs)
    repo = _workspace() / path
    if not repo.is_dir():
        pytest.skip(f"{path} is not cloned here")
    registered = bd.registered_intents(repo, refs[path])
    assert registered, f"{path}@{refs[path][:7]} registers nothing"
    pairs = _by_skill()[skill_id]
    news = {new for _, new in pairs if new in registered}
    olds = {old for old, _ in pairs if old in registered}
    if len(news) == len(pairs) and not olds:
        return
    # A bridge may land before the skill's rename merges (laugh#131 is open
    # while m2v#215 is on dev). Then the pin registers every old name and
    # no new one, the bridge is inert, and that is allowed as long as the
    # repository's dev head registers the old names too. Once dev carries
    # the rename, a pin behind it trains the label the bridge maps away.
    assert len(olds) == len(pairs) and not news, (
        f"{skill_id}: the pin {refs[path][:7]} registers new {sorted(news)} "
        f"and old {sorted(olds)} of {len(pairs)} bridged names")
    on_dev = bd.registered_intents(repo, "origin/dev")
    renamed_on_dev = {new for _, new in pairs if new in on_dev}
    assert not renamed_on_dev, (
        f"{skill_id}: dev registers {sorted(renamed_on_dev)} but the pin "
        f"{refs[path][:7]} still registers {sorted(olds)}; move the pin past "
        f"the rename")


def test_positive_control_a_pre_rename_sha_shows_the_old_names():
    """The reader sees the old names where they were: hello-world before #140."""
    repo = _workspace() / "ovos/skills/adult/ovos-skill-hello-world"
    if not repo.is_dir():
        pytest.skip("ovos-skill-hello-world is not cloned here")
    before = bd.registered_intents(repo, "a71f4e826f3a6d8c837deb41b1801db8ca86e063")
    assert "HelloWorldIntent" in before and "hello_world_intent" not in before
