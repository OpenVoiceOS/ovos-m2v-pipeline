"""Every `LABEL_ALIASES` value that names a skill must be a label the skill
registers at the head the builder reads.

`train/build_dataset.py` states its own contract on `LABEL_ALIASES`: each
right-hand side is what the skill registers today. A right-hand side the
skill does not register rescues nothing -- the corpus rows that alias was
meant to save still drop as `unresolved_labels`, and the comment lies about
what the table does.

Two checks, one offline and one over the network:

- `FIXTURE_REGISTRATIONS` is an intent set per aliased skill, read once from
  the real repositories with `build_dataset.registered_intents`. It does not
  derive from `LABEL_ALIASES`, so an alias cannot pass because the table
  claims it. The offline test rebuilds those skills as fixture git repos
  carrying exactly that set and runs `build_dataset.build_registry` over
  them as the real builder does. It proves the registry maps the aliases;
  it says nothing about today's skills.
- `test_every_skill_alias_resolves_at_the_dev_head` reads each real skill at
  the branch `train/sources.yaml` names for it, over the network, and checks
  every alias against what it registers there. It runs only when
  `M2V_NETWORK=1`. A head that moves on is not a failure; a head that stops
  registering an alias target is.

The positive control proves the fixture registry builder finds a label that
is present.
"""
import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

BUILDER = Path(__file__).resolve().parents[1] / "train" / "build_dataset.py"

spec = importlib.util.spec_from_file_location("build_dataset", BUILDER)
bd = importlib.util.module_from_spec(spec)
sys.modules["build_dataset"] = bd
spec.loader.exec_module(bd)


def _alias_items() -> dict:
    """Read `LABEL_ALIASES` out of the builder's source as a literal dict,
    the same way `test_renamed_label_aliases.py` does, so this test does not
    depend on `build_dataset` importing cleanly in every test environment."""
    tree = ast.parse(BUILDER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "LABEL_ALIASES":
            return {ast.literal_eval(k): ast.literal_eval(v)
                    for k, v in zip(node.value.keys, node.value.values)}
    raise AssertionError("LABEL_ALIASES not found in build_dataset.py")


#: `LABEL_ALIASES` values naming a skill repo (excludes the four whose value
#: names a pipeline-plugin family -- ocp/common_query/persona -- which are
#: not registered from a skill's locale resources).
_ALIASES = _alias_items()
PIPELINE_IDS = {"ocp", "common_query", "stop", "persona", "common_reading"}
SKILL_ALIASES = {k: v for k, v in _ALIASES.items()
                  if v.partition(":")[0] not in PIPELINE_IDS}

#: The intent set each aliased skill registered when this fixture was read,
#: from a checkout of the real skill repo with
#: `build_dataset.registered_intents`. Independent of `LABEL_ALIASES`: it is
#: not filtered to the names the alias table happens to claim.
FIXTURE_REGISTRATIONS = {
    "ovos-skill-alerts.openvoiceos": {
        "add_list_subitems", "calendar_list", "cancel_alert",
        "change_media_properties", "change_priority", "change_repeat",
        "change_until", "create_alarm", "create_event", "create_list",
        "create_reminder", "create_timer", "dav_sync", "delete_list",
        "delete_list_entries",
        "list_alerts", "missed_alerts", "query_list_entries",
        "query_list_names", "reschedule_alert", "timer_status",
    },
    "ovos-skill-confucius-quotes.openvoiceos": {
        "confucius_lifespan", "confucius_quote", "who",
    },
    "ovos-skill-fuster-quotes.openvoiceos": {
        "fuster_lifespan", "fuster_quotes", "who",
    },
    "ovos-skill-mark1-ctrl.openvoiceos": {
        "blink", "brightness", "crazy_eyes", "custom_eye_color", "eye_color",
        "listen", "look_down", "look_left", "look_left_right", "look_right",
        "look_up", "look_up_down", "narrow_eyes", "reset", "smile", "spin",
        "think",
    },
    "ovos-skill-volume.openvoiceos": {
        "change_volume", "current_volume", "increase_volume", "less_volume",
        "volume_level", "volume_max_boost", "volume_mute",
        "volume_mute_toggle", "volume_reset", "volume_unmute",
    },
    "ovos-skill-weather.openvoiceos": {
        "humidity", "is_wind", "next_rain", "sunrise",
        "sunset", "temperature", "weather", "weather_condition",
    },
    "ovos-skill-wolfie.openvoiceos": {
        "search_wolfie",
    },
    "ovos-skill-wordnet.openvoiceos": {
        "search_wordnet",
    },
    "skill-ovos-wallpapers.openvoiceos": {
        "make_wallpaper", "next_picture", "picture_about", "picture_random",
        "previous_picture", "wallpaper_about", "wallpaper_random",
    },
}

#: The `skill_refs` path of each aliased skill.
ALIASED_REPOS = {
    "ovos-skill-alerts.openvoiceos": "ovos/skills/ovos-skill-alerts",
    "ovos-skill-confucius-quotes.openvoiceos": "ovos/skills/ovos-skill-confucius-quotes",
    "ovos-skill-fuster-quotes.openvoiceos": "ovos/skills/ovos-skill-fuster-quotes",
    "ovos-skill-mark1-ctrl.openvoiceos": "ovos/skills/ovos-skill-mark1-ctrl",
    "ovos-skill-volume.openvoiceos": "ovos/skills/ovos-skill-volume",
    "ovos-skill-weather.openvoiceos": "ovos/skills/ovos-skill-weather",
    "ovos-skill-wolfie.openvoiceos": "ovos/skills/ovos-skill-wolfie",
    "ovos-skill-wordnet.openvoiceos": "ovos/skills/ovos-skill-wordnet",
    "skill-ovos-wallpapers.openvoiceos": "ovos/skills/ovos-skill-wallpapers",
}

SOURCES = Path(__file__).resolve().parents[1] / "train" / "sources.yaml"

#: Every skill a `LABEL_ALIASES` value names must have a frozen registration
#: set above, or the test below would silently skip checking it.
_ALIASED_SKILLS = {v.partition(":")[0] for v in SKILL_ALIASES.values()}


def git_repo(path: Path, files: dict) -> str:
    path.mkdir(parents=True)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True,
                                     env=env, capture_output=True)
    run("init", "-q", "-b", "main")
    for name, text in files.items():
        f = path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "fixture")
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()


def _build_fixture_registry(tmp_path: Path, fixture_intents: dict):
    """Build a `build_dataset` registry from fixture repos carrying exactly
    `fixture_intents` (skill_id -> {intent stems}), the same shape
    `build_registry` reads a real `skill_refs` workspace as."""
    ws = tmp_path / "ws"
    refs = {}
    for i, (skill_id, intents) in enumerate(sorted(fixture_intents.items())):
        files = {"pyproject.toml":
                 f'[project.entry-points."ovos.plugin.skill"]\n'
                 f'"{skill_id}" = "s:S"\n'}
        for intent in sorted(intents):
            files[f"locale/en-US/{intent}.intent"] = "fixture utterance\n"
        repo_path = ws / f"repo{i}"
        refs[f"repo{i}"] = git_repo(repo_path, files)
    cfg = {
        "skill_refs": {"refs": refs},
        "git_sources": [],
    }
    return bd.build_registry(cfg, ws)


def test_positive_control_registry_contains_a_known_label(tmp_path):
    """Sanity check the fixture-registry builder itself: a label the fixture
    obviously carries must come back, or every result below is worthless."""
    labels, _, _ = _build_fixture_registry(
        tmp_path, {"a_skill.openvoiceos": {"known.intent"}})
    assert "a_skill.openvoiceos:known.intent" in labels


def test_every_aliased_skill_has_a_frozen_registration_set():
    """Guards `FIXTURE_REGISTRATIONS` itself: an alias naming a skill not
    listed there would be silently unchecked by the test below."""
    missing = _ALIASED_SKILLS - set(FIXTURE_REGISTRATIONS)
    assert not missing, f"no frozen registration set for {missing}"


def test_every_skill_alias_resolves_in_the_fixture_registry(tmp_path):
    labels, _, _ = _build_fixture_registry(tmp_path, FIXTURE_REGISTRATIONS)
    checked = sorted(SKILL_ALIASES.items())
    failing = [(k, v) for k, v in checked if v not in labels]
    assert not failing, (
        f"{len(failing)}/{len(checked)} LABEL_ALIASES values do not resolve "
        f"in the registry built from the fixture: {failing}")


LEDGER = Path(__file__).resolve().parents[1] / "train" / "labels.yaml"

#: Corpus spellings older than any ledger label that must reach the same
#: survivor as the ledger entry they predate. The localize export keys on
#: file names; the older exports carry the Adapt intent name.
OLD_SPELLINGS = {
    "ovos-skill-alerts.openvoiceos:CreateAlarmAlt": "ovos-skill-alerts.openvoiceos:create_alarm",
}

#: Ledger removals this builder sends somewhere else. `INTENT_ALIASES` maps
#: the corpora's `volume.mute` to `volume.mute.toggle`, which resolves to
#: `volume_mute_toggle`; the ledger, read from the skill's own files, names
#: `volume_mute`. Which intent the corpus rows mean is a separate question
#: from the folds this test guards.
LEDGER_EXCEPTIONS = {"ovos-skill-volume.openvoiceos:volume.mute"}


def _folds_to_check() -> dict:
    """Every label the ledger removes from a skill with a frozen registration
    set, with the label its phrasings went to. A removal whose label the
    fixture still registers has not happened yet and is left out."""
    import yaml
    ledger = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    folds = {}
    for entry in ledger["removed"]:
        label, target = entry["label"], entry.get("merged_into")
        skill, _, intent = label.partition(":")
        if not target or skill not in FIXTURE_REGISTRATIONS:
            continue
        if intent in FIXTURE_REGISTRATIONS[skill] or label in LEDGER_EXCEPTIONS:
            continue
        folds[label] = target
    return {**folds, **OLD_SPELLINGS}


def test_every_folded_label_reaches_its_survivor(tmp_path):
    """A corpus row under a label the skill folded away must train the label
    it folded into. Without an alias the row resolves to nothing and drops
    as `unknown-intent`, and no other check notices, because the alias that
    is missing has no right-hand side to test."""
    registry = _build_fixture_registry(tmp_path, FIXTURE_REGISTRATIONS)
    folds = _folds_to_check()
    assert "ovos-skill-alerts.openvoiceos:create_reminder_recurring" in folds
    wrong = {}
    for label, target in sorted(folds.items()):
        skill, _, intent = label.partition(":")
        got = bd.resolve_label(bd.make_label(skill, intent), *registry, {})
        if got[0] != target:
            wrong[label] = {"want": target, "got": got}
    assert not wrong, (
        f"{len(wrong)}/{len(folds)} folded labels do not reach their survivor: "
        f"{wrong}")


def _branches() -> dict:
    import yaml
    sys.path.insert(0, str(BUILDER.parent))
    import source_refs
    return source_refs.skill_entries(
        yaml.safe_load(SOURCES.read_text(encoding="utf-8")))


def test_every_aliased_skill_is_a_listed_skill_repository():
    assert set(ALIASED_REPOS) == set(FIXTURE_REGISTRATIONS)
    missing = sorted(p for p in ALIASED_REPOS.values() if p not in _branches())
    assert not missing, f"aliased skills train/sources.yaml does not list: {missing}"


def _fetch(url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "m2v-alias-census"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _registered_at(repo: str, ref: str) -> set:
    """`build_dataset.registered_intents`, read from GitHub at *ref*."""
    import json
    sha = json.loads(_fetch(
        f"https://api.github.com/repos/{repo}/commits/{ref}"))["sha"]
    tree = json.loads(_fetch(
        f"https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1"))
    assert not tree.get("truncated"), f"{repo}@{sha}: tree listing truncated"
    declared, stems = set(), set()
    for entry in tree["tree"]:
        path = entry["path"]
        if entry["type"] != "blob" or path.split("/")[0] in {"test", "tests"}:
            continue
        if path.endswith(".intent"):
            stems.add(Path(path).stem)
        elif path.endswith(".py"):
            src = _fetch(f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"
                         ).decode("utf-8", "replace")
            declared.update(bd._INTENT_BUILDER_RE.findall(src))
            declared.update(n[:-len(".intent")]
                            for n in bd._INTENT_FILE_RE.findall(src))
    folds = {bd.fold(n) for n in declared}
    return declared | {s for s in stems if bd.fold(s) not in folds}


@pytest.mark.skipif(os.environ.get("M2V_NETWORK") != "1",
                    reason="reads the real skill repos; set M2V_NETWORK=1")
def test_every_skill_alias_resolves_at_the_dev_head():
    branches = _branches()
    failing = []
    for skill_id, path in sorted(ALIASED_REPOS.items()):
        real = _registered_at(f"OpenVoiceOS/{Path(path).name}", branches[path])
        failing += [(k, v) for k, v in sorted(SKILL_ALIASES.items())
                    if v.partition(":")[0] == skill_id
                    and v.partition(":")[2] not in real]
    assert not failing, (
        f"LABEL_ALIASES values the skills do not register at their heads: {failing}")
