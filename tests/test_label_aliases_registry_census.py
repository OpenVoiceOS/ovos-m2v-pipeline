"""Every `LABEL_ALIASES` value that names a skill must be a label the pinned
skill revision actually registers.

`train/build_dataset.py` states its own contract on `LABEL_ALIASES`: "each
right-hand side is what the pinned revision registers today." A right-hand
side that the pinned sha does not register rescues nothing -- the corpus
rows that alias was meant to save still drop as `unresolved_labels`, and the
comment lies about what the table does.

This test does not reach the network or a local skill clone, and it does not
derive its ground truth from `LABEL_ALIASES` itself -- that would make every
alias trivially pass no matter what it claimed. `PINNED_REGISTRATIONS` below
is the real intent set each skill in `train/sources.yaml`'s `skill_refs`
registers at its pinned sha, read once (offline, from a local workspace
checkout of the real skill repos already fetched to those shas) and frozen
here as fixture data. The test rebuilds those skills as fixture git repos
carrying exactly that intent set, then runs `build_dataset.build_registry` /
the alias table over the fixtures exactly as the real builder does. An alias
edit that points at a label outside this frozen set fails here; a real
skill's registrations moving without a matching update to this fixture is a
gap this test cannot see, so a wave that re-pins `sources.yaml` must refresh
`PINNED_REGISTRATIONS` from the new sha at the same time.
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

#: The full intent set each skill registers at the sha `train/sources.yaml`
#: pins for it today, read offline from a local checkout of the real skill
#: repo at that exact sha with `build_dataset.registered_intents`. This is
#: ground truth independent of `LABEL_ALIASES` -- it is not filtered to the
#: names the alias table happens to claim.
PINNED_REGISTRATIONS = {
    "ovos-skill-alerts.openvoiceos": {
        "AddListSubitems", "CalendarList", "CancelAlert",
        "ChangeMediaProperties", "ChangePriority", "ChangeRepeat",
        "ChangeUntil", "CreateAlarm", "CreateAlarmAlt", "CreateEvent",
        "CreateList", "CreateReminder", "CreateTimer", "DAVSync",
        "DeleteList", "DeleteListEntries", "DeleteTodoEntries", "ListAlerts",
        "QueryListEntries", "QueryListNames", "QueryTodoEntries",
        "RescheduleAlert", "TimerStatus", "create_reminder_recurring",
        "missed_alerts",
    },
    "ovos-skill-confucius-quotes.openvoiceos": {
        "ConfuciusQuote", "confucius_lifespan", "who",
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
        "volume.max.boost", "volume.mute", "volume.mute.toggle",
        "volume.reset", "volume.unmute", "volume_level",
    },
    "ovos-skill-weather.openvoiceos": {
        "do-i-need-an-umbrella", "do.i.need.an.umbrella", "humidity",
        "is_hot_or_cold", "is_wind", "next_rain", "sunrise", "sunset",
        "temperature", "weather", "weather_condition",
    },
    "ovos-skill-wolfie.openvoiceos": {"search_wolfie"},
    "ovos-skill-wordnet.openvoiceos": {"search_wordnet"},
    "skill-ovos-wallpapers.openvoiceos": {
        "make_wallpaper", "next_picture", "picture_about", "picture_random",
        "previous_picture", "wallpaper_about", "wallpaper_random",
    },
}

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
    labels, by_skill_fold, intents_by_skill = bd.build_registry(cfg, ws)
    return labels


def test_positive_control_registry_contains_a_known_label(tmp_path):
    """Sanity check the fixture-registry builder itself: a label the fixture
    obviously carries must come back, or every result below is worthless."""
    labels = _build_fixture_registry(
        tmp_path, {"a_skill.openvoiceos": {"known.intent"}})
    assert "a_skill.openvoiceos:known.intent" in labels


def test_every_aliased_skill_has_a_frozen_registration_set():
    """Guards `PINNED_REGISTRATIONS` itself: an alias naming a skill not
    listed there would be silently unchecked by the test below."""
    missing = _ALIASED_SKILLS - set(PINNED_REGISTRATIONS)
    assert not missing, f"no frozen registration set for {missing}"


def test_every_skill_alias_resolves_at_the_pinned_revision(tmp_path):
    labels = _build_fixture_registry(tmp_path, PINNED_REGISTRATIONS)
    checked = sorted(SKILL_ALIASES.items())
    failing = [(k, v) for k, v in checked if v not in labels]
    assert not failing, (
        f"{len(failing)}/{len(checked)} LABEL_ALIASES values do not resolve "
        f"at the pinned revision they claim to: {failing}")
