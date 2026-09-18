"""Every `LABEL_ALIASES` value that names a skill must be a label the pinned
skill revision actually registers.

`train/build_dataset.py` states its own contract on `LABEL_ALIASES`: "each
right-hand side is what the pinned revision registers today." A right-hand
side that the pinned sha does not register rescues nothing -- the corpus
rows that alias was meant to save still drop as `unresolved_labels`, and the
comment lies about what the table does.

`PINNED_REGISTRATIONS` below is the real intent set each aliased skill
registers, read with `build_dataset.registered_intents` from the real skill
repo and frozen here. It does not derive from `LABEL_ALIASES`, so an alias
cannot pass because the table claims it. The snapshot is keyed by the sha it
was read at, in `PINNED_REFS`. The test rebuilds those skills as fixture git
repos carrying exactly that intent set, then runs
`build_dataset.build_registry` over them as the real builder does.

Three checks keep the snapshot honest:

- `test_the_snapshot_is_keyed_by_the_shas_sources_yaml_pins` reads
  `train/sources.yaml`. A re-pin of an aliased skill fails it until
  `PINNED_REFS` and `PINNED_REGISTRATIONS` are refreshed from the new sha.
- `test_the_snapshot_matches_the_real_repos_at_the_pinned_shas` reads each
  real repo at the sha `sources.yaml` pins, over the network. It runs only
  when `M2V_NETWORK=1`.
- the positive control proves the fixture registry builder finds a label
  that is present.
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
        "DeleteList", "DeleteListEntries", "DeleteTodoEntries",
        "ListAlerts", "QueryListEntries", "QueryListNames",
        "QueryTodoEntries", "RescheduleAlert", "TimerStatus",
        "create_reminder_recurring", "missed_alerts",
    },
    "ovos-skill-confucius-quotes.openvoiceos": {
        "confucius_lifespan", "confucius_quote", "who",
    },
    "ovos-skill-fuster-quotes.openvoiceos": {
        "fuster_lifespan", "fuster_quotes", "who",
    },
    "ovos-skill-mark1-ctrl.openvoiceos": {
        "blink", "brightness", "crazy_eyes", "custom_eye_color",
        "eye_color", "listen", "look_down", "look_left", "look_left_right",
        "look_right", "look_up", "look_up_down", "narrow_eyes", "reset",
        "smile", "spin", "think",
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

#: The `skill_refs` entry and the sha each `PINNED_REGISTRATIONS` set was
#: read at. A re-pin in `train/sources.yaml` must update both tables.
PINNED_REFS = {
    "ovos-skill-alerts.openvoiceos":
        ("ovos/skills/ovos-skill-alerts", "ab3e82b516aefab3d03be6989548dabebd4594ee"),
    "ovos-skill-confucius-quotes.openvoiceos":
        ("ovos/skills/ovos-skill-confucius-quotes", "eb0dd4cdfffa99b7e369378a4b3150a92f1e73d2"),
    "ovos-skill-fuster-quotes.openvoiceos":
        ("ovos/skills/ovos-skill-fuster-quotes", "adcd72cbc2e883f434a81cafe818a76dde829d22"),
    "ovos-skill-mark1-ctrl.openvoiceos":
        ("ovos/skills/ovos-skill-mark1-ctrl", "3d3e5abe2f9bef4e9a8a9a042abda4827b40fd57"),
    "ovos-skill-volume.openvoiceos":
        ("ovos/skills/ovos-skill-volume", "94a3c3ac7841edffb15546abe50f16b404561092"),
    "ovos-skill-weather.openvoiceos":
        ("ovos/skills/ovos-skill-weather", "64900360ba0eb1a59cbdad638ec3780c63e20b41"),
    "ovos-skill-wolfie.openvoiceos":
        ("ovos/skills/ovos-skill-wolfie", "4db152bf949e9c87bae6cd1b3c0fe233cfc0bb44"),
    "ovos-skill-wordnet.openvoiceos":
        ("ovos/skills/ovos-skill-wordnet", "6efb78c2250c3ab0c4b754fa75be99d5ad260c6e"),
    "skill-ovos-wallpapers.openvoiceos":
        ("ovos/skills/ovos-skill-wallpapers", "370344daf3c17c837c3713ddf457d5d677232327"),
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


def _sources_refs() -> dict:
    import yaml
    return yaml.safe_load(SOURCES.read_text(encoding="utf-8"))["skill_refs"]["refs"]


def test_the_snapshot_is_keyed_by_the_shas_sources_yaml_pins():
    assert set(PINNED_REFS) == set(PINNED_REGISTRATIONS)
    refs = _sources_refs()
    stale = {skill_id: (path, sha, refs.get(path))
             for skill_id, (path, sha) in sorted(PINNED_REFS.items())
             if refs.get(path) != sha}
    assert not stale, (
        "train/sources.yaml re-pins an aliased skill; refresh PINNED_REFS and "
        f"PINNED_REGISTRATIONS from the new sha: {stale}")


def _fetch(url: str) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "m2v-alias-census"})
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def _registered_at(repo: str, sha: str) -> set:
    """`build_dataset.registered_intents`, read from GitHub at *sha*."""
    import json
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
def test_the_snapshot_matches_the_real_repos_at_the_pinned_shas():
    refs = _sources_refs()
    wrong = {}
    for skill_id, (path, _sha) in sorted(PINNED_REFS.items()):
        sha = refs[path]
        real = _registered_at(f"OpenVoiceOS/{Path(path).name}", sha)
        if real != PINNED_REGISTRATIONS[skill_id]:
            wrong[skill_id] = {"missing": sorted(real - PINNED_REGISTRATIONS[skill_id]),
                               "extra": sorted(PINNED_REGISTRATIONS[skill_id] - real)}
    assert not wrong, f"frozen registrations differ from the pinned repos: {wrong}"
