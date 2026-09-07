#!/usr/bin/env python3
"""Build the ovos-m2v-pipeline intent training corpus from pinned sources.

Every source in ``sources.yaml`` names an immutable revision. Nothing here
reaches for a branch tip: a git revision that the local clone does not carry,
or a Hugging Face revision that no longer resolves, is a hard error rather
than a silent substitution. The same manifest therefore always yields the
same corpus.

Labels follow ``docs/labels.md``: ``<skill_id>:<intent_name>`` exactly as
``ovos_m2v_pipeline`` registers it at runtime.
"""
import argparse
import collections
import csv
import fnmatch
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
from ovos_spec_tools.expansion import iter_expand

from ovos_m2v_pipeline.slots import expand_entities, MAX_ENTITY_EXPANSIONS

HERE = Path(__file__).resolve().parent

#: A training row is one example, not a live registration -- an evenly
#: strided sample of this many entity-filled variants per template is
#: enough to teach the classifier the slot is filled, without multiplying
#: the corpus by an entity's full value count on every matching row.
TEMPLATE_FILL_CAP = 3

#: Sentences one padatious template line may expand to. A template with
#: several alternation groups multiplies out combinatorially; past this many
#: variants the extra sentences are near-duplicates that only reweight the
#: corpus toward the skill that happens to write the longest templates.
TEMPLATE_EXPANSION_CAP = 64

# ---------------------------------------------------------------- labels ----

#: Pipeline-plugin families. Everything else is family ``skill``.
PIPELINE_IDS = {"ocp", "common_query", "stop", "persona", "common_reading"}

#: Bare pipeline ids and legacy corpus spellings that must not gain the
#: ``.openvoiceos`` suffix a real skill id carries.
BARE_SKILL_IDS = PIPELINE_IDS | {
    "ovos-ocp-pipeline-plugin",
    "ovos-common-query-pipeline-plugin",
    "ovos-common-reading-pipeline-plugin",
    "ovos-option-matcher-fuzzy-plugin",
    "ovos-persona",
    "ovos-core",
}

#: Cross-source aliases: a label some corpus spells one way that denotes an
#: intent another corpus (and the runtime) spells another way. Left-hand
#: sides are post-normalisation labels.
LABEL_ALIASES = {
    "ovos-skill-ddg.openvoiceos:search_wolfie": "ovos-skill-wolfie.openvoiceos:search_wolfie",
    "ovos-skill-ddg.openvoiceos:common_query": "common_query:common_query",
    "ovos-skill-confucius-quotes.openvoiceos:common_query": "ovos-skill-confucius-quotes.openvoiceos:who",
    "ovos-skill-fuster-quotes.openvoiceos:common_query": "ovos-skill-fuster-quotes.openvoiceos:who",
    "ovos-skill-volume.openvoiceos:volume.mute.intent.toggle": "ovos-skill-volume.openvoiceos:volume.mute.toggle",
    "ovos-common-query-pipeline-plugin:search_fakewiki": "common_query:search_fakewiki",
    "ovos-ocp-pipeline-plugin:play": "ocp:play",
    "ovos-persona:ask": "persona:ask",

    # Renames and folds the corpora predate. Each left-hand side is a
    # spelling a skill registered before the Adapt-to-`.intent` migration or
    # the unification wave; each right-hand side is what the pinned revision
    # registers today. See docs/labels.md, "Renames, merges, and the
    # unification wave".

    # ovos-skill-mark1-ctrl#45: the 14 Adapt IntentBuilder names became
    # lowercase file-intents, one for one.
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureEyesBlink": "ovos-skill-mark1-ctrl.openvoiceos:blink",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureEyesSpin": "ovos-skill-mark1-ctrl.openvoiceos:spin",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureEyesNarrow": "ovos-skill-mark1-ctrl.openvoiceos:narrow_eyes",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureCrazyEyes": "ovos-skill-mark1-ctrl.openvoiceos:crazy_eyes",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureMouthSmile": "ovos-skill-mark1-ctrl.openvoiceos:smile",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureMouthListen": "ovos-skill-mark1-ctrl.openvoiceos:listen",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureMouthThink": "ovos-skill-mark1-ctrl.openvoiceos:think",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureLookUp": "ovos-skill-mark1-ctrl.openvoiceos:look_up",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureLookDown": "ovos-skill-mark1-ctrl.openvoiceos:look_down",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureLookLeft": "ovos-skill-mark1-ctrl.openvoiceos:look_left",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureLookRight": "ovos-skill-mark1-ctrl.openvoiceos:look_right",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureLookUpDown": "ovos-skill-mark1-ctrl.openvoiceos:look_up_down",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureLookLeftRight": "ovos-skill-mark1-ctrl.openvoiceos:look_left_right",
    "ovos-skill-mark1-ctrl.openvoiceos:EnclosureReset": "ovos-skill-mark1-ctrl.openvoiceos:reset",

    # ovos-skill-wallpapers#82: the context-gated Adapt intents became file
    # intents. The skill registers under `skill-ovos-wallpapers`.
    "ovos-skill-wallpapers.openvoiceos:MakeWallpaperIntent": "skill-ovos-wallpapers.openvoiceos:make_wallpaper",
    "ovos-skill-wallpapers.openvoiceos:NextPictureIntent": "skill-ovos-wallpapers.openvoiceos:next_picture",
    "ovos-skill-wallpapers.openvoiceos:PrevPictureIntent": "skill-ovos-wallpapers.openvoiceos:previous_picture",

    # ovos-skill-weather#230: six condition intents became one, dispatched
    # from the shared weather_condition vocabulary.
    "ovos-skill-weather.openvoiceos:is_clear": "ovos-skill-weather.openvoiceos:weather_condition",
    "ovos-skill-weather.openvoiceos:is_cloudy": "ovos-skill-weather.openvoiceos:weather_condition",
    "ovos-skill-weather.openvoiceos:is_fog": "ovos-skill-weather.openvoiceos:weather_condition",
    "ovos-skill-weather.openvoiceos:is_rain": "ovos-skill-weather.openvoiceos:weather_condition",
    "ovos-skill-weather.openvoiceos:is_snow": "ovos-skill-weather.openvoiceos:weather_condition",
    "ovos-skill-weather.openvoiceos:is_stormy": "ovos-skill-weather.openvoiceos:weather_condition",

    # ovos-skill-weather#243: the forecast and temperature families each fold
    # to one label, and is_hot/is_cold (#231's split of the older is_hot_cold
    # Adapt intent) fold back together.
    "ovos-skill-weather.openvoiceos:current_weather": "ovos-skill-weather.openvoiceos:weather",
    "ovos-skill-weather.openvoiceos:hourly_forecast": "ovos-skill-weather.openvoiceos:weather",
    "ovos-skill-weather.openvoiceos:daily_forecast": "ovos-skill-weather.openvoiceos:weather",
    "ovos-skill-weather.openvoiceos:N_days_forecast": "ovos-skill-weather.openvoiceos:weather",
    "ovos-skill-weather.openvoiceos:weekend_forecast": "ovos-skill-weather.openvoiceos:weather",
    "ovos-skill-weather.openvoiceos:current_temperature": "ovos-skill-weather.openvoiceos:temperature",
    "ovos-skill-weather.openvoiceos:hourly_temperature": "ovos-skill-weather.openvoiceos:temperature",
    "ovos-skill-weather.openvoiceos:high_temperature": "ovos-skill-weather.openvoiceos:temperature",
    "ovos-skill-weather.openvoiceos:low_temperature": "ovos-skill-weather.openvoiceos:temperature",
    "ovos-skill-weather.openvoiceos:is_hot_cold": "ovos-skill-weather.openvoiceos:is_hot_or_cold",

    # ovos-skill-volume#137: the discrete level intents became one
    # {level}-slot intent. volume.max and volume.default are deliberately
    # absent: each also fed a phrasing to a new sibling intent
    # (volume.max.boost, volume.reset), so their rows need an utterance-level
    # rule rather than one label.
    "ovos-skill-volume.openvoiceos:volume.low": "ovos-skill-volume.openvoiceos:volume_level",
    "ovos-skill-volume.openvoiceos:volume.high": "ovos-skill-volume.openvoiceos:volume_level",

    # ovos-skill-alerts#197 and #204: the alias intents fold into their base
    # intent. CreateOcpAlarmAlt carried the wake-with-media phrasings, which
    # #199 split out again as CreateAlarmAlt.
    "ovos-skill-alerts.openvoiceos:CreateReminderAlt": "ovos-skill-alerts.openvoiceos:CreateReminder",
    "ovos-skill-alerts.openvoiceos:RescheduleAlertAlt": "ovos-skill-alerts.openvoiceos:RescheduleAlert",
    "ovos-skill-alerts.openvoiceos:CreateOcpAlarmAlt": "ovos-skill-alerts.openvoiceos:CreateAlarmAlt",

    # ovos-skill-confucius-quotes#70 and ovos-skill-fuster-quotes#38: birth,
    # death and "when did X live" answer one lifespan question.
    "ovos-skill-confucius-quotes.openvoiceos:ConfuciusBirth": "ovos-skill-confucius-quotes.openvoiceos:confucius_lifespan",
    "ovos-skill-confucius-quotes.openvoiceos:ConfuciusDeath": "ovos-skill-confucius-quotes.openvoiceos:confucius_lifespan",
    "ovos-skill-confucius-quotes.openvoiceos:ConfuciusLive": "ovos-skill-confucius-quotes.openvoiceos:confucius_lifespan",
    "ovos-skill-fuster-quotes.openvoiceos:FusterBirth": "ovos-skill-fuster-quotes.openvoiceos:fuster_lifespan",
    "ovos-skill-fuster-quotes.openvoiceos:FusterDeath": "ovos-skill-fuster-quotes.openvoiceos:fuster_lifespan",
    "ovos-skill-fuster-quotes.openvoiceos:FusterLive": "ovos-skill-fuster-quotes.openvoiceos:fuster_lifespan",
}

#: Intent-name merges that predate the unification wave: the corpus attests
#: several spellings of one runtime intent.
INTENT_ALIASES = {
    "what.date.is.it": "current_date",
    "handle_query_date_simple": "current_date",
    "handle_day_for_date": "weekday.for.date",
    "handle_query_relative_date": "time.until",
    "handle_show_time": "what.time.is.it",
    "handle_query_time": "what.time.is.it",
    "handle_weekday": "what.weekday.is.it",
    "howto": "wikihow",
    "HowAreYou": "Greetings",
    "current_wind": "is_wind",
    "do-i-need-an-umbrella": "is_rain",
    "do.i.need.an.umbrella": "is_rain",
    "volume.mute": "volume.mute.toggle",
}

_SUFFIX_RE = re.compile(r"\.(intent|voc)$")

#: Region-less language codes, resolved to the single locale the corpus
#: ships for that language. A bare code is only folded when the corpus
#: attests exactly one region for it AND the source is not mixing dialects
#: under the bare code - see docs/labels.md. `es`, `nl` and `pt` are
#: deliberately absent: the corpus carries es-ES and es-419, nl-NL and nl-BE,
#: pt-PT and pt-BR, and the tracker's bare `pt` rows are Brazilian-leaning.
#: `an` (Aragonese) folds to `an-ES` because the corpus, like the other
#: single-dialect entries here, has exactly one Aragonese locale directory
#: to attest -- there is no `an-AR`/`an-FR` split to disambiguate.
LANG_REGIONS = {"ca": "ca-ES", "da": "da-DK", "de": "de-DE", "en": "en-US",
                "eu": "eu-ES", "fr": "fr-FR", "gl": "gl-ES", "it": "it-IT",
                "an": "an-ES"}

#: Language codes collapsed to one bucket regardless of region tag. Kabyle
#: ships as both bare `kab` and `kab-DZ` across the corpus (the skills
#: register under `kab`; `kab-DZ` is a duplicated code some `.intent`
#: trees still carry) -- they denote the same language and must never be
#: trained as two disjoint labels.
LANG_MERGE = {"kab": "kab", "kab-dz": "kab"}


def norm_lang(lang: str) -> str:
    """Return a full BCP-47 code; bare language subtags gain their corpus region."""
    lang = str(lang).strip().strip('"').strip()
    if lang.lower() in LANG_MERGE:
        return LANG_MERGE[lang.lower()]
    if lang in LANG_REGIONS:
        return LANG_REGIONS[lang]
    if "-" in lang:
        base, _, region = lang.partition("-")
        if base.lower() in LANG_MERGE:
            return LANG_MERGE[base.lower()]
        return f"{base.lower()}-{region.upper()}"
    return lang.lower()
_SLOT_RE = re.compile(r"\{[^}]*\}")


def norm_intent(name: str) -> str:
    """Strip the resource-file suffix, keeping the intent name's case."""
    name = str(name).strip().strip('"').strip("'").strip()
    # some corpora key on a repo-relative path rather than a bare file name
    name = name.rsplit("/", 1)[-1]
    name = _SUFFIX_RE.sub("", name)
    name = name.replace(".intent.intent", ".intent")
    return INTENT_ALIASES.get(name, name)


def norm_skill(skill_id: str) -> str:
    """Canonicalise a skill id to the form the skill registers on the bus."""
    s = str(skill_id).strip().strip('"').strip("'").strip().lower()
    s = s.replace(".openvoiceos.openvoiceos", ".openvoiceos")
    if s in BARE_SKILL_IDS or "." in s:
        return s
    if s.startswith("ovos-skill-") or s.startswith("skill-"):
        return s + ".openvoiceos"
    return s


def make_label(skill_id: str, intent: str) -> str:
    label = f"{norm_skill(skill_id)}:{norm_intent(intent)}"
    return LABEL_ALIASES.get(label, label)


def family_of(label: str) -> str:
    head = label.split(":", 1)[0]
    return head if head in PIPELINE_IDS else "skill"


def norm_utterance(text: str) -> str:
    text = str(text).strip().strip('"').strip("'").strip("`")
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------ expansion ----

def expand_template(line: str, cap: int = TEMPLATE_EXPANSION_CAP):
    """Expand padatious template syntax into concrete sentences.

    Handles ``(a|b)`` alternation and ``[optional]`` groups, and replaces
    ``{slot}`` placeholders with the literal slot name so the sentence keeps
    its shape without inventing entity values. Deterministic and capped.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return []
    out = [""]

    def _flush(variants):
        merged = []
        for prefix in out:
            for v in variants:
                merged.append(prefix + v)
                if len(merged) >= cap:
                    return merged
        return merged

    i = 0
    buf = ""
    while i < len(line):
        ch = line[i]
        if ch in "([":
            close = ")" if ch == "(" else "]"
            depth = 1
            j = i + 1
            while j < len(line) and depth:
                if line[j] == ch:
                    depth += 1
                elif line[j] == close:
                    depth -= 1
                j += 1
            inner = line[i + 1:j - 1]
            out = [p + buf for p in out]
            buf = ""
            variants = inner.split("|")
            if ch == "[":
                variants = variants + [""]
            sub = []
            for v in variants:
                sub.extend(expand_template(v, cap) or [""])
            out = _flush(sub)
            i = j
            continue
        buf += ch
        i += 1
    out = [p + buf for p in out]
    return [re.sub(r"\s+", " ", s).strip() for s in out[:cap]]


# -------------------------------------------------------------- registry ----
# The pinned skill and pipeline refs are the ground truth for what labels can
# exist. A class the runtime cannot produce must never be trained.

_ENTRY_KEY_RE = re.compile(r'^\s*"?([A-Za-z0-9_.\-]+\.[A-Za-z0-9_\-]+)"?\s*=', re.M)
_URL_RE = re.compile(r'https?://github\.com/([A-Za-z0-9_.\-]+)/([A-Za-z0-9_.\-]+)')
_INTENT_BUILDER_RE = re.compile(r"""IntentBuilder\(\s*['"]([^'"]+)['"]""")
_INTENT_FILE_RE = re.compile(r"""@intent_handler\(\s*['"]([^'"]+\.intent)['"]""")

def fold(name: str) -> str:
    """Folded intent name: separators and case carry no identity."""
    return re.sub(r"[.\-_\s]", "", str(name)).lower()


def fold_skill(skill_id: str) -> str:
    """Folded skill id, insensitive to the order of the name's own words.

    Repos disagree on whether the package is `ovos-skill-wallpapers` or
    `skill-ovos-wallpapers`, and the entry point follows the repo name, so
    the corpora carry both orders for one skill. Matching on the token
    multiset bridges that without a hand-maintained table.
    """
    name, _, author = str(skill_id).lower().rpartition(".")
    if not name:
        name, author = author, ""
    return "|".join(sorted(re.split(r"[-_.\s]+", name))) + "@" + author


def skill_id_from_repo(repo: Path, rev: str) -> str:
    """The skill id the repo's entry point declares at *rev*.

    pyproject declares it literally. setup.py derives it from the GitHub URL
    as ``<repo-name>.<author>``, both lowercased, so it is read back the same
    way rather than executed.
    """
    try:
        toml = git_show(repo, rev, "pyproject.toml")
    except subprocess.CalledProcessError:
        toml = ""
    for group in ("ovos.plugin.skill", "opm.skill"):
        marker = f'entry-points."{group}"'
        if marker not in toml:
            continue
        tail = toml.split(marker, 1)[1]
        m = _ENTRY_KEY_RE.search(tail.split("[", 1)[0] if "[" in tail else tail)
        if m:
            return m.group(1).lower()
    try:
        setup = git_show(repo, rev, "setup.py")
    except subprocess.CalledProcessError:
        setup = ""
    skill_id = _entry_point_from_setup(setup)
    if skill_id:
        return skill_id
    m = _URL_RE.search(setup or toml)
    if m and "{" not in m.group(2):
        return f"{m.group(2).lower()}.{m.group(1).lower()}"
    raise SystemExit(f"[registry] cannot determine skill_id for {repo} at {rev}")


def _entry_point_from_setup(setup: str):
    """The left-hand side of ``PLUGIN_ENTRY_POINT``, resolved statically.

    Skills build the entry point from module-level string constants, often
    derived from the GitHub URL. Those few forms are read back rather than
    executed - importing a skill's setup.py to learn its id is not something
    a dataset builder should do.
    """
    if "PLUGIN_ENTRY_POINT" not in setup:
        return None
    env = dict(re.findall(r"^([A-Z_]+)\s*=\s*f?['\"]([^'\"]+)['\"]", setup, re.M))
    url = _URL_RE.search(env.get("URL", ""))
    if url:
        author, repo = url.group(1), url.group(2)
        # `AUTHOR, NAME = URL.split(".com/")[-1].split("/")`
        m = re.search(r"^([A-Z_]+),\s*([A-Z_]+)\s*=\s*URL\.split", setup, re.M)
        if m:
            env[m.group(1)], env[m.group(2)] = author, repo
        # `NAME = URL.split("/")[-1]`
        for name in re.findall(r'^([A-Z_]+)\s*=\s*URL\.split\(["\']/["\']\)\[-1\]',
                                setup, re.M):
            env[name] = repo
    m = re.search(r"PLUGIN_ENTRY_POINT\s*=\s*\(?\s*f?['\"]([^'\"]*?)=", setup, re.S)
    if not m:
        return None

    placeholder = re.compile(r"\{([A-Z_]+)(?:\.lower\(\))?\}")

    def sub(match, strict=True):
        value = env.get(match.group(1))
        if value is None:
            if not strict:
                return match.group(0)
            raise SystemExit(
                f"[registry] unresolved {match.group(1)!r} in PLUGIN_ENTRY_POINT")
        return value.lower() if ".lower()" in match.group(0) else value

    # constants may be defined in terms of each other; settle them first.
    # A constant that stays unresolved here only matters if the entry point's
    # left-hand side actually references it.
    for _ in range(4):
        if not any(placeholder.search(v) for v in env.values()):
            break
        env = {k: placeholder.sub(lambda mm: sub(mm, False), v)
               for k, v in env.items()}
    skill_id = placeholder.sub(sub, m.group(1)).lower()
    if "{" in skill_id:
        raise SystemExit(f"[registry] could not resolve PLUGIN_ENTRY_POINT "
                         f"to a literal skill id, got {skill_id!r}")
    return skill_id


def registered_intents(repo: Path, rev: str) -> set:
    """Intent names the repo registers at *rev*.

    Adapt intents are the names passed to ``IntentBuilder``; Padatious
    intents are declared by ``@intent_handler("name.intent")``. Locale files
    contribute their stem too, since not every skill decorates every intent -
    but a stem that differs from a declared name only in case or separators
    loses to the declaration. A locale file whose stem does not match its
    handler is a typo in the skill (ovos-skill-count ships an `it-IT`
    `count_to_N.intent` against a `count_to_n` handler) and the runtime never
    registers it. Test fixtures are excluded: a skill's own test skills are
    not registered by the real skill.
    """
    declared, stems = set(), set()
    for path in git_ls_all(repo, rev):
        if path.split("/")[0] in {"test", "tests"}:
            continue
        if path.endswith(".intent"):
            stems.add(Path(path).stem)
        elif path.endswith(".py"):
            try:
                src = git_show(repo, rev, path)
            except subprocess.CalledProcessError:
                continue
            declared.update(_INTENT_BUILDER_RE.findall(src))
            declared.update(n[:-len(".intent")]
                            for n in _INTENT_FILE_RE.findall(src))
    declared_folds = {fold(n) for n in declared}
    return declared | {s for s in stems if fold(s) not in declared_folds}


def build_registry(cfg, ws):
    """Canonical labels, plus the lookup tables used to resolve corpus labels.

    Returns ``(labels, by_skill_fold, intents_by_skill)`` where ``labels`` is
    the set of `skill_id:intent` the pinned refs actually register.
    """
    labels = set()
    intents_by_skill = {}
    for repo_name, rev in sorted(cfg["skill_refs"]["refs"].items()):
        repo = ws / repo_name
        assert_rev(repo, rev, f"skill:{repo_name}")
        skill_id = skill_id_from_repo(repo, rev)
        names = registered_intents(repo, rev)
        intents_by_skill.setdefault(skill_id, set()).update(names)
    for src in cfg["git_sources"]:
        if src["kind"] != "plugin_intents":
            continue
        repo = ws / src["path"]
        names = {Path(n).stem for n in git_ls(repo, src["revision"], src["files"])
                 if n.split("/")[0] not in {"test", "tests"}}
        names.update(src.get("extra_intents") or [])
        intents_by_skill.setdefault(src["pipeline_id"], set()).update(names)
    for skill_id, names in intents_by_skill.items():
        labels.update(f"{skill_id}:{n}" for n in names)

    by_skill_fold = {}
    for skill_id in intents_by_skill:
        by_skill_fold.setdefault(fold_skill(skill_id), []).append(skill_id)
    return labels, by_skill_fold, intents_by_skill


def _entity_values(repo: Path, rev: str, path: str) -> List[str]:
    """Values an ``.entity`` file at *path* attests, alternation-expanded.

    Uses the same ``iter_expand`` the runtime calls in
    ``_handle_intent4_register_entity`` on a live entity registration, so a
    static ``.entity`` file yields the same value set a running skill would
    register at match time.
    """
    values = []
    try:
        text = git_show(repo, rev, path)
    except subprocess.CalledProcessError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            for v in iter_expand(line):
                v = v.strip()
                if v:
                    values.append(v)
        except Exception:
            continue
    return values


def collect_entities(cfg, ws) -> Dict[str, List[str]]:
    """Entity value-sets attested by the pinned refs, keyed by entity name.

    Mirrors ``Model2VecIntentPipeline.entities``: one flat, skill-agnostic
    dict, exactly what the runtime accumulates from every skill's own
    ``ENTITY_REGISTER`` calls. Feeds :func:`ovos_m2v_pipeline.slots.expand_entities`
    so the corpus builder fills ``{slot}`` templates the same way the
    runtime prototype pipeline does.
    """
    entities: Dict[str, List[str]] = {}

    def add(name: str, values: List[str]):
        if not values:
            return
        bucket = entities.setdefault(name.lower(), [])
        seen = set(bucket)
        for v in values:
            if v not in seen and len(bucket) < MAX_ENTITY_EXPANSIONS:
                bucket.append(v)
                seen.add(v)

    for repo_name, rev in sorted(cfg["skill_refs"]["refs"].items()):
        repo = ws / repo_name
        for path in git_ls_all(repo, rev):
            if path.split("/")[0] in {"test", "tests"}:
                continue
            if path.endswith(".entity"):
                add(Path(path).stem, _entity_values(repo, rev, path))
    for src in cfg["git_sources"]:
        if src["kind"] != "plugin_intents":
            continue
        repo = ws / src["path"]
        entity_glob = src["files"].replace(".intent", ".entity")
        for path in git_ls(repo, src["revision"], entity_glob):
            if path.split("/")[0] in {"test", "tests"}:
                continue
            add(Path(path).stem, _entity_values(repo, src["revision"], path))
    return entities


def reduce_skill_id(skill_id: str, by_skill_fold: dict, aliases: dict = {}):
    """Resolve a corpus skill id to a registered one, or None.

    Handles the spellings the corpora actually carry: a doubled author suffix
    (`<name>.krisgesling.openvoiceos`), a numeric disambiguator a source
    appended to the repo name (`ovos-skill-days-in-history_1.openvoiceos`),
    and pure case or separator differences.
    """
    skill_id = aliases.get(skill_id, skill_id)
    seen = set()
    queue = [skill_id]
    while queue:
        cand = queue.pop(0)
        if cand in seen:
            continue
        seen.add(cand)
        hits = by_skill_fold.get(fold_skill(cand))
        if hits and len(hits) == 1:
            return hits[0]
        head, _, tail = cand.rpartition(".")
        if head and tail:
            queue.append(head)
        stripped = re.sub(r"_\d+(?=\.|$)", "", cand)
        if stripped != cand:
            queue.append(stripped)
    return None


def resolve_label(label: str, registry, by_skill_fold, intents_by_skill, aliases):
    """``(canonical_label, how)``; ``how`` is None when the label is exact."""
    if label in registry:
        return label, None
    skill_id, _, intent = label.partition(":")
    resolved_skill = reduce_skill_id(skill_id, by_skill_fold, aliases)
    if resolved_skill is None:
        return None, "unknown-skill"
    names = intents_by_skill[resolved_skill]
    if intent in names:
        return f"{resolved_skill}:{intent}", "skill-id"
    matches = [n for n in names if fold(n) == fold(intent)]
    if len(matches) == 1:
        how = "intent-spelling" if resolved_skill == skill_id else "skill-id+intent-spelling"
        return f"{resolved_skill}:{matches[0]}", how
    if len(matches) > 1:
        return None, "ambiguous-intent-fold"
    return None, "unknown-intent"


# --------------------------------------------------------------- readers ----

def git_show(repo: Path, rev: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{rev}:{path}"],
        check=True, capture_output=True, text=True).stdout


def git_ls_all(repo: Path, rev: str):
    return subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", rev],
        check=True, capture_output=True, text=True).stdout.splitlines()


def git_ls(repo: Path, rev: str, pattern: str):
    for name in git_ls_all(repo, rev):
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, "**/" + pattern):
            yield name


def assert_rev(repo: Path, rev: str, src_id: str):
    if not repo.is_dir():
        raise SystemExit(f"[{src_id}] missing clone: {repo}")
    r = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", rev + "^{commit}"],
                       capture_output=True)
    if r.returncode:
        raise SystemExit(
            f"[{src_id}] pinned revision {rev} is not present in {repo}. "
            f"Fetch the repo, or the pin in sources.yaml is wrong.")


def lang_from_name(name: str, default: str) -> str:
    m = re.search(r"_([a-z]{2}-[A-Z]{2})\.jsonl$", name)
    return m.group(1) if m else default


def read_localize(src, ws, rows, stats):
    repo = ws / src["path"]
    assert_rev(repo, src["revision"], src["id"])
    drop_types = set((src.get("exclude") or {}).get("file_type") or [])
    for name in sorted(git_ls(repo, src["revision"], src["files"])):
        for line in git_show(repo, src["revision"], name).splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("file_type") in drop_types:
                stats["dropped_voc"] += 1
                continue
            utt = norm_utterance(d.get("text", ""))
            rows.append((norm_lang(d.get("lang") or "en-US"), make_label(d["skill"], d["intent"]),
                         utt, src["id"], utt))


def read_tracker(src, ws, rows, stats):
    repo = ws / src["path"]
    assert_rev(repo, src["revision"], src["id"])
    for lang in src["langs"]:
        name = src["files"].format(lang=lang)
        text = git_show(repo, src["revision"], name)
        for r in csv.DictReader(io.StringIO(text)):
            utt = norm_utterance(r["utterance"])
            rows.append((norm_lang(lang), make_label(r["domain"], r["intent"]),
                         utt, src["id"], utt))


def read_plugin_intents(src, ws, rows, stats):
    repo = ws / src["path"]
    assert_rev(repo, src["revision"], src["id"])
    pid = src["pipeline_id"]
    for name in sorted(git_ls(repo, src["revision"], src["files"])):
        if name.split("/")[0] in {"test", "tests"}:
            # a plugin's own test skill is not a registration
            continue
        stem = Path(name).stem
        lang = Path(name).parent.name
        # ISO 639-2/3 language subtag (2-3 letters), optionally followed by
        # a BCP-47 script (4 letters) and/or region (2 letters or 3 digits)
        # subtag, e.g. `kab`, `zsm`, `cmn`, `sr-Latn`, `es-419`.
        if not re.fullmatch(r"[a-z]{2,3}(-[a-z]{4})?(-([a-z]{2}|\d{3}))?",
                            lang, re.I):
            raise SystemExit(
                f"[{src['id']}] cannot read a locale from {name!r}; the "
                f"`files` glob must select paths under a locale directory")
        label = make_label(pid, stem)
        for line in git_show(repo, src["revision"], name).splitlines():
            template_key = f"{name}:{line.strip()}"
            for sent in expand_template(line):
                rows.append((norm_lang(lang), label, norm_utterance(sent), src["id"],
                             template_key))


#: A locale directory under a skill's `locale/` root: an ISO 639-2/3 subtag,
#: optionally with a BCP-47 script and/or region subtag.
_LOCALE_DIR_RE = re.compile(r"[a-z]{2,3}(-[a-z]{4})?(-([a-z]{2}|\d{3}))?$", re.I)

_SKILL_LOCALE_RE = re.compile(r"(?:^|.*/)locale/([^/]+)/(?:.*/)?[^/]+\.intent$")


def read_skill_intents(cfg, ws, rows, stats):
    """Templates from the `.intent` files the pinned skills themselves ship.

    The external corpora are translations of a snapshot; a skill's own
    `locale/<lang>/` tree is what the runtime loads, so it is the only source
    that attests every locale a skill actually supports. Both layouts skills
    use are read -- a top-level `locale/` and a `<package>/locale/` -- at any
    depth beneath the locale directory, and the locale name is normalised
    through :func:`norm_lang`, so a tree carrying `kab-DZ` merges with one
    carrying `kab` instead of training as a second language.

    Templates expand exactly as the pipeline plugins' do; `{slot}` fills come
    later, from the entity values the same pinned refs attest.
    """
    for repo_name, rev in sorted(cfg["skill_refs"]["refs"].items()):
        repo = ws / repo_name
        assert_rev(repo, rev, f"skill:{repo_name}")
        skill_id = skill_id_from_repo(repo, rev)
        source = f"skill-intents:{repo_name.rsplit('/', 1)[-1]}"
        for name in sorted(git_ls_all(repo, rev)):
            if name.split("/")[0] in {"test", "tests"}:
                continue
            m = _SKILL_LOCALE_RE.match(name)
            if not m:
                continue
            if not _LOCALE_DIR_RE.fullmatch(m.group(1)):
                stats["skill_intents_bad_locale"] += 1
                print(f"WARNING [skill-intents] {repo_name}: {name!r} is not "
                      f"under a locale directory; skipped", file=sys.stderr)
                continue
            lang = norm_lang(m.group(1))
            label = make_label(skill_id, Path(name).stem)
            for line in git_show(repo, rev, name).splitlines():
                sents = expand_template(line)
                if len(sents) >= TEMPLATE_EXPANSION_CAP:
                    stats["skill_intents_capped_templates"] += 1
                template_key = f"{repo_name}/{name}:{line.strip()}"
                for sent in sents:
                    rows.append((lang, label, norm_utterance(sent), source,
                                 template_key))


def read_hf(src, rows, stats):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(src["repo_id"], src["file"], repo_type="dataset",
                           revision=src["revision"])
    df = pd.read_csv(path)
    cols = src["columns"]
    const_label = src.get("constant_label")
    const_skill = src.get("constant_skill_id")
    const_lang = src.get("constant_lang")
    unmappable = []
    for r in df.to_dict("records"):
        utt = norm_utterance(r.get(cols["utterance"], ""))
        lang = const_lang or str(r.get(cols.get("lang", ""), "")).strip().strip('"').strip()
        if const_label:
            label = const_label
        else:
            skill = const_skill or r.get(cols.get("skill_id", ""), "")
            intent = r.get(cols.get("intent", ""), "")
            if not str(skill).strip() or not str(intent).strip():
                unmappable.append(f"{skill!r}:{intent!r}")
                continue
            label = make_label(skill, intent)
        if not utt:
            unmappable.append(f"<empty utterance> {label}")
            continue
        rows.append((norm_lang(lang or "en-US"), label, utt, src["id"], utt))
    stats["unmappable"][src["id"]] = unmappable


def read_golden(cfg, ws, rows, stats):
    g = cfg["golden"]
    shared = ws / g["shared"]["path"]
    if not shared.is_file():
        raise SystemExit(f"[golden] missing shared corpus: {shared}")
    seen = set()

    def consume(text, source, default_lang, name=""):
        n = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("needs_manual"):
                stats["golden_needs_manual"] += 1
                continue
            if not d.get("intent_label"):
                # fallback-skill corpora assert a dialog, not an intent
                stats["golden_no_intent"] += 1
                continue
            label = make_label(d["skill_id"], d["intent_label"])
            utt = norm_utterance(d["utterance"])
            lang = norm_lang(d.get("lang") or lang_from_name(name, default_lang))
            key = (label, utt.lower(), lang)
            if key in seen:
                continue
            seen.add(key)
            rows.append((lang, label, utt, source, utt))
            n += 1
        return n

    stats["golden_shared"] = consume(shared.read_text(encoding="utf-8"),
                                     "golden:ovoscope-shared",
                                     g["shared"]["default_lang"])
    pat = g["per_skill"]["files"]
    for repo_name, rev in sorted(cfg["skill_refs"]["refs"].items()):
        repo = ws / repo_name
        assert_rev(repo, rev, f"skill:{repo_name}")
        for name in sorted(git_ls(repo, rev, pat)):
            n = consume(git_show(repo, rev, name), f"golden:{repo_name.rsplit(chr(47), 1)[-1]}",
                        g["per_skill"]["default_lang"], name)
            stats["golden_per_skill"] += n
    return {r[3] for r in rows if r[3].startswith("golden:")}


# ----------------------------------------------------------------- main ----

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: Files a dataset build writes to `--out`; the exact set `push_dataset`
#: uploads to the Hub, additively, one commit per file.
DATASET_FILES = ("train.parquet", "train.jsonl", "test.parquet", "test.jsonl",
                 "labels.json", "manifest.json")


def push_dataset(out: Path, repo_id: str, dry_run: bool = False) -> None:
    """Upload the corpus at *out* to a Hugging Face dataset repo.

    Additive: `upload_file` creates or updates exactly the named path in one
    commit each and never touches or deletes anything else in the repo. The
    token comes from the `HF_TOKEN` environment variable, never hardcoded.
    """
    if dry_run:
        print(f"[push] would upload to dataset repo {repo_id!r} (dry run):")
        for f in DATASET_FILES:
            p = out / f
            size = f"{p.stat().st_size} bytes" if p.is_file() else "not built yet"
            print(f"  {f} ({size})")
        return

    missing = [f for f in DATASET_FILES if not (out / f).is_file()]
    if missing:
        raise SystemExit(f"[push] {out} is missing {missing}; run the build first")

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    for f in DATASET_FILES:
        api.upload_file(path_or_fileobj=str(out / f), path_in_repo=f,
                        repo_id=repo_id, repo_type="dataset",
                        commit_message=f"update {f}")
        print(f"[push] uploaded {f} -> {repo_id}")


def template_split(df: pd.DataFrame, label_col: str, test_size: float, seed: int):
    """Stratified train/test split at the TEMPLATE level, not the row level.

    Every row expanded from the same source template (an `.intent` line's
    `(alt|alt)`/`[optional]`/`{slot}` expansions, or one already-atomic
    utterance from the other readers) carries the same ``(lang, template)``
    key and is assigned to train or test as one unit, so no template's
    expansions straddle the split -- the failure mode that inflates held-out
    accuracy by letting a near-identical sentence leak across sides.
    """
    from sklearn.model_selection import train_test_split
    group_cols = ["lang", "template"]
    groups = df[group_cols + [label_col]].drop_duplicates(subset=group_cols)
    counts = groups[label_col].value_counts()
    # a label attested by only one template group cannot be stratified across
    # both sides; keep it whole in train rather than fail the split.
    unsplittable = set(counts[counts < 2].index)
    forced_train = groups[groups[label_col].isin(unsplittable)]
    splittable = groups[~groups[label_col].isin(unsplittable)]
    train_g, test_g = train_test_split(splittable, test_size=test_size,
                                       random_state=seed,
                                       stratify=splittable[label_col])
    train_keys = set(map(tuple, pd.concat([train_g, forced_train])[group_cols].values))
    is_train = df[group_cols].apply(tuple, axis=1).isin(train_keys)
    return df[is_train], df[~is_train]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(HERE / "sources.yaml"))
    ap.add_argument("--workspace", default=None,
                    help="root the git clones live under (overrides sources.yaml)")
    ap.add_argument("--out", default=str(HERE / "dataset"))
    ap.add_argument("--skill-ref-list", default=None,
                    help="file of '<repo-name> <sha>' lines replacing skill_refs.refs; "
                         "use after the Adapt->.intent PRs merge")
    ap.add_argument("--allow-ambiguous", action="store_true",
                    help="keep rows whose (utterance, lang) carries several "
                         "labels instead of dropping them")
    ap.add_argument("--dry-run", action="store_true",
                    help="report counts and write nothing")
    ap.add_argument("--push-to", default=None,
                    help="Hugging Face dataset repo id to upload --out to "
                         "after the build (e.g. OpenVoiceOS/ovos-m2v-intents); "
                         "additive, combine with --dry-run to preview")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8"))
    ws = Path(os.path.expanduser(args.workspace or cfg["workspace"]))

    if args.skill_ref_list:
        refs = {}
        for line in Path(args.skill_ref_list).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, sha = line.split()
            refs[name] = sha
        cfg["skill_refs"]["refs"] = refs

    rows = []
    stats = collections.Counter()
    stats["unmappable"] = {}
    per_source_raw = collections.Counter()

    readers = {"localize_jsonl": read_localize, "tracker_csv": read_tracker,
               "plugin_intents": read_plugin_intents}
    for src in cfg["git_sources"]:
        before = len(rows)
        readers[src["kind"]](src, ws, rows, stats)
        per_source_raw[src["id"]] = len(rows) - before
    for src in cfg["hf_sources"]:
        before = len(rows)
        read_hf(src, rows, stats)
        per_source_raw[src["id"]] = len(rows) - before
    before = len(rows)
    read_skill_intents(cfg, ws, rows, stats)
    per_source_raw["skill-intents"] = len(rows) - before
    before = len(rows)
    read_golden(cfg, ws, rows, stats)
    per_source_raw["golden"] = len(rows) - before

    df = pd.DataFrame(rows, columns=["lang", "label", "utterance", "source", "template"],
                      dtype=object)
    n_raw = len(df)

    # ---- blacklists
    df["skill_id"] = df["label"].str.split(":").str[0]
    df["intent"] = df["label"].str.split(":", n=1).str[1]
    n_black = int(df["skill_id"].isin(cfg["skill_blacklist"]).sum()
                  + df["intent"].isin(cfg["intent_blacklist"]).sum())
    df = df[~df["skill_id"].isin(cfg["skill_blacklist"])]
    df = df[~df["intent"].isin(cfg["intent_blacklist"])]

    # ---- fill {slot} placeholders the same way the runtime prototype
    # pipeline does (ovos_m2v_pipeline.slots.expand_entities), from entity
    # values attested by the pinned refs' `.entity` files. A row whose slot
    # has no registered entity is left with the placeholder literal by
    # `expand_entities` (entities are an optional hint, same as at runtime);
    # such rows carry no natural-language signal and are dropped below,
    # same treatment as `drop_bare_slot`. `expand_entities` bounds a single
    # call at MAX_ENTITY_EXPANSIONS (2000) -- fine for one live registration,
    # but every one of the corpus's already-expanded template rows calls it
    # independently, so an entity with hundreds of values (e.g. `color`,
    # `date`) would multiply the corpus by that count per row, per locale.
    # `TEMPLATE_FILL_CAP` takes an evenly-strided sample of that result
    # instead, keeping a bounded, deterministic number of fills per template.
    entities = collect_entities(cfg, ws)
    n_slot_templates = int(df["utterance"].str.contains("{", regex=False).sum())

    def _fill(u):
        if "{" not in u:
            return [u]
        filled = expand_entities([u], entities)
        if len(filled) <= TEMPLATE_FILL_CAP:
            return filled
        step = (len(filled) - 1) / (TEMPLATE_FILL_CAP - 1)
        return [filled[round(i * step)] for i in range(TEMPLATE_FILL_CAP)]

    df = df.assign(utterance=df["utterance"].map(_fill))
    df = df.explode("utterance", ignore_index=False)
    still_literal = df["utterance"].str.contains("{", regex=False, na=False)
    n_unfilled_slot = int(still_literal.sum())
    df = df[~still_literal]

    # ---- content filters
    f = cfg["filters"]
    u = df["utterance"]
    keep = u.str.len().between(f["min_chars"], f["max_chars"])
    if f["drop_bare_slot"]:
        keep &= ~u.str.fullmatch(r"\{[^}]*\}", na=False)
    if f["drop_non_alpha"]:
        # `Series.str.contains` on a compiled pattern goes through pandas'
        # accessor regex engine, which silently stops matching non-Latin
        # scripts (Cyrillic, Greek, CJK, ...) once the column is promoted
        # to pandas' `str` dtype (e.g. by an upstream `.explode()`) instead
        # of staying `object` -- confirmed to zero out every Cyrillic row
        # under pandas 3.x. `str.isalpha()` on each plain Python string,
        # applied element-wise, is Unicode-aware by construction and never
        # touches pandas' own regex engine, so the column's dtype cannot
        # change the result.
        keep &= u.map(lambda s: any(ch.isalpha() for ch in str(s)))
    n_filtered = int((~keep).sum())
    df = df[keep]

    # ---- resolve every label against what the pinned refs register
    registry, by_skill_fold, intents_by_skill = build_registry(cfg, ws)
    aliases = cfg.get("skill_id_aliases") or {}
    counts = df["label"].value_counts()
    resolution, merges, unresolved = {}, [], {}
    for label, n in counts.items():
        canonical, how = resolve_label(label, registry, by_skill_fold,
                                       intents_by_skill, aliases)
        if canonical is None:
            unresolved[label] = {"rows": int(n), "reason": how}
            continue
        resolution[label] = canonical
        if how:
            merges.append({"corpus_label": label, "rows": int(n),
                           "canonical": canonical, "resolved_by": how})
    n_unresolved = int(sum(v["rows"] for v in unresolved.values()))
    df = df[df["label"].isin(resolution)]
    df["label"] = df["label"].map(resolution).astype(object)

    # ---- an unknown-registry drop is a coverage gap, not a hygiene detail:
    # warn loudly per skill, with the row count, instead of only recording it
    # in `unresolved_labels` where a dry run has to be read to notice it.
    unknown_skill_rows = collections.Counter()
    for label, info in unresolved.items():
        if info["reason"] == "unknown-skill":
            unknown_skill_rows[label.split(":", 1)[0]] += info["rows"]
    for skill_id, n in sorted(unknown_skill_rows.items(), key=lambda kv: -kv[1]):
        print(f"WARNING [registry] {skill_id!r} is not in skill_refs/"
              f"skill_id_aliases; {n} corpus row(s) dropped unresolved",
              file=sys.stderr)

    # ---- exact dedup on (utterance, label, lang)
    n_before = len(df)
    df = df.sort_values(["source", "label", "utterance"], kind="mergesort")
    df["_k"] = (df["label"].astype(str) + "\x1f" + df["utterance"].str.lower().astype(str)
               + "\x1f" + df["lang"].astype(str))
    df = df[~df.duplicated("_k", keep="first")].drop(columns=["_k"])
    n_dedup = n_before - len(df)

    df["family"] = df["label"].map(family_of)
    df["skill_id"] = df["label"].str.split(":").str[0]
    df = df.sort_values(["label", "lang", "utterance"], kind="mergesort").reset_index(drop=True)

    # ---- one (utterance, lang) must denote exactly one label
    # A sentence that carries two labels is not a hard example, it is a
    # contradiction: whichever way the model resolves it, the corpus says it
    # is wrong. Most of these are genuine template overlaps between adjacent
    # intents in one skill's own locale files (`ocp:pause` and
    # `ocp:media_stop` share phrasings), so they cannot be aliased away - one
    # is not a misspelling of the other. They are dropped, and the label
    # pairs are reported so the overlap can be fixed upstream in the skill.
    key = df["utterance"].str.lower().astype(str) + "\x1f" + df["lang"].astype(str)
    nlabels = df.groupby(key)["label"].transform("nunique")
    ambiguous_mask = nlabels > 1
    n_ambiguous_groups = int(key[ambiguous_mask].nunique())
    n_ambiguous_rows = int(ambiguous_mask.sum())
    pair_counts = collections.Counter(
        tuple(sorted(set(g))) for g in
        df.loc[ambiguous_mask].groupby(key[ambiguous_mask])["label"].apply(list))
    ambiguous_pairs = [{"labels": list(k), "groups": v}
                       for k, v in pair_counts.most_common(30)]
    if not args.allow_ambiguous:
        df = df[~ambiguous_mask]
        residual = 0
    else:
        residual = n_ambiguous_groups
    if not args.allow_ambiguous:
        check = df.groupby(df["utterance"].str.lower().astype(str) + "\x1f"
                          + df["lang"].astype(str))["label"].nunique()
        if int((check > 1).sum()):
            raise SystemExit("ambiguity filter left residual groups; this is a bug")

    # ---- labels too rare to stratify
    counts = df["label"].value_counts()
    rare = sorted(counts[counts < f["min_rows_per_label"]].index)
    df = df[~df["label"].isin(rare)]

    # ---- no row may ever reach the export with a literal `{slot}` in it.
    # A regression in the entity-fill step above must fail the build loudly
    # rather than silently ship bracket-syntax training rows again.
    leftover = df[df["utterance"].str.contains("{", regex=False, na=False)]
    if len(leftover):
        raise SystemExit(
            f"[slots] {len(leftover)} row(s) still contain a literal '{{' "
            f"after entity-fill; this is a bug, e.g. {leftover['utterance'].iloc[0]!r}")

    report = {
        "rows_raw": n_raw,
        "rows_final": len(df),
        "rows_per_source_raw": dict(per_source_raw),
        "rows_per_source_final": df["source"].value_counts().to_dict(),
        "rows_per_family": df["family"].value_counts().to_dict(),
        "rows_per_lang": df["lang"].value_counts().to_dict(),
        "labels": int(df["label"].nunique()),
        "labels_per_family": df.groupby("family")["label"].nunique().to_dict(),
        "langs": int(df["lang"].nunique()),
        "dropped_blacklist": n_black,
        "slot_templates_filled": n_slot_templates,
        "dropped_unfilled_slot": n_unfilled_slot,
        "dropped_filters": n_filtered,
        "dropped_localize_voc": int(stats["dropped_voc"]),
        "skill_intent_templates_capped": int(stats["skill_intents_capped_templates"]),
        "dropped_exact_duplicates": int(n_dedup),
        "dropped_unresolved_labels": n_unresolved,
        "dropped_rare_labels": {"labels": len(rare), "examples": rare[:40]},
        "golden_needs_manual_excluded": int(stats["golden_needs_manual"]),
        "golden_rows_without_intent_label": int(stats["golden_no_intent"]),
        "golden_shared_rows": int(stats["golden_shared"]),
        "golden_per_skill_rows": int(stats["golden_per_skill"]),
        "canonical_labels_registered": len(registry),
        "label_resolutions": sorted(merges, key=lambda m: -m["rows"]),
        "unresolved_labels": {"labels": len(unresolved), "rows": n_unresolved,
                              "detail": dict(sorted(unresolved.items(),
                                                    key=lambda kv: -kv[1]["rows"]))},
        "unmappable_rows": {k: {"count": len(v), "examples": sorted(set(v))[:20]}
                            for k, v in stats["unmappable"].items() if v},
        "revisions": {
            **{s["id"]: s["revision"] for s in cfg["git_sources"]},
            **{s["id"]: f'{s["repo_id"]}@{s["revision"]}' for s in cfg["hf_sources"]},
            "skill_refs": cfg["skill_refs"]["refs"],
        },
        "ambiguous_utterance_groups": n_ambiguous_groups,
        "ambiguous_rows_dropped": 0 if args.allow_ambiguous else n_ambiguous_rows,
        "ambiguous_residual_groups": residual,
        "ambiguous_label_pairs": ambiguous_pairs,
        "split": cfg["split"],
    }

    if args.dry_run:
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False, sort_keys=False)
        print()
        if args.push_to:
            push_dataset(Path(args.out), args.push_to, dry_run=True)
        return 0

    sp = cfg["split"]
    train, test = template_split(df, sp["stratify"], test_size=sp["test_size"],
                                 seed=sp["seed"])
    train = train.drop(columns=["template"])
    test = test.drop(columns=["template"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, part in (("train", train.sort_index()), ("test", test.sort_index())):
        for ext, writer in (("parquet", lambda p: part.to_parquet(p, index=False)),
                            ("jsonl", lambda p: part.to_json(p, orient="records",
                                                             lines=True, force_ascii=False))):
            p = out / f"{name}.{ext}"
            writer(p)
            written[p.name] = sha256_file(p)

    labels = sorted(df["label"].unique())
    # m2v#73 manifest shape. The training scheme is identical to the runtime
    # registration string, so no label needs remapping and the manifest
    # carries no flat label_map. `families` rides alongside the allow-list
    # because the plugin's per-family claim filter needs to know which family
    # a label belongs to, and that is a property of the trained head rather
    # than of the plugin. The three special labels stay raw here; the plugin
    # maps them to their bus topics at match time.
    manifest_labels = {"valid_labels": labels,
                       "families": {lbl: family_of(lbl) for lbl in labels}}
    p = out / "labels.json"
    p.write_text(json.dumps(manifest_labels, indent=2, ensure_ascii=False), encoding="utf-8")
    written[p.name] = sha256_file(p)

    report["outputs"] = written
    report["golden_in_split"] = {
        "train": int(train["source"].str.startswith("golden:").sum()),
        "test": int(test["source"].str.startswith("golden:").sum()),
    }
    (out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("rows_final", "labels", "langs", "outputs")},
                     indent=2))
    if args.push_to:
        push_dataset(out, args.push_to, dry_run=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
