#!/usr/bin/env python3
"""Locale parity census for the ovos-skill fleet.

The training set is the locale files. A locale that is missing an intent, a
dialog or a vocabulary that the reference language ships is a locale the
model cannot learn, and nothing in a skill's own suite reports it: every
skill is green against its own resources.

Two layers, and the difference between them matters:

* **Layer A, per locale.** The conformance rules OVOS-INTENT-1 and
  OVOS-INTENT-2 state: resource names, duplicate base names within a role,
  `<name>` resolution, required-slot and slot-type validity. This layer is
  NOT re-implemented here. `ovos_spec_tools.lint.lint_locale` is called and
  its findings are reported as it returns them, so the census cannot drift
  from the linter.

* **Layer B, across locales.** Mirroring each locale against the reference
  language. `knowledge/wiki/concepts/locale-parity.md` records that NO
  clause requires this: no specification names a reference locale or says a
  locale must mirror it. Every Layer B row is therefore labelled SHOULD and
  is policy, not conformance, until Miro rules on the seven questions in
  that page's §5.

Two rules the census obeys deliberately:

* A `{slot}` with no `<slot>.entity` is **never** a violation. OVOS-INTENT-1
  §5.4 says a slot with no value set still fills. It is reported as
  inventory, in its own table, because the training pipeline wants to know.
* The `.blacklist` column follows OVOS-INTENT-2 §4.3, which pairs a
  blacklist with the intent, entity or slot of the same base name. It does
  NOT come from the linter's blacklist differential: that warning fires on a
  blacklist with no `.intent` even when an `.entity` pairs it, which
  locale-parity.md §5 Q6 records as a lint defect (architecture T-2965).

Typed slots need no special case here: `declared_slots` maps `{type:name}`
to `name` for us (OVOS-INTENT-1 §5.6).

    python tools/parity_census.py --workspace ~/AgentWorkspaces
    python tools/parity_census.py --workspace ~/AgentWorkspaces --at-pins

Without `--at-pins` every skill is read at `origin/dev`, which measures the
fleet as it stands, and each skill's sha is printed. With `--at-pins` the
revisions in `train/sources.yaml` are read instead, which measures the tree
the published corpus was built from.

Exit status is 0 when the census is produced. It is 1 when a Layer A finding
of severity `error` exists, so a build job can refuse to publish on one.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REFERENCE = "en-US"
ROLES = (".intent", ".dialog", ".voc", ".entity", ".blacklist", ".required")


# ----------------------------------------------------------------- git ----

def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"git {' '.join(args)} in {repo}: {out.stderr.strip()}")
    return out.stdout


def rev_of(repo: Path, rev: str) -> str:
    return git(repo, "rev-parse", rev).strip()


def files_at(repo: Path, rev: str) -> list[str]:
    return [line for line in git(repo, "ls-tree", "-r", "--name-only", rev).splitlines()
            if line]


def read_at(repo: Path, rev: str, path: str) -> str:
    return git(repo, "show", f"{rev}:{path}")


# ------------------------------------------------------- base language ----

# Miro ruled on locale-parity.md §5 Q1: the reference locale is policy and
# not specification, and each skill has the language it was WRITTEN in.
# ovos-skill-fuster-quotes is Catalan, not English. Parity is measured
# against that base language, so a Catalan skill is not reported as 13
# locales short of an en-US it never had.
#
# No skill declares its base language today: no `skill.json` in the fleet
# carries a language field. So the base is inferred, and every row records
# WHICH signal decided it. A guess that cannot be traced is worse than the
# en-US default it replaces.

DEFAULT_BASE = "en-US"

# Language names as a README writes them, mapped to the subtag a locale
# directory uses. Only languages the fleet actually ships.
README_NAMES = {
    "catalan": "ca", "valencian": "ca", "basque": "eu", "euskara": "eu",
    "galician": "gl", "portuguese": "pt", "spanish": "es", "castilian": "es",
    "german": "de", "french": "fr", "italian": "it", "dutch": "nl",
    "danish": "da", "swedish": "sv", "finnish": "fi", "russian": "ru",
    "polish": "pl", "czech": "cs", "hungarian": "hu", "persian": "fa",
    "kabyle": "kab", "occitan": "oc", "aragonese": "an", "arabic": "ar",
    # english is in the map ON PURPOSE. A README that names English is not
    # evidence of a non-English base: ovos-skill-alerts says it is "tested
    # mainly in German and, to a lesser extent, English", which made German
    # the only hit and moved the whole skill onto a de-DE base. With English
    # in the map that README names two languages, the signal declines to
    # answer, and the default takes it.
    "english": "en",
}


def _subtag(lang: str) -> str:
    return lang.split("-")[0].lower()


def first_locale_commit_langs(repo: Path, rev: str) -> set:
    """The locale languages present in the earliest commit that adds one."""
    log = git(repo, "log", "--reverse", "--format=%H", "--diff-filter=A",
              rev, "--", "locale")
    shas = [line for line in log.splitlines() if line]
    if not shas:
        return set()
    names = git(repo, "show", "--stat=400", "--format=", shas[0])
    out = set()
    for line in names.splitlines():
        match = re.search(r"locale/([^/|\s]+)/", line)
        if match:
            out.add(match.group(1))
    return out


def readme_language(repo: Path, rev: str, langs: dict) -> str | None:
    """A language named in the README that the skill also ships."""
    for name in ("README.md", "readme.md", "README.rst"):
        try:
            text = read_at(repo, rev, name).lower()
        except RuntimeError:
            continue
        hits = {sub for word, sub in README_NAMES.items() if word in text}
        shipped = {sub: lang for lang in langs for sub in [_subtag(lang)]}
        found = sorted(hits & set(shipped))
        if len(found) == 1:
            return shipped[found[0]]
        return None
    return None


def base_language(repo: Path, rev: str, langs: dict) -> tuple:
    """``(base language, the signal that decided it)``.

    Signals in order. The first that gives ONE shipped locale wins:

    * ``history``: the earliest commit that adds a locale file carries a
      single language. A skill imported with several locales at once gives
      no answer here, which is the common case.
    * ``readme``: the README names exactly one language the skill ships.
    * ``default``: ``en-US``, recorded as the default and not as a finding.
    """
    if not langs:
        return DEFAULT_BASE, "default (no locale directory)"

    try:
        first = first_locale_commit_langs(repo, rev)
    except RuntimeError:
        first = set()
    subtags = {_subtag(lang) for lang in first}
    if len(subtags) == 1:
        only = subtags.pop()
        for lang in sorted(langs):
            if _subtag(lang) == only:
                return lang, "history (only language in the first locale commit)"

    named = readme_language(repo, rev, langs)
    if named:
        return named, "readme (the one shipped language the README names)"

    if DEFAULT_BASE in langs:
        return DEFAULT_BASE, "default (no signal; en-US is shipped)"
    return sorted(langs)[0], "default (no signal and no en-US)"


# ------------------------------------------------------------- locales ----

def locale_map(paths: list[str]) -> dict[str, dict[str, set]]:
    """``{lang: {role: {base name}}}`` for every locale file in the listing."""
    out: dict[str, dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for path in paths:
        parts = path.split("/")
        if "locale" not in parts:
            continue
        index = parts.index("locale")
        if index + 2 > len(parts) - 1:
            continue
        lang = parts[index + 1]
        name = parts[-1]
        for role in ROLES:
            if name.endswith(role):
                out[lang][role].add(name[: -len(role)])
                break
    return out


def layer_b(langs: dict[str, dict[str, set]],
            reference_lang: str = REFERENCE) -> list[dict]:
    """Per locale, what the base language has and this one does not.

    SHOULD, not MUST: no clause requires mirroring (locale-parity.md §1).
    The reference is the skill's own base language, not always en-US
    (locale-parity.md §5 Q1, ruled by Miro).
    """
    reference = langs.get(reference_lang) or {}
    rows = []
    for lang in sorted(langs):
        if lang == reference_lang:
            continue
        row = {"lang": lang}
        for role in ROLES:
            missing = sorted((reference.get(role) or set()) - (langs[lang].get(role) or set()))
            extra = sorted((langs[lang].get(role) or set()) - (reference.get(role) or set()))
            row[role] = {"missing": missing, "extra": extra}
        rows.append(row)
    return rows


# ------------------------------------------------- §2.3 parity defects ----

# OVOS-INTENT-2 §2.3, in architecture#272. NOT MERGED at the time of
# writing: read at head 32f0973, which is not the 80a7d5f the task cited,
# so the clause moved once already. These two reports are produced beside
# the existing Layer B rows and change no verdict until the clause lands.
#
# What §2.3 says, and what makes it different from Layer B above:
#
# * A (role, base name) in ANY locale must be in EVERY other. Both
#   directions. So the "extra" column of Layer B, which that layer calls a
#   locale-specific addition and never a failure, becomes a defect of every
#   OTHER locale, reported against the locale that lacks the file.
# * `.blacklist` and `.prompt` are excepted. A blacklist is a property of
#   one language, and a prompt is language-model input.
# * A missing `.voc` is a parity gap ONLY when no `.intent` of that locale
#   references it inline. When one carries `<name>`, that `.intent` is
#   malformed under OVOS-INTENT-1 §3.6 and the linter reports the error.
#   §2.3: "One file is never both."
# * An intent's available slot set, the union over that locale's templates,
#   must be identical in every locale. Reported by intent name.

PARITY_EXEMPT = (".blacklist", ".prompt")
INLINE_REF = re.compile(r"<([^<>]+)>")


def inline_voc_refs(repo: Path, rev: str, paths: list[str], lang: str) -> set:
    """Every `<name>` an .intent of this locale references inline."""
    out = set()
    for path in paths:
        if not path.endswith(".intent"):
            continue
        parts = path.split("/")
        if "locale" not in parts or parts[parts.index("locale") + 1] != lang:
            continue
        try:
            text = read_at(repo, rev, path)
        except (RuntimeError, UnicodeDecodeError):
            continue
        out.update(m.group(1).strip() for m in INLINE_REF.finditer(text))
    return out


def parity_defects(repo: Path, rev: str, paths: list[str],
                   langs: dict) -> list[dict]:
    """§2.3 file-set defects, both directions, per locale that lacks a pair."""
    universe: dict[str, set] = collections.defaultdict(set)
    for lang, roles in langs.items():
        for role in ROLES:
            if role in PARITY_EXEMPT:
                continue
            universe[role] |= (roles.get(role) or set())
    rows = []
    for lang in sorted(langs):
        missing = []
        for role in ROLES:
            if role in PARITY_EXEMPT:
                continue
            gap = sorted(universe[role] - (langs[lang].get(role) or set()))
            if role == ".voc" and gap:
                # §2.3: a .voc an .intent of this locale references inline is
                # an INTENT-1 §3.6 error, reported by the linter, not here.
                referenced = inline_voc_refs(repo, rev, paths, lang)
                gap = [name for name in gap if name not in referenced]
            missing += [{"role": role, "name": name} for name in gap]
        if missing:
            rows.append({"lang": lang, "missing": missing})
    return rows


def slot_set_defects(slots: list[dict]) -> list[dict]:
    """§2.3 slot-set differences for one intent, reported by intent name."""
    by_intent: dict[str, dict[str, set]] = collections.defaultdict(
        lambda: collections.defaultdict(set))
    for row in slots:
        if row.get("slot") is None:
            continue                      # a malformed template, already an error
        base = row["intent"]
        if base.endswith(".intent"):
            base = base[: -len(".intent")]
        by_intent[base][row["lang"]].add(row["slot"])
    rows = []
    for intent in sorted(by_intent):
        per_lang = by_intent[intent]
        union = set().union(*per_lang.values())
        differing = {lang: sorted(union - names)
                     for lang, names in per_lang.items() if names != union}
        if differing:
            rows.append({"intent": intent, "union": sorted(union),
                         "missing_per_lang": differing})
    return rows


# ---------------------------------------------------------------- slots ----

def slot_rows(repo: Path, rev: str, paths: list[str]) -> list[dict]:
    """One row per (lang, intent file, slot). Inventory, never a verdict."""
    from ovos_spec_tools.lint import declared_slots
    from ovos_spec_tools.resources import read_resource_file

    by_lang = locale_map(paths)
    rows = []
    for path in paths:
        if not path.endswith(".intent") or "/locale/" not in f"/{path}":
            continue
        parts = path.split("/")
        lang = parts[parts.index("locale") + 1]
        text = read_at(repo, rev, path)
        with tempfile.NamedTemporaryFile("w", suffix=".intent", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(text)
            tmp = Path(handle.name)
        try:
            slots = declared_slots(read_resource_file(tmp))
        except Exception as exc:                      # a malformed template
            rows.append({"lang": lang, "intent": parts[-1], "slot": None,
                         "error": str(exc)[:120]})
            continue
        finally:
            tmp.unlink(missing_ok=True)
        for slot in sorted(slots):
            rows.append({
                "lang": lang,
                "intent": parts[-1],
                "slot": slot,
                "entity": slot in (by_lang.get(lang, {}).get(".entity") or set()),
                # OVOS-INTENT-2 §4.3: a blacklist pairs by base name with the
                # intent, the entity or the slot. Not the linter's differential.
                "blacklist": slot in (by_lang.get(lang, {}).get(".blacklist") or set()),
            })
    return rows


# ---------------------------------------------------------------- lint ----

def layer_a(repo: Path, rev: str, paths: list[str], scratch: Path) -> list[dict]:
    """Call the linter on a materialised locale tree; never re-implement it."""
    from ovos_spec_tools.lint import lint_locale

    root = scratch / "locale"
    wrote = False
    for path in paths:
        parts = path.split("/")
        if "locale" not in parts:
            continue
        index = parts.index("locale")
        target = root.joinpath(*parts[index + 1:])
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(read_at(repo, rev, path), encoding="utf-8")
            wrote = True
        except (RuntimeError, UnicodeDecodeError):
            continue
    if not wrote:
        return []
    return [{"severity": f.severity, "path": str(f.path), "message": f.message}
            for f in lint_locale(root)]


# --------------------------------------------------------------- census ----

def census_one(repo: Path, rev: str) -> dict:
    paths = files_at(repo, rev)
    langs = locale_map(paths)
    base, how = base_language(repo, rev, langs)
    slots = slot_rows(repo, rev, paths)
    with tempfile.TemporaryDirectory(prefix="parity-census-") as scratch:
        findings = layer_a(repo, rev, paths, Path(scratch))
    return {
        "skill": repo.name,
        "sha": rev_of(repo, rev),
        "languages": sorted(langs),
        "base_language": base,
        "base_language_signal": how,
        "reference_present": REFERENCE in langs,
        "counts": {lang: {role: len(langs[lang].get(role) or ()) for role in ROLES}
                   for lang in sorted(langs)},
        "layer_a": findings,
        "layer_b": layer_b(langs, base) if base in langs else [],
        # what the census said before the Q1 ruling, so a row that moves is
        # visible instead of silently rewritten
        "layer_b_vs_en_us": layer_b(langs) if REFERENCE in langs else [],
        "slots": slots,
        # OVOS-INTENT-2 §2.3, architecture#272, not merged: reported, and
        # no verdict depends on it yet
        "parity_defects_2_3": parity_defects(repo, rev, paths, langs),
        "slot_set_defects_2_3": slot_set_defects(slots),
    }


def pinned_revisions(sources: Path) -> dict[str, str]:
    """``{repo name: revision}`` from train/sources.yaml, read as text."""
    import re
    text = sources.read_text(encoding="utf-8")
    out = {}
    for match in re.finditer(r"^\s+([\w./-]*ovos-skill-[\w-]+):\s*([0-9a-f]{7,40})\s*$",
                             text, re.M):
        out[match.group(1).rsplit("/", 1)[-1]] = match.group(2)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True,
                        help="directory holding the ovos-skill-* clones")
    parser.add_argument("--glob", default="ovos/skills/ovos-skill-*",
                        help="where the clones are, relative to the workspace")
    parser.add_argument("--at-pins", action="store_true",
                        help="read each skill at its train/sources.yaml revision "
                             "instead of origin/dev")
    parser.add_argument("--sources", default=None,
                        help="path to train/sources.yaml, for --at-pins")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="write the census as JSON instead of a table")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser()
    repos = sorted(p for p in workspace.glob(args.glob) if (p / ".git").exists())
    if not repos:
        raise SystemExit(f"[census] no clones under {workspace / args.glob}")

    pins = {}
    if args.at_pins:
        if not args.sources:
            raise SystemExit("[census] --at-pins needs --sources")
        pins = pinned_revisions(Path(args.sources).expanduser())

    results, skipped = [], []
    for repo in repos:
        rev = pins.get(repo.name, "") if args.at_pins else ""
        if not rev:
            # Not every skill calls its default branch dev; the census must
            # not drop a repository over the name of its branch.
            for candidate in ("origin/dev", "origin/master", "origin/main"):
                try:
                    rev_of(repo, candidate)
                    rev = candidate
                    break
                except RuntimeError:
                    continue
        if not rev:
            skipped.append({"skill": repo.name,
                            "reason": "no origin/dev, origin/master or origin/main"})
            continue
        try:
            results.append(census_one(repo, rev))
        except RuntimeError as exc:
            skipped.append({"skill": repo.name, "reason": str(exc)[:160]})

    report = {
        "skills_in": len(repos),
        "rows_out": len(results),
        "skipped": skipped,
        "read_at": "train/sources.yaml pins" if args.at_pins else "origin/dev",
        "skills": results,
    }
    if args.as_json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(f"{len(repos)} skills in, {len(results)} rows out, "
              f"read at {report['read_at']}")
        for skipped_one in skipped:
            print(f"  SKIPPED {skipped_one['skill']}: {skipped_one['reason']}")
        for entry in results:
            errors = [f for f in entry["layer_a"] if f["severity"] == "error"]
            gaps = sum(len(row[role]["missing"]) for row in entry["layer_b"]
                       for role in ROLES)
            was = sum(len(row[role]["missing"]) for row in entry["layer_b_vs_en_us"]
                      for role in ROLES)
            moved = "" if was == gaps else f"  (en-US said {was})"
            p23 = sum(len(row["missing"]) for row in entry["parity_defects_2_3"])
            s23 = len(entry["slot_set_defects_2_3"])
            print(f"  {entry['skill']:<38} {entry['sha'][:8]} "
                  f"base={entry['base_language']:<6} "
                  f"langs={len(entry['languages']):>3} "
                  f"layerA_errors={len(errors):>3} layerB_missing={gaps:>5} "
                  f"slots={len(entry['slots']):>4}{moved}\n"
                  f"      §2.3 file_defects={p23:>5}  slot_set_defects={s23:>3}")

    errors = sum(1 for e in results for f in e["layer_a"] if f["severity"] == "error")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
