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

A gap an OPEN pull request already fills is reported as **in flight**, in its
own column, and is not counted as missing. The census reads `origin/dev`, and
a branch under review is not on `origin/dev`, so without this the same gap is
filed as a task until the branch merges. One `gh` call per repository pays for
it; `--pr-cache` reuses a reading and `--no-in-flight` skips it.

A pushed branch with NO pull request is read the same way, from the local
remote-tracking refs named `t<task>-*`, because a skill takes one pull request
at a time and the waiting branches are this lane's own work. Such a pair reads
`in-flight:branch:<name>`. A pull request number wins over a branch, and
`--no-branch-signal` skips this reading. No network call pays for it, so a
branch this clone never fetched is not seen.

Without `--at-pins` every skill is read at `origin/dev`, which measures the
fleet as it stands, and each skill's sha is printed. With `--at-pins` the
revisions in `train/sources.yaml` are read instead, which measures the tree
the published corpus was built from.

The count is taken against the clones on disk, and the report says so: the
first line names the listing (`<workspace>/<glob>`). A skill the workspace
does not hold produces no row, so a workspace census is a complete census of
the workspace and a partial one of the fleet. To tell the two apart, give the
census the fleet listing and it names what it did not see:

    python tools/parity_census.py --workspace ~/AgentWorkspaces --org OpenVoiceOS
    python tools/parity_census.py --workspace ~/AgentWorkspaces --fleet names.txt

`--org` reads the organisation's repositories through `gh`; `--fleet` reads a
file with one repository name per line. Either way every `ovos-skill-*` name
in the listing with no clone is printed as `NOT CLONED`, and every clone that
the listing does not carry as `NOT IN FLEET`.

An archived repository is not a gap: cloning it does not fix anything, so
`--org` excludes it from the listing before the comparison runs. The report
names how many it excluded, so the count is auditable.

Exit status is 0 when the census is produced. It is 1 when a Layer A finding
of severity `error` exists, so a build job can refuse to publish on one. A
fleet name that is not cloned does not change the exit status: the census
did not measure it, and the report says so.
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
            reference_lang: str = REFERENCE,
            in_flight: dict | None = None) -> list[dict]:
    """Per locale, what the base language has and this one does not.

    SHOULD, not MUST: no clause requires mirroring (locale-parity.md §1).
    The reference is the skill's own base language, not always en-US
    (locale-parity.md §5 Q1, ruled by Miro).

    ``in_flight`` maps ``(lang, file name)`` to the number of an OPEN pull
    request that already adds that file. Such a pair leaves ``missing`` and
    is reported in its own ``in_flight`` list instead, so no task is filed
    for work that is written and waiting for review.
    """
    reference = langs.get(reference_lang) or {}
    index = in_flight or {}
    rows = []
    for lang in sorted(langs):
        if lang == reference_lang:
            continue
        row = {"lang": lang}
        for role in ROLES:
            gap = sorted((reference.get(role) or set()) - (langs[lang].get(role) or set()))
            extra = sorted((langs[lang].get(role) or set()) - (reference.get(role) or set()))
            missing, flying = [], []
            for name in gap:
                source = index.get((lang, name + role))
                if source is None:
                    missing.append(name)
                elif isinstance(source, int):
                    flying.append({"name": name, "pr": source})
                else:
                    flying.append({"name": name,
                                   "branch": str(source).split(":", 1)[-1]})
            row[role] = {"missing": missing, "extra": extra, "in_flight": flying}
        rows.append(row)
    return rows


# -------------------------------------------------------------- in flight ----

# Four parity tasks in a row were filed for pairs an open pull request already
# carried (T-4107, T-4109, T-4111, T-4113). The census could not know: it
# reads `origin/dev`, and a branch under review is not on `origin/dev`. So the
# open pull requests are read too, and a gap a branch already fills is
# reported as in flight rather than as a gap.


def repo_slug(repo: Path) -> str | None:
    """``owner/name`` from the origin remote, or None if there is no remote."""
    try:
        url = git(repo, "remote", "get-url", "origin").strip()
    except RuntimeError:
        return None
    match = re.search(r"[:/]([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
    return f"{match.group(1)}/{match.group(2)}" if match else None


#: `gh pr list --json files` returns at most this many files per pull
#: request, with no flag to raise it and no marker saying it truncated. A
#: locale pull request easily passes it: moviemaster#105 carries 85 locale
#: files inside its first 100 and more beyond them. A list of exactly this
#: length is therefore assumed to be cut, and is read again from the API,
#: which pages.
GH_PR_LIST_FILE_CAP = 100


def open_pr_files(slug: str, runner=None) -> list[dict]:
    """``[{"number": n, "files": [path]}]`` for every OPEN pull request.

    One `gh` call per repository, and one more for each pull request that
    hits the file cap. A failure is not fatal to the census as a whole: the
    caller reports the skill as unread, and every gap in it is then counted
    as a gap, which is the safe direction.
    """
    run = runner or (lambda cmd: subprocess.run(cmd, capture_output=True, text=True))
    out = run(["gh", "pr", "list", "--repo", slug, "--state", "open",
               "--json", "number,headRefOid,files", "--limit", "100"])
    if out.returncode:
        raise RuntimeError(f"gh pr list {slug}: {out.stderr.strip()[:160]}")
    prs = []
    for pr in json.loads(out.stdout or "[]"):
        files = [f["path"] for f in pr.get("files") or []]
        truncated = len(files) >= GH_PR_LIST_FILE_CAP
        if truncated:
            files = all_pr_files(slug, pr["number"], run)
        prs.append({"number": pr["number"],
                    "head": pr.get("headRefOid", ""),
                    "files": files,
                    "refetched": truncated})
    return prs


def all_pr_files(slug: str, number: int, run) -> list[str]:
    """Every file of one pull request, paged, because the list API caps."""
    out = run(["gh", "api", f"repos/{slug}/pulls/{number}/files",
               "--paginate", "--jq", ".[].filename"])
    if out.returncode:
        raise RuntimeError(f"gh api pulls/{number}/files {slug}: "
                           f"{out.stderr.strip()[:160]}")
    return [line for line in out.stdout.splitlines() if line]


def in_flight_index(prs: list[dict]) -> dict:
    """``{(lang, file name): pr number}`` for the locale files those add.

    The path shape is not fixed: a skill may write `locale/<lang>/x.intent`
    or `locale/<lang>/intents/x.intent`. Only the segment after `locale`
    and the base name are read, which is exactly what layer B compares.
    The LOWEST pull request number wins, so the answer does not depend on
    the order GitHub returns.
    """
    index: dict = {}
    for pr in sorted(prs, key=lambda p: p["number"]):
        for path in pr["files"]:
            parts = path.split("/")
            if "locale" not in parts:
                continue
            position = parts.index("locale")
            if position + 2 > len(parts) - 1:
                continue
            key = (parts[position + 1], parts[-1])
            index.setdefault(key, pr["number"])
    return index


# ------------------------------------------------------- branch signal ----

# The pull request signal above cannot see a branch that was PUSHED and has
# NO pull request. This lane holds such branches on purpose: a skill takes one
# pull request at a time, so the second and third branch wait. Their pairs
# read as gaps every run, which is the same false report the pull request
# signal was written to stop (parity-census-in-flight.md, "What the column
# cannot see").
#
# So a second signal reads them. A branch counts when its name matches
# `t<digits>-`, which is the lane's own naming for a task branch, and when it
# adds a locale file that `origin/dev` does not have.

#: A task branch of this lane: `t4108-kab-parity`, `t4112-kab-parity`.
TASK_BRANCH = re.compile(r"^t\d+-")


def task_branches(repo: Path, remote: str = "origin") -> list[str]:
    """Every local remote-tracking branch of `remote` named `t<digits>-*`.

    Local refs only, so no network call. A clone that never fetched the
    branch sees nothing here, which is why `remote_task_branches` exists:
    measured on 2026-09-25, all three branches this lane holds were on the
    remote and in no clone.
    """
    out = git(repo, "for-each-ref", "--format=%(refname:short)",
              f"refs/remotes/{remote}")
    names = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "/" not in line:
            continue
        short = line.split("/", 1)[1]
        if TASK_BRANCH.match(short):
            names.append(short)
    return sorted(set(names))


def remote_task_branches(repo: Path, remote: str = "origin") -> list[str]:
    """Every `t<digits>-*` head on the remote itself, from `ls-remote`.

    One git call per repository, and no `gh` quota. This is the authority on
    what exists: a pushed branch is on the remote whether or not any clone
    fetched it.
    """
    out = git(repo, "ls-remote", "--heads", remote)
    names = []
    for line in out.splitlines():
        parts = line.split("refs/heads/", 1)
        if len(parts) != 2:
            continue
        name = parts[1].strip()
        if TASK_BRANCH.match(name):
            names.append(name)
    return sorted(set(names))


def fetch_branch(repo: Path, remote: str, name: str) -> None:
    """Put one remote branch in this clone, as the tracking ref.

    The refspec is forced. A task branch is force-pushed when it is
    squashed or rebased, and a stale tracking ref then makes the plain
    refspec fail as a non-fast-forward, which drops the branch from the
    signal and reports its pairs as gaps. The tracking ref must follow the
    remote, because the remote is what a reader sees.
    """
    git(repo, "fetch", "--quiet", remote,
        f"+refs/heads/{name}:refs/remotes/{remote}/{name}")


def branch_locale_files(repo: Path, rev: str, branch: str,
                        remote: str = "origin") -> list[str]:
    """Locale files the branch adds that `rev` does not have.

    The comparison is against the merge base, so a change that landed on
    `rev` after the branch was cut is not read as the branch's work.
    """
    out = git(repo, "diff", "--name-only", "--diff-filter=AMR",
              f"{rev}...{remote}/{branch}")
    return [line for line in out.splitlines()
            if line and "locale" in line.split("/")]


def branch_index(repo: Path, rev: str, remote: str = "origin",
                 branches: list[str] | None = None,
                 fetch: bool = False, notes: list | None = None) -> dict:
    """``{(lang, file name): "branch:<name>"}`` for pushed branches with work.

    The key shape is the pull request index's, so the two merge. The first
    branch in name order wins, so the answer does not depend on ref order.

    With ``fetch``, the remote is asked what exists and a branch this clone
    does not hold is fetched. Without it, only the local tracking refs are
    read. A branch that cannot be read is named in ``notes``, never skipped
    in silence.
    """
    index: dict = {}
    if branches is None:
        branches = remote_task_branches(repo, remote) if fetch \
            else task_branches(repo, remote)
    for branch in branches:
        if fetch:
            try:
                fetch_branch(repo, remote, branch)
            except RuntimeError as error:
                if notes is not None:
                    notes.append({"branch": branch, "reason": str(error)[:160]})
                continue
        for path in branch_locale_files(repo, rev, branch, remote):
            parts = path.split("/")
            position = parts.index("locale")
            if position + 2 > len(parts) - 1:
                continue
            index.setdefault((parts[position + 1], parts[-1]), f"branch:{branch}")
    return index


def merge_signals(pr_index: dict, branch_index_: dict) -> dict:
    """The pull request signal wins; a branch only fills what it leaves.

    A pair that a pull request carries keeps its number, because a number is
    a stronger statement than a pushed branch: it is under review.
    """
    merged = dict(pr_index)
    for key, value in branch_index_.items():
        merged.setdefault(key, value)
    return merged


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

def census_one(repo: Path, rev: str, in_flight: dict | None = None,
               in_flight_prs: list | None = None,
               in_flight_branches: list | None = None) -> dict:
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
        "layer_b": layer_b(langs, base, in_flight) if base in langs else [],
        # what the census said before the Q1 ruling, so a row that moves is
        # visible instead of silently rewritten
        "layer_b_vs_en_us": layer_b(langs, REFERENCE, in_flight) if REFERENCE in langs else [],
        # which open pull requests were read, so a reader can check the
        # in-flight column rather than trust it
        "open_prs_read": sorted(pr["number"] for pr in (in_flight_prs or [])),
        "branches_read": sorted(in_flight_branches or []),
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


def fleet_from_org(org: str) -> tuple[list[str], int]:
    """``ovos-skill-*`` repositories the organisation holds, via ``gh``.

    An archived repository is excluded: it names a gap the workspace cannot
    close by cloning it. The count of excluded rows is returned alongside the
    names so the report can say how many it dropped.
    """
    out = subprocess.run(["gh", "api", f"/orgs/{org}/repos", "--paginate",
                          "--jq", ".[] | .name + \"\\t\" + (.archived | tostring)"],
                         capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"[census] gh api /orgs/{org}/repos: {out.stderr.strip()}")
    return _split_archived(out.stdout)


def _split_archived(raw: str) -> tuple[list[str], int]:
    """``(names, archived_excluded)`` from ``name<TAB>archived`` lines."""
    names, archived_excluded = [], 0
    for line in raw.splitlines():
        if not line.strip():
            continue
        name, _, archived = line.rpartition("\t")
        if not name.startswith("ovos-skill-"):
            continue
        if archived == "true":
            archived_excluded += 1
            continue
        names.append(name)
    return sorted(names), archived_excluded


def fleet_from_file(path: Path) -> list[str]:
    """One repository name per line; blank lines and ``#`` comments skipped."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.rsplit("/", 1)[-1])
    return sorted(names)


def fleet_diff(fleet: list[str], cloned: list[str]) -> dict:
    """What the listing has and the workspace lacks, and the reverse."""
    fleet_set, cloned_set = set(fleet), set(cloned)
    return {
        "named": len(fleet),
        "not_cloned": sorted(fleet_set - cloned_set),
        "not_in_fleet": sorted(cloned_set - fleet_set),
    }


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
    parser.add_argument("--org", default=None,
                        help="GitHub organisation whose ovos-skill-* listing "
                             "the clones are checked against (needs gh)")
    parser.add_argument("--fleet", default=None,
                        help="file with one repository name per line, the "
                             "listing the clones are checked against")
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="write the census as JSON instead of a table")
    parser.add_argument("--no-in-flight", dest="in_flight", action="store_false",
                        help="do not read the open pull requests; every gap is "
                             "then reported as a gap, including one a branch "
                             "already fills")
    parser.add_argument("--no-branch-signal", dest="branch_signal",
                        action="store_false",
                        help="do not read pushed t<task>-* branches; a pair that "
                             "sits on a branch with no pull request is then "
                             "reported as a gap")
    parser.add_argument("--no-fetch-branches", dest="fetch_branches",
                        action="store_false",
                        help="read only the branches this clone already holds; "
                             "without it the remote is asked what exists and a "
                             "missing branch is fetched")
    parser.add_argument("--pr-cache", default=None,
                        help="JSON file of {repo: [pull request]} to read the "
                             "open pull requests from, and to write after a "
                             "live read; one gh call per repository is spent "
                             "on a cold run")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).expanduser()
    repos = sorted(p for p in workspace.glob(args.glob) if (p / ".git").exists())
    if not repos:
        raise SystemExit(f"[census] no clones under {workspace / args.glob}")

    fleet, archived_excluded = None, None
    if args.org and args.fleet:
        raise SystemExit("[census] give --org or --fleet, not both")
    if args.org:
        fleet, archived_excluded = fleet_from_org(args.org)
    elif args.fleet:
        fleet = fleet_from_file(Path(args.fleet).expanduser())

    pins = {}
    if args.at_pins:
        if not args.sources:
            raise SystemExit("[census] --at-pins needs --sources")
        pins = pinned_revisions(Path(args.sources).expanduser())

    cache, cache_path = {}, None
    if args.pr_cache:
        cache_path = Path(args.pr_cache).expanduser()
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))

    results, skipped, pr_notes = [], [], []
    branch_notes, moved_by_branch = [], 0
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
        prs = []
        if args.in_flight:
            slug = repo_slug(repo)
            if slug is None:
                pr_notes.append({"skill": repo.name, "reason": "no origin remote"})
            elif repo.name in cache:
                prs = cache[repo.name]
            else:
                try:
                    prs = open_pr_files(slug)
                    cache[repo.name] = prs
                except (RuntimeError, ValueError) as exc:
                    # Reported, never swallowed: without it every gap in this
                    # skill is counted as a gap, and a reader must know that
                    # the in-flight column is empty because nothing was read.
                    pr_notes.append({"skill": repo.name, "reason": str(exc)[:160]})
        branches, from_branches = [], {}
        if args.branch_signal:
            try:
                branches = (remote_task_branches(repo) if args.fetch_branches
                            else task_branches(repo))
                unread: list = []
                from_branches = branch_index(repo, rev, branches=branches,
                                             fetch=args.fetch_branches,
                                             notes=unread)
                for one in unread:
                    branch_notes.append({"skill": repo.name,
                                         "reason": f"{one['branch']}: {one['reason']}"})
            except RuntimeError as exc:
                branch_notes.append({"skill": repo.name, "reason": str(exc)[:160]})
        pr_only = in_flight_index(prs)
        signals = merge_signals(pr_only, from_branches)
        moved_by_branch += sum(1 for key in signals
                               if key not in pr_only and key in from_branches)
        try:
            results.append(census_one(repo, rev, signals, prs, branches))
        except RuntimeError as exc:
            skipped.append({"skill": repo.name, "reason": str(exc)[:160]})

    listing = str(workspace / args.glob)
    if cache_path is not None:
        cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True),
                              encoding="utf-8")

    report = {
        "listing": listing,
        "skills_in": len(repos),
        "rows_out": len(results),
        "skipped": skipped,
        "in_flight_read": bool(args.in_flight),
        "in_flight_not_read": pr_notes,
        "branch_signal_read": bool(args.branch_signal),
        "branch_signal_fetched": bool(args.fetch_branches),
        "branch_signal_not_read": branch_notes,
        "pairs_moved_by_a_branch": moved_by_branch,
        "read_at": "train/sources.yaml pins" if args.at_pins else "origin/dev",
        "fleet": None,
        "skills": results,
    }
    if fleet is not None:
        report["fleet"] = fleet_diff(fleet, [r.name for r in repos])
        report["fleet"]["source"] = f"org {args.org}" if args.org else args.fleet
        report["fleet"]["archived_excluded"] = archived_excluded
    if args.as_json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    else:
        print(f"listing: {listing} ({len(repos)} clones)")
        print(f"{len(repos)} skills in, {len(results)} rows out, "
              f"read at {report['read_at']}")
        for skipped_one in skipped:
            print(f"  SKIPPED {skipped_one['skill']}: {skipped_one['reason']}")
        for note in pr_notes:
            print(f"  NO PULL REQUESTS READ {note['skill']}: {note['reason']}")
        for note in branch_notes:
            print(f"  NO BRANCHES READ {note['skill']}: {note['reason']}")
        print(f"  {moved_by_branch} pair(s) in flight on a pushed branch "
              f"with no pull request")
        if fleet is not None:
            diff = report["fleet"]
            print(f"fleet: {diff['source']} names {diff['named']}, "
                  f"{len(diff['not_cloned'])} not cloned, "
                  f"{len(diff['not_in_fleet'])} clones not in the listing")
            if diff["archived_excluded"] is not None:
                print(f"fleet: {diff['archived_excluded']} archived "
                      f"repositories excluded from the listing")
            for name in diff["not_cloned"]:
                print(f"  NOT CLONED {name}")
            for name in diff["not_in_fleet"]:
                print(f"  NOT IN FLEET {name}")
        else:
            print("fleet: not checked; the count is the clones on disk only "
                  "(give --org or --fleet)")
        for entry in results:
            errors = [f for f in entry["layer_a"] if f["severity"] == "error"]
            gaps = sum(len(row[role]["missing"]) for row in entry["layer_b"]
                       for role in ROLES)
            was = sum(len(row[role]["missing"]) for row in entry["layer_b_vs_en_us"]
                      for role in ROLES)
            moved = "" if was == gaps else f"  (en-US said {was})"
            flying = sum(len(row[role]["in_flight"]) for row in entry["layer_b"]
                         for role in ROLES)
            p23 = sum(len(row["missing"]) for row in entry["parity_defects_2_3"])
            s23 = len(entry["slot_set_defects_2_3"])
            print(f"  {entry['skill']:<38} {entry['sha'][:8]} "
                  f"base={entry['base_language']:<6} "
                  f"langs={len(entry['languages']):>3} "
                  f"layerA_errors={len(errors):>3} layerB_missing={gaps:>5} "
                  f"in_flight={flying:>4} "
                  f"slots={len(entry['slots']):>4}{moved}\n"
                  f"      §2.3 file_defects={p23:>5}  slot_set_defects={s23:>3}")

    errors = sum(1 for e in results for f in e["layer_a"] if f["severity"] == "error")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
