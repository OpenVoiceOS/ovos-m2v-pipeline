#!/usr/bin/env python3
"""Typed slots per skill per locale, and what each language can fill.

Two questions, one pass:

* Which `{type:name}` slots does each skill declare, in which locale?
* For each language, how many surface values does each type yield?

Per skill per locale the answer is reported as N in, N out: the typed slots
declared, and the ones the locale can actually fill. A slot whose type the
locale has no generator for is OUT of the corpus, and the cell names the
type, so a zero in a locale is a row the corpus will not contain and the
reason it will not.

The second is what decides whether a typed template can enter the corpus at
all. A language with no generator for a type drops that template rather than
filling it with another language's words, so a zero in that table is a row
the corpus will not contain, and the reason it will not.

    python train/typed_slot_census.py --workspace ~/AgentWorkspaces

By default the skills are the ones the build reads: every `skill_refs` entry
of `train/sources.yaml`, at the revision it pins. A census taken over a
directory glob at `origin/dev` instead describes a different corpus than the
one that ships -- 13 of the 64 pinned skills live outside the old
`ovos/skills/ovos-skill-*` glob, and a pinned skill's head is not its pin.
`--glob` still takes that sweep, for a look at the fleet as it stands now.

Exit status is 0 always: this is a census, not a gate. The gate is
`tests/test_typed_slot_rows.py`, which fails when a placeholder survives.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import typed_slots  # noqa: E402


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr.strip())
    return out.stdout


def declared(repo: Path, rev: str):
    """Every (lang, intent file, slot, type) the repository declares."""
    from ovos_spec_tools.lint import declared_slot_types, read_resource_file

    rows = []
    for path in git(repo, "ls-tree", "-r", "--name-only", rev).split():
        if not path.endswith(".intent") or "/locale/" not in f"/{path}":
            continue
        parts = path.split("/")
        lang = parts[parts.index("locale") + 1]
        try:
            text = git(repo, "show", f"{rev}:{path}")
        except RuntimeError:
            continue
        with tempfile.NamedTemporaryFile("w", suffix=".intent", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(text)
            tmp = Path(handle.name)
        try:
            for name, slot_type in (declared_slot_types(read_resource_file(tmp)) or {}).items():
                rows.append({"lang": lang, "intent": parts[-1],
                             "slot": name, "type": slot_type})
        except Exception:
            continue
        finally:
            tmp.unlink(missing_ok=True)
    return rows


def in_out(rows):
    """Per locale: typed slots declared, filled, and dropped with the type.

    "In" counts the declarations the skill ships for that locale. "Out"
    counts the ones the locale can fill, which is what enters the corpus.
    The two differ only where a type has no generator for that language, so
    the gap is never a mystery: `dropped_types` names it.
    """
    per_lang = {}
    for row in rows:
        cell = per_lang.setdefault(row["lang"], {"in": 0, "out": 0,
                                                 "dropped_types": {}})
        cell["in"] += 1
        if typed_slots.values_for(row["type"], row["lang"]):
            cell["out"] += 1
        else:
            cell["dropped_types"][row["type"]] = \
                cell["dropped_types"].get(row["type"], 0) + 1
    return per_lang


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--sources", default=str(Path(__file__).resolve().parent
                                                 / "sources.yaml"))
    parser.add_argument("--glob", default=None,
                        help="sweep this glob at origin/dev instead of the "
                             "pinned skill_refs")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    ws = Path(args.workspace).expanduser()
    if args.glob:
        pinned = {p: None for p in sorted(ws.glob(args.glob))
                  if (p / ".git").exists()}
    else:
        import yaml
        refs = yaml.safe_load(Path(args.sources).read_text(
            encoding="utf-8"))["skill_refs"]["refs"]
        pinned = {ws / key: rev for key, rev in sorted(refs.items())
                  if (ws / key / ".git").exists()}
    repos = list(pinned)
    per_skill = {}
    langs = set()
    for repo in repos:
        rev = pinned[repo] or ""
        if not rev:
            for candidate in ("origin/dev", "origin/master", "origin/main"):
                try:
                    git(repo, "rev-parse", "--verify", "-q", candidate)
                    rev = candidate
                    break
                except RuntimeError:
                    continue
        if not rev:
            continue
        try:
            rows = declared(repo, rev)
        except RuntimeError:
            continue
        if rows:
            per_skill[repo.name] = rows
        for row in rows:
            langs.add(row["lang"])

    fill = {lang: typed_slots.coverage(lang) for lang in sorted(langs or {"en-US"})}
    report = {
        "skills_in": len(repos),
        "skills_read_at": "the pins in sources.yaml" if not args.glob
                          else f"origin/dev under {args.glob}",
        "skills_declaring_a_typed_slot": len(per_skill),
        "registry": sorted(typed_slots.registry()),
        "resolvable_by_the_runtime": sorted(typed_slots.resolvable()),
        "registered_but_not_resolvable": sorted(typed_slots.unresolvable()),
        "declarations": per_skill,
        "in_out_per_skill_per_locale": {skill: in_out(rows)
                                        for skill, rows in per_skill.items()},
        "fill_coverage_per_language": fill,
    }
    if args.as_json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0

    print(f"{len(repos)} skills in, {len(per_skill)} declare a typed slot")
    print(f"registry: {sorted(typed_slots.registry())}")
    print(f"runtime resolves: {sorted(typed_slots.resolvable())}")
    if typed_slots.unresolvable():
        print(f"registered but NOT resolvable, so never sampled: "
              f"{sorted(typed_slots.unresolvable())}")
    for skill, rows in sorted(per_skill.items()):
        by_type = collections.Counter(r["type"] for r in rows)
        cells = in_out(rows)
        went_in = sum(c["in"] for c in cells.values())
        came_out = sum(c["out"] for c in cells.values())
        print(f"  {skill:<34} {went_in:>5} in {came_out:>5} out, "
              f"{len(cells):>3} locales, {dict(by_type)}")
        for lang, cell in sorted(cells.items()):
            if cell["dropped_types"]:
                print(f"      {lang:<8} {cell['in']:>4} in {cell['out']:>4} "
                      f"out, no generator for {dict(cell['dropped_types'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
