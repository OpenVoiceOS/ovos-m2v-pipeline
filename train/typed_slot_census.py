#!/usr/bin/env python3
"""Typed slots per skill per locale, and what each language can fill.

Two questions, one pass:

* Which `{type:name}` slots does each skill declare, in which locale?
* For each language, how many surface values does each type yield?

The second is what decides whether a typed template can enter the corpus at
all. A language with no generator for a type drops that template rather than
filling it with another language's words, so a zero in that table is a row
the corpus will not contain, and the reason it will not.

    python train/typed_slot_census.py --workspace ~/AgentWorkspaces

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--glob", default="ovos/skills/ovos-skill-*")
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args(argv)

    repos = sorted(p for p in Path(args.workspace).expanduser().glob(args.glob)
                   if (p / ".git").exists())
    per_skill = {}
    langs = set()
    for repo in repos:
        rev = ""
        for candidate in ("origin/dev", "origin/master", "origin/main"):
            try:
                git(repo, "rev-parse", candidate)
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
        "skills_declaring_a_typed_slot": len(per_skill),
        "registry": sorted(typed_slots.registry()),
        "resolvable_by_the_runtime": sorted(typed_slots.resolvable()),
        "registered_but_not_resolvable": sorted(typed_slots.unresolvable()),
        "declarations": per_skill,
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
        by_lang = len({r["lang"] for r in rows})
        print(f"  {skill:<34} {len(rows):>4} declarations, {by_lang:>3} locales, {dict(by_type)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
