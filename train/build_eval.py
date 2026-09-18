#!/usr/bin/env python3
"""Build the evaluation set from the skills' own golden files.

The eval side of the corpus is the gold sentences the fleet ships: every
``test/end2end/golden_utterances*.jsonl`` of every skill in ``skill_refs``,
read at the same pinned revision the train side reads the skill's resources
at. Nothing else is an eval source: no hand-kept tree, no shared fleet file,
no model-written set.

Every row's label comes from ``skill_labels``, the one label function the
train side uses too: the skill id the repository's entry point declares and
the intent name without its ``.intent`` suffix. A gold row's own ``skill_id``
field is not read for the label. When the gold side names an intent the train
side does not carry, that is a naming defect in the skill or its gold file;
``census_gold_labels`` refuses it (``--train``), this builder does not resolve
it.

Output, under ``--out``: ``{lang}/test.jsonl`` (the layout the Hub repo
``OpenVoiceOS/ovos-intents-v5-eval`` publishes), a flat ``test.jsonl`` for the
census, ``census.json`` (rows in, rows out and the reason for each row dropped,
per skill per locale) and ``manifest.json``.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import skill_labels  # noqa: E402
from census_gold_labels import census_paths  # noqa: E402

#: Gold lives beside the end-to-end suite that reads it.
GOLD = re.compile(r"(?:^|.*/)golden_utterances(?:_([A-Za-z-]+))?\.jsonl$")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout


def tree(repo: Path, rev: str):
    return git(repo, "ls-tree", "-r", "--name-only", rev).split("\n")


def show(repo: Path, rev: str, path: str) -> str:
    try:
        return git(repo, "show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return ""


def normalize_lang(tag: str) -> str:
    """The train side's fold (build_from_skills.normalize_lang), imported
    lazily to avoid the circular import at module load."""
    from build_from_skills import normalize_lang as _fold
    return _fold(tag)


def read_gold(repo: Path, rev: str, repo_name: str, stats: collections.Counter,
              census: dict = None):
    """Gold rows a skill ships at *rev*, labelled by the shared label
    function. *census* (per skill per locale) records rows in, rows out and
    why a row was dropped."""
    census = census if census is not None else {}
    skill_id = skill_labels.skill_id_from_repo(repo, rev, repo_name)
    if not skill_labels.declares_skill_id(repo, rev):
        stats["skill_id_assumed_from_repo_name"] += 1
    rows = []
    seen = set()
    for path in tree(repo, rev):
        m = GOLD.match(path or "")
        if not m:
            continue
        from_name = m.group(1)
        for line in show(repo, rev, path).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                stats["gold_unparsable"] += 1
                _count(census, repo_name, from_name or "?", "unparsable")
                continue
            lang = normalize_lang(d.get("lang") or from_name or "en-US")
            _count(census, repo_name, lang, "in")
            if d.get("needs_manual"):
                stats["gold_needs_manual"] += 1
                _count(census, repo_name, lang, "needs_manual")
                continue
            utterance = (d.get("utterance") or "").strip()
            intent = d.get("intent_label") or d.get("expected_intent")
            if not utterance:
                stats["gold_without_utterance"] += 1
                _count(census, repo_name, lang, "no_utterance")
                continue
            if not intent:
                # A fallback skill asserts the dialog it speaks, because no
                # intent claimed the utterance. That is a real assertion and
                # this corpus cannot score it: there is no label to predict.
                stats["gold_asserts_a_dialog_not_an_intent"] += 1
                _count(census, repo_name, lang, "no_intent")
                continue
            label = skill_labels.label(skill_id, str(intent))
            key = (lang, label, utterance.lower())
            if key in seen:
                stats["gold_duplicate_rows_removed"] += 1
                _count(census, repo_name, lang, "duplicate")
                continue
            seen.add(key)
            rows.append({"lang": lang, "label": label, "utterance": utterance,
                         "source": f"gold:{repo_name}"})
            _count(census, repo_name, lang, "out")
    return rows


def _count(census: dict, skill: str, lang: str, key: str) -> dict:
    cell = census.setdefault(skill, {}).setdefault(lang, {"in": 0, "out": 0})
    cell[key] = cell.get(key, 0) + 1
    return cell


def build(sources: Path, workspace: Path = None):
    cfg = yaml.safe_load(sources.read_text(encoding="utf-8"))
    ws = Path(workspace or cfg["workspace"]).expanduser()
    refs = cfg["skill_refs"]["refs"]
    stats: collections.Counter = collections.Counter()
    census: dict = {}
    rows = []
    for key, rev in sorted(refs.items()):
        repo, repo_name = ws / key, key.rsplit("/", 1)[-1]
        if not repo.is_dir():
            stats["skill_missing_clone"] += 1
            continue
        rows.extend(read_gold(repo, rev, repo_name, stats, census))
    return rows, census, stats, refs


def write(out: Path, rows, census, stats, refs, sources: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    by_lang = collections.defaultdict(list)
    for row in rows:
        by_lang[row["lang"]].append(row)
    for lang, lang_rows in sorted(by_lang.items()):
        (out / lang).mkdir(exist_ok=True)
        with (out / lang / "test.jsonl").open("w", encoding="utf-8") as fh:
            for row in lang_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "test.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "test_rows": len(rows),
        "languages_test": len(by_lang),
        "labels_scored": len({r["label"] for r in rows}),
        "skills_with_gold": sorted({r["source"].split(":", 1)[1] for r in rows}),
        "rows_per_language": {lang: len(v) for lang, v in sorted(by_lang.items())},
        "stats": dict(sorted(stats.items())),
        "sources": str(sources),
        "skill_refs": dict(sorted(refs.items())),
        "label_function": "train/skill_labels.py",
    }
    (out / "census.json").write_text(json.dumps(census, indent=2, sort_keys=True))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(HERE / "sources.yaml"))
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--out", required=True, help="directory for eval/{lang}/test.jsonl")
    ap.add_argument("--train", default=None,
                    help="train.jsonl to gate against: a gold label with no train "
                         "row fails the build (census_gold_labels)")
    ap.add_argument("--min-test-rows", type=int, default=1)
    args = ap.parse_args(argv)

    rows, census, stats, refs = build(Path(args.sources), args.workspace)
    if len(rows) < args.min_test_rows:
        print(f"[eval] {len(rows)} rows, floor {args.min_test_rows}", file=sys.stderr)
        return 1
    out = Path(args.out)
    manifest = write(out, rows, census, stats, refs, Path(args.sources))
    print(json.dumps({k: v for k, v in manifest.items() if k != "skill_refs"}, indent=2))
    if args.train:
        train, test, missing = census_paths(Path(args.train), out / "test.jsonl")
        if missing:
            affected = sum(missing.values())
            print(f"[eval] {len(missing)} test labels have no train rows, "
                  f"{affected} rows affected:", file=sys.stderr)
            for lbl, n in sorted(missing.items(), key=lambda kv: -kv[1]):
                print(f"  {n:6}  {lbl}", file=sys.stderr)
            return 1
        print(f"[eval] every one of the {len(test)} test labels has train rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
