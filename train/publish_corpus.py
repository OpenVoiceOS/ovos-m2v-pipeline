#!/usr/bin/env python3
"""Publish a built corpus to its Hugging Face dataset repo.

The corpus ships as ONE repo holding both splits. Each locale directory holds
`train_templates.jsonl`, and the locales that have gold also hold
`test.jsonl`. That is the layout the previous release already uses for the
training split, so a consumer that pins the file pattern `{lang}/test.jsonl`
or `{lang}/train_templates.jsonl` keeps working across a version bump. That
compatibility is the reason the layout is not up for discussion here.

One repo, not two. An earlier draft of this script published the templates
and the gold to separate repos. A single dataset is easier to pin and easier
to keep consistent: two repos can disagree about which build they came from,
and nothing in the pair records that they are meant to be read together. Tag
the commit instead, and a consumer pins a revision rather than a repo name.

Not every locale has gold. A locale with training rows and no gold utterance
has no `test.jsonl`, and this script never invents an empty one to even them
up: an empty `test.jsonl` reads as "measured, found nothing" when the truth
is "never measured". So the locale set of each split comes from what the
staged directory holds and is never a list written down in this file.

`upload_folder` creates or updates the files it is given and deletes nothing
else, so a locale directory from an earlier publish outlives every build that
does not write it. That is how `arb` and `es-419` reached the `v6` tag on the
v5 row schema: they were pushed on 2026-09-06 and no later publish removed
them, while the `v6` manifest recorded 52 training languages against 54
directories in the repository.

The upload therefore prunes. `delete_patterns` names the locale payloads, so
a `.jsonl` under a locale directory that this build did not stage is deleted
in the same commit. The pattern never matches the repository root, so the
dataset card, the manifest and `.gitattributes` are kept whatever the staged
tree holds.

The token comes from the `HF_TOKEN` environment variable, never from an
argument, so it stays out of shell history and out of any log this prints.

Usage:

    python train/publish_corpus.py \
        --corpus-dir staging/m2v-v6 \
        --repo OpenVoiceOS/ovos-intents-v5-templates \
        --tag v6 \
        --dry-run

Drop `--dry-run` to upload. Every check runs against the whole staged tree
before anything is uploaded, so a failed check cannot leave the repo holding
one split and not the other.
"""
import argparse
import os
import sys
from pathlib import Path

TEMPLATES_FILE = "train_templates.jsonl"
EVAL_FILE = "test.jsonl"
REQUIRED_ROOT_FILES = ("README.md", "manifest.json")


def locale_files(root: Path, filename: str) -> dict:
    """Map locale -> path for every `{lang}/<filename>` under *root*.

    The locale set is read from the directory, never from a hardcoded list.
    """
    found = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / filename
        if candidate.is_file():
            found[child.name] = candidate
    return found


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def check_root(root: Path) -> None:
    """The staged tree carries its own card and manifest, or it is not published."""
    if not root.is_dir():
        raise SystemExit(f"[publish] {root} is not a directory")
    missing = [f for f in REQUIRED_ROOT_FILES if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"[publish] {root} is missing {missing}")


def inspect(root: Path, filename: str) -> dict:
    """Read one split of a staged directory and report what it holds.

    Raises SystemExit when the split could not be published as it stands.
    """
    files = locale_files(root, filename)
    if not files:
        raise SystemExit(f"[publish] {root} holds no {filename} under any locale directory")
    per_locale = {lang: count_rows(path) for lang, path in sorted(files.items())}
    empty = [lang for lang, rows in per_locale.items() if rows == 0]
    if empty:
        raise SystemExit(
            f"[publish] {root} has an empty {filename} for {empty}; "
            "remove the file instead, an empty file claims a measurement that "
            "was never made")
    return {"root": root, "per_locale": per_locale, "rows": sum(per_locale.values())}


def report(name: str, found: dict, expect_rows: int | None) -> None:
    per_locale = found["per_locale"]
    print(f"[publish] {name}: {len(per_locale)} locales, {found['rows']} rows")
    for lang, rows in per_locale.items():
        print(f"    {lang:<10} {rows:>9}")
    if expect_rows is not None and found["rows"] != expect_rows:
        raise SystemExit(
            f"[publish] {name} holds {found['rows']} rows, expected {expect_rows}; "
            "the staged directory and the build do not agree")


#: Remote paths the upload may delete: a locale payload the staged tree no
#: longer holds. One directory level only, so nothing at the repository root
#: is ever matched.
LOCALE_PAYLOADS = ["*/*.jsonl"]


def upload(root: Path, repo_id: str, message: str, tag: str | None) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "[publish] HF_TOKEN is not set; the write token is not read from "
            "an argument or a file")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(root), repo_id=repo_id,
                      repo_type="dataset", commit_message=message,
                      delete_patterns=LOCALE_PAYLOADS)
    print(f"[publish] uploaded {root} -> {repo_id}")
    if tag:
        api.create_tag(repo_id, tag=tag, repo_type="dataset",
                       tag_message=message, exist_ok=False)
        print(f"[publish] tagged {repo_id} {tag}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True,
                    help="staged directory holding {lang}/train_templates.jsonl "
                         "and, where there is gold, {lang}/test.jsonl")
    ap.add_argument("--repo", required=True,
                    help="dataset repo id that holds both splits")
    ap.add_argument("--tag", default=None,
                    help="tag to put on the published commit, so a consumer "
                         "pins a revision instead of a repo name")
    ap.add_argument("--expect-train-rows", type=int, default=None,
                    help="fail if the staged templates do not hold this many rows")
    ap.add_argument("--expect-test-rows", type=int, default=None,
                    help="fail if the staged gold rows do not hold this many rows")
    ap.add_argument("--commit-message", default="publish corpus",
                    help="commit message for the upload")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be uploaded, touch no network")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    root = Path(args.corpus_dir)
    check_root(root)
    templates = inspect(root, TEMPLATES_FILE)
    gold = inspect(root, EVAL_FILE)
    report("templates", templates, args.expect_train_rows)
    report("gold", gold, args.expect_test_rows)

    no_gold = sorted(set(templates["per_locale"]) - set(gold["per_locale"]))
    if no_gold:
        rows = sum(templates["per_locale"][lang] for lang in no_gold)
        print(f"[publish] {len(no_gold)} locales have training rows and no gold "
              f"({rows} rows): {', '.join(no_gold)}")
    gold_only = sorted(set(gold["per_locale"]) - set(templates["per_locale"]))
    if gold_only:
        raise SystemExit(
            f"[publish] {gold_only} have gold rows and no training rows; "
            "a model cannot be scored on a locale it was never given")

    if args.dry_run:
        print("[publish] dry run, nothing uploaded")
        return 0

    upload(root, args.repo, args.commit_message, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
