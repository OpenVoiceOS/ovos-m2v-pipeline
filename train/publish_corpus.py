#!/usr/bin/env python3
"""Publish a built corpus to its two Hugging Face dataset repos.

The corpus ships as two repos, not one: the training templates and the gold
evaluation rows. They are separate because consumers pin them separately --
the plugin arena pins a revision of the eval repo to score against while
training moves on -- and because a single repo makes it easy to hand somebody
the gold by accident. Each repo holds one file per locale, `{lang}/
train_templates.jsonl` and `{lang}/test.jsonl`, which is the layout the v5
repos already use. A consumer that pins that file pattern keeps working
across a version bump, and that compatibility is the reason the layout is not
up for discussion here.

The two repos do not hold the same locales. A locale with training rows and
no gold utterance appears in the templates repo and not in the eval repo, and
this script never invents an empty file to even them up: an empty
`test.jsonl` reads as "measured, found nothing" when the truth is "never
measured". So the locale set comes from what the staged directories actually
contain and is never a list written down in this file.

The upload is additive. `upload_folder` creates or updates the files it is
given in one commit and deletes nothing else in the repo. The token comes
from the `HF_TOKEN` environment variable, never from an argument, so it stays
out of shell history and out of any log this prints.

Usage:

    python train/publish_corpus.py \
        --templates-dir staging/m2v-v6-templates \
        --eval-dir staging/m2v-v6-eval \
        --templates-repo OpenVoiceOS/ovos-intents-v6-templates \
        --eval-repo OpenVoiceOS/ovos-intents-v6-eval \
        --dry-run

Drop `--dry-run` to upload. Nothing is uploaded until every check passes for
both directories, so a half-published pair is not a state this can reach by
failing partway.
"""
import argparse
import json
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


def inspect(root: Path, filename: str) -> dict:
    """Read a staged directory and report what it holds.

    Raises SystemExit when the directory could not be published as it stands.
    """
    if not root.is_dir():
        raise SystemExit(f"[publish] {root} is not a directory")
    missing = [f for f in REQUIRED_ROOT_FILES if not (root / f).is_file()]
    if missing:
        raise SystemExit(f"[publish] {root} is missing {missing}")
    files = locale_files(root, filename)
    if not files:
        raise SystemExit(f"[publish] {root} holds no {filename} under any locale directory")
    per_locale = {lang: count_rows(path) for lang, path in sorted(files.items())}
    empty = [lang for lang, rows in per_locale.items() if rows == 0]
    if empty:
        raise SystemExit(
            f"[publish] {root} has an empty {filename} for {empty}; "
            "remove the locale instead, an empty file claims a measurement that "
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


def upload(found: dict, repo_id: str, message: str) -> None:
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "[publish] HF_TOKEN is not set; the write token is not read from "
            "an argument or a file")
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(found["root"]), repo_id=repo_id,
                      repo_type="dataset", commit_message=message)
    print(f"[publish] uploaded {found['root']} -> {repo_id}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates-dir", required=True,
                    help="staged directory holding {lang}/train_templates.jsonl")
    ap.add_argument("--eval-dir", required=True,
                    help="staged directory holding {lang}/test.jsonl")
    ap.add_argument("--templates-repo", required=True,
                    help="dataset repo id for the training templates")
    ap.add_argument("--eval-repo", required=True,
                    help="dataset repo id for the gold evaluation rows")
    ap.add_argument("--expect-train-rows", type=int, default=None,
                    help="fail if the staged templates do not hold this many rows")
    ap.add_argument("--expect-test-rows", type=int, default=None,
                    help="fail if the staged eval rows do not hold this many rows")
    ap.add_argument("--commit-message", default="publish corpus",
                    help="commit message for both repos")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be uploaded, touch no network")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    templates = inspect(Path(args.templates_dir), TEMPLATES_FILE)
    gold = inspect(Path(args.eval_dir), EVAL_FILE)
    report("templates", templates, args.expect_train_rows)
    report("eval", gold, args.expect_test_rows)

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

    upload(templates, args.templates_repo, args.commit_message)
    upload(gold, args.eval_repo, args.commit_message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
