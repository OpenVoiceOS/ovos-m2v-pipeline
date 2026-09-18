#!/usr/bin/env python3
"""Census of a built corpus: every test label must be trainable.

A gold row whose label has no train row teaches nothing and can never be
answered. The model has no class for it, so the row counts as a miss at
evaluation time whatever the model does. Such a label is always a naming
mismatch: the gold file and the train side spell the same intent two ways,
or one of them names an intent the skill does not register at all.

The census prints the mismatch and exits non-zero, so a build job can refuse
to publish on it.

    python train/census_gold_labels.py --dataset train/dataset

Exit status is 0 when every test label is present in train, 1 otherwise.
"""
import argparse
import collections
import json
import sys
from pathlib import Path


def read_labels(path: Path, by_lang: bool = False) -> collections.Counter:
    """Label counts; with ``by_lang`` the key is ``(lang, label)``."""
    counts: collections.Counter = collections.Counter()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: {exc}") from exc
            try:
                key = (row["lang"], row["label"]) if by_lang else row["label"]
            except KeyError as exc:
                raise SystemExit(f"{path}:{number}: row has no {exc}")
            counts[key] += 1
    return counts


def census_paths(train_path: Path, test_path: Path):
    """The gate on two files: every test label must have a train row."""
    train = read_labels(train_path)
    test = read_labels(test_path)
    missing = {label: test[label] for label in sorted(test) if label not in train}
    return train, test, missing


def census_per_locale(train_path: Path, test_path: Path):
    """The second gate: every (lang, label) on the test side must have a
    train row in the SAME locale.

    The flat census passes while a per-locale model still has no class for
    a gold row, when the locale's intent file holds only the sentences the
    gold file repeats: the gold-overlap removal takes every train row the
    gold matches, the template is silenced, and the locale keeps gold and
    no train. v6.1 carries 24 such pairs in 7 locales, every one that
    shape. The fix is in the skill (gold rows that are not template lines,
    or templates the gold does not copy), so this gate reports by default
    and refuses only with ``--per-locale``, until the fleet is clean and
    the default flips.
    """
    train = read_labels(train_path, by_lang=True)
    test = read_labels(test_path, by_lang=True)
    missing = {key: test[key] for key in sorted(test) if key not in train}
    return missing


def census(dataset: Path):
    return census_paths(dataset / "train.jsonl", dataset / "test.jsonl")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="directory holding train.jsonl and test.jsonl")
    parser.add_argument("--quiet", action="store_true",
                        help="print the mismatch only")
    parser.add_argument("--per-locale", action="store_true",
                        help="also refuse a test (lang, label) with no train row in "
                             "the same locale; without it the per-locale census is "
                             "reported and does not change the exit status")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset).expanduser()
    train, test, missing = census(dataset)
    per_locale = census_per_locale(dataset / "train.jsonl", dataset / "test.jsonl")

    if not args.quiet:
        print(f"train: {sum(train.values())} rows, {len(train)} labels")
        print(f"test:  {sum(test.values())} rows, {len(test)} labels")

    status = 0
    if not missing:
        if not args.quiet:
            print("every test label has train rows")
    else:
        rows = sum(missing.values())
        print(f"{len(missing)} test labels have no train rows, {rows} rows affected:")
        for label, count in sorted(missing.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {count:5d}  {label}")
        status = 1

    if not per_locale:
        if not args.quiet:
            print("every test (lang, label) has a train row in the same locale")
    else:
        rows = sum(per_locale.values())
        langs = sorted({lang for lang, _ in per_locale})
        print(f"{len(per_locale)} test (lang, label) pairs have no train row in the same "
              f"locale, {rows} rows in {len(langs)} locales"
              + ("" if args.per_locale else " (reported, not refused: pass --per-locale)") + ":")
        for (lang, label), count in sorted(per_locale.items(), key=lambda item: (item[0][0], -item[1], item[0][1])):
            print(f"  {count:5d}  {lang:8s} {label}")
        if args.per_locale:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
