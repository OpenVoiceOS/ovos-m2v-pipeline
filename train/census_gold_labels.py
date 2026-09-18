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


def read_labels(path: Path) -> collections.Counter:
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
                counts[row["label"]] += 1
            except KeyError:
                raise SystemExit(f"{path}:{number}: row has no 'label'")
    return counts


def census(dataset: Path):
    train = read_labels(dataset / "train.jsonl")
    test = read_labels(dataset / "test.jsonl")
    missing = {label: test[label] for label in sorted(test) if label not in train}
    return train, test, missing


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True,
                        help="directory holding train.jsonl and test.jsonl")
    parser.add_argument("--quiet", action="store_true",
                        help="print the mismatch only")
    args = parser.parse_args(argv)

    dataset = Path(args.dataset).expanduser()
    train, test, missing = census(dataset)

    if not args.quiet:
        print(f"train: {sum(train.values())} rows, {len(train)} labels")
        print(f"test:  {sum(test.values())} rows, {len(test)} labels")

    if not missing:
        if not args.quiet:
            print("every test label has train rows")
        return 0

    rows = sum(missing.values())
    print(f"{len(missing)} test labels have no train rows, {rows} rows affected:")
    for label, count in sorted(missing.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {count:5d}  {label}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
