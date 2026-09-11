#!/usr/bin/env python3
"""Validate a semantic evaluation set against the built corpus.

The train/test split is by template, so both halves are expansions of the same
`.intent` lines and a score on the test half measures memorisation of those
lines. The evaluation set under `train/eval/` is written independently of the
templates and must stay that way: an utterance that also appears in the corpus
turns the set back into a memorisation probe, so any overlap is an error.

Checks: no eval utterance appears in train or test, every corpus label is
covered, no label outside the corpus vocabulary appears, every label carries at
least MIN_ROWS rows, and no utterance is repeated.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

#: A label below this many rows is too thin to read a per-label score from.
MIN_ROWS = 8


def normalise(utterance: str) -> str:
    return re.sub(r"\s+", " ", utterance.strip()).casefold()


def read_eval(path: Path) -> list:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def corpus_utterances(corpus: Path) -> set:
    seen = set()
    for name in ("train.parquet", "test.parquet"):
        frame = pd.read_parquet(corpus / name, columns=["utterance"])
        seen.update(normalise(u) for u in frame["utterance"])
    return seen


def validate(eval_path: Path, corpus: Path) -> list:
    rows = read_eval(eval_path)
    valid = set(json.loads((corpus / "labels.json").read_text())["valid_labels"])
    in_corpus = corpus_utterances(corpus)

    errors = []
    counts = {}
    seen = {}
    for row in rows:
        label, utterance = row["label"], row["utterance"]
        key = normalise(utterance)
        counts[label] = counts.get(label, 0) + 1
        if label not in valid:
            errors.append(f"unknown label {label!r} on {utterance!r}")
        if key in in_corpus:
            errors.append(f"{utterance!r} ({label}) also appears in the corpus")
        if key in seen:
            errors.append(f"{utterance!r} is repeated ({seen[key]} and {label})")
        seen[key] = label

    for label in sorted(valid - set(counts)):
        errors.append(f"no rows for label {label}")
    for label, count in sorted(counts.items()):
        if count < MIN_ROWS:
            errors.append(f"{label} has {count} rows, fewer than {MIN_ROWS}")

    print(f"{len(rows)} rows over {len(counts)} labels")
    kinds = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items())))
    for label, count in sorted(counts.items()):
        near = sum(1 for r in rows if r["label"] == label and r["kind"] == "near-miss")
        print(f"  {label}: {count} ({near} near-miss)")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=HERE / "eval" / "en-US.jsonl",
                        help="evaluation set to validate")
    parser.add_argument("--corpus", type=Path, required=True,
                        help="directory holding train.parquet, test.parquet and labels.json")
    args = parser.parse_args()

    errors = validate(args.eval, args.corpus)
    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
