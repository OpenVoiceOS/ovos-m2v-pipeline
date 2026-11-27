#!/usr/bin/env python3
"""Train a Model2Vec intent classifier from a corpus built by build_dataset.py.

The corpus already carries the train/test split, the language column and the
per-row provenance, so this script only picks a slice, fits, scores, and
writes the model out together with the `labels.json` manifest the pipeline
reads (m2v#73).

Training is on hold until the Adapt-to-`.intent` refactors merge; see
`docs/training.md`.
"""
import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LOG = logging.getLogger("train")


def load(dataset: Path, lang: str | None, family: list[str] | None):
    train = pd.read_parquet(dataset / "train.parquet")
    test = pd.read_parquet(dataset / "test.parquet")
    if lang:
        train = train[train["lang"] == lang]
        test = test[test["lang"] == lang]
    if family:
        train = train[train["family"].isin(family)]
        test = test[test["family"].isin(family)]
    if train.empty:
        raise SystemExit("empty training slice - check --lang / --family")
    # a class the test slice cannot score is still worth learning, but a class
    # absent from training must not appear in the test set
    test = test[test["label"].isin(set(train["label"]))]
    return train, test


def cap_per_label(train: pd.DataFrame, max_per_label: int, seed: int) -> tuple[pd.DataFrame, dict]:
    """Downsample every label with more than *max_per_label* rows.

    Sampling is stratified by ``lang`` within the label, so a multilingual
    label keeps its language mix: each language present gets a proportional
    share of the cap (largest-remainder rounding), with at least one row per
    language when the cap allows it. Labels at or below the cap are left
    untouched. Sampling is deterministic for a given *seed*.
    """
    before = train["label"].value_counts().to_dict()
    parts = []
    for label, group in train.groupby("label", sort=False):
        if len(group) <= max_per_label:
            parts.append(group)
            continue
        lang_counts = group["lang"].value_counts().sort_index()
        alloc = {lang: 0 for lang in lang_counts.index}
        remaining = max_per_label
        if max_per_label >= len(lang_counts):
            for lang in lang_counts.index:
                alloc[lang] = 1
            remaining -= len(lang_counts)
        total = int(lang_counts.sum())
        raw_share = {lang: remaining * (count / total) for lang, count in lang_counts.items()}
        for lang, share in raw_share.items():
            alloc[lang] += int(share)
        leftover = remaining - sum(int(v) for v in raw_share.values())
        by_remainder = sorted(raw_share.items(),
                              key=lambda kv: (-(kv[1] - int(kv[1])), kv[0]))
        i = 0
        while leftover > 0 and by_remainder:
            lang = by_remainder[i % len(by_remainder)][0]
            alloc[lang] += 1
            leftover -= 1
            i += 1
        # never allocate more than a language actually has
        for lang in alloc:
            alloc[lang] = min(alloc[lang], int(lang_counts[lang]))
        shortfall = min(max_per_label, len(group)) - sum(alloc.values())
        if shortfall > 0:
            for lang in lang_counts.index:
                room = int(lang_counts[lang]) - alloc[lang]
                take = min(room, shortfall)
                alloc[lang] += take
                shortfall -= take
                if shortfall <= 0:
                    break
        # sort by content, not position, so the same corpus samples the same
        # rows for a given seed no matter what row order it arrives in
        sampled = [group[group["lang"] == lang]
                  .sort_values(["utterance", "source"], kind="stable")
                  .reset_index(drop=True)
                  .sample(n=k, random_state=seed)
                  for lang, k in alloc.items() if k > 0]
        parts.append(pd.concat(sampled))
    capped = pd.concat(parts).reset_index(drop=True)
    after = capped["label"].value_counts().to_dict()
    return capped, {"max_per_label": max_per_label, "seed": seed, "before": before, "after": after}


def print_cap_table(report: dict) -> None:
    print(f"[cap] --max-per-label {report['max_per_label']} --seed {report['seed']}")
    print(f"{'label':<40} {'before':>8} {'after':>8}")
    for label in sorted(report["before"]):
        before = report["before"][label]
        after = report["after"].get(label, 0)
        print(f"{label:<40} {before:>8} {after:>8}")


def push_model(out: Path, dataset: Path, repo_id: str, dry_run: bool = False) -> None:
    """Upload the trained pipeline at *out* to a Hugging Face model repo.

    Uploads the whole *out* tree (`skops`/safetensors/config.json at the
    root, `onnx/` alongside if present -- the layout the published
    `OpenVoiceOS/ovos-m2v-intents-*` repos already use) plus a
    `training_manifest.json` recording the dataset's own per-file sha256
    (read back from `dataset/manifest.json`, written by `build_dataset.py`)
    and the `model2vec` version training ran with, so a published model
    always names the exact corpus and library version that produced it.
    `upload_folder` creates or updates files in one commit and never
    deletes anything else already in the repo. The token comes from the
    `HF_TOKEN` environment variable, never hardcoded.
    """
    import model2vec

    dataset_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    training_manifest = {
        "dataset_shas": dataset_manifest.get("outputs", {}),
        "model2vec_version": model2vec.__version__,
    }
    (out / "training_manifest.json").write_text(
        json.dumps(training_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    if dry_run:
        print(f"[push] would upload {out} -> model repo {repo_id!r} (dry run):")
        for p in sorted(out.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(out)} ({p.stat().st_size} bytes)")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model",
                      commit_message=f"train: {out.name}")
    print(f"[push] uploaded {out} -> {repo_id}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(HERE / "dataset"),
                    help="directory build_dataset.py wrote")
    ap.add_argument("--base-model", default="minishlab/potion-base-32M")
    ap.add_argument("--lang", default=None,
                    help="train on one locale only, e.g. en-US")
    ap.add_argument("--family", action="append", default=None,
                    help="restrict to a label family (repeatable)")
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--max-per-label", type=int, default=None,
                    help="downsample any label with more rows than this to "
                         "this many, stratified by lang, after the --lang/"
                         "--family slice and before fitting")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for --max-per-label sampling")
    ap.add_argument("--out", default=None)
    ap.add_argument("--push-to", default=None,
                    help="Hugging Face model repo id to upload the trained "
                         "pipeline to (e.g. OpenVoiceOS/ovos-m2v-intents-en); "
                         "combine with --dry-run to preview")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the training summary (including the "
                         "--max-per-label table, if given) without fitting "
                         "or uploading")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    dataset = Path(args.dataset)
    train, test = load(dataset, args.lang, args.family)
    tag = args.lang or "mul"
    out = Path(args.out or HERE / f"model_{tag}_{args.base_model.split('/')[-1]}")

    cap_report = None
    if args.max_per_label is not None:
        train, cap_report = cap_per_label(train, args.max_per_label, args.seed)
        print_cap_table(cap_report)

    LOG.info(f"{len(train)} train / {len(test)} test rows, "
             f"{train['label'].nunique()} labels, base={args.base_model}")

    if args.dry_run:
        if cap_report:
            out.mkdir(parents=True, exist_ok=True)
            (out / "label_cap_report.json").write_text(
                json.dumps(cap_report, indent=2), encoding="utf-8")
        return 0

    from model2vec.train import StaticModelForClassification
    from sklearn.metrics import (accuracy_score, classification_report,
                                 cohen_kappa_score, f1_score,
                                 matthews_corrcoef)

    clf = StaticModelForClassification.from_pretrained(model_name=args.base_model)
    clf.fit(train["utterance"].tolist(), train["label"].tolist(),
            max_epochs=args.max_epochs)
    pipeline = clf.to_pipeline()
    pipeline.save_pretrained(str(out))

    labels = sorted(train["label"].unique())
    (out / "labels.json").write_text(
        json.dumps({"valid_labels": labels}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    shutil.copy(dataset / "manifest.json", out / "dataset_manifest.json")

    y_true = test["label"].tolist()
    y_pred = pipeline.predict(test["utterance"].tolist())
    metrics = {
        "base_model": args.base_model,
        "lang": args.lang, "family": args.family,
        "n_train": len(train), "n_test": len(test), "n_labels": len(labels),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if cap_report:
        metrics["label_cap"] = cap_report
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, zero_division=0), encoding="utf-8")
    # golden-only slice: the rows generated against a skill's live registration
    golden = test[test["source"].str.startswith("golden:")]
    if not golden.empty:
        metrics["accuracy_golden_slice"] = accuracy_score(
            golden["label"].tolist(),
            pipeline.predict(golden["utterance"].tolist()))
        (out / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if args.push_to:
        push_model(out, dataset, args.push_to, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
