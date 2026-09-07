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


def main(argv=None):
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
    ap.add_argument("--out", default=None)
    ap.add_argument("--push-to", default=None,
                    help="Hugging Face model repo id to upload the trained "
                         "pipeline to (e.g. OpenVoiceOS/ovos-m2v-intents-en); "
                         "combine with --dry-run to preview")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --push-to, print what would be uploaded "
                         "instead of uploading it; training still runs")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from model2vec.train import StaticModelForClassification
    from sklearn.metrics import (accuracy_score, classification_report,
                                 cohen_kappa_score, f1_score,
                                 matthews_corrcoef)

    dataset = Path(args.dataset)
    train, test = load(dataset, args.lang, args.family)
    tag = args.lang or "mul"
    out = Path(args.out or HERE / f"model_{tag}_{args.base_model.split('/')[-1]}")

    LOG.info(f"{len(train)} train / {len(test)} test rows, "
             f"{train['label'].nunique()} labels, base={args.base_model}")
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
