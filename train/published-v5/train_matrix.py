#!/usr/bin/env python3
"""Train one cell of the m2v classifier retrain matrix and evaluate it.

Mechanism (recovered from ovos_m2v_pipeline + model2vec.train):
  - ovos_m2v_pipeline's "classifier" mode loads a frozen
    StaticModelPipeline (potion embeddings + trained linear head) via
    StaticModelPipeline.from_pretrained(). That pipeline is produced by
    model2vec.train.StaticModelForClassification: load a potion
    StaticModel, .fit(X, y) a small (Linear->ReLU->Linear) head on top of
    the frozen(-ish) embeddings, then .to_pipeline().save_pretrained(dir).
  - Dataset files are Adapt/Padatious OVOS-INTENT-1 templates
    ("(a|b) [optional]"), expanded to literal utterances with
    ovos_spec_tools.expansion.expand before they are usable as X.

Usage (one cell):
    python train_matrix.py --base minishlab/potion-base-8M \
        --train /home/miro/tmp/m2v-dataset-v5/train_en.jsonl \
        --test /home/miro/tmp/m2v-dataset-v5/test_en.jsonl \
        --ood /home/miro/tmp/m2v-dataset-v5/ood_en.jsonl \
        --out ~/tmp/m2v-models-v5/en-8M

Resumable: a cell whose --out directory already contains model.safetensors
is skipped (treated as done) unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MAX_EXPANSIONS_PER_TEMPLATE = 20


def load_jsonl_expanded(path: str, cap: int = MAX_EXPANSIONS_PER_TEMPLATE) -> tuple[list[str], list[str]]:
    """Read a {label, text} jsonl of OVOS-INTENT-1 templates, expand each
    line's (a|b)/[optional] syntax to literal utterances (capped per line
    so a combinatorial template can't blow up the set), and return
    parallel (X, y) lists.
    """
    from ovos_spec_tools.expansion import iter_expand
    from itertools import islice

    X: list[str] = []
    y: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row["label"]
            template = row["text"]
            try:
                variants = list(islice(iter_expand(template), cap))
            except Exception as exc:
                print(f"WARN: skipping malformed template {template!r}: {exc}")
                continue
            if not variants:
                continue
            for v in variants:
                X.append(v)
                y.append(label)
    return X, y


def train_one(base: str, train_path: str, test_path: str, ood_path: str | None,
              out_dir: str, force: bool = False) -> dict:
    out = Path(out_dir).expanduser()
    marker = out / "model.safetensors"
    if marker.exists() and not force:
        print(f"SKIP {out} (already trained)")
        return {"skipped": True, "out": str(out)}

    from model2vec.train import StaticModelForClassification

    print(f"[{out.name}] loading base {base}")
    model = StaticModelForClassification.from_pretrained(base)

    print(f"[{out.name}] loading + expanding train set {train_path}")
    X_train, y_train = load_jsonl_expanded(train_path)
    print(f"[{out.name}] {len(X_train)} training utterances, "
          f"{len(set(y_train))} labels")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"[{out.name}] trained in {train_time:.1f}s")

    out.mkdir(parents=True, exist_ok=True)
    pipeline = model.to_pipeline()
    pipeline.save_pretrained(str(out))
    print(f"[{out.name}] saved -> {out}")

    result = {
        "base": base,
        "train_path": train_path,
        "out": str(out),
        "n_train_utterances": len(X_train),
        "n_labels": len(set(y_train)),
        "train_time_s": train_time,
    }

    for split_name, split_path in (("in_dist", test_path), ("ood", ood_path)):
        if not split_path:
            continue
        X_eval, y_eval = load_jsonl_expanded(split_path)
        preds = pipeline.predict(X_eval)
        correct = sum(1 for p, g in zip(preds, y_eval) if p == g)
        acc = correct / len(y_eval) if y_eval else 0.0
        result[f"{split_name}_accuracy"] = acc
        result[f"{split_name}_n"] = len(y_eval)
        print(f"[{out.name}] {split_name} accuracy: {acc:.4f} "
              f"({correct}/{len(y_eval)})")

    (out / "eval_report.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="potion base model id, e.g. minishlab/potion-base-8M")
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--ood", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    result = train_one(args.base, args.train, args.test, args.ood, args.out, force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
