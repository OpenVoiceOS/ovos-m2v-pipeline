#!/usr/bin/env python3
"""Score a trained intent model on the fleet's own golden utterances.

The held-out number comes from each pinned skill's end-to-end gold files --
``test/end2end/golden_utterances[_<lang>].jsonl`` at the sha
``train/sources.yaml`` pins -- not from a set written for the purpose. The
design, with the measurements behind each choice, is
``knowledge/wiki/audits/m2v-training/golden-eval-design.md`` in the workspace.

A gold row is scored only when all of these hold:

* it asserts an intent (``intent_label``) and is not ``needs_manual``;
* its label resolves to a class the model predicts: exact, then
  ``RENAMED_LABELS``, then the unambiguous underscore fold. An unresolved row
  is listed, never scored as wrong: the model cannot emit a label it never
  trained on;
* its text is not a training row's text. ``build_from_skills.py`` removes
  gold text from the training side, but a model may have been trained on any
  corpus, so the eval reads the model's own training rows and refuses every
  gold row they contain, under the punctuation-insensitive fold the runtime
  normalizer applies. Without the training rows the eval does not run.

    golden_eval.py --model <dir> --train <train.jsonl|parquet> --out <dir>
    golden_eval.py --train <train.jsonl|parquet> --out <dir> --dry-run
"""
import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_from_skills import (normalize_utterance_punct_insensitive,  # noqa: E402
                               read_gold, resolve_gold_label)
from ovos_m2v_pipeline.renames import RENAMED_LABELS  # noqa: E402


def collect_gold(refs: dict, ws: Path, stats: collections.Counter):
    """Gold rows of every pinned skill at its pin, and the skills with none."""
    rows, without = [], []
    for key, rev in sorted(refs.items()):
        repo, repo_name = ws / key, key.rsplit("/", 1)[-1]
        if not repo.is_dir():
            stats["skill_missing_clone"] += 1
            without.append(repo_name)
            continue
        gold = read_gold(repo, rev, repo_name, stats)
        if not gold:
            without.append(repo_name)
        for row in gold:
            row["skill_ref"] = f"{key}@{rev}"
        rows.extend(gold)
    return rows, sorted(without)


def resolve_label(label: str, classes: set):
    """The model class a gold label names, or None."""
    if label in classes:
        return label
    renamed = RENAMED_LABELS.get(label)
    if renamed in classes:
        return renamed
    skill, _, intent = label.partition(":")
    shipped = {c.split(":", 1)[1] for c in classes if c.startswith(skill + ":")}
    folded = resolve_gold_label(intent, shipped)
    if folded != intent:
        return f"{skill}:{folded}"
    return None


def select(gold, classes: set, train_texts: set, stats: collections.Counter):
    """Rows the eval may score, and the rows it refuses with the reason."""
    scored, unresolved, refused, seen = [], collections.Counter(), [], set()
    for row in gold:
        label = resolve_label(row["label"], classes)
        if label is None:
            stats["unresolved_label"] += 1
            unresolved[row["label"]] += 1
            continue
        folded = normalize_utterance_punct_insensitive(row["utterance"])
        if folded in train_texts:
            stats["refused_train_overlap"] += 1
            refused.append({"lang": row["lang"], "label": label,
                            "utterance": row["utterance"]})
            continue
        key = (label, row["lang"], folded)
        if key in seen:
            stats["gold_duplicates"] += 1
            continue
        seen.add(key)
        scored.append({**row, "label": label})
    return scored, unresolved, refused


def _group(scored, predictions, key):
    out = collections.defaultdict(lambda: {"rows": 0, "correct": 0})
    for row, pred in zip(scored, predictions):
        g = out[key(row)]
        g["rows"] += 1
        g["correct"] += int(pred == row["label"])
    for g in out.values():
        g["accuracy"] = round(g["correct"] / g["rows"], 4)
    return dict(sorted(out.items()))


def evaluate(scored, predict):
    """Accuracy overall, per language, per skill and per label."""
    predictions = list(predict([r["utterance"] for r in scored])) if scored else []
    per_label = _group(scored, predictions, lambda r: r["label"])
    wrong = collections.defaultdict(collections.Counter)
    langs = collections.defaultdict(set)
    for row, pred in zip(scored, predictions):
        langs[row["label"]].add(row["lang"])
        if pred != row["label"]:
            wrong[row["label"]][pred] += 1
    for label, g in per_label.items():
        g["languages"] = sorted(langs[label])
        if wrong[label]:
            pred, n = wrong[label].most_common(1)[0]
            g["most_frequent_wrong"] = {"label": pred, "rows": n}
    correct = sum(int(p == r["label"]) for r, p in zip(scored, predictions))
    return {
        "rows_scored": len(scored),
        "correct": correct,
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "macro_accuracy_over_labels": (
            round(sum(g["accuracy"] for g in per_label.values()) / len(per_label), 4)
            if per_label else None),
        "per_language": _group(scored, predictions, lambda r: r["lang"]),
        "per_skill": _group(scored, predictions, lambda r: r["label"].split(":", 1)[0]),
        "per_label": per_label,
    }


def read_train_texts(path: Path) -> set:
    """Folded text of every training row the model saw."""
    if path.suffix == ".parquet":
        import pandas as pd
        utterances = pd.read_parquet(path, columns=["utterance"])["utterance"]
    else:
        utterances = [json.loads(l)["utterance"]
                      for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {normalize_utterance_punct_insensitive(u) for u in utterances}


def fleet_cross_check(harness: Path, gold) -> dict:
    """The vendored fleet copy against the pinned files (en-US, as the copy has no lang)."""
    fleet_dir = harness / "test" / "skills_fleet"
    if not (fleet_dir / "golden_utterances.jsonl").is_file():
        return {"harness": str(harness), "read": False}

    def load(name):
        p = fleet_dir / name
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                if l.strip()] if p.is_file() else []

    fold = normalize_utterance_punct_insensitive
    fleet = {(d["skill_id"], fold(d["utterance"])) for d in load("golden_utterances.jsonl")}
    quarantine = {(d["skill_id"], fold(d["utterance"])) for d in load("quarantine.jsonl")}
    pinned = {(r["label"].split(":", 1)[0], fold(r["utterance"]))
              for r in gold if r["lang"] == "en-US"}
    return {"harness": str(harness), "read": True,
            "fleet_rows_not_in_pinned_files": len(fleet - pinned),
            "pinned_en_rows_not_in_fleet_file": len(pinned - fleet),
            "pinned_rows_the_harness_quarantines": len(pinned & quarantine)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_markdown(report: dict) -> str:
    lines = ["# Golden evaluation", "",
             f"Model: `{report['model']}`. Train rows: `{report['train']}`.", "",
             f"Scored {report['rows_scored']} of {report['gold_rows_read']} gold rows: "
             f"accuracy {report['accuracy']}, macro over labels "
             f"{report['macro_accuracy_over_labels']}.", "",
             "| Not scored | Rows |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(report["excluded"].items())]
    for title, key in (("Language", "per_language"), ("Skill", "per_skill")):
        lines += ["", f"| {title} | Rows | Correct | Accuracy |", "|---|---:|---:|---:|"]
        lines += [f"| {k} | {g['rows']} | {g['correct']} | {g['accuracy']} |"
                  for k, g in report[key].items()]
    lines += ["", "| Label | Languages | Rows | Correct | Accuracy | Most frequent wrong |",
              "|---|---|---:|---:|---:|---|"]
    for k, g in report["per_label"].items():
        w = g.get("most_frequent_wrong")
        lines.append(f"| {k} | {', '.join(g['languages'])} | {g['rows']} | {g['correct']} | "
                     f"{g['accuracy']} | {w['label'] + ' x' + str(w['rows']) if w else ''} |")
    return "\n".join(lines) + "\n"


def main(argv=None, predictor=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=str(Path(__file__).resolve().parent / "sources.yaml"))
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--model", default=None, help="model2vec StaticModelPipeline directory")
    ap.add_argument("--train", required=True,
                    help="the training rows the model saw (train.jsonl or train.parquet)")
    ap.add_argument("--harness", default=None,
                    help="ovos-test-harness checkout, for the fleet-file cross-check")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-scored-rows", type=int, default=1,
                    help="fail when fewer gold rows are scored: a moved gold glob "
                         "scores nothing and would report a clean number")
    ap.add_argument("--dry-run", action="store_true",
                    help="read and filter the gold side against the training rows; "
                         "score nothing, load no model")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8"))
    ws = Path(args.workspace or cfg["workspace"]).expanduser()
    refs = cfg["skill_refs"]["refs"]
    stats = collections.Counter()
    gold, without = collect_gold(refs, ws, stats)
    train_path = Path(args.train)
    train_texts = read_train_texts(train_path)

    if args.dry_run:
        classes = {r["label"] for r in gold}
        predict = None
    else:
        if predictor is None:
            if not args.model:
                ap.error("--model is required unless --dry-run")
            from model2vec.inference import StaticModelPipeline
            model = StaticModelPipeline.from_pretrained(args.model)
            predictor = model
        classes = set(getattr(predictor, "classes_", []))
        predict = predictor.predict

    scored, unresolved, refused = select(gold, classes, train_texts, stats)
    result = evaluate(scored, predict) if predict else {
        "rows_scored": len(scored), "correct": None, "accuracy": None,
        "macro_accuracy_over_labels": None, "per_language": {}, "per_skill": {},
        "per_label": {}}
    labels_json = Path(args.model) / "labels.json" if args.model else None
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    report = {
        "pipeline_commit": commit or None,
        "model": args.model,
        "model_labels_json_sha256": sha256(labels_json) if labels_json and labels_json.is_file() else None,
        "train": str(train_path),
        "train_sha256": sha256(train_path),
        "skill_refs": refs,
        "gold_rows_read": len(gold) + stats["gold_needs_manual"]
                          + stats["gold_asserts_a_dialog_not_an_intent"]
                          + stats["gold_without_utterance"] + stats["gold_unparsable"],
        "excluded": {
            "needs_manual": stats["gold_needs_manual"],
            "no_intent_label": stats["gold_asserts_a_dialog_not_an_intent"],
            "no_utterance": stats["gold_without_utterance"],
            "unparsable": stats["gold_unparsable"],
            "unresolved_label": stats["unresolved_label"],
            "refused_train_overlap": stats["refused_train_overlap"],
            "gold_duplicates": stats["gold_duplicates"],
        },
        **result,
        "unresolved_labels": dict(unresolved.most_common()),
        "refused_train_overlap_rows": refused,
        "skills_without_gold": without,
        "fleet_cross_check": fleet_cross_check(Path(args.harness), gold) if args.harness else None,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "golden_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
    if predict:
        (out / "golden_eval.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("gold_rows_read", "excluded", "rows_scored",
                                             "accuracy", "macro_accuracy_over_labels")},
                     indent=2))
    if report["rows_scored"] < args.min_scored_rows:
        print(f"[gate] only {report['rows_scored']} gold rows scored, floor "
              f"{args.min_scored_rows}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
