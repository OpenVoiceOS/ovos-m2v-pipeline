#!/usr/bin/env python3
"""Build the classifier and the prototype artifact from one corpus, together.

The two artifacts describe the same intents in different ways: the classifier
carries a label head frozen at fit time, and the prototype artifact carries
centroids a registering skill's cache key unlocks. They are published as a
pair, so their label sets have to agree. Produced separately they drift, and
nothing downstream notices: a deployment loading one and reading the other's
label list would silently disagree about what the model knows.

This runs both producers against the same dataset directory into a staging
area, compares the label sets, and publishes only when they match. When they
do not, nothing is written and the failure names the labels that differ rather
than only reporting that they do.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dataset_identity(dataset: Path) -> str:
    """The sha256 of the dataset manifest, as the shared build identity.

    `manifest.json` already records the sha256 of every output and the
    revisions used, so hashing it names the corpus exactly. When a dataset
    ships no manifest the identity is empty rather than invented: a made-up
    id would pair two artifacts that nothing actually pairs.
    """
    manifest = dataset / "manifest.json"
    if not manifest.is_file():
        return ""
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def classifier_labels(model_dir: Path) -> set:
    return set(json.loads((model_dir / "labels.json").read_text())["valid_labels"])


def prototype_labels(artifact_dir: Path) -> set:
    return set(json.loads((artifact_dir / "manifest.json").read_text())["labels"])


def describe_drift(classifier: set, prototype: set) -> str:
    only_classifier = sorted(classifier - prototype)
    only_prototype = sorted(prototype - classifier)
    lines = [f"label sets differ: {len(classifier)} in the classifier, "
             f"{len(prototype)} in the prototype artifact"]
    if only_classifier:
        lines.append(f"  only in the classifier ({len(only_classifier)}): "
                     + ", ".join(only_classifier[:20])
                     + (" ..." if len(only_classifier) > 20 else ""))
    if only_prototype:
        lines.append(f"  only in the prototype artifact ({len(only_prototype)}): "
                     + ", ".join(only_prototype[:20])
                     + (" ..." if len(only_prototype) > 20 else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(HERE / "dataset"),
                    help="directory build_dataset.py wrote")
    ap.add_argument("--out", required=True,
                    help="directory to publish 'classifier' and 'prototypes' into")
    ap.add_argument("--classifier-base", default="minishlab/potion-base-32M")
    ap.add_argument("--prototype-base", default="minishlab/M2V_multilingual_output")
    ap.add_argument("--lang", default=None,
                    help="build one locale only, e.g. en-US")
    ap.add_argument("--prototype-k", type=int, default=None,
                    help="cap prototypes kept per label")
    args = ap.parse_args(argv)

    out = Path(args.out)
    staging = out.with_name(out.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    train_cmd = [sys.executable, str(HERE / "train.py"),
                 "--dataset", args.dataset,
                 "--base-model", args.classifier_base,
                 "--out", str(staging / "classifier")]
    export_cmd = [sys.executable, "-m", "ovos_m2v_pipeline.cli", "export",
                  "--from-dataset", args.dataset,
                  "--model", args.prototype_base,
                  "--out", str(staging / "prototypes")]
    if args.lang:
        train_cmd += ["--lang", args.lang]
        export_cmd += ["--lang", args.lang]
    if args.prototype_k is not None:
        export_cmd += ["--prototype-k", str(args.prototype_k)]

    def abandon(message: str, code: int) -> int:
        """Report, drop the staging tree, and return the exit code.

        A refused run used to leave `<out>.staging` behind holding whatever the
        producers wrote, which for a real corpus is a full classifier and a full
        prototype artifact. The next run deleted it, so nothing accumulated, but
        a rejected build sat on disk unannounced and looked like output.
        """
        print(message, file=sys.stderr)
        shutil.rmtree(staging, ignore_errors=True)
        return code

    for cmd in (train_cmd, export_cmd):
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return abandon(
                f"error: {' '.join(cmd)} exited {result.returncode}; "
                "nothing published", result.returncode)

    classifier = classifier_labels(staging / "classifier")
    prototype = prototype_labels(staging / "prototypes")
    if classifier != prototype:
        return abandon(
            "error: refusing to publish.\n" + describe_drift(classifier, prototype), 1)

    # CONFIRMED 2: stamp both sides with one identity. The driver proved the
    # label sets agreed at build time and then published two files with nothing
    # in common, so a consumer that loads a classifier from one place and a
    # prototype artifact from another could not tell they were built together.
    # The dataset manifest's sha256 is that identity: it names the corpus both
    # producers read.
    build_id = dataset_identity(Path(args.dataset))
    for side in ("classifier", "prototypes"):
        stamp_path = staging / side / "build.json"
        stamp_path.write_text(json.dumps(
            {"dataset_manifest_sha256": build_id, "sides": ["classifier", "prototypes"]},
            indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if out.exists():
        shutil.rmtree(out)
    staging.rename(out)
    print(f"published {len(classifier)} label(s) to {out} "
          "(classifier/ and prototypes/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
