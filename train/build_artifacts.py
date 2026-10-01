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

#: The one backbone both sides are built on by default. The classifier and
#: the prototypes are compared label by label and published as a pair, so a
#: pair built on two different embedding spaces is a pair in name only. The
#: three entry points here used to default to three different models:
#: `--classifier-base` to potion-base-32M, `--prototype-base` to
#: M2V_multilingual_output, and the exporter's own `--model` to
#: OpenVoiceOS/ovos-m2v-intents-multilingual. The corpus is multilingual, so
#: the multilingual backbone is the one default. The exporter's own default
#: is left alone: it is that package's public interface, not this driver's,
#: and the driver names the model on every call.
DEFAULT_BACKBONE = "minishlab/M2V_multilingual_output"


def canonical_json(value) -> str:
    """One serialisation for one value, so a hash over it is stable."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def dataset_identity(dataset: Path) -> str:
    """The sha256 of the dataset manifest, as the shared build identity.

    `manifest.json` already records the sha256 of every output and the
    revisions used, so hashing it names the corpus exactly. When a dataset
    ships no manifest the identity is empty rather than invented: a made-up
    id would pair two artifacts that nothing actually pairs.

    The hash is taken over the manifest PARSED AND RE-SERIALISED in one
    canonical form, the same form the stamp below is written in. Hashing the
    raw bytes renames the corpus when a writer changes its indent or its key
    order, and two builds of one dataset would then refuse to look like one
    dataset. A manifest that is not JSON has no canonical form, so it takes
    the empty identity a missing manifest takes, for the same reason: an id
    that cannot be reproduced pairs nothing.
    """
    manifest = dataset / "manifest.json"
    if not manifest.is_file():
        return ""
    try:
        parsed = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ""
    return hashlib.sha256(canonical_json(parsed).encode("utf-8")).hexdigest()


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
    ap.add_argument("--classifier-base", default=DEFAULT_BACKBONE,
                    help="the backbone the classifier trains on "
                         f"(default {DEFAULT_BACKBONE})")
    ap.add_argument("--prototype-base", default=DEFAULT_BACKBONE,
                    help="the backbone the prototypes are built with "
                         f"(default {DEFAULT_BACKBONE})")
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

    published = False
    try:
        for cmd in (train_cmd, export_cmd):
            result = subprocess.run(cmd)
            if result.returncode != 0:
                return abandon(
                    f"error: {' '.join(cmd)} exited {result.returncode}; "
                    "nothing published", result.returncode)

        try:
            classifier = classifier_labels(staging / "classifier")
            prototype = prototype_labels(staging / "prototypes")
        except (OSError, KeyError, json.JSONDecodeError) as err:
            # A producer that exits 0 and writes no labels.json, or writes one
            # this driver cannot read, used to raise straight out of main()
            # past every cleanup and leave the staging tree on disk. It is the
            # same non-publishing exit as a refusal and it ends the same way.
            return abandon(
                f"error: a producer exited 0 but its labels could not be "
                f"read ({type(err).__name__}: {err}); nothing published", 1)

        if classifier != prototype:
            return abandon(
                "error: refusing to publish.\n" + describe_drift(classifier, prototype), 1)

        # Stamp both sides with one identity. The driver proved the label
        # sets agreed at build time and then published two files with
        # nothing in common, so a consumer that loads a classifier from one
        # place and a prototype artifact from another could not tell they
        # were built together. The dataset manifest's sha256 is that
        # identity: it names the corpus both producers read.
        build_id = dataset_identity(Path(args.dataset))
        stamp = {"dataset_manifest_sha256": build_id,
                 "sides": ["classifier", "prototypes"],
                 # which backbone each side was built on, so a reader of one
                 # published pair can tell without re-running anything
                 "backbones": {"classifier": args.classifier_base,
                               "prototypes": args.prototype_base}}
        for side in ("classifier", "prototypes"):
            stamp_path = staging / side / "build.json"
            stamp_path.write_text(json.dumps(stamp, indent=2, sort_keys=True)
                                  + "\n", encoding="utf-8")

        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
        published = True
        print(f"published {len(classifier)} label(s) to {out} "
              "(classifier/ and prototypes/)")
        return 0
    finally:
        # Every exit that does not publish drops the staging tree, including
        # one this driver did not foresee: a raise anywhere above, a
        # KeyboardInterrupt, a write that fails while stamping. abandon()
        # already removed it on the paths it knows; rmtree is idempotent
        # under ignore_errors, and after a successful rename the path is
        # gone, so the guard costs nothing on the happy path.
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
