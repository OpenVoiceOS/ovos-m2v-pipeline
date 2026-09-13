#!/usr/bin/env python3
"""Distil the sentence encoders in ``backbones.yaml`` into Model2Vec models.

A Model2Vec model is a static embedding table, so distilling one is cheap and
the result is small enough to ship per language: a base-sized encoder becomes a
few tens of megabytes and loads without a neural runtime.

Every backbone is distilled from a pinned Hugging Face revision. ``--resolve``
fills an empty ``revision`` in the manifest from the Hub and writes it back, so
a rebuild months later reads the same weights rather than whatever the branch
points at.

Each distilled model is written to ``<out>/<backbone id>`` beside a
``distill.json`` recording the source model, its revision, the embedding
dimension, the vocabulary size and the model2vec version that produced it. A
backbone whose directory already carries a matching ``distill.json`` is skipped,
so an interrupted run resumes instead of starting again.

    python train/distill.py --lang gl              # every backbone for one language
    python train/distill.py --all --out distilled  # everything in the manifest
    python train/distill.py --resolve              # pin revisions, distil nothing
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import yaml

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "backbones.yaml"
LOG = logging.getLogger("distill")


def load_manifest(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_revision(model: str) -> str:
    """Return the current commit sha of a Hugging Face model repository."""
    from huggingface_hub import HfApi
    return HfApi().model_info(model).sha


def pin_revisions(path: Path) -> int:
    """Fill every empty ``revision`` in the manifest. Returns how many changed."""
    text = path.read_text()
    manifest = yaml.safe_load(text)
    pinned = 0
    for entry in manifest["backbones"]:
        if entry.get("revision"):
            continue
        sha = resolve_revision(entry["model"])
        # rewrite in place so the file keeps its comments and ordering
        text = text.replace(f"    model: {entry['model']}\n    revision:\n",
                            f"    model: {entry['model']}\n    revision: {sha}\n")
        LOG.info(f"pinned {entry['id']} at {sha}")
        pinned += 1
    if pinned:
        path.write_text(text)
    return pinned


def selected_backbones(manifest: dict, lang: Optional[str],
                       multilingual: bool = True) -> list[dict]:
    """The backbones to build: one language's candidates, or all of them.

    A language's candidates are its own backbones plus every multilingual one,
    because the multilingual model is the baseline each language-specific
    backbone has to beat before it is worth shipping. Drop the baseline with
    ``multilingual=False`` when it is already built, or when the box cannot
    hold it: a multilingual vocabulary is an order of magnitude larger than a
    single language's, and the distillation peaks with the whole vocabulary
    encoded in memory.
    """
    if not lang:
        return list(manifest["backbones"])
    # a backbone may name a language (``pt``) or a locale (``pt-PT``); a
    # request for a locale takes both, a request for a language takes only its
    # own, since pt-PT and pt-BR are different training targets rather than
    # spellings of one
    base = lang.split("-")[0]
    out = [b for b in manifest["backbones"]
           if lang in b["languages"] or base in b["languages"]
           or (multilingual and "mul" in b["languages"])]
    if not out:
        raise SystemExit(f"no backbone in {MANIFEST.name} serves lang {lang!r}")
    return out


def already_built(target: Path, entry: dict) -> bool:
    record = target / "distill.json"
    if not record.is_file():
        return False
    built = json.loads(record.read_text())
    return (built.get("model") == entry["model"]
            and built.get("revision") == entry.get("revision"))


def distil_one(entry: dict, out_dir: Path, trust_remote_code: bool = False,
               max_vocab: Optional[int] = None) -> Optional[Path]:
    """Distil one backbone. Returns its directory, or None when it was skipped."""
    target = out_dir / entry["id"]
    if already_built(target, entry):
        LOG.info(f"{entry['id']}: already built at this revision")
        return None
    if entry.get("distilled") is False:
        LOG.info(f"{entry['id']}: already a Model2Vec model, nothing to distil")
        return None

    from model2vec import __version__ as m2v_version
    from model2vec.distill import distill_from_model
    from transformers import AutoModel, AutoTokenizer

    revision = entry.get("revision")
    if not revision:
        raise SystemExit(f"{entry['id']} has no revision; run --resolve first")
    if entry.get("custom_code") and not trust_remote_code:
        LOG.warning(f"{entry['id']}: skipped, its repository ships code that "
                    f"loading the model executes; pass --trust-remote-code "
                    f"once you have read {entry['model']}")
        return None
    # model2vec's own `distill` takes a model name and resolves it to whatever
    # the branch points at, so the encoder is loaded here at the pinned
    # revision and handed over already materialised.
    try:
        tokenizer = AutoTokenizer.from_pretrained(entry["model"], revision=revision,
                                                  trust_remote_code=trust_remote_code)
    except Exception as exc:
        # a backbone the installed transformers cannot load is one backbone's
        # problem; the rest of the manifest still builds
        LOG.warning(f"{entry['id']}: skipped, its tokenizer does not load "
                    f"({type(exc).__name__}: {exc})")
        return None
    # the vocabulary is what distillation costs: every token is encoded, and
    # peak memory follows its size rather than the encoder's parameter count. A
    # name is no guide, since an encoder called "small" can carry a multilingual
    # vocabulary an order of magnitude larger than a base model's.
    vocab = len(tokenizer.get_vocab())
    if max_vocab and vocab > max_vocab:
        LOG.warning(f"{entry['id']}: skipped, vocabulary of {vocab} tokens is "
                    f"over the --max-vocab of {max_vocab}")
        return None
    LOG.info(f"{entry['id']}: distilling {vocab} tokens")
    try:
        encoder = AutoModel.from_pretrained(entry["model"], revision=revision,
                                            trust_remote_code=trust_remote_code)
    except Exception as exc:
        LOG.warning(f"{entry['id']}: skipped, its weights do not load "
                    f"({type(exc).__name__}: {exc})")
        return None
    model = distill_from_model(model=encoder, tokenizer=tokenizer)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(target))
    (target / "distill.json").write_text(json.dumps({
        "backbone_id": entry["id"],
        "model": entry["model"],
        "revision": revision,
        "languages": entry["languages"],
        "license": entry.get("license"),
        "embedding_dim": int(model.dim),
        "vocab_size": len(model.tokens),
        "model2vec_version": m2v_version,
    }, indent=2, sort_keys=True) + "\n")
    LOG.info(f"{entry['id']}: dim={model.dim} vocab={len(model.tokens)} -> {target}")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--out", type=Path, default=HERE / "distilled")
    parser.add_argument("--lang", help="build the backbones serving this language")
    parser.add_argument("--all", action="store_true", help="build every backbone")
    parser.add_argument("--max-vocab", type=int, default=None,
                        help="skip a backbone whose vocabulary is larger than "
                             "this, since peak memory follows vocabulary size")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="load backbones whose repository ships code that "
                             "is executed when the model loads")
    parser.add_argument("--no-multilingual", action="store_true",
                        help="with --lang, skip the multilingual baselines")
    parser.add_argument("--resolve", action="store_true",
                        help="pin every empty revision from the Hub and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.resolve:
        LOG.info(f"pinned {pin_revisions(args.manifest)} revision(s)")
        return
    if not args.lang and not args.all:
        raise SystemExit("pass --lang <code> or --all")

    manifest = load_manifest(args.manifest)
    built = 0
    for entry in selected_backbones(manifest, args.lang,
                                    multilingual=not args.no_multilingual):
        if distil_one(entry, args.out, args.trust_remote_code,
                      args.max_vocab) is not None:
            built += 1
    LOG.info(f"{built} backbone(s) built into {args.out}")


if __name__ == "__main__":
    main()
