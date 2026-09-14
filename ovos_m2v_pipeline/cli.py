"""``ovos-m2v-prototypes`` command line entry point.

Prebuilds a prototype-mode artifact (see ``ovos_m2v_pipeline.prebuilt``)
without booting OVOS, so a developer can generate it once on a desktop and
ship the result to a resource-constrained device.
"""
import argparse
import os
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, List, Optional

from ovos_utils.log import LOG


def _discover_intent_files(lang_dir: Path) -> List[Path]:
    """Find every ``.intent`` file under a skill's per-language directory.

    A skill lays its intent templates out either flat
    (``locale/<lang>/foo.intent``) or nested under a resource subdirectory
    (``locale/<lang>/intents/foo.intent``, or the legacy ``vocab/`` scheme) --
    ``ovos_workshop.resource_files.ResourceFile._locate`` resolves a single
    named resource the same way, by walking the whole language directory
    rather than assuming a flat layout. Mirrors that walk order (top-down,
    sorted per directory) so discovery here always matches what a live
    registration would find, deduplicated and in a deterministic order.
    """
    seen = set()
    found: List[Path] = []
    for directory, dirnames, filenames in os.walk(lang_dir):
        dirnames.sort()
        for name in sorted(filenames):
            if name.endswith(".intent"):
                path = Path(directory, name)
                if path not in seen:
                    seen.add(path)
                    found.append(path)
    return found


def _build_from_skill_dir(
    skill_dir: Path, skill_id: str, model_id: str, lang: Optional[str],
    k: Optional[int], strategy: str,
) -> "tuple":
    """Encode a skill's ``locale/<lang>/*.intent`` templates into a store.

    Mirrors the expansion ``padatious:register_intent`` runs at registration
    time (``ovos_m2v_pipeline._parse_intent_file`` / ``_raw_intent_lines``),
    without needing a running OVOS instance or its entity registrations --
    a ``{slot}`` placeholder in a template is left as literal text, so a
    skill whose intents rely heavily on registered entities gets weaker
    prototypes from this path than a live registration would; its cache key
    will then simply miss on boot and the pipeline re-encodes normally,
    exactly like any other registration-input change.
    """
    import model2vec
    from model2vec import StaticModel

    from ovos_m2v_pipeline import (
        PrototypeIntentStore, _parse_intent_file, _raw_intent_lines,
        MAX_ENTITY_EXPANSIONS,
    )
    from ovos_m2v_pipeline.cache import compute_cache_key
    from ovos_m2v_pipeline.strategies import PrototypeStrategy

    model = StaticModel.from_pretrained(model_id)
    model2vec_version = getattr(model2vec, "__version__", "")
    store = PrototypeIntentStore(strategy=PrototypeStrategy(strategy))

    locale_root = skill_dir / "locale"
    if not locale_root.is_dir():
        raise FileNotFoundError(f"no 'locale' directory under {skill_dir}")

    langs = [lang] if lang else sorted(
        p.name for p in locale_root.iterdir() if p.is_dir())
    if not langs:
        raise FileNotFoundError(f"no locale subdirectories under {locale_root}")

    cache_keys: Dict[str, str] = {}
    for one_lang in langs:
        lang_dir = locale_root / one_lang
        if not lang_dir.is_dir():
            LOG.warning(f"no locale directory for '{one_lang}' under {locale_root}")
            continue
        for intent_file in _discover_intent_files(lang_dir):
            intent_name = intent_file.stem
            label = f"{skill_id}:{intent_name}"
            raw_samples = _raw_intent_lines(str(intent_file))
            sentences = _parse_intent_file(str(intent_file))
            if not sentences:
                LOG.warning(f"skipping empty/malformed '{intent_file}'")
                continue
            cache_key = compute_cache_key(
                model_id, model2vec_version,
                {"k": k, "strategy": strategy,
                 "max_expansions": MAX_ENTITY_EXPANSIONS},
                raw_samples, lang=one_lang,
            )
            store.add(model, label, sentences, k=k, cache_key=cache_key,
                      lang=one_lang)
            cache_keys[label] = cache_key
    return store, model_id, model2vec_version, cache_keys


def _build_from_cache(cache_dir: Path, model_id: str, model2vec_version: str,
                       strategy: str) -> "tuple":
    """Fold an existing on-disk ``PrototypeCache`` directory into a store.

    Every registration a running install already encoded (and cached) is
    reusable as-is: this just concatenates the per-label ``.npz`` entries
    without touching the encoder again.
    """
    import numpy as np

    from ovos_m2v_pipeline import PrototypeIntentStore
    from ovos_m2v_pipeline.strategies import PrototypeStrategy

    store = PrototypeIntentStore(strategy=PrototypeStrategy(strategy))
    cache_keys: Dict[str, str] = {}
    for npz_file in sorted(cache_dir.glob("*/*.npz")):
        skill_id = npz_file.parent.name
        stem = npz_file.stem
        lang = None
        intent_name = stem
        if "@" in stem:
            intent_name, _, lang = stem.rpartition("@")
        label = f"{skill_id}:{intent_name}"
        try:
            data = np.load(npz_file, allow_pickle=False)
            embeddings = data["embeddings"].astype(np.float32)
            key = str(data["key"].item())
        except (OSError, ValueError, KeyError) as exc:
            LOG.warning(f"skipping unreadable cache entry '{npz_file}': {exc}")
            continue
        store._add_anchors(store._compose(label, lang), embeddings)
        cache_keys[label] = key
    return store, model_id, model2vec_version, cache_keys


def _cmd_export(args: argparse.Namespace) -> int:
    from ovos_m2v_pipeline.version import __version__ as plugin_version

    if args.skill_dir:
        store, model_id, model2vec_version, cache_keys = _build_from_skill_dir(
            Path(args.skill_dir), args.skill_id, args.model, args.lang,
            args.prototype_k, args.prototype_strategy,
        )
    elif args.from_cache:
        import model2vec
        store, model_id, model2vec_version, cache_keys = _build_from_cache(
            Path(args.from_cache), args.model,
            getattr(model2vec, "__version__", ""), args.prototype_strategy,
        )
    else:
        print("error: one of --skill-dir or --from-cache is required",
              file=sys.stderr)
        return 2

    if args.labels:
        wanted = set(args.labels)
        for label in set(store.unique_labels) - wanted:
            store.remove(label)
        cache_keys = {k: v for k, v in cache_keys.items() if k in wanted}

    if len(store.unique_labels) == 0:
        print("error: no labels found to export "
              "(no readable '.intent' files under the given source)",
              file=sys.stderr)
        return 1

    store.export(
        args.out, model_id=model_id, model2vec_version=model2vec_version,
        cache_keys=cache_keys, plugin_version=plugin_version,
    )
    print(f"exported {len(store.unique_labels)} label(s) -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ovos-m2v-prototypes",
        description="Prebuild/export ovos-m2v-pipeline prototype-mode "
                     "centroids as a shareable artifact.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser(
        "export", help="build and export a prebuilt prototype artifact")
    export.add_argument("--out", required=True,
                         help="output directory for the artifact")
    export.add_argument("--model", default="OpenVoiceOS/ovos-m2v-intents-multilingual",
                         help="Model2Vec repo id or local path the "
                              "centroids are built with")
    export.add_argument("--skill-dir",
                         help="path to a skill's repo (containing a "
                              "'locale/<lang>/*.intent' tree)")
    export.add_argument("--skill-id",
                         help="skill id to prefix labels with "
                              "(required with --skill-dir)")
    export.add_argument("--lang", default=None,
                         help="only export this locale from --skill-dir "
                              "(default: every locale under 'locale/')")
    export.add_argument("--from-cache",
                         help="path to an existing PrototypeCache directory "
                              "(prototype_cache_dir) to export as-is")
    export.add_argument("--labels", nargs="*", default=None,
                         help="only export these canonical labels "
                              "('skill_id:intent'); default: all")
    export.add_argument("--prototype-k", dest="prototype_k", type=int,
                         default=None, help="cap prototypes kept per label")
    export.add_argument("--prototype-strategy", dest="prototype_strategy",
                         default="max_over_all",
                         help="anchor-selection strategy (default: "
                              "max_over_all)")
    export.set_defaults(func=_cmd_export)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "export" and bool(args.skill_dir) == bool(args.from_cache):
        parser.error("pass exactly one of --skill-dir or --from-cache")
    if args.command == "export" and args.skill_dir and not args.skill_id:
        parser.error("--skill-id is required with --skill-dir")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
