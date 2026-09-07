"""On-disk cache for prototype embeddings.

Encoding every registered skill's example utterances on every boot is the
whole cost this module exists to avoid: for inputs that have not changed
since the last run (same model, same templates, same entity values), the
embeddings are read back from a small ``.npz`` file instead of being
recomputed by ``model.encode()``.

The cache is keyed on the *inputs* to a registration (model id, model2vec
version, the strategy/anchor-selection parameters, and the raw pre-expansion
sample lines), never on the resulting embeddings themselves, so a changed
template, a bumped model, or a different anchor-selection setting is a plain
cache miss that falls through to normal ingest -- no separate invalidation
logic is needed for the common case of "something about this registration
changed". A cached entry whose embedding dimension disagrees with the
currently-loaded model (e.g. the artifact behind an unchanged model id was
retrained in place) is likewise treated as a miss: loading it in regardless
would poison the live store with vectors of the wrong shape.

Removal (``remove()`` / ``remove_skill()``) is the one case that needs an
explicit delete: nothing about the registration inputs changes when a skill
unloads, so a stale on-disk entry would otherwise resurrect a removed
skill's intents on the next boot.

All reads/writes are local disk only; this module never touches the network.
"""
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ovos_utils.log import LOG

#: Characters allowed verbatim in a cache path component; everything else
#: (notably ``:`` in ``skill_id:intent_name`` labels) is folded to ``_``.
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]")


def compute_cache_key(
    model_id: str,
    model2vec_version: str,
    params: Dict[str, Any],
    raw_samples: List[str],
    entity_values: Optional[Dict[str, List[str]]] = None,
    lang: Optional[str] = None,
) -> str:
    """Hash the inputs of a prototype registration into a cache key.

    ``raw_samples`` are sorted before hashing: they are the pre-expansion
    template lines, which are already stable, but ``entity_values`` (the
    live-registered entity value sets, sourced from a ``set`` upstream) are
    NOT stable in iteration order across runs -- sorting both here is what
    makes the key reproducible across restarts for otherwise-identical
    inputs.

    ``lang`` is the registration's BCP-47 tag (``None`` for a caller that
    does not partition by language). It is folded into the hash -- on top
    of the on-disk path already being per-language (see
    ``PrototypeCache._path``) -- so that two registrations that differ only
    in language never hash to the same key even if their raw inputs happen
    to collide.
    """
    payload = {
        "model_id": model_id,
        "model2vec_version": model2vec_version,
        "params": params,
        "samples": sorted(raw_samples),
        "entities": {k: sorted(v) for k, v in sorted((entity_values or {}).items())},
        "lang": lang,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class PrototypeCache:
    """Per-label on-disk store of cached prototype embeddings.

    Layout is one subdirectory per (sanitized) skill id, with one ``.npz``
    file per intent name inside it: ``<cache_dir>/<skill>/<intent>.npz``.
    Labels are always ``<skill_id>:<intent_name>`` (the plugin's own
    namespace, including the special ``ocp:play``-style labels), so the
    split on the first ``:`` is unambiguous. This layout -- rather than a
    flat directory globbed by a sanitized ``<skill_id>:`` prefix -- exists
    because folding ``:`` to ``_`` for a flat filename is not injective:
    ``_`` is itself legal in a skill id, so a flat prefix glob for skill
    ``"skill"`` also matches files that actually belong to an unrelated
    skill ``"skill_extra"``. A subdirectory per skill has no such collision,
    and skill removal becomes a single directory delete rather than a glob.

    A key mismatch (changed inputs), a dimension mismatch against the
    currently-loaded model, or an unreadable/corrupt file are all treated as
    a miss, never as an error.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def _split_label(label: str) -> Tuple[str, str]:
        skill_id, sep, intent_name = label.partition(":")
        if not sep:
            # no namespace separator -- not a label this plugin produces,
            # but never a reason to crash; treat the whole label as the
            # "intent" part of an unnamespaced skill bucket.
            return "", skill_id
        return skill_id, intent_name

    def _skill_dir(self, skill_id: str) -> Path:
        safe = _UNSAFE_RE.sub("_", skill_id) or "_"
        return self.cache_dir / safe

    def _path(self, label: str, lang: Optional[str] = None) -> Path:
        skill_id, intent_name = self._split_label(label)
        safe_intent = _UNSAFE_RE.sub("_", intent_name) or "_"
        if lang:
            # a distinct filename per language partition: a bare (no-lang)
            # path collides between e.g. an en-US and a pt-PT registration
            # of the same label, and a partitioned entry must never be read
            # back as a hit for the other language's registration. A cache
            # written before per-language partitioning existed used the
            # bare, lang-less filename -- it is simply never looked up
            # again once a caller passes `lang`, so it is ignored rather
            # than misread as belonging to whichever language asks first.
            safe_lang = _UNSAFE_RE.sub("_", lang) or "_"
            safe_intent = f"{safe_intent}@{safe_lang}"
        return self._skill_dir(skill_id) / f"{safe_intent}.npz"

    def load(
        self, label: str, key: str, lang: Optional[str] = None,
        expected_dim: Optional[int] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return ``(embeddings, labels)`` for *label* if the cached entry's
        key matches and (when *expected_dim* is given) its embedding
        dimension agrees with the currently-loaded model, else ``None``
        (miss -- including on any read error).

        *expected_dim* should be the live model's known output dimension
        (e.g. ``model2vec.StaticModel.dim``) when available. Passing
        ``None`` skips the dimension check (used when the caller cannot
        determine it, e.g. in tests against a mock model) -- callers that
        CAN determine it should always pass it, since a stale entry from a
        model swapped in place at the same id/version is otherwise
        indistinguishable from a valid hit and would poison the live store
        with wrong-dimension vectors.
        """
        path = self._path(label, lang)
        if not path.exists():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            if str(data["key"].item()) != key:
                return None
            embeddings = data["embeddings"].astype(np.float32)
            labels = data["labels"]
        except (OSError, ValueError, KeyError, EOFError) as exc:
            LOG.warning(
                f"prototype cache: unreadable entry for {label!r}, "
                f"re-encoding ({exc})"
            )
            return None
        if (expected_dim is not None
                and (embeddings.ndim != 2 or embeddings.shape[1] != expected_dim)):
            LOG.warning(
                f"prototype cache: dimension mismatch for {label!r} "
                f"(cached {embeddings.shape}, current model dim "
                f"{expected_dim}); discarding stale entry and re-encoding"
            )
            self.remove(label, lang)
            return None
        return embeddings, labels

    def save(self, label: str, key: str, embeddings: np.ndarray, labels: np.ndarray,
              lang: Optional[str] = None) -> None:
        """Persist *embeddings*/*labels* for *label* under *key*.

        A failure to write (e.g. read-only filesystem) is logged and
        swallowed: the cache is a fast path, not a requirement for the
        registration to have succeeded.
        """
        try:
            path = self._path(label, lang)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                key=np.array(key),
                embeddings=embeddings,
                labels=labels.astype(str),
            )
        except OSError as exc:
            LOG.warning(f"prototype cache: failed to persist entry for {label!r}: {exc}")

    def remove(self, label: str, lang: Optional[str] = None) -> None:
        """Delete the cached entry for *label*, if any.

        With *lang* given, only that language partition's entry is removed.
        Without it, every entry for *label* is removed -- the bare
        (pre-partition) file, if any, plus every ``<intent>@<lang>.npz``
        partition -- since a caller removing a whole label (e.g. skill
        detach) has no reliable way to know which language(s) it was
        registered under.
        """
        try:
            if lang is not None:
                self._path(label, lang).unlink(missing_ok=True)
                return
            skill_id, intent_name = self._split_label(label)
            safe_intent = _UNSAFE_RE.sub("_", intent_name) or "_"
            skill_dir = self._skill_dir(skill_id)
            (skill_dir / f"{safe_intent}.npz").unlink(missing_ok=True)
            for p in skill_dir.glob(f"{safe_intent}@*.npz"):
                p.unlink(missing_ok=True)
        except OSError as exc:
            LOG.warning(f"prototype cache: failed to remove entry for {label!r}: {exc}")

    def remove_skill(self, skill_id: str) -> None:
        """Delete every cached entry belonging to *skill_id*.

        A single directory removal (see the layout note on the class), so
        there is no prefix-matching ambiguity between e.g. ``"skill"`` and
        ``"skill_extra"``.
        """
        d = self._skill_dir(skill_id)
        try:
            shutil.rmtree(d)
        except FileNotFoundError:
            pass
        except OSError as exc:
            LOG.warning(f"prototype cache: failed to remove skill dir '{d}': {exc}")
