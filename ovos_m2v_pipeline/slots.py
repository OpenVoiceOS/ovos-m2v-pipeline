"""Shared ``{slot}`` template expansion (OVOS-INTENT-4 §7).

Pulled out of ``Model2VecIntentPipeline._expand_entities`` so the training
corpus builder (``train/build_dataset.py``) fills slots exactly the way the
runtime prototype pipeline does, from the same function, instead of carrying
a second implementation that can silently drift from it.
"""
import itertools
import re
from typing import Dict, Iterable, List

from ovos_utils.log import LOG

#: Regex matching ``{slot}`` placeholders in OVOS-INTENT-1 template samples.
SLOT_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

#: Upper bound on entity-filled samples generated per template. The
#: cartesian product over registered entity values is unbounded input
#: (auto-registered .entity files can carry thousands of values each);
#: everything past this bound is a deterministic evenly-strided sample of
#: the combination space.
MAX_ENTITY_EXPANSIONS = 2000

#: ``(template, total_combinations)`` pairs recorded by :func:`expand_entities`
#: with ``record_oversample=True`` since the last :func:`reset_oversample_stats`
#: call. A bulk caller that fills thousands of templates from large entity sets
#: (the corpus builder) hits the per-call bound on most of them; that caller
#: opts in and reads this list to log one summary instead of one WARNING per
#: call. A default call records nothing, so a long-lived runtime holds no list.
_oversampled: List[tuple] = []


def oversample_stats() -> List[tuple]:
    """Return the oversampling events recorded since the last
    :func:`reset_oversample_stats` call."""
    return list(_oversampled)


def reset_oversample_stats() -> None:
    """Clear the recorded oversampling events, e.g. before a fresh corpus
    build."""
    _oversampled.clear()


def expand_entities(samples: Iterable[str],
                    entities: Dict[str, List[str]],
                    record_oversample: bool = False) -> List[str]:
    """Fill ``{slot}`` placeholders in template *samples* with registered
    entity values (OVOS-INTENT-4 §7). Samples without placeholders, or whose
    entity is unregistered, are passed through with the placeholder left
    literal (entities are an optional hint, §7).
    """
    samples = list(samples)
    if not entities:
        return samples
    out: List[str] = []
    for tmpl in samples:
        slots = SLOT_RE.findall(tmpl)
        if not slots:
            out.append(tmpl)
            continue
        slot_values: List[List[str]] = []
        for slot in slots:
            vals = entities.get(slot.lower())
            slot_values.append(vals if vals else ["{" + slot + "}"])
        # the cartesian product over large value sets explodes: two
        # ~2000-value entities in one template is ~4M strings, all
        # materialized and embedded on every registration — enough to
        # swap out and OOM-kill a capped service. Engines own bounding
        # unbounded entity data: take a deterministic, evenly-strided
        # sample of the combination space instead.
        sizes = [len(v) for v in slot_values]
        total = 1
        for n in sizes:
            total *= n
        if total > MAX_ENTITY_EXPANSIONS:
            msg = (f"template {tmpl!r} expands to {total} combinations; "
                   f"sampling {MAX_ENTITY_EXPANSIONS} evenly")
            if record_oversample:
                # Routine at bulk-fill scale (a corpus builder calling this
                # once per template row against large `.entity` files) --
                # WARNING here would drown itself; the caller summarizes
                # from `oversample_stats()` instead.
                LOG.debug(msg)
                _oversampled.append((tmpl, total))
            else:
                # A live registration: the truncation is news, and nothing
                # reads `oversample_stats()` here, so log it and keep no state.
                LOG.warning(msg)
            step = (total - 1) / (MAX_ENTITY_EXPANSIONS - 1)
            indices = {round(i * step) for i in range(MAX_ENTITY_EXPANSIONS)}
            combos = []
            for idx in sorted(indices):
                combo = []
                rem = idx
                for n in reversed(sizes):
                    combo.append(rem % n)
                    rem //= n
                combos.append(tuple(slot_values[d][c] for d, c in
                                    enumerate(reversed(combo))))
        else:
            combos = itertools.product(*slot_values)
        for combo in combos:
            filled = tmpl
            for slot, val in zip(slots, combo):
                filled = filled.replace("{" + slot + "}", val, 1)
            out.append(filled.strip())
    return out
