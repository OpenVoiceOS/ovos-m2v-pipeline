"""Embedding must stay in-process: model2vec spawns os.cpu_count() loky
workers for large batches unless use_multiprocessing=False — observed as 24
subprocesses inside a 2G service cgroup and an OOM-kill at skill load."""
from unittest import mock

import numpy as np

from ovos_m2v_pipeline import PrototypeIntentStore


def test_store_add_disables_multiprocessing():
    store = PrototypeIntentStore()
    model = mock.Mock()
    model.encode.return_value = np.ones((2, 4), dtype=np.float32)
    store.add(model, "label", ["a b", "c d"])
    assert model.encode.called
    _, kwargs = model.encode.call_args
    assert kwargs.get("use_multiprocessing") is False


def test_entity_expansion_is_bounded():
    """Two large entities in one template must not materialize the full
    cartesian product (observed: ~4.8M strings, 23G swap, OOM-kill)."""
    from ovos_m2v_pipeline import MAX_ENTITY_EXPANSIONS, Model2VecIntentPipeline
    p = Model2VecIntentPipeline.__new__(Model2VecIntentPipeline)
    p.entities = {"mon": [f"mon{i}" for i in range(2200)],
                  "move": [f"move{i}" for i in range(2200)]}
    out = p._expand_entities(["can {mon} learn {move}"])
    assert len(out) == MAX_ENTITY_EXPANSIONS
    assert len(set(out)) == len(out)
    # deterministic: same input, same sample
    assert out == p._expand_entities(["can {mon} learn {move}"])
    # endpoints of the combination space are kept
    assert "can mon0 learn move0" in out
    assert "can mon2199 learn move2199" in out


def test_small_expansion_untouched():
    from ovos_m2v_pipeline import Model2VecIntentPipeline
    p = Model2VecIntentPipeline.__new__(Model2VecIntentPipeline)
    p.entities = {"city": ["porto", "lisbon"]}
    out = p._expand_entities(["weather in {city}"])
    assert sorted(out) == ["weather in lisbon", "weather in porto"]


def test_store_ingest_is_bounded_regardless_of_source():
    """The cap must hold at the store, not one expansion path: real
    deployments feed pre-expanded padatious samples straight to add()."""
    import numpy as np
    from ovos_m2v_pipeline import (MAX_ENTITY_EXPANSIONS,
                                   PrototypeIntentStore)
    store = PrototypeIntentStore()
    model = mock.Mock()
    model.encode.side_effect = \
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
    n = store.add(model, "big", [f"sentence {i}" for i in range(50000)])
    assert n <= MAX_ENTITY_EXPANSIONS
    sents_passed = model.encode.call_args[0][0]
    assert len(sents_passed) == MAX_ENTITY_EXPANSIONS
    assert sents_passed[0] == "sentence 0"
    assert sents_passed[-1] == "sentence 49999"


def test_padatious_labels_dealias_to_canonical():
    """Dual-contract registration must collapse to ONE label, or the store
    holds every prototype twice (observed live: 83 -> 116 labels, ~1.1M
    prototypes)."""
    import numpy as np
    from ovos_utils.fakebus import FakeBus
    from ovos_bus_client.message import Message
    from ovos_m2v_pipeline import Model2VecPrototypePipeline
    p = Model2VecPrototypePipeline.__new__(Model2VecPrototypePipeline)
    p.ignore_labels = set()
    p.intents = set()
    p._context_gates = {}
    p._intent_slots = {}
    p.model = mock.Mock()
    p.model.encode.side_effect = \
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
    from ovos_m2v_pipeline import PrototypeIntentStore
    p.prototype_store = PrototypeIntentStore()
    p._prototype_k = None
    p.entities = {}
    p._prototype_cache_enabled = False
    p._handle_register_padatious(Message(
        "padatious:register_intent",
        {"name": "skill.test:go.intent", "samples": ["go to work"]}))
    assert "skill.test:go" in p.intents
    assert "skill.test:go.intent" not in p.intents
    # detach with the suffixed name removes the canonical entry
    p._handle_detach_intent(Message(
        "detach_intent", {"intent_name": "skill.test:go.intent"}))
    assert len(p.prototype_store) == 0


def test_store_add_is_amortized_no_per_add_vstack():
    """Each add must not copy the whole store (O(n^2) build churn: a 1.2GB
    array reallocated per registration on real installs). Adds buffer into
    pending chunks; consolidation happens once on read."""
    import numpy as np
    from ovos_m2v_pipeline import PrototypeIntentStore
    store = PrototypeIntentStore()
    model = mock.Mock()
    model.encode.side_effect = \
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
    for i in range(50):
        store.add(model, f"label{i}", [f"s{i}a", f"s{i}b"])
        assert len(store._pending) == i + 1  # no consolidation during adds
    assert len(store) == 100
    labels = store.labels  # first read consolidates, once
    assert len(labels) == 100
    assert len(store._pending) == 0
    # re-registration replaces, through the consolidation path
    store.add(model, "label0", ["new"])
    assert (store.labels == "label0").sum() == 1
    assert len(store) == 99


def test_scores_does_not_consolidate():
    """scores() runs on the bus dispatch thread for every live utterance; it
    must never trigger _consolidate()'s full-backlog copy, or a live
    registration burst wedges utterance matching behind it (observed:
    add-lines complete, then a 2G-cgroup install pins at 100% CPU for 25+
    minutes with zero further intents matched)."""
    store = PrototypeIntentStore()
    model = mock.Mock()
    model.encode.side_effect = \
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
    for i in range(50):
        store.add(model, f"label{i}", [f"s{i}a", f"s{i}b"])
    assert len(store._pending) == 50
    scores = store.scores(np.ones(4, dtype=np.float32))
    assert len(scores) == 50
    assert len(store._pending) == 50  # untouched: no consolidation happened


def test_consolidate_out_of_memory_is_a_hard_failure_not_a_retry_loop():
    """A MemoryError while folding pending chunks into the contiguous store
    must be logged and left as-is, not retried -- a live install kept
    spinning at 100% CPU for 25+ minutes on the same allocation instead of
    failing fast and staying usable with whatever it had already stored."""
    store = PrototypeIntentStore()
    model = mock.Mock()
    model.encode.side_effect = \
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
    store.add(model, "label0", ["a", "b"])
    assert len(store._pending) == 1
    with mock.patch("numpy.empty", side_effect=MemoryError("simulated OOM")):
        store._consolidate()  # must return, not raise or loop
    # nothing pending was dropped: the batch is still there to retry later
    assert len(store._pending) == 1
    assert len(store._labels) == 0

    # once memory is available again, a normal call still succeeds
    store._consolidate()
    assert len(store._pending) == 0
    assert len(store._labels) == 2


def test_reregistration_drops_stale_pending_chunk_after_failed_consolidate():
    """A re-registration must retire the label's OLD phrasing even when the
    consolidation attempt it triggers fails with MemoryError: relying on
    _consolidate() to fold (and thus filter) the old rows leaves a stale
    pending chunk sitting next to the fresh one, and scores() (which reads
    consolidated + pending) keeps matching retired phrasing forever."""
    store = PrototypeIntentStore()
    old_vec = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    new_vec = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    model = mock.Mock()
    model.encode.side_effect = lambda sents, **kw: (
        old_vec if sents == ["turn on the lights"] else new_vec
    )

    store.add(model, "L", ["turn on the lights"])
    assert len(store._pending) == 1

    with mock.patch("numpy.empty", side_effect=MemoryError("simulated OOM")):
        n = store.add(model, "L", ["completely different phrase"])
    assert n == 1

    # exactly one pending chunk for L must survive: the new one
    label_chunks = [l[0] for _, l in store._pending if len(l)]
    assert label_chunks.count("L") == 1

    # the retired phrasing must no longer match L at all
    stale_scores = store.scores(old_vec[0])
    assert stale_scores.get("L", 0.0) < 0.5, (
        "stale pending chunk from before the re-registration still "
        f"matches: {stale_scores}"
    )
    # the new phrasing does match
    fresh_scores = store.scores(new_vec[0])
    assert fresh_scores["L"] == 1.0
