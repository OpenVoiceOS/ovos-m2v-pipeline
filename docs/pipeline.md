# Pipeline Internals

## Class: `Model2VecIntentPipeline`

Defined in `ovos_m2v_pipeline/__init__.py`.

Extends `ConfidenceMatcherPipeline` from `ovos-plugin-manager`, which provides the three-tier match interface (`match_high`, `match_medium`, `match_low`) and integrates with the OVOS pipeline priority system.

---

## Operating Modes

The pipeline supports two modes selected via `config["mode"]`:

### Classifier mode (default)

Loads a `StaticModelPipeline` (embedding backbone + trained linear classifier head). Inference returns softmax probabilities over the label set baked into the model.

The active intent set (`self.intents`) is synchronised from the bus on every registration or detach event. Only labels present in this set (plus the special-case remaps below) are returned.

### Prototype mode

Loads a bare `StaticModel` (embeddings only, no classifier head). An empty `PrototypeIntentStore` is created at startup and populated incrementally as skills register their Padatious intents at boot. Inference uses cosine nearest-neighbour against all stored prototypes.

---

## Initialisation

```python
Model2VecIntentPipeline(bus=None, config=None)
```

On startup the pipeline:

1. Reads configuration from `mycroft.conf` under `intents.ovos_m2v_pipeline` (or uses the `config` kwarg).
2. Loads the model (`StaticModelPipeline` in classifier mode, `StaticModel` in prototype mode).
3. Registers message bus event handlers.

### Bus Events: Classifier Mode

| Event | Handler |
|-------|---------|
| `mycroft.ready` | `handle_sync_intents` |
| `padatious:register_intent` | `handle_sync_intents` |
| `register_intent` | `handle_sync_intents` |
| `detach_intent` | `handle_sync_intents` |
| `detach_skill` | `handle_sync_intents` |

### Bus Events: Prototype Mode

| Event | Handler |
|-------|---------|
| `mycroft.ready` | `_handle_ready_prototype` (logs store statistics) |
| `padatious:register_intent` | `_handle_register_padatious` |
| `register_intent` | `_handle_register_adapt` |
| `detach_intent` | `_handle_detach_intent` |
| `detach_skill` | `_handle_detach_skill` |

---

## Intent Synchronization: Classifier Mode

`handle_sync_intents` is called whenever skills are loaded or unloaded. It:

1. Applies a 3-second debounce (`_syncing` flag) to avoid redundant requests during bulk skill loading.
2. Requests the Adapt manifest via `intent.service.adapt.manifest.get`.
3. Requests the Padatious manifest via `intent.service.padatious.manifest.get`.
4. Merges the two lists into `self.intents` (a `set`), excluding any labels in `ignore_intents`.

---

## Prototype Registration: Prototype Mode

### `_handle_register_padatious`

Called for every `padatious:register_intent` event. Steps:

1. Extract `name` from `message.data`. Skip it if it is in `ignore_labels`.
2. Prefer inline `message.data["samples"]` if present. Otherwise read the file at `message.data["file_name"]`.
3. Apply `ovos_utils.bracket_expansion.expand_template` to every line so that template syntax is expanded into concrete utterances:
   - `(turn on|switch on) the lights` → `["turn on the lights", "switch on the lights"]`
   - `[please] play music` → `["please play music", "play music"]`
4. Embed all expanded examples. `select_anchors()` (`ovos_m2v_pipeline/strategies.py:131`) reduces them to the subset or aggregation dictated by `prototype_strategy`. All resulting anchors are stored in `PrototypeIntentStore` (capped per label by `prototype_k` when set).
5. Add the label to `self.intents`.

### `_handle_register_adapt`

Adapt intents carry no example sentences, only keyword rules. The label is added to `self.intents` but no prototypes are created. These intents are not matched in prototype mode.

### `_handle_detach_intent`

Removes all prototypes for the given `intent_name` and discards the label from `self.intents`.

### `_handle_detach_skill`

Removes all prototypes whose label starts with `<skill_id>:` and removes matching labels from `self.intents`. `skill_id` is read from `message.data` with `message.context` as a fallback.

---

## `PrototypeIntentStore`

Defined in `ovos_m2v_pipeline/__init__.py:34`. A mutable store of L2-normalised prototype embeddings. The store starts empty and is populated incrementally at runtime.

```python
from ovos_m2v_pipeline.strategies import PrototypeStrategy

store = PrototypeIntentStore(
    strategy=PrototypeStrategy.MAX_OVER_ALL,  # default
    top_k=3,
    tau=0.1,
)
store.add(model, "skill_a:my.intent", ["example 1", "example 2"], k=5)
store.remove("skill_a:my.intent")
store.remove_skill("skill_a")

scores = store.scores(query_embedding)  # {label: score}
```

The `strategy` kwarg determines the scoring algorithm, and therefore what `scores()` returns. The default `MAX_OVER_ALL` returns the maximum cosine similarity across all stored anchors per label. This is byte-compatible with the store's behavior before strategies were introduced.

The store can optionally be persisted and reloaded with any strategy:

```python
store.save("prototypes.npz")
store = PrototypeIntentStore.load(
    "prototypes.npz",
    strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
    top_k=3,
    tau=0.05,
)
```

See [Configuration: Prototype Strategies](configuration.md#prototype-strategies-prototype-mode-only) for a full description of each strategy.

---

## Inference (`_match`)

`_match(utterance)` dispatches to `_match_classifier` or `_match_prototype` depending on the mode. Both yield `(skill_id, label, score)` tuples sorted by score descending.

### Special-case label remapping

Regardless of mode, three labels are remapped before being returned:

| Training label | `skill_id` returned | `label` returned |
|----------------|---------------------|-----------------|
| `ocp:play` | `ovos.common_play` | `ovos.common_play.play_search` |
| `common_query:common_query` | `common_query.openvoiceos` | `common_query.question` |
| `stop:stop` | `stop.openvoiceos` | `mycroft.stop` |

For all other labels `skill_id` is the prefix before `:`.

### Classifier mode filter

After remapping, a label is skipped unless `skill_id` is one of the three special-case IDs above **or** the label appears in `self.intents`.

### Prototype mode filter

The `PrototypeIntentStore` only ever contains labels that were explicitly registered via `_handle_register_padatious`. No additional `self.intents` filter is applied. Labels in `ignore_labels` are skipped.

---

## Match Methods

All three methods share the same logic via `_first_match`: call `_match`, take the first result, check it against the threshold.

```
match_high   → conf_high   (default 0.70)
match_medium → conf_medium (default 0.50)
match_low    → conf_low    (default 0.15)
```

OVOS does **not** call all three tiers automatically. Each tier is a separate entry in the OVOS pipeline list, identified by a suffix: `ovos-m2v-pipeline-high`, `ovos-m2v-pipeline-medium`, `ovos-m2v-pipeline-low`. Only the tiers you add to the pipeline list are ever called. See [OVOS Pipeline Plugin](ovos_pipeline.md) for configuration examples.

If the utterance list is empty, or the top result's score is below the threshold, `None` is returned. Otherwise an `IntentHandlerMatch` is returned with:

| Field | Value |
|-------|-------|
| `match_type` | The intent label |
| `match_data` | `{"utterance": ..., "confidence": ...}` |
| `skill_id` | Extracted from the label prefix (before `:`) |
| `utterance` | The original utterance string |

---

## Limitations

- **Classifier mode**: the model is pretrained on a fixed dataset. Skills absent from the training corpus will not be recognised, even if their intents are registered at runtime.
- **Prototype mode**: only Padatious intents with example utterances produce prototypes. Adapt intents (keyword-based) are tracked but never matched.
- **Intent sync debounce** (classifier mode): a fixed 3-second sleep may leave `self.intents` empty briefly after startup.
- Only the **first utterance** in the input list is evaluated (`utterances[0]`).

---
[← Configuration](configuration.md) · [Home](README.md) · [Prototype Strategies →](strategies.md)
