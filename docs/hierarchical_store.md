# Hierarchical Prototype Pipeline & Store

This page documents two layers that ship together:

* **`Model2VecHierarchicalPrototypePipeline`** — the OPM-discoverable pipeline class. Entry point: `ovos-m2v-hierarchical-prototype-pipeline`. Subclasses the flat prototype pipeline; the only difference is the store shape (below). Intents are grouped into a domain == skill_id at registration time and matched in two stages. The embedding model comes from `load_shared_model`, so this plugin and the flat prototype plugin on the same model hold one `StaticModel`.
* **`HierarchicalPrototypeIntentStore`** — the two-stage, domain-routed prototype store used internally by that pipeline.

A separate entry point keeps the two prototype pipelines (flat, hierarchical) independently selectable in `default_pipeline` ordering, each with its own `intents.<key>` config block.

## Enabling

Add it to your OVOS config and place it in your pipeline order alongside (or in place of) the other prototype pipelines:

```json
{
  "intents": {
    "ovos-m2v-hierarchical-prototype-pipeline": {
      "model": "OpenVoiceOS/ovos-m2v-intents-multilingual",
      "intent_strategy": "softmax_weighted",
      "intent_tau": 0.1,
      "domain_threshold": 0.2
    }
  }
}
```

Configuration keys are read from `intents.ovos_m2v_hierarchical_prototype_pipeline`. The pipeline accepts every key the flat plugin does, plus `intent_strategy` / `intent_top_k` / `intent_tau` for the per-domain sub-stores and `domain_threshold` — the minimum router score required to route a query.

## Architecture

`HierarchicalPrototypeIntentStore` groups intents into *domains* and routes queries in **two stages**. Stage one is a top-level router. Each domain has a *fingerprint* per registration language: the anchors of every intent in the domain, reduced by `prototype_strategy`. The router scores the fingerprints and selects the single best domain. The fingerprint reuses the anchors the sub-store already holds, so no sample is encoded twice. The on-disk prototype cache and prebuilt artifacts work as they do for the flat store. Stage two resolves the intent only inside that domain's sub-store — exactly one sub-store runs per query.

```
              query embedding
                    │
                    ▼
        ┌───────────────────────┐
        │  domain fingerprints  │  stage 1: pick ONE domain
        │  {media: .., home: ..}│  argmax cosine similarity
        └───────────────────────┘
                    │
            domain_threshold gate
                    │
                    ▼
            ┌───────────────┐
            │ routed domain │       stage 2: PrototypeIntentStore
            │   {scores}    │       strategy = intent_strategy
            └───────────────┘
                    │
                    ▼
              global argmax
```

## Why two-stage routing

The router decides the domain before any per-intent scoring happens. Two consequences:

1. **Off-topic rejection.** When the best domain's fingerprint score is below `domain_threshold`, the query is rejected outright and no intent is returned. `0.0` (default) disables the gate.
2. **Cheaper inference with many domains.** Only one sub-store is scored per query rather than every intent across every domain.

No training step required: the static encoder produces the per-domain intent embeddings, and the fingerprints are built from them.

The router strategy is `prototype_strategy` and defaults to `mean_centroid`. The sub-store strategy is `intent_strategy` and defaults to `max_over_all`. The router routes on one anchor per domain, so it is cheap, and the two stages can disagree with the flat store.

With `max_over_all` for the router, the router keeps every anchor. It picks the domain that holds the best single anchor, so the top match equals the flat store's top match. The query then pays for both stages, and the result is the flat matcher at 8.6 times the latency. Use that setting only to reproduce the flat matcher.

The flat prototype pipeline stays the default pipeline. This variant is opt-in. Measured on the 1985-row en-US gold set (191 labels, 45 skills; the open set leaves every fifth skill unregistered, 425 rows), with the accept threshold at 0.65:

| variant | top-1 | false accepts, closed set | false accepts, open set | mean latency |
|---|---|---|---|---|
| flat prototype | 0.5073 | 495 (0.2494) | 101/425 (0.2376) | 25.46 ms |
| hierarchical, `max_over_all` router | 0.5073 | 495 (0.2494) | 101/425 (0.2376) | 220.12 ms |
| hierarchical, `mean_centroid` router (default) | 0.3849 | 356 (0.1793) | 42/425 (0.0988) | 16.39 ms |

The default router costs 12 points of top-1 accuracy. It halves false accepts for skills that are not installed, and it is faster than the flat matcher.

## Configuration

`HierarchicalPrototypeIntentStore` exposes the [`PrototypeStrategy`](ovos_pipeline.md#prototype-strategies) for the per-domain sub-stores via the `intent_*` keys, the router via the `domain_*` keys, and the rejection gate via `domain_threshold`.

```python
from model2vec import StaticModel
from ovos_m2v_pipeline import HierarchicalPrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy

model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
store = HierarchicalPrototypeIntentStore(
    intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
    intent_tau=0.1,
    domain_threshold=0.2,
)
```

## Usage

```python
store.add(model, "media:play",     ["play {song}", "put on {song}"], lang="en-US")
store.add(model, "media:pause",    ["pause", "pause the music"], lang="en-US")
store.add(model, "home:lights_on", ["turn on the lights", "lights on"], lang="en-US")

query = model.encode(["play africa"])[0]
store.calc_domain(query, lang="en-US")   # → "media"
scores = store.scores(query, lang="en-US")
# only the routed domain is scored: {"media:play": ..., "media:pause": ...}
best = max(scores, key=scores.get)       # "media:play"
```

### Restricting to one domain

A label's domain is its `skill_id`, the part before the first `:`. Pass `domain=...` to `add()` to group a label differently.

Pass `domain=...` to `scores()` (or `calc_intent()`) to skip the top-level router and score only inside a specific domain. This also bypasses the `domain_threshold` gate:

```python
scores = store.scores(query_embedding, domain="home")
```

### Convenience argmax

```python
store.calc_intent(query_embedding)                 # → "media:play" / None
store.calc_intent(query_embedding, domain="home")  # restricted to home
```

### Lifecycle

```python
store.remove("media:pause")   # drop an intent
store.remove_skill("media")    # drop every label of one skill
store.remove_domain("media")   # drop a whole domain (intents + fingerprint)
```

## Strategy round-trip

Every `PrototypeStrategy` works inside every per-domain sub-store: they're all `PrototypeIntentStore` instances, so they inherit the full strategy machinery described in [Prototype strategies](ovos_pipeline.md#prototype-strategies). Tests in `tests/test_hierarchical_store.py` round-trip all seven strategies.

## See also

- [Prototype strategies](ovos_pipeline.md#prototype-strategies): the scoring strategies the underlying stores expose.
- [OVOS Pipeline Plugin](ovos_pipeline.md): how `PrototypeIntentStore` is wired into the bus.
- [Configuration](configuration.md) — top-level OVOS config keys.
