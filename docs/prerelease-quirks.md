# Pre-release quirks

Behavior changes since the last stable release, newest first. This file is
reset at each stable release.

## 0.8.2a2

- The default model changed from the deprecated
  `Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2` to
  `OpenVoiceOS/ovos-m2v-intents-multilingual` for every language. There is
  no fallback to the `Jarbas/` classifier; its model card carries a pointer
  to the replacement. Deployments with an explicit `model` config key are
  unaffected and keep loading whatever they already point at. A new
  `models` config key (`{locale_or_lang: repo_id}`) lets a deployment pick
  a different model per language without setting `model` outright; it is
  matched against the full locale first, then the primary language subtag,
  and is only consulted when `model` is unset.
  `OpenVoiceOS/ovos-m2v-intents-en` is a smaller (16 MB), English-only
  model from the same roster; it is not the default (nor wired into any
  built-in per-language table) purely to keep that table small, since its
  held-out accuracy is comparable to the multilingual model. It remains
  available via `model` or `models["en"]` for deployments where the
  smaller footprint is worth trading off broader language coverage.
  Both models are frozen at dataset manifest `c6831002…` (pipeline commit
  `173c1fe`) and will be refit after the Adapt → `.intent` refactors merge.
- `valid_labels` is no longer consulted in prototype mode. It is an
  allow-list drawn from a model manifest's frozen training vocabulary, and
  prototype labels are registered at runtime rather than trained into the
  model, so every runtime-registered prototype label was legitimately
  absent from that manifest and was being silently discarded at match time
  - even at cosine 1.0. The prototype store already only holds
  explicitly-registered labels, so it is its own allow-list.

## 0.8.0a2

- The model no longer loads in the constructor. ovos-core builds every
  installed pipeline plugin at boot regardless of whether it is configured
  or ever used, so an eager `StaticModel`/`StaticModelPipeline` load there
  paid full model memory and (in prototype mode) encoded every buffered
  registration on every boot. The model now loads on first use — first
  match, or the first registration once no match has happened yet — behind
  a lock so concurrent callers share a single load. Registrations that
  arrive before the model exists buffer their raw samples and encode once,
  when the deferred load completes; `detach_intent`/`detach_skill` (and
  their OVOS-INTENT-4 equivalents) drop a still-buffered label without ever
  encoding it. A match that lands during the cold-start load waits up to
  `model_load_budget` seconds (default 0.5s); past that it returns no match
  for that one utterance (logging once that the model is warming up) while
  the load keeps running in the background, and normal matching resumes
  automatically once it finishes. Set `preload_model: true` to restore the
  previous eager-load-at-construction behavior for deployments that would
  rather pay the cost at boot than on first query.
- A failed deferred load retries on its own instead of leaving the plugin silently dead until restart: a failure clears the load thread and schedules the next attempt no sooner than a backoff window (30s, doubling on each consecutive failure, capped at 15 minutes), logging each failed attempt at ERROR.

## 0.8.0a1

- `valid_labels` is now checked against the raw model label, before
  `label_map` resolution, instead of after. A `labels.json` listing the raw
  training label vocabulary (as `ocp:play`, `stop:stop`, and
  `common_query:common_query` are stored) previously had every OCP-play,
  stop and common-query match silently discarded, because the allow-list
  check ran against the already-remapped canonical labels
  (`ovos.common_play.play_search`, `mycroft.stop`,
  `common_query.question`), which never appeared in that list.
- Prototype mode caches each label's encoded prototypes to disk, keyed on
  the registration's inputs (model id, `model2vec` version,
  anchor-selection parameters, raw pre-expansion template lines, referenced
  entity values). A registration whose inputs are unchanged since the last
  boot loads its embeddings from the cache instead of re-encoding through
  the embedding model; any other input change is a plain cache miss.
  `detach_intent`/`detach_skill` (and their OVOS-INTENT-4 equivalents)
  delete the corresponding cache entry so a removed skill's intents are not
  resurrected from a stale cache on the next boot. Enabled by default
  (`prototype_cache: false` disables it); see the README's "Prototype
  cache" section for the on-disk layout and the `prototype_cache_dir`
  override.
- OVOS-CONTEXT-1 `requires_context` / `excludes_context` gating now also
  applies to the legacy `padatious:register_intent` wire contract, not only
  OVOS-INTENT-4 template registrations: a Padatious `.intent` registration
  carrying either field is gated the same way at match time, and the gate
  is cleared on `detach_intent`/`detach_skill` and replaced on
  re-registration. Previously `requires_context`/`excludes_context` on that
  path were silently ignored.

## 0.7.0a1

- Classifier-mode label remapping is configurable instead of a hardcoded OCP
  / common-query / stop table: `label_map` (config) maps a raw model label to
  its canonical `skill_id:intent` label, merged over the built-in defaults
  and over any `labels.json` the loaded model ships alongside it (defaults <
  model manifest < user config). A trained model's frozen label head is model
  metadata, not plugin code, so the model itself is now the natural place to
  document what its labels mean.
- `valid_labels` (config) is the allow-list counterpart of `ignore_intents`:
  when set, only the listed canonical labels are eligible to match. Both
  filters apply after `label_map` resolution. With no `label_map` /
  `valid_labels` config and no `labels.json` on the model, behavior is
  unchanged.
- A `label_map` target that is not a `skill_id:intent` string (no colon) is
  logged as a warning (once per label) and used as-is; the plugin never
  invents a bus topic.
- `labels.json` lookup for a hub-id model never touches the network:
  construction only consults the local HF cache (`local_files_only=True`),
  riding whatever cache entry the model's own weights were already fetched
  into. A model not yet cached, or with no manifest, is silently treated as
  having none - the manifest is never a reason for pipeline construction to
  block on a network round trip.

## 0.6.1a1

- `PrototypeIntentStore` consolidation no longer needs a transient copy of
  the whole store: it pre-allocates the final-size array once and copies
  the existing store plus each pending chunk into it in turn, dropping
  every source as soon as it is copied, instead of `np.vstack`-ing the old
  array and every pending chunk at once (~2x peak over the final store
  size). A `MemoryError` while doing so is logged and the pending batch is
  left untouched rather than raised or silently dropped, so the store stays
  usable with whatever it already consolidated. On a real deployment this
  transient doubling, hit on the bus dispatch thread handling the first
  utterance after a registration burst, pinned a 2G-capped service at 100%
  CPU for 25+ minutes with no further intents matched. `scores()` (called
  on every live utterance) also no longer consolidates at all: it scores
  the already-consolidated store and each pending chunk separately and
  merges the per-label results, since a label's prototypes are always
  fully contained in exactly one of them.

## 0.5.8a2

- Declared template slots fill from live intent context (OVOS-CONTEXT-1
  §7, `ovos-spec-tools` `context_slot_candidates`): a registered template's
  `{slot}` placeholders are recorded at registration time, and any slot with
  a live non-null context entry fills `match_data[slot]` when the utterance
  itself never produces a value for it, independently of any
  `requires_context` declaration on the same intent. m2v never extracts a
  slot value from the utterance (it is a label classifier), so context is
  the only source of a filled value and the OVOS-INTENT-4 per-slot entity
  `.blacklist` remains a no-op here — there is nothing utterance-supplied to
  exclude.

## 0.5.8a1

- Template expansion in every registration path is lazy and bounded:
  `islice(iter_expand(...))` (ovos-spec-tools 1.8.0a1) replaces
  materializing `expand` calls, so a combinatorial template costs what the
  bound takes — registration-time expansion previously materialized full
  cartesian products transiently (~3GB and ~20 minutes measured on a real
  deployment) even though the store then kept at most 2000 samples.

## 0.5.7a1

- Prototype-store growth is amortized: registrations buffer into pending
  chunks and consolidate once on first read (match, save, or property
  access). Previously every registration re-stacked the entire embedding
  array — quadratic build cost, measured as a 1.2GB array reallocated per
  skill on a real install, which is why store construction took tens of
  minutes with one core pegged. Each registration also logs an INFO
  progress line (prototypes added, running totals) so long builds are
  visible instead of looking like an idle hang.

## 0.5.6a1

- The sample bound applies at store ingest, whatever path materialized the
  samples: the 2000-cap previously applied only to entity slot-filling, so
  pre-expanded padatious registrations flowed in unbounded — a real
  deployment still built a ~1.1M-prototype store on 0.5.5a1 and swapped its
  cgroup to death. `PrototypeIntentStore.add` now evenly samples any batch
  over the bound.
- Padatious-contract labels are dealiased: the legacy `.intent`-suffixed
  name folds onto the canonical suffixless label (registration, detach and
  ignore_labels accept both forms), so the dual-emit from ovos-workshop no
  longer stores every prototype twice, and matches dispatch on the same
  canonical id padatious itself uses.

## 0.5.5a1

- Embedding runs strictly in-process: every `model.encode` call passes
  `use_multiprocessing=False`. model2vec otherwise spawns `os.cpu_count()`
  loky worker subprocesses for sentence batches over its threshold, which
  inside a memory-capped service cgroup meant dozens of workers, deep swap
  and an OOM-kill during skill loading on a real deployment. Large template
  batches now encode sequentially — slightly slower for huge batches, never
  a process fan-out.
- Entity-filled template expansion is bounded: the cartesian product over a
  template's registered entity values caps at 2000 combinations, sampled
  deterministically and evenly across the space (endpoints kept). Two
  ~2200-value entities in one template previously materialized ~4.8M
  strings and embedded them all on every registration — tens of GB of swap
  and an OOM-kill on a real deployment.
