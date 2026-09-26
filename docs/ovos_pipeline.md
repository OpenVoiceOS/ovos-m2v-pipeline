# OVOS Pipeline Plugin

This package ships two `opm.pipeline` entry points, both classes of
`Model2VecIntentPipeline`:

```
ovos-m2v-pipeline           = ovos_m2v_pipeline:Model2VecIntentPipeline
ovos-m2v-prototype-pipeline = ovos_m2v_pipeline:Model2VecPrototypePipeline
```

| Plugin | Mode | Needs training? |
|---|---|---|
| `ovos-m2v-pipeline` | classifier (default) | Yes, a pre-trained `StaticModelPipeline` |
| `ovos-m2v-prototype-pipeline` | prototype (forced) | No, it embeds skill examples at boot |

Each reads its own key under `mycroft.conf["intents"]`, so both can run at
once. Every config key is in [Configuration](configuration.md).

## Classifier mode

`ovos-m2v-pipeline` loads a `StaticModelPipeline`: an embedding model plus a
linear classifier head, trained ahead of time. Inference returns a softmax
probability over the labels baked into the head. Registering a new intent at
runtime gates or allow-lists a trained label. It never teaches the head a
label it was not trained on.

## Prototype mode

`ovos-m2v-prototype-pipeline` loads a bare `StaticModel`, embeddings only,
no head. At boot it builds an empty `PrototypeIntentStore` and fills it as
skills register Padatious and OVOS-INTENT-4 template intents: each example
utterance is template-expanded, embedded, and reduced to anchors by
`prototype_strategy`. Inference scores the query against the stored anchors
with cosine similarity. Adapt intents carry no example utterances, so they
are tracked but never matched here.

One embedding model loads per process per resolved model path
(`load_shared_model`): the classifier's head, its `-low` prototype stage,
and a standalone prototype plugin that names the same model all share one
`StaticModel` object.

## Confidence scale

Classifier scores are softmax probabilities that sum to 1 across all labels.
Prototype scores are cosine similarities between L2-normalised embeddings.
The two scales are not comparable, so each mode has its own `conf_high` /
`conf_medium` / `conf_low` default (0.7 / 0.5 / 0.15 for classifier, 0.85 /
0.7 / 0.65 for prototype). A configured `conf_*` key overrides the default in
either mode.

## A label the model lacks

Classifier mode filters trained labels down to the ones registered
plus three special labels (`ocp:play`, `common_query:common_query`,
`stop:stop`), gated on the matching pipeline being present in the caller's
session. A trained label with no active registration never matches, and a
registered label the head was never trained on never appears in its output:
the head is frozen, and it can only ever emit a label from its training
set.

Prototype mode has no such gap. Its store holds only labels a skill
registered, so every label it can return is one it was actually given
examples for.

## Confidence tiers and the pipeline list

Each tier is a separate entry in `mycroft.conf["intents"]["pipeline"]`,
identified by a `-high`, `-medium` or `-low` suffix on the entry-point name.
OVOS reads the list top to bottom and stops at the first match. A tier not
in the list is never called and costs nothing.

| Pipeline entry | Method | Threshold |
|---|---|---|
| `ovos-m2v-pipeline-high` | `match_high()` | `conf_high` |
| `ovos-m2v-pipeline-medium` | `match_medium()` | `conf_medium` |
| `ovos-m2v-pipeline-low` | `match_low()` | `conf_low`, or the `-low` prototype stage's own `conf_low` when `low_tier` is `"prototype"` |
| `ovos-m2v-prototype-pipeline-high` / `-medium` / `-low` | same methods, on the standalone prototype plugin | its own `conf_*` |

By default `ovos-m2v-pipeline-low` runs a prototype stage built from the same
model and the loaded skills' own templates: the head answers `-high` and
`-medium`, and any label the head declined or was never trained on falls to
the skill's own examples at `-low`. Set `low_tier: "classifier"` to run the
head itself at `conf_low` instead.

Nothing in the plugin tracks whether `-high` or `-medium` ran first. The
`-low` entry checks only its own threshold. It behaves as a fallback purely
because the session's `pipeline` list stops at the first match: with
`-high` and `-medium` above it, `-low` is reached only after both declined.
Placed alone, with no `-high` or `-medium` entry in the list, `-low`
answers every utterance that clears its own confidence, whether or not a
higher tier would have matched it too.

## Place padacioso before the classifier

The classifier stage is frozen. It answers only with labels present in its
training set. A skill can register an exact Padatious template line for a
label the classifier never saw. Put `ovos-padacioso-pipeline-plugin-high`
ahead of the classifier stages so it claims that exact line first. Add the
prototype stage after the classifier, since it reads every loaded skill's
templates and needs no training.

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "OpenVoiceOS/ovos-m2v-intents-multilingual"
    },
    "pipeline": [
      "ovos-padacioso-pipeline-plugin-high",
      "ovos-m2v-pipeline-high",
      "ovos-m2v-prototype-pipeline-medium"
    ]
  }
}
```

These three entries are the m2v part of the list, not the whole list.
ovos-config's shipped default runs `ovos-stop-pipeline-plugin-high`,
`ovos-converse-pipeline-plugin`, `ovos-ocp-pipeline-plugin-high`,
`ovos-padatious-pipeline-plugin-high` and `ovos-adapt-pipeline-plugin-high`
before `ovos-m2v-pipeline-high`; stop, converse and the exact matchers keep
their place ahead of m2v. Fallback and common-query plugins go after it.

Placing a prototype tier ahead of the classifier's own tiers of the same
name costs accuracy instead: the prototype store holds only
runtime-registered labels, so for an utterance whose label the classifier
WAS trained on, the nearest prototype is always some other, wrong label. A
prototype tier run first claims that utterance before the classifier gets
to see it. Give the classifier's own tiers first refusal, and let a
prototype tier answer only what the classifier already declined. The
`Model2VecPrototypePipeline` docstring in `ovos_m2v_pipeline/__init__.py`
carries the measured test-suite cost of the reverse order, with its own
provenance tag.

## Prototype strategies

`prototype_strategy` picks how a label's registered examples become stored
anchors and how those anchors turn into a match score, via `PrototypeStrategy`
in `ovos_m2v_pipeline/strategies.py`.

| Value | Anchors stored | Score |
|---|---|---|
| `max_over_all` | every sample (default) | max cosine over anchors |
| `mean_centroid` | one, the mean of all samples | cosine to centroid |
| `medoid` | one, the sample closest to the centroid | cosine to medoid |
| `top_k_mean` | every sample | mean of the top `prototype_top_k` cosines |
| `farthest_point` | up to `prototype_k`, maximin sampling | max cosine |
| `kmeans_centers` | up to `prototype_k` spherical k-means centroids | max cosine |
| `softmax_weighted` | every sample | softmax-weighted average, temperature `prototype_tau` |

`max_over_all` is the default, so an existing `.npz` cache and tuned
thresholds stay valid under it.

## Prototype cache

Prototype mode re-encodes a skill's templates on every boot unless a cache
holds them. Each label's encoded prototypes are cached to disk, keyed on the
model id, the installed `model2vec` version, the anchor-selection
parameters, the raw template lines, and any entity values the templates
reference. An unchanged registration loads from the cache. Any other change
is a plain miss. `detach_intent` / `detach_skill` delete the corresponding
entry so a removed skill's intents are never resurrected from a stale
cache. Cache files live under `prototype_cache_dir`, one small `.npz` file
per label. Set `prototype_cache: false` to disable it.

## Prebuilt prototypes

`prebuilt_prototypes` points at a directory or Hugging Face repo built ahead
of time with the `ovos-m2v-prototypes` command, so a resource-constrained
device never has to encode a skill's templates itself:

```bash
ovos-m2v-prototypes export \
  --out ./my-skill-prototypes \
  --model OpenVoiceOS/ovos-m2v-intents-multilingual \
  --skill-dir /path/to/my-skill \
  --skill-id my-skill.openvoiceos
```

The corpus that `train/build_dataset.py` builds is the other source, and it
is the one to use for a published artifact. It carries the translated,
tracker, golden and augmented rows a skill's own templates do not, and its
slots are filled rather than left literal, so prototypes built from it
describe the same data a classifier is fitted on:

```bash
ovos-m2v-prototypes export \
  --out ./intent-prototypes \
  --model OpenVoiceOS/ovos-m2v-intents-multilingual \
  --from-dataset train/dataset
```

Every `(label, lang)` pair in `train.parquet` becomes one prototype set.
`--lang` narrows the export to a single locale.

The command discovers every `.intent` file under the skill's
`locale/<lang>/` directories the same way a live registration does. The
output holds `prototypes.npz` and a `manifest.json` naming the model id and
revision, the `model2vec` version, the embedding dimension, the strategy and
its parameters, and a per-label cache key. At boot the manifest is checked
against the running model's id and version. A mismatch is logged and the
artifact is ignored, falling back to encoding from scratch. A label the
skill does not register never becomes matchable, even if the artifact
carries it. The artifact only removes the one-time registration encode: a
live query still needs the embedding model to embed the incoming utterance.

## Trained models document their labels

A classifier's label head is frozen at training time, so which bus intent
each label denotes is a property of that model, not of the plugin. A model
repo or directory can ship a `labels.json` alongside its weights:

```json
{
  "valid_labels": ["my_domain:book_flight", "my_domain:cancel_flight"],
  "families": {
    "my_domain:book_flight": "skill",
    "my_domain:cancel_flight": "skill"
  }
}
```

`labels.json` has the same shape as `label_map`, plus an optional
`valid_labels` list and an optional `families` map (`skill`, `ocp`,
`common_query`, `stop`, or `persona`, see [Label scheme](labels.md)). Three
layers combine, later overriding earlier: the built-in OCP/common-query/stop
remaps, the loaded model's own `labels.json`, then the deployment's
`label_map` / `valid_labels` config. For a Hugging Face repo id,
`labels.json` is read only from the local cache already populated by the
model download. A missing or corrupt manifest is logged and ignored. A
`label_map` target with no colon is used as-is and logged once as a
warning.

## Typed slots

The prototype plugin never reads a slot value from the utterance text
itself. A template that declares a typed slot (`set the brightness to
{number:b}`, OVOS-INTENT-1 §5.6) fills it from the `typed_slots` map the
core computed for the utterance:

- one entry of the declared type whose span holds on the utterance fills
  the slot with its surface;
- of several, the entry that follows the template's literal word before the
  slot fills it. When that word occurs more than once, the occurrence
  nearest the slot's position in its templates wins;
- with no such literal-word anchor, the entry whose position in the
  utterance is nearest the slot's position in its templates fills it;
- no entry, no map, or a slot already filled from session context: the slot
  stays as it was.

`Match.slots[name]` carries the surface string.

---
[Home](../README.md) · [Configuration →](configuration.md)
