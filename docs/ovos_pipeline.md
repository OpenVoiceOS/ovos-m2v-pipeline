# OVOS Pipeline Plugin

This package registers two pipeline plugins, each suited for a different deployment scenario. Both are discovered automatically by OVOS via `opm.pipeline` entry points.

## Entry Points

```
ovos-m2v-pipeline           = ovos_m2v_pipeline:Model2VecIntentPipeline
ovos-m2v-prototype-pipeline = ovos_m2v_pipeline:Model2VecPrototypePipeline
```

| Plugin | Class | Mode | Requires training? |
|--------|-------|------|--------------------|
| `ovos-m2v-pipeline` | `Model2VecIntentPipeline` | Classifier | Yes, a pre-trained `StaticModelPipeline` |
| `ovos-m2v-prototype-pipeline` | `Model2VecPrototypePipeline` | Prototype | No, it embeds examples at boot |

## Configuration

Each plugin reads from its own key under `"intents"` in `mycroft.conf`, so both can coexist in the same OVOS instance.

### Classifier plugin: `ovos-m2v-pipeline`

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE",
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": [],
      "timeout": 1
    }
  }
}
```

### Prototype plugin: `ovos-m2v-prototype-pipeline`

```json
{
  "intents": {
    "ovos-m2v-prototype-pipeline": {
      "model": "minishlab/M2V_multilingual_output",
      "prototype_strategy": "max_over_all",
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": []
    }
  }
}
```

### Configuration Keys

| Key | Type | Default | Applies to | Description |
|-----|------|---------|------------|-------------|
| `model` | `str` | unset (defaults to `OpenVoiceOS/ovos-m2v-intents-multilingual`) | both | Hugging Face repo ID or local path. Classifier mode requires a `StaticModelPipeline`. Prototype mode accepts any bare `StaticModel`. Set explicitly to override the default (e.g. the smaller, English-only `OpenVoiceOS/ovos-m2v-intents-en`, which trades language coverage for size at comparable held-out accuracy). |
| `models` | `dict[str, str]` | `{}` | both | Per-language default override, `{locale_or_lang: repo_id}`. Matched against the full `lang` locale, then its primary subtag. Only consulted when `model` is unset. |
| `prototype_k` | `int` | unset (keep all) | prototype | Maximum prototype embeddings stored per intent label. Unset keeps every registered sample so exact training samples always match. Set an integer to cap memory. |
| `conf_high` | `float` | `0.7` | both | Minimum score for `match_high`. |
| `conf_medium` | `float` | `0.5` | both | Minimum score for `match_medium`. |
| `conf_low` | `float` | `0.15` | both | Minimum score for `match_low`. |
| `ignore_intents` | `list[str]` | `[]` | both | Labels to always discard. |
| `timeout` | `int` | `1` | classifier | Seconds to wait for Adapt / Padatious manifest responses. |

## Messagebus Events

### Classifier plugin

| Event | Handler | Description |
|-------|---------|-------------|
| `mycroft.ready` | `handle_sync_intents` | Initial intent sync after all skills load. |
| `padatious:register_intent` | `handle_sync_intents` | Re-sync when a new Padatious intent registers. |
| `register_intent` | `handle_sync_intents` | Re-sync when a new Adapt intent registers. |
| `detach_intent` | `handle_sync_intents` | Re-sync after an intent is removed. |
| `detach_skill` | `handle_sync_intents` | Re-sync after a skill unloads. |

Sync is debounced with a 3-second sleep and the `_syncing` flag to coalesce bursts of registrations during bulk skill loading.

### Prototype plugin

| Event | Handler | Description |
|-------|---------|-------------|
| `mycroft.ready` | `_handle_ready_prototype` | Logs store statistics when system is ready. |
| `padatious:register_intent` | `_handle_register_padatious` | Reads `file_name` or inline `samples`, expands templates, embeds the expanded examples per label (capped by `prototype_k` when set). |
| `register_intent` | `_handle_register_adapt` | Tracks Adapt label in `self.intents`. No prototypes are created, since Adapt uses keywords, not examples. |
| `detach_intent` | `_handle_detach_intent` | Removes prototypes and label for the detached intent. |
| `detach_skill` | `_handle_detach_skill` | Removes all prototypes and labels for the skill. `skill_id` is taken from `message.data` with `message.context` as fallback. |

## Template Expansion

Padatious `.intent` files support bracket template syntax. The prototype plugin expands templates before embedding so every concrete variant is represented:

| Template line | Expanded variants |
|---------------|-------------------|
| `(turn on\|switch on) the lights` | `turn on the lights`, `switch on the lights` |
| `[please] play music` | `please play music`, `play music` |

Inline `samples` in `padatious:register_intent` messages are expanded the same way.

## Typed Slots

The prototype plugin never reads a slot value from the utterance. A template that declares a typed slot (`set the brightness to {number:b}`, OVOS-INTENT-1 section 5.6) gets its value from the `typed_slots` map on the utterance message, when the core computed one:

- one entry of the declared type whose span holds on the utterance fills the slot with its surface;
- of several, the entry that follows the template's literal word before the slot fills it (`to` in the template above, so `set the brightness to twenty five please not fifty` gives `twenty five`); when that word occurs more than once with an entry after it, the occurrence nearest the slot's position in its templates wins; an entry fills at most one slot, so two slots of one type never collapse onto one reading;
- when no template puts a literal word before the slot, or no entry follows one in this utterance, the entry whose relative position in the utterance is nearest the slot's position in its templates fills it;
- no entry, no map, or a slot already filled from session context: the slot stays as it was.

`Match.slots[name]` is the surface string. The normalized value stays in the map, keyed by that surface.

## Confidence Tiers and the Pipeline List

OVOS does **not** call all three tiers of a plugin automatically. Instead, each tier is a separate named entry in the `pipeline` list, identified by a `-high`, `-medium`, or `-low` suffix:

| Pipeline entry | Method called | Threshold key | Default |
|----------------|---------------|---------------|---------|
| `ovos-m2v-pipeline-high` | `match_high()` | `conf_high` | `0.70` |
| `ovos-m2v-pipeline-medium` | `match_medium()` | `conf_medium` | `0.50` |
| `ovos-m2v-pipeline-low` | `match_low()` | prototype stage, `low_prototype.conf_low` | `0.65` cosine |

`ovos-m2v-pipeline-low` runs prototype mode by default (`low_tier: "prototype"`): the `-high` and `-medium` tiers answer from the trained head, and the `-low` tier answers from the loaded skills' own templates, on the same embedding model. Set `low_tier: "classifier"` to run the head at `conf_low` (`0.15`) instead.

**What the `-low` stage accepts.** Every label a skill registers, trained labels included. The stage denies only `ignore_intents` (the classifier's own list, plus any list under `low_prototype`); nothing compares a registration against the model's classes. So a label the model was trained on, `ovos-skill-alerts.openvoiceos:AddListSubitems` for example, is in the `-low` prototype store as well as in the head.

That is the intent. The `-low` stage is the fallback the head did not answer: it runs only after `match_high` declined at `conf_high` and `match_medium` declined at `conf_medium`. That holds because all three tiers are in the pipeline list: OVOS calls `ovos-m2v-pipeline-low` only after the `-high` and `-medium` entries above it returned nothing. A list that names `-low` without the other two gets a prototype stage with no head in front of it, which is the standalone prototype plugin's layout, and the T-1621 warning below applies to it. For a trained label the head answers first, above those thresholds, and the stage never sees the utterance. Below them the head has said it does not know, and the skill's own templates are the better source of an answer than a class the head scored under 0.50.

This is why the warning in the standalone plugin's docstring (T-1621: a prototype stage that runs *before* the classifier takes the utterance away from a head that would have matched it, measured as 7 of 117 alerts handler tests and 6 of 53 volume golden rows) does not apply here. That warning is about stage ORDER. At the `-low` position the classifier has already had both of its tiers. A deny list of the trained classes would be the opposite trade: it would leave a declined utterance unanswered.

Known cost: one global lock covers the first `from_pretrained` of a model (`load_shared_model`), so two plugins that name two DIFFERENT models and boot at the same time wait for each other. The first load measured 13.33s. No deadlock is possible (no path holds an instance lock inside the global one) and OVOS boots pipeline plugins in sequence, so this is latency under a concurrent boot, not a failure.

The same tier suffixes apply to `ovos-m2v-prototype-pipeline-high/medium/low`, the standalone prototype plugin.

### An exact template line goes to the prototype stage

A trained head can only answer with a label it was trained on. When a skill
declares an intent the model does not know, and the utterance is one of that
intent's own template lines, the head still has an answer: the nearest label
it can emit. That answer is wrong, and it is wrong at a high confidence.
This was measured on `ovos-skill-volume` with the classifier before the
prototype stage: `crank the volume up`, an exact line of
`volume.max.boost.intent`, routed to `increase_volume` at 1.00.

So `match_high` and `match_medium` yield such an utterance. The head keeps a
map of the expanded template lines it sees registered. It yields when all
these hold:

- the utterance is one of those lines, ignoring case and punctuation;
- every label that declares the line is registered and is a label this head
  cannot emit (it is not in the model's classes, or it is in
  `ignore_intents`). A line that a trained label also declares is ambiguous,
  and the head answers it;
- a prototype stage can take the utterance: the `-low` stage of this
  instance, or an `ovos-m2v-prototype-pipeline` entry in the caller's
  session pipeline.

The result does not depend on where the prototype stage sits: the skill
author's own line wins, and the head still wins every line it was trained
for. A template line that declares a `{slot}` is never an exact line, since
the utterance carries a value the author did not write. Set
`exact_prototype_first: false` to switch this off.

Every plugin instance that names the same model shares one embedding object: the classifier, its `-low` stage and the standalone prototype plugin load the model once per process (`load_shared_model`).

You control which tiers are active and where they sit relative to other matchers by placing (or omitting) these entries in the `pipeline` list. OVOS evaluates the list top-to-bottom and stops at the first match.

In classifier mode the scores are softmax probabilities (0-1). In prototype mode the scores are cosine similarities (0-1 in practice). You may need to tune `conf_*` downward.

An empty utterance list always returns `None` without attempting inference.

### Example: classifier at high, prototype as medium fallback

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE",
      "conf_high": 0.7
    },
    "ovos-m2v-prototype-pipeline": {
      "model": "minishlab/M2V_multilingual_output",
      "conf_medium": 0.5
    },
    "pipeline": [
      "ovos-stop-pipeline-plugin-high",
      "ovos-converse-pipeline-plugin",
      "ovos-ocp-pipeline-plugin-high",
      "ovos-adapt-pipeline-plugin-high",
      "ovos-m2v-pipeline-high",
      "ovos-ocp-pipeline-plugin-medium",
      "ovos-fallback-pipeline-plugin-high",
      "ovos-m2v-prototype-pipeline-medium",
      "ovos-fallback-pipeline-plugin-medium",
      "ovos-fallback-pipeline-plugin-low"
    ]
  }
}
```

Here the classifier runs at high confidence after Adapt. The prototype plugin runs at medium confidence only if all high-tier matchers have already failed.

## Mixing Both Plugins

The classifier plugin is faster at inference (single matrix multiply + softmax) but is limited to skills present in its training data. The prototype plugin handles any skill that registers Padatious intents with example utterances, at the cost of slightly more memory (one embedding per prototype).

A typical setup runs the classifier first and falls back to the prototype plugin for unrecognised intents:

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE"
    },
    "ovos-m2v-prototype-pipeline": {
      "model": "minishlab/M2V_multilingual_output"
    },
    "pipeline": [
      "ovos-adapt-pipeline-plugin-high",
      "ovos-m2v-pipeline-high",
      "ovos-m2v-prototype-pipeline-high",
      "ovos-fallback-pipeline-plugin-high",
      "ovos-fallback-pipeline-plugin-medium",
      "ovos-fallback-pipeline-plugin-low"
    ]
  }
}
```

---
[← Installation](installation.md) · [Home](README.md) · [Configuration →](configuration.md)
