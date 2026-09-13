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

## Confidence Tiers and the Pipeline List

OVOS does **not** call all three tiers of a plugin automatically. Instead, each tier is a separate named entry in the `pipeline` list, identified by a `-high`, `-medium`, or `-low` suffix:

| Pipeline entry | Method called | Threshold key | Default |
|----------------|---------------|---------------|---------|
| `ovos-m2v-pipeline-high` | `match_high()` | `conf_high` | `0.70` |
| `ovos-m2v-pipeline-medium` | `match_medium()` | `conf_medium` | `0.50` |
| `ovos-m2v-pipeline-low` | `match_low()` | `conf_low` | `0.15` |

The same applies to `ovos-m2v-prototype-pipeline-high/medium/low`.

You control which tiers are active and where they sit relative to other matchers by placing (or omitting) these entries in the `pipeline` list. OVOS evaluates the list top-to-bottom and stops at the first match.

In classifier mode the scores are softmax probabilities (0-1). In prototype mode the scores are cosine similarities (0-1 in practice), which cluster in a lower and narrower band than softmax output. `Model2VecPrototypePipeline` (and any classifier-entry-point pipeline configured with `"mode": "prototype"`) therefore uses its own defaults instead of the classifier ones: `conf_high` `0.60`, `conf_medium` `0.45`, `conf_low` `0.30`. These were calibrated against the default multilingual model's cosine-similarity distribution for matching vs. non-matching utterances (see `scripts/calibrate_prototype_thresholds.py`). Set `conf_*` explicitly in your config to override them.

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
