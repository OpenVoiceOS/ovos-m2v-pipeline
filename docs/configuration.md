# Configuration

Every key lives under `mycroft.conf["intents"]["ovos-m2v-pipeline"]` (classifier
plugin) or `mycroft.conf["intents"]["ovos-m2v-prototype-pipeline"]` (standalone
prototype plugin). The table lists every key the code reads.

## Both modes

| Key | Type | Default | Meaning |
|---|---|---|---|
| `model` | `str` | unset, resolves per language (see [models.md](models.md)) | Hugging Face repo ID or local path to load. |
| `models` | `dict[str, str]` | `{}` | Per-language override, `{locale_or_lang: repo_id}`. Skipped when `model` is set. |
| `revision` | `str` | unset | Git revision of the `model` repo to pin. Ignored for a local path. |
| `mode` | `str` | `"classifier"` | `"classifier"` or `"prototype"`. Forced to `"prototype"` when the plugin loads as `ovos-m2v-prototype-pipeline`. |
| `conf_high` | `float` | `0.7` classifier, `0.85` prototype | Minimum score for `match_high`. |
| `conf_medium` | `float` | `0.5` classifier, `0.7` prototype | Minimum score for `match_medium`. |
| `conf_low` | `float` | `0.15` classifier, `0.65` prototype | Minimum score for `match_low`. |
| `ignore_intents` | `list[str]` | `[]` | Canonical labels to always discard, after `label_map`. |
| `label_map` | `dict[str, str or [str, str]]` | `{}` | Maps a raw model label to a canonical `skill_id:intent` label. Merges over the built-in OCP/common-query/stop remaps and any `labels.json` the model ships. |
| `valid_labels` | `list[str]` | unset | Allow-list of raw model labels, checked before `label_map`. Classifier mode only; a prototype store is its own allow-list. |
| `timeout` | `int` | `1` | Seconds to wait for an Adapt or Padatious manifest response. |
| `preload_model` | `bool` | `false` | Load the model at construction instead of on first use. |
| `model_load_budget` | `float` | `0.5` | Seconds a match call waits for a cold-start model load before returning no match for that utterance. |

## Classifier mode only

| Key | Type | Default | Meaning |
|---|---|---|---|
| `low_tier` | `str` | `"prototype"` | Engine behind `ovos-m2v-pipeline-low`: `"prototype"` (skill templates) or `"classifier"` (the trained head at `conf_low`). |
| `low_prototype` | `dict` | `{}` | Prototype-mode keys for the `-low` stage (`conf_high`, `conf_medium`, `conf_low`, `ignore_intents`, `prototype_k`, ...). The stage always loads this plugin's own model. A `model` key here is overwritten, and logged as discarded, only when it names a different model than the classifier's; naming the same model is a no-op with no warning. |
| `renormalize` | `bool` | `false` | Renormalize softmax probabilities over the surviving label subset after filtering to registered intents. |

## Prototype mode only

| Key | Type | Default | Meaning |
|---|---|---|---|
| `prototype_k` | `int` | unset, keep every sample | Cap on stored prototype embeddings per label. |
| `prototype_strategy` | `str` | `"max_over_all"` | Anchor and scoring algorithm. See [ovos_pipeline.md](ovos_pipeline.md#prototype-strategies). |
| `prototype_top_k` | `int` | `3` | K for the `top_k_mean` strategy. |
| `prototype_tau` | `float` | `0.1` | Softmax temperature for the `softmax_weighted` strategy. |
| `prototype_cache` | `bool` | `true` | Cache each label's encoded prototypes to disk across restarts. |
| `prototype_cache_dir` | `str` | `{XDG_DATA_HOME}/mycroft/m2v_prototypes/` | Override the cache directory. |
| `prebuilt_prototypes` | `str` | unset | Local directory or Hugging Face repo id holding a prebuilt artifact from `ovos-m2v-prototypes export`. |

---
[← OVOS Pipeline Plugin](ovos_pipeline.md) · [Home](../README.md) · [Models →](models.md)
