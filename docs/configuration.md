# Configuration

The pipeline is configured inside `mycroft.conf` under the `intents` section.

## Minimal Configuration: Classifier Mode

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE"
    }
  }
}
```

## Minimal Configuration: Prototype Mode

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "mode": "prototype",
      "model": "minishlab/M2V_multilingual_output"
    }
  }
}
```

Any bare `StaticModel` on Hugging Face (or a local path) can be used as the embedding backbone for prototype mode. No classifier head is needed.

## Full Configuration Reference

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE",
      "mode": "classifier",
      "prototype_strategy": "max_over_all",
      "prototype_top_k": 3,
      "prototype_tau": 0.1,
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": [],
      "timeout": 1
    }
  }
}
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | `str` | unset (defaults to `OpenVoiceOS/ovos-m2v-intents-multilingual`, see below) | Hugging Face repo ID or local path. In classifier mode this must be a `StaticModelPipeline`. In prototype mode any bare `StaticModel` works. Set explicitly to override the default. |
| `models` | `dict[str, str]` | `{}` | Per-language default override, `{locale_or_lang: repo_id}` (e.g. `{"pt": "my-org/pt-model"}`). Matched against the full `lang` locale first, then its primary subtag. Only consulted when `model` is unset. |
| `mode` | `str` | `"classifier"` | Operating mode: `"classifier"` or `"prototype"`. |
| `prototype_k` | `int` | unset (keep all) | Maximum number of prototype embeddings stored per intent label (prototype mode only). Unset keeps every registered sample so exact training samples always match. Set an integer to cap memory. |
| `prototype_strategy` | `str` | `"max_over_all"` | Scoring strategy for prototype mode. See [Prototype Strategies](#prototype-strategies-prototype-mode-only) below. |
| `prototype_top_k` | `int` | `3` | Number of top cosine similarities averaged by the `top_k_mean` strategy. Also the default `k` for `softmax_weighted` when used in scoring. |
| `prototype_tau` | `float` | `0.1` | Temperature for the `softmax_weighted` strategy. Lower values sharpen the distribution toward the maximum. Higher values flatten it toward the mean. |
| `conf_high` | `float` | `0.7` | Minimum score for a `match_high` result. |
| `conf_medium` | `float` | `0.5` | Minimum score for a `match_medium` result. |
| `conf_low` | `float` | `0.15` | Minimum score for a `match_low` result. |
| `ignore_intents` | `list[str]` | `[]` | Intent labels to always discard, regardless of confidence. |
| `timeout` | `int` | `1` | Seconds to wait for Adapt / Padatious manifest responses (classifier mode only). |

## Default model resolution

With no `model` config, the plugin resolves to
`OpenVoiceOS/ovos-m2v-intents-multilingual` for every language, unless
`models[<full locale>]` or `models[<primary subtag>]` says otherwise.
Setting `model` explicitly always wins and skips this resolution entirely.
There is no fallback to the deprecated
`Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2` model;
deployments that need it must set `model` themselves.

`OpenVoiceOS/ovos-m2v-intents-en` is a smaller (16 MB), English-only
alternative available through `model` or `models["en"]`. It is not wired in
as anyone's default, purely to keep the built-in per-language table small;
its held-out accuracy is comparable to the multilingual model. Pick it
where the smaller footprint is worth trading off broader language
coverage.

## Prototype Strategies (prototype mode only)

`prototype_strategy` selects the algorithm used to aggregate a label's sample embeddings into match scores. Defined in `ovos_m2v_pipeline/strategies.py:51` as `PrototypeStrategy`.

| Value | Storage | Scoring | Notes |
|-------|---------|---------|-------|
| `max_over_all` | Every sample per label (random subsample when `prototype_k` is set) | Max cosine over stored anchors | Default. |
| `mean_centroid` | 1 anchor = mean of all samples | Cosine to centroid | Cheapest storage and inference. A classic prototype baseline. |
| `medoid` | 1 anchor = sample closest to centroid | Cosine to medoid | Not sensitive to outliers. Avoids averaging blur. |
| `top_k_mean` | All samples kept | Mean of top-`prototype_top_k` cosines | Combines sharpness of max with smoothing. |
| `farthest_point` | Up to `prototype_k` samples via maximin sampling | Max cosine | Anchors span the example space. Good for diverse phrasings. |
| `kmeans_centers` | Up to `prototype_k` spherical k-means centroids | Max cosine | Useful when a label has multi-modal phrasings. |
| `softmax_weighted` | All samples kept | Softmax-weighted average of cosines | `prototype_tau` controls sharpness. |

The default `max_over_all` produces byte-identical results to the store behavior before strategies were introduced. Existing `.npz` files and tuned thresholds remain valid.

## Confidence Thresholds

Each `conf_*` key sets the minimum score required for the corresponding tier method to return a match:

- **`conf_high`**: threshold for `match_high()`, called when `ovos-m2v-pipeline-high` appears in the pipeline list.
- **`conf_medium`**: threshold for `match_medium()`, called when `ovos-m2v-pipeline-medium` appears.
- **`conf_low`**: threshold for `match_low()`, called when `ovos-m2v-pipeline-low` appears.

OVOS evaluates the pipeline list top-to-bottom and stops at the first match. Only the tiers you add to the pipeline list are ever invoked. Unused tiers consume no resources.

In classifier mode the scores are softmax probabilities (sum to 1 across all labels). In prototype mode the scores are cosine similarities (typically 0-1). You may need to tune thresholds downward.

Only the top-ranked valid intent is returned at each tier. If its score is below the threshold, `None` is returned and the OVOS pipeline moves to the next entry.

## Ignoring Intents

Use `ignore_intents` to suppress specific labels that cause repeated false matches:

```json
{
  "ignore_intents": [
    "ovos-skill-hello-world.openvoiceos:Greetings.intent"
  ]
}
```

Labels in this list are filtered out before any confidence check.

## Using a Local Model

Set `model` to an absolute path to use a locally saved model:

```json
{
  "model": "/opt/models/m2v_intents_LaBSE"
}
```

See [Models](models.md) for available pre-trained models and [Training](training.md) to produce your own.

---
[← OVOS Pipeline Plugin](ovos_pipeline.md) · [Home](README.md) · [Pipeline Internals →](pipeline.md)
