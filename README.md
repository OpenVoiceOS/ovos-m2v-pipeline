[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenVoiceOS/ovos-m2v-pipeline)

# OVOS Model2Vec Intent Pipeline

An intent-matching pipeline plugin for [OpenVoiceOS](https://openvoiceos.org),
built on [Model2Vec](https://github.com/MinishLab/model2vec) static
embeddings. It classifies an utterance against the intents that loaded
skills registered at runtime, either through a pre-trained classifier head
or through the skills' own example utterances with no training step. Use it
as a semantic fallback behind exact matchers such as Adapt and padacioso.

## Install

```bash
uv pip install --prerelease=allow ovos-m2v-pipeline
```

## Configure

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

`intents.pipeline` replaces the whole pipeline list, not only its m2v
entries; see [the pipeline order](docs/ovos_pipeline.md#place-padacioso-before-the-classifier)
for where the other plugins a deployment installs belong in it.

`ovos-padacioso-pipeline-plugin-high` runs first: a frozen classifier only
answers with a label it was trained on, so an exact template line of a
skill it never saw is claimed earlier, and correctly, by an exact matcher.
See [the pipeline order](docs/ovos_pipeline.md#place-padacioso-before-the-classifier)
for the full reasoning.

## Tiers

| Entry-point suffix | Method | Answers from |
|---|---|---|
| `-high` | `match_high()` | the trained head, or the skill's own examples for the standalone prototype plugin |
| `-medium` | `match_medium()` | same, at a lower threshold |
| `-low` | `match_low()` | the skill's own examples by default (`low_tier: "prototype"`) |

## Documentation

| Page | Covers |
|---|---|
| [Configuration](docs/configuration.md) | Every config key, its default and its type |
| [OVOS Pipeline Plugin](docs/ovos_pipeline.md) | Entry points, tiers, modes, caching |
| [Models](docs/models.md) | Published model ids and picking one |
| [Training](docs/training.md) | Building the corpus and fitting a model |
| [Label scheme](docs/labels.md) | Label format, families, dedup and renames |
| [Pre-release quirks](docs/prerelease-quirks.md) | Behaviour changes by pre-release version, reset at each stable release |

Training your own model starts at [Training](docs/training.md).

## License

Apache 2.0. See [LICENSE](LICENSE).

## Credits

First built by [TigreGótico](https://tigregotico.pt) for
[OpenVoiceOS](https://openvoiceos.org) under the
[ILENIA](https://proyectoilenia.es) project, and extended through the
[NGI0 Commons Fund](https://nlnet.nl/commonsfund).

<img src="./ilenia.png" width="128"/>

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)
