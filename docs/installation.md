# Installation

## Install from PyPI

```bash
pip install ovos-m2v-pipeline
```

This installs the plugin and registers it as an `opm.pipeline` entry point under the name `ovos-m2v-pipeline`.

## Dependencies

| Package | Purpose |
|---------|---------|
| `model2vec[inference]` | Static embedding model and classification pipeline |
| `ovos-plugin-manager` | OVOS plugin infrastructure and base pipeline class |
| `ovos-bus-client` | Message bus communication |
| `ovos-config` | Reading `mycroft.conf` configuration |
| `ovos-utils` | Logging and bus utilities |
| `ovos-workshop` | OVOS skill framework compatibility |

## Model Download

The model is **not bundled** with the package. It is downloaded on first use from Hugging Face Hub. The default model for every language is `OpenVoiceOS/ovos-m2v-intents-multilingual`.

`OpenVoiceOS/ovos-m2v-intents-en` is a smaller (16 MB), English-only alternative for size-constrained deployments, with comparable held-out accuracy to the multilingual model; it is not the default purely to keep the built-in per-language table small. Opt into it (or any other model) with the `model` key, or per-language with the `models` key, in your configuration (see [Configuration](configuration.md)).

## Verifying the Installation

After installing, confirm the entry point is registered:

```bash
python -c "from ovos_m2v_pipeline import Model2VecIntentPipeline; print('OK')"
```

---
[Home](README.md) · [OVOS Pipeline Plugin →](ovos_pipeline.md)
