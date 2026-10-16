# Models

## Classifier models

`OpenVoiceOS/ovos-m2v-intents-multilingual` is the default for every
language with no `models` override. It is a `StaticModelPipeline`: an
embedding model plus a trained classifier head, distilled from a
multilingual base and fit on the OVOS skill intent corpus (see
[Training](training.md)).

`OpenVoiceOS/ovos-m2v-intents-en` is an English-only alternative. It holds
one model per potion English base: potion-base-32M at the root (126 MB),
with `fp16`, `int8` and `onnx` copies beside it, and `potion-base-8M`,
`potion-base-4M` and `potion-base-2M` folders down to 8 MB. Its model card
lists each one's gold accuracy on English next to the multilingual
model's. It is not the default for any language; opt in with
`models["en"]` (or `model`), and pick a folder with `subfolder`:

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "models": {"en": "OpenVoiceOS/ovos-m2v-intents-en"},
      "subfolder": "int8"
    }
  }
}
```

`subfolder` applies to every model the plugin loads, so set it only when
each of them has that folder. Both repositories carry `fp16` and `int8`.
The English models are measured in classifier mode only.

Check each repo's model card on Hugging Face for its exact size, language
list and accuracy.

## Prototype mode

Prototype mode needs no classifier head: any Model2Vec `StaticModel` works
as the embedding backbone, including the base a classifier model was
distilled from. `train/distill.py` lists the base models this repo
distills; a base's own model card names its source Sentence Transformer and
language coverage.

## Pointing at another model

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "/path/to/my_custom_model"
    }
  }
}
```

`model` accepts a Hugging Face repo id or a local directory holding a
`StaticModelPipeline` checkpoint (classifier mode) or a bare `StaticModel`
(prototype mode). `models` sets a different `model` per language. See
[Configuration](configuration.md) for both keys, and
[Training](training.md) to produce a model of your own.

---
[← Configuration](configuration.md) · [Home](../README.md) · [Training →](training.md)
