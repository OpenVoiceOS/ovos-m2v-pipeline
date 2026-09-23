# Models

## Classifier models

`OpenVoiceOS/ovos-m2v-intents-multilingual` is the default for every
language with no `models` override. It is a `StaticModelPipeline`: an
embedding model plus a trained classifier head, distilled from a
multilingual base and fit on the OVOS skill intent corpus (see
[Training](training.md)).

`OpenVoiceOS/ovos-m2v-intents-en` is a smaller, English-only alternative
with comparable held-out accuracy, but its model card reports that
end-to-end dispatch testing ranks paraphrases worse in prototype mode than
the multilingual model does on the same cases. That is why it is not wired
in as the default for any language, English included. Opt in with `model`
or `models["en"]` only after reading the card's Trade-offs section and
accepting that weakness. Check each repo's model card on Hugging Face for
its exact size, language list and held-out accuracy.

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
