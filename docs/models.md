# Pre-trained Models

All models are published on Hugging Face in the [ovos-model2vec-intents](https://huggingface.co/collections/Jarbas/ovos-model2vec-intents-681c478aecb9979e659b17f8) collection.

They are trained on OVOS skill intent examples from [GitLocalize](https://gitlocalize.com/users/OpenVoiceOS) and the `ovos_intent_examples` / `music_queries_templates` datasets.

---

## Multilingual Model (Default)

| Model | Base | Languages | Approx. Size |
|-------|------|-----------|-------------|
| `Jarbas/ovos-model2vec-intents-LaBSE` | `minishlab/M2V_multilingual_output` (distilled from LaBSE) | en, pt, eu, es, gl, nl, fr, de, ca, it, da | ~500 MB |

### Benchmark

| Language | Accuracy | F1 Score |
|----------|----------|----------|
| multilingual | 0.9916 | 0.9911 |

---

## English Models

Distilled from the [Potion](https://huggingface.co/collections/minishlab/potion-6721e0abd4ea41881417f062) family of English static models.

| Hugging Face Repo | Base Model | Approx. Size | Accuracy | F1 Score |
|-------------------|-----------|-------------|----------|----------|
| `Jarbas/ovos-model2vec-intents-potion-base-2M` | `minishlab/potion-base-2M` | ~8 MB | 0.9233 | 0.9127 |
| `Jarbas/ovos-model2vec-intents-potion-base-4M` | `minishlab/potion-base-4M` | ~16 MB | 0.9129 | 0.9076 |
| `Jarbas/ovos-model2vec-intents-potion-base-8M` | `minishlab/potion-base-8M` | ~32 MB | 0.9303 | 0.9255 |
| `Jarbas/ovos-model2vec-intents-potion-base-32M` | `minishlab/potion-base-32M` | ~128 MB | 0.9338 | 0.9302 |
| `Jarbas/ovos-model2vec-intents-potion-retrieval-32M` | `minishlab/potion-retrieval-32M` | ~128 MB | 0.9408 | 0.9352 |

> Benchmarks were measured on a 10% held-out split of the English training data.

---

## Choosing a Model

| Use case | Recommendation |
|----------|----------------|
| Multilingual OVOS instance | Default multilingual (`LaBSE`-based) |
| English-only, resource-constrained | `potion-base-2M` (~8 MB, loads in ~130 ms) |
| English-only, best accuracy | `potion-retrieval-32M` |
| English-only, balanced | `potion-base-8M` or `potion-base-32M` |

---

## Prototype Mode: Embedding-Only Models

In prototype mode the plugin uses a bare `StaticModel` (no classifier head). Any Model2Vec embedding model works, including the distilled bases used to train the classifier models above:

| Model | Languages | Approx. Size |
|-------|-----------|-------------|
| `minishlab/M2V_multilingual_output` | multilingual | ~500 MB |
| `minishlab/potion-base-2M` | English | ~8 MB |
| `minishlab/potion-base-8M` | English | ~32 MB |
| `minishlab/potion-base-32M` | English | ~128 MB |

Prototype mode requires no training step. Point `model` at any of the above and set `"mode": "prototype"`.

---

## Using a Custom Model

Point `model` at any local directory or Hugging Face repo that contains a `StaticModelPipeline` checkpoint (classifier mode) or a bare `StaticModel` (prototype mode):

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "/path/to/my_custom_model"
    }
  }
}
```

See [Training](training.md) to produce your own model.

---
[← Prototype Strategies](strategies.md) · [Home](README.md) · [Training →](training.md)
