# Domain (Parallel-Argmax) Trained Classifier Pipeline

This page documents two layers that ship together:

* **`Model2VecDomainIntentPipeline`** — the OPM-discoverable pipeline class. Entry point: `ovos-m2v-domain-intent-pipeline`. Runs in classifier mode (registration tracking, confidence tiers, the `-low` prototype stage); matching is delegated to a parallel-argmax trained classifier (one classifier per domain, no top-level router). The encoder is the bare embedding of `model`, taken from `load_shared_model`. Only labels a loaded skill registered can match.
* **`DomainIntentClassifier`** — the supervised classifier used internally by that pipeline. Same training shape as [`HierarchicalIntentClassifier`](hierarchical_classifier.md) but without the domain router stage; at inference time every per-domain head scores the query and a single global argmax over their softmax outputs picks the winner.

A separate entry point keeps the domain-trained pipeline independently selectable in `default_pipeline` ordering, with its own `intents.<key>` config block.

## When to pick this over flat / hierarchical

* **vs flat** — each per-domain classifier fits on a different sample set, so its decision boundary genuinely differs from a single global classifier; adding a new skill only requires retraining one per-domain head, not the whole flat model.
* **vs hierarchical** — no top-level router to misroute on. Inference runs every domain head (more compute) but cannot reject a query because of a router mistake.

## Enabling

```json
{
  "intents": {
    "ovos-m2v-domain-intent-pipeline": {
      "model": "minishlab/potion-multilingual-128M",
      "model_path": "/path/to/m2v_domain_intents_bundle"
    }
  }
}
```

* `model` — the bare `StaticModel` encoder used at training time.
* `model_path` — local directory containing the saved `DomainIntentClassifier` bundle.

## Architecture

```
              query embedding
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │domain A│  │domain B│  │domain C│   per-domain LR
   │  LR    │  │  LR    │  │  LR    │   classifiers
   └────────┘  └────────┘  └────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
            global argmax over
            all softmax outputs
```

## Bundle layout

`DomainIntentClassifier.save(path)` produces:

```
<bundle>/
    manifest.json
    intent/
        <domain_a>/classifier.joblib
        <domain_b>/classifier.joblib
        ...
```

`manifest.json` records the list of domains. Each `classifier.joblib` is a single scikit-learn `LogisticRegression` fitted on raw model2vec embeddings of the sentences for that domain.

## Programmatic use

```python
import numpy as np
from model2vec import StaticModel
from ovos_m2v_pipeline.domain_classifier import DomainIntentClassifier

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

sentences = ["play africa", "pause music", "lights on", "set thermostat to 21"]
labels    = ["media:play", "media:pause", "home:lights", "home:thermo"]

X = np.asarray(model.encode(sentences))
clf = DomainIntentClassifier.train(X, labels)
clf.save("my_bundle")

clf2 = DomainIntentClassifier.load("my_bundle")
intent, conf = clf2.predict(model.encode(["put on africa"])[0])
```

## Training from a CSV

See [`train/train_domain.py`](training.md#hierarchical-and-domain-trained-classifiers) for the CLI. It reads the same `train.parquet` as the flat and hierarchical trainers.

## See also

- [Training](training.md) — full training pipeline including the flat and hierarchical trainers.
- [Hierarchical Trained Classifier](hierarchical_classifier.md) — two-stage routed counterpart.
- [OVOS Pipeline Plugin](ovos_pipeline.md): how the pipelines plug into the OVOS bus.
