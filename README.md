[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenVoiceOS/ovos-m2v-pipeline)

# OVOS Model2Vec Intent Pipeline

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org). It uses the [Model2Vec](https://github.com/MinishLab/model2vec) model for intent classification.

This plugin uses a pretrained Model2Vec model to classify natural language utterances into intent labels registered with the system (Adapt, Padatious, and plugin-specific labels). It only considers intents from loaded skills and ignores labels from unregistered intents. Use this pipeline when deterministic engines fail to give a high-confidence match.

---

## Features

* Model2Vec drives intent classification.
* The plugin integrates directly with OVOS pipelines.
* The Model2Vec models train on [GitLocalize](https://gitlocalize.com/users/OpenVoiceOS) exports.
* English models come in several sizes, distilled from [Potion](https://huggingface.co/collections/minishlab/potion-6721e0abd4ea41881417f062).
* The multilingual model is distilled from [LaBSE](https://huggingface.co/minishlab/M2V_multilingual_output).
* The plugin syncs Adapt and Padatious intents dynamically at runtime.
* The plugin considers only intents from loaded skills and ignores unregistered labels.

> English models range from 8 MB to 150 MB. The multilingual model (the default) is over 500 MB.

---

## Installation

Install the plugin with `pip`:

```bash
pip install ovos-m2v-pipeline
```

---

## Configuration

In your `mycroft.conf`:

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE",
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": []
    }
  }
}
```

* `model`: Path to your pretrained Model2Vec model or huggingface repo.
* `conf_xxx`: Minimum confidence threshold for intent matching.
* `ignore_intents`: List of canonical labels to exclude from matching (deny-list, applied after `label_map`).
* `valid_labels`: Classifier-mode only. List of raw model labels eligible to match (allow-list, checked before `label_map` is applied). When unset, every label is eligible. Not consulted in prototype mode: the prototype store already only holds runtime-registered labels, so it is its own allow-list.
* `label_map`: Maps a raw model label to its canonical `skill_id:intent` label. Merges over (and can override) the built-in OCP/common-query/stop remaps and any labels the model itself declares in `labels.json`; see [Trained models document their labels](#trained-models-document-their-labels).
* `prototype_strategy`: Scoring strategy for prototype mode (default `"max_over_all"`, back-compatible). See [docs/strategies.md](docs/strategies.md).
* `prototype_top_k`: Top-k cosines averaged by the `top_k_mean` strategy (default `3`).
* `prototype_tau`: Softmax temperature for the `softmax_weighted` strategy (default `0.1`).
* `preload_model`: Load the embedding model at construction instead of on first use (default `false`). ovos-core builds every installed pipeline plugin at boot, so the default defers the load to the first registration or match; set this if a deployment would rather pay the load cost once at boot than on the first query. See `model_load_budget` below for what happens to that first query when the model is still loading.
* `model_load_budget`: Seconds a match call waits for a cold-start model load before giving up on that one utterance (default `0.5`). The load keeps running in the background regardless; matching resumes automatically once it completes. If the load itself fails (bad model id, unreachable host), it retries on its own after a backoff that starts at 30s and doubles on each consecutive failure, capped at 15 minutes.

> The Model2Vec model is pretrained on GitLocalize exports. It **cannot learn new skills** dynamically.

---

## Prototype cache (prototype mode)

Prototype mode rebuilds its label set from scratch on every boot: each
registered skill's example utterances are re-encoded through the embedding
model as `padatious:register_intent` / OVOS-INTENT-4 template registrations
arrive. On an install with many skills, re-encoding the same, unchanged
templates on every restart is pure repeated work.

To avoid that, each label's encoded prototypes are cached to disk, keyed on
the *inputs* of its registration: the model id, the installed `model2vec`
version, the anchor-selection parameters (`prototype_k`, `prototype_strategy`,
the entity-expansion cap), the registration's raw (pre-expansion) template
lines, and any registered entity values its `{slot}` placeholders reference.
When a label's next registration hashes to the same key, its embeddings are
loaded from the cache instead of being re-encoded; any other change to those
inputs is a plain cache miss, so nothing needs to be told explicitly to
invalidate a stale entry when a skill updates its templates or the model is
swapped.

Removal (`detach_intent`, `detach_skill`, and their OVOS-INTENT-4
equivalents) is the one case that *does* need explicit invalidation: nothing
about a removed skill's registration inputs changes when it unloads, so its
cache entry is deleted immediately rather than left to resurrect the skill's
intents on the next boot.

Cache files live under `{XDG_DATA_HOME}/mycroft/m2v_prototypes/` by default
(one small `.npz` file per label) and are local-disk-only: nothing here ever
touches the network. A corrupt or unreadable entry is logged and treated as a
miss, never as a fatal error.

Configuration (under the same `intents.<entrypoint-name>` block as above):

* `prototype_cache`: Enable/disable the cache (default `true`).
* `prototype_cache_dir`: Override the cache directory.

---

## Which entrypoint do I want?

This plugin ships two `opm.pipeline` entrypoints. Both use the same
`Model2VecIntentPipeline` class, running in different modes:

* **`ovos-m2v-pipeline`** (`Model2VecIntentPipeline`, `mode: "classifier"`, the
  default) loads a pretrained, **frozen** classification head with a fixed
  label set baked in at training time. It is fast and needs no runtime
  fitting, but it can only ever return the labels it was trained on. It still
  tracks OVOS-INTENT-4 `ovos.intent.register.template` registrations from
  skills so it can gate/allowlist a trained label, but registering a new
  intent that was not part of training does **not** teach it to that skill.
  That intent will never be matched.
* **`ovos-m2v-prototype-pipeline`** (`Model2VecPrototypePipeline`, `mode:
  "prototype"`) loads a bare embedding model with no classification head and
  builds its label set entirely at runtime, from the example utterances
  supplied by Adapt/Padatious registrations and OVOS-INTENT-4 template
  registrations. Use this entrypoint whenever skills need to register new
  intents (including custom/dynamically-created skills) that must actually be
  matched.

You can enable both entrypoints together. Configure each independently under
its own `intents.<entrypoint-name>` key (see the `Model2VecPrototypePipeline`
docstring for an example), so a deployment can keep the fast frozen
classifier for its core trained intents while the prototype matcher picks up
everything else.

---

## Trained models document their labels

In classifier mode the label head is frozen at training time, so which bus
intent each label denotes (`ocp:play` -> `ovos.common_play:ovos.common_play.play_search`,
for example) is a property of that particular trained model, not of the
plugin code. A model can ship this mapping alongside its weights as a
`labels.json` file in its repo/directory:

```json
{
  "valid_labels": ["my_domain:book_flight", "my_domain:cancel_flight"],
  "families": {
    "my_domain:book_flight": "skill",
    "my_domain:cancel_flight": "skill"
  }
}
```

A canonical label carries no `.intent` suffix: the suffix names the resource a
skill ships, not the intent, and the legacy Padatious handler strips it before
the label reaches the bus.

`labels.json` has the same shape as the `label_map` config option, plus an
optional `valid_labels` list and an optional `families` map. `valid_labels` lists the model's raw labels -
the ones it was actually trained on - not the `label_map` targets; the
allow-list check happens before `label_map` resolution, so it also covers
`ocp:play` / `stop:stop` / `common_query:common_query` before those get
rewritten to their bus topics. `families`
names the family each canonical label belongs to (`skill`, `ocp`,
`common_query`, `stop` or `persona`), which is what the per-family claim
filter keys on. A label missing from the map is logged once and treated as
`skill`.

`valid_labels` only gates classifier mode: a prototype-mode label is registered at runtime rather than trained into the model, so it is legitimately absent from `labels.json`, and the prototype store itself is the allow-list there. When present, list the labels a model was trained on in its model card too, so users know what to expect without downloading it first.

Three layers combine, each overriding the previous on a per-key basis:

1. Built-in defaults (the OCP / common-query / stop remaps that predate this
   mechanism).
2. The loaded model's own `labels.json`, if it ships one.
3. The deployment's `label_map` / `valid_labels` config.

For a Hugging Face hub model id, `labels.json` rides the same local cache
as the model's own weights: the plugin never fetches it over the network at
construction time, and only ever consults the cache entry already populated
by whatever downloaded the model. If the model has not been cached yet, or
the cached copy has no `labels.json`, the manifest layer is treated as empty
- it is never a reason for plugin construction to touch the network or block
on one.

A missing or corrupt `labels.json` is logged and ignored; the plugin falls
back to the layers below it rather than failing to load. A `label_map` target
that is not a `skill_id:intent` string (no colon) is logged as a warning
(once per label) and used as-is - the plugin never invents a bus topic
from it.

---

## Usage

The `Model2VecIntentPipeline` class integrates with the OVOS intent system. It:

1. Receives an utterance (text).
2. Predicts intent labels using the pretrained Model2Vec model.
3. Filters out intents that are not part of the loaded skills.
4. Returns a match for the highest-confidence intent from the list of valid intents.

---

## Tips

* Tune `min_conf` to control the confidence threshold for intent matching.
* Use the `ignore_intents` list to filter out specific problematic intents from predictions.
* The plugin syncs Adapt and Padatious intents automatically at runtime, over the OVOS message bus.

> Pre-trained models are available in the [ovos-model2vec-intents](https://huggingface.co/collections/Jarbas/ovos-model2vec-intents-681c478aecb9979e659b17f8) Hugging Face collection.

---

## Related projects

* [OpenVoiceOS](https://github.com/OpenVoiceOS): the OVOS org, and the intent-pipeline system this plugin extends.
* [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): the base pipeline class and plugin infrastructure.
* [ovos-workshop](https://github.com/OpenVoiceOS/ovos-workshop): the skill framework this plugin integrates with.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## Credits

The model2vec intent pipeline was first prototyped by
[TigreGótico](https://tigregotico.pt) under the [ILENIA](https://proyectoilenia.es)
project for [OpenVoiceOS](https://openvoiceos.org) and later extended
with an embeddings-only mode and new models, through the NGI0 Commons Fund.

<img src="./ilenia.png" width="128"/>

> This project was funded by the Ministerio para la Transformación Digital y de la Función Pública and Plan de Recuperación, Transformación y Resiliencia - Funded by EU, NextGenerationEU within the framework of the project [ILENIA](https://proyectoilenia.es) with reference 2022/TL22/00215337

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).

---

## Training your own model

`train/` builds the intent corpus from pinned sources and fits a classifier on
it. See [docs/training.md](docs/training.md) for the end-to-end recipe and the
current hold on training runs, and [docs/labels.md](docs/labels.md) for the
label scheme every model must follow.
