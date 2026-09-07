# Training

`train/` holds the reproducible pipeline that builds the intent corpus and
fits a classifier on it. You only need it to produce a custom model; the
[pre-trained models](models.md) already cover the standard OVOS skill corpus.

The Adapt-to-`.intent` refactors and the unification wave that merged the
weather condition intents, the alerts create and list families, and the
volume levels are merged in the revisions `sources.yaml` pins, so `train.py`
runs against those pins. A classifier's label head is frozen at fit time: the
day a skill renames or folds an intent again, its pin has to be refreshed and
the model refit before that skill's labels are trusted.

```
train/
├── sources.yaml        # every source, pinned to an immutable revision
├── build_dataset.py    # resolve, normalise, dedup, split, write the manifest
├── train.py            # fit a classifier on the built corpus
├── distill.py          # distill a Sentence Transformer into a Model2Vec base
└── predict.py          # inference smoke test
```

Labels are `<skill_id>:<intent_name>` exactly as the pipeline registers them at
runtime. The scheme, the pipeline families, the dedup rules, and the procedure
for renames and merges are in [Label scheme](labels.md). Read that page before
changing anything in `sources.yaml`.

## Reproducing end to end

The builder reads git sources from local clones, so it needs a workspace with
the OVOS repos checked out; a pinned revision the clone does not carry is a
hard error, not a fallback to the branch tip. The Hugging Face sources are
downloaded, pinned revision by pinned revision, through the shared cache. So
the builder fetches only what a pin names, and a pin that has moved fails the
build rather than quietly changing the corpus.

```bash
python -m venv .venv && . .venv/bin/activate
pip install pandas pyarrow pyyaml huggingface_hub scikit-learn model2vec

# 1. count what the pinned revisions currently yield, writing nothing
python train/build_dataset.py --dry-run --workspace ~/AgentWorkspaces

# 2. build it
python train/build_dataset.py --workspace ~/AgentWorkspaces --out train/dataset

# 3. fit against the pinned revisions
python train/train.py --dataset train/dataset --base-model minishlab/potion-base-32M
```

`--allow-ambiguous` keeps rows whose `(utterance, lang)` carries more than one
label; by default they are dropped and the label pairs are reported.

Step 2 writes `train.parquet`, `test.parquet`, their JSONL twins,
`labels.json`, and `manifest.json`. The manifest records the row counts per
source, label, language and family, every drop the filters made, the case
duplicates that were collapsed, the revisions actually used, and the sha256 of
each output. Two runs from the same pins produce the same shas.

Each row carries `lang`, `label`, `utterance`, `source`, `skill_id` and
`family`. `source` is the provenance tag: `golden:` rows come from a skill's
own end-to-end corpus, `skill-intents:` rows from the `.intent` templates that
skill ships.

`labels.json` is the manifest the pipeline reads beside a model (m2v#73). Ship
it with the model so the plugin can restrict matching to the label set the
model was actually trained on.

## Regenerating after skills merge

When skill repos move, refresh their pins. Either edit `skill_refs` in
`sources.yaml`, or pass a generated list:

```bash
for d in ~/AgentWorkspaces/ovos/skills/ovos-skill-*; do
  echo "$(basename $d) $(git -C $d rev-parse origin/dev)"
done > skill-refs.txt

python train/build_dataset.py --skill-ref-list skill-refs.txt --dry-run
```

Diff the new manifest against the old one. A label count that moved, rows
shifted by an alias, or a new entry in the rare-label list all mean the corpus
changed shape and the model has to be refit.

The pins are also the label vocabulary: the builder reads each pinned repo's
entry point and registered intents and drops any corpus label those refs do
not attest. `unresolved_labels` in the manifest is where an unpinned or
archived skill shows up. Adding a skill to `skill_refs` is how you add its
intents to the vocabulary.

## Sources

`sources.yaml` is the authority; each entry carries its revision, its license
note, and the column mapping into `(skill_id, intent, utterance, lang)`. In
summary the corpus comes from the ovos-localize classification export and the
lang-support tracker CSVs, the legacy GitLocalize export, the OCP music query
templates, the common-query and weather intent corpora, an LLM-augmented
balancing set, the locale intent files of the OCP, common-query, persona and
stop pipelines, the `locale/<lang>/**/*.intent` templates of the pinned skills
themselves, and their golden end-to-end corpora.

The skills' own locale trees are what the runtime loads, so they are the only
source that attests every locale a skill supports; the external exports carry
only the languages a translation round happened to reach. Both layouts skills
use are read, a top-level `locale/` and a `<package>/locale/`, at any depth
beneath the locale directory. Templates expand the same way the pipeline
plugins' do, and their `{slot}` placeholders are filled from the entity values
the same pinned revisions attest. These rows are ordinary training data rather
than golden: a `.intent` line is a template a skill ships, not a sentence
checked against a live registration.

## Distilling a new base model

If you want to start from a Sentence Transformer with no Model2Vec distillate
yet, edit the model list at the top of `distill.py` and run it. The result can
be passed to `train.py --base-model`.

## Publishing

Both `build_dataset.py` and `train.py` can push their output straight to the
Hub with `--push-to`, given a token in the `HF_TOKEN` environment variable
(`huggingface-cli login` also works; either way, never put a token in a
config file or a command line argument). Every upload is additive — it
creates or updates the named files in one commit each and never deletes
anything else already in the target repo.

```bash
export HF_TOKEN=hf_...

# corpus: train.parquet, train.jsonl, test.parquet, test.jsonl,
# labels.json and manifest.json
python train/build_dataset.py --push-to YourOrg/your-dataset-name

# trained pipeline: everything train.py wrote to --out, including an
# onnx/ subdirectory if one is present, plus a training_manifest.json
# that records the dataset's own file hashes and the model2vec version
# training ran with
python train/train.py --push-to YourOrg/your-model-name
```

Add `--dry-run` to either command to print what would be uploaded without
touching the network — useful for checking a corpus or a model repo id
before spending a real commit on it. `train.py --dry-run --push-to ...`
still trains and writes `--out` locally; only the upload is skipped.

Point the `model` key in your OVOS configuration at the new model repo once
it is published.
