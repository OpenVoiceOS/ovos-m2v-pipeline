# Training

`train/` holds the reproducible pipeline that builds the intent corpus and
fits a classifier on it. You only need it to produce a custom model; the
[pre-trained models](models.md) already cover the standard OVOS skill corpus.

> **Training is on hold.** The Adapt-to-`.intent` refactors change intent names
> across the default skills, and a later unification wave will merge the
> weather condition intents, the alerts create and list families, and the
> volume levels. A classifier's label head is frozen at fit time, so a model
> trained before those land is stale the day they merge. Build the dataset and
> read the manifest as often as you like; run `train.py` only once the
> refactors are merged and the skill pins in `sources.yaml` have been
> regenerated against them.

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

# 3. fit (only once the hold above is lifted)
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
`family`. `source` is the provenance tag — `golden:` rows come from a skill's
own end-to-end corpus.

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
stop pipelines, and the golden end-to-end corpora of the pinned skills.

## Distilling a new base model

If you want to start from a Sentence Transformer with no Model2Vec distillate
yet, edit the model list at the top of `distill.py` and run it. The result can
be passed to `train.py --base-model`.

## Publishing

```bash
huggingface-cli login
python -c "
from model2vec.inference import StaticModelPipeline
m = StaticModelPipeline.from_pretrained('train/model_mul_potion-base-32M')
m.push_to_hub('YourOrg/your-model-name')
"
```

Upload `labels.json` alongside the weights, then point the `model` key in your
OVOS configuration at the new repo.
