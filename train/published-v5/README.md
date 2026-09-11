# The scripts that built the published v5 models

Twenty model repositories sit under the `OpenVoiceOS` account on the Hub.
Eighteen of them were produced by the three scripts in this directory. They are
committed exactly as they ran, with no edit of any kind, so each one can be
checked against the artefacts it produced. A tidied script cannot be.

They are not the current path. `train/build_from_skills.py` is. These are here
so a published model can be traced to the code that made it.

## build_v5.py

Reads the skill list at `~/tmp/m2v-retrain-v4/ovos_org_skills.txt` and walks
each skill's `locale/` directory under `~/AgentWorkspaces/ovos/skills`, taking
`.intent` files as labelled templates and `.entity` files as slot values. Fills
`{slot}` placeholders, falling back to a small built-in value set for common
slot names. Writes `train_<lang>.jsonl`, `test_<lang>.jsonl` and
`build_report_v5.json` into `~/tmp/m2v-dataset-v5`.

It stores templates unexpanded, so a row can read
`add (items|entries) to the list`. That is deliberate: expansion happens at
training time. A reader who mistakes those rows for corrupt training text will
reach the wrong conclusion, and the row count settles it, since the trainer
reports about three times as many utterances as this file holds lines.

## gen_ood_v5.py

Reads `train_en.jsonl` and `test_en.jsonl` from the same directory and builds a
held-out English paraphrase set by synonym and register substitution. Writes
`ood_en.jsonl` and `ood_report_v5.json`. Only the four English models carry an
out-of-distribution figure, so only they used it.

## train_matrix.py

Reads a `{label, text}` template file, expands each line through
`ovos_spec_tools.expansion.iter_expand` with a per-line cap, trains a
model2vec classification head on the result, and writes the model together with
`eval_report.json`. That report carries `train_path`, `base`,
`n_train_utterances` and the accuracy figures, and `train_path` is what makes
each published model traceable.

## Which model came from which script

Eighteen models name a `train_path` under `~/tmp/m2v-dataset-v5`, so their
corpus came from `build_v5.py` and their training from `train_matrix.py`: the
four `ovos-m2v-intents-en-*-v5` sizes, the six `ovos-m2v-intents-multi-*-v5`
variants, and the eight per-language models for `ca-ES`, `da-DK`, `de-DE`,
`es-ES`, `gl-ES`, `nl-NL`, `pt-BR` and `pt-PT`. The four English models also
used `gen_ood_v5.py`.

## The two models these scripts do not account for

`ovos-m2v-intents-multilingual` and `ovos-m2v-intents-en` predate the v5 line.
Neither carries an `eval_report.json`, so neither records a `train_path`, and
their builder cannot be read from the artefact at all. Both model cards say
their corpus is repeatable through `train/build_dataset.py`.

Treat those two as suspect. `train/build_dataset.py` collected a template
expansion into a shared list before validation could reject it, so a template
that failed part-way through expansion contributed its prefix as an entity
value. Measurement over the shipped `.entity` files puts the count of affected
values at zero, which makes the defect latent rather than active, but the claim
about these two models rests on a card sentence rather than on anything the
artefact records.

That is the real provenance gap. Not that these scripts lived outside the
repository, which this directory fixes, but that two published models record
nothing about where they came from.
