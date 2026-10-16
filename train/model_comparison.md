# Model Comparison

| Language     | Model                             |   Accuracy |   F1 Score |
|:-------------|:----------------------------------|-----------:|-----------:|
| multilingual | minishlab/M2V_multilingual_output |   0.991617 |   0.991138 |

The row above comes from a row-level train/test split: individual rows are shuffled and
divided regardless of which source template produced them. Many rows in this corpus are
expansions of the same `.intent` template line, so a row-level split puts near-identical
phrasings on both sides and scores the model on wording it has already seen. It is not a
generalisation figure.

`train/build_dataset.py` splits at the template level: every expansion of one template
is kept whole on one side, so a held-out hit means the model placed a phrasing it never saw
nearest the right intent. Measured that way, per language, on the same backbone:

| Language | Accuracy | Labels | Held-out rows |
|---|---:|---:|---:|
| en-US | 0.8027 | 189 | 4000 |
| nl-NL | 0.6740 | 164 | 4000 |
| sv-SE | 0.6690 | 176 | 4000 |
| pt-BR | 0.6607 | 147 | 4000 |
| it-IT | 0.6028 | 171 | 4000 |
| da-DK | 0.5950 | 180 | 4000 |
| ca-ES | 0.5735 | 182 | 4000 |
| es-ES | 0.5630 | 181 | 4000 |
| eu-ES | 0.5025 | 176 | 4000 |
| pt-PT | 0.4898 | 156 | 4000 |
| de-DE | 0.4838 | 183 | 4000 |
| fr-FR | 0.4612 | 198 | 4000 |

Each row is 4,000 held-out rows scored against roughly 180 of that language's own labels,
with centroids capped at 300 rows per label. These come from an unmerged pin set and will
move once the pending pull requests land. The spread between languages is not a
difficulty ranking: en-US carries 692,290 training rows against gl-ES's 51,219, and every
non-English locale in this corpus is machine-translated and unvouched, so a low number may
be measuring the text rather than the model.
