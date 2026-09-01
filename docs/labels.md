# Label scheme

A trained classifier's label head is frozen. Whatever string it emits has to be
the string the pipeline sees registered on the bus at runtime, or the match is
discarded before it ever reaches a skill. This page is the contract between the
training corpus and `ovos_m2v_pipeline`.

## The canonical form

A label is `<skill_id>:<intent_name>`, byte-identical to what the plugin
registers.

The plugin builds that string in two places, and neither one normalises
anything. The OVOS-INTENT-4 handler joins the two `§3.2` fields verbatim
(`_intent4_label` in `ovos_m2v_pipeline/__init__.py`). The legacy Padatious
handler takes the registration's `name` as-is, with one exception: because
ovos-workshop dual-registers each intent under both the suffixed legacy name
and the suffixless spec name, a trailing `.intent` is stripped so the prototype
store does not hold every intent twice. Classifier mode then masks
`model.classes_` against the registered set by exact string equality.

Three rules follow.

**Suffix policy: no suffix.** `.intent` and `.voc` are filename extensions of
the resource a skill ships, not part of the intent's identity, and the legacy
handler strips `.intent` before storing the label. Training on
`what.time.is.it.intent` — as the currently deployed model does — produces a
label that matches nothing once the suffix is folded away. Sources keyed on
file names (ovos-localize, the lang-support tracker, the GitLocalize export)
have the suffix stripped on read.

**Case policy: preserved in the intent name.** The runtime lowercases nothing.
Neither the INTENT-4 handler nor the legacy Padatious handler touches case, so
a label's case is whatever the skill registered: Adapt intents come in under
the handler's CamelCase name (`CreateAlarm`), Padatious intents under their
file stem (`what.time.is.it`), and both spellings are legitimate labels that
coexist in one vocabulary.

The builder does lowercase the skill-id half. That is a builder-side
corpus-hygiene assumption, not a description of the runtime: entry-point skill
ids are lowercase by ecosystem convention, and the corpora disagree there only
in case, never meaningfully. The intent half is left exactly as registered.

**Skill-id policy: read it from the entry point.** A skill's id is the key of
its `ovos.plugin.skill` (or `opm.skill`) entry point, and nothing else. Most
resolve to `<repo-name>.<author>`, but the repo name is not always the one you
expect: `ovos-skill-wallpapers` declares `skill-ovos-wallpapers.openvoiceos`,
because its `setup.py` derives the id from the GitHub URL. Guessing the id
from a corpus spelling is how 62k wallpaper rows ended up under a skill id the
runtime never registers. The builder reads the entry point at the pinned
revision instead — see below.

## Families

Every label carries a `family` attribute, emitted as a column in the dataset
and derived from the part before the colon:

| family | label prefix | source of the intents |
|---|---|---|
| `skill` | `<skill_id>` | the skill's own locale files and golden corpus |
| `ocp` | `ocp:` | ovos-ocp-pipeline-plugin locale intents, music query templates |
| `common_query` | `common_query:` | ovos-common-query-pipeline-plugin, the common-query question corpus |
| `stop` | `stop:` | the stop pipeline's locale intents in ovos-core |
| `persona` | `persona:` | ovos-persona locale intents |

The four pipeline families are first-class members of the vocabulary, with all
of their sub-intents, not just the umbrella labels the plugin special-cases.
Whether a family may match is the plugin's decision, taken per family from
config and from the caller's `session.pipeline`; the model's job is to be able
to recognise them at all. Training a model that cannot express `ocp:next`
would make that config flag unimplementable.

Their labels live under the bare pipeline id rather than a skill id, following
the PIPELINE-1 rule that a pipeline's namespace is its pipeline id and matching
how the plugin already names the umbrellas in `_SPECIAL_LABELS`:

- `ocp:` — `play`, `next`, `prev`, `pause`, `resume`, `media_stop`, `open`,
  `read`, `featured`, `like_song`, `play_favorites`, `load_game`, `save_game`,
  from `ocp_pipeline/locale/<lang>/` (`play.intent`, `next.intent`,
  `media_stop.intent`, …)
- `common_query:` — `common_query` alone. The pipeline ships no intent
  resource of its own; the umbrella is declared in `sources.yaml`
- `stop:` — `stop`, `global_stop`, from
  `ovos_core/intent_services/locale/<lang>/` (`stop.intent`,
  `global_stop.intent`)
- `persona:` — `ask`, `summon`, `active_persona`, `list_personas`, from
  `ovos_persona/locale/<lang>/` (`ask.intent`, `summon.intent`,
  `list_personas.intent`, …)

Each list is read from the pinned plugin revision at build time, not written
down here; this page names them so a reader can check the two agree.

### The three special labels

`ocp:play`, `common_query:common_query` and `stop:stop` bypass the
registered-intent check: no skill ever registers them, so a model that emits
them would otherwise never be believed. The plugin gates each one on its
downstream pipeline being present in the session, then rewrites it through the
built-in `label_map` layer into the bus topic that actually routes
(`ovos.common_play.play_search`, `common_query.question`, `mycroft.stop`).

Which utterances carry them:

- `ocp:play` — media playback requests naming a title, artist, station, genre
  or media type. Fed by the music query templates and by
  `ovos-ocp-pipeline-plugin`'s own `play.intent`.
- `common_query:common_query` — open-domain factual questions with no skill of
  their own ("how tall is the Sagrada Familia"). Fed by the common-query
  question corpus and by rows older corpora mislabelled onto a specific
  question skill.
- `stop:stop` — bare stop/cancel commands. Fed by `stop.intent` in ovos-core's
  intent-service locale. `global_stop.intent` is a distinct, narrower
  utterance class and keeps its own label.

## How label_map resolves aliases

The plugin merges three layers, later winning: the built-in special-label
defaults, the model's own `labels.json`, then user config. A `labels.json`
entry maps a model label to a `skill_id:intent` string; a target without a
colon is used as-is with a warning, and never invented into a topic.

Because the training scheme above is by construction identical to the runtime
registration string, no label needs remapping. The `labels.json` the builder
writes therefore carries no flat label map at all. Aliasing is resolved at
build time instead, in the corpus, where it can be reviewed — not at match
time, where it would be invisible.

What it does carry is two keys. `valid_labels` is the allow-list the plugin
applies after mapping. `families` gives each canonical label its family:

```json
{
  "valid_labels": ["ocp:play", "ovos-skill-hello-world.openvoiceos:hello.world"],
  "families": {
    "ocp:play": "ocp",
    "ovos-skill-hello-world.openvoiceos:hello.world": "skill"
  }
}
```

The family ships with the model rather than living only as a dataset column,
because the plugin's per-family claim filter has to know which family a label
belongs to and that is a property of the trained head, not of the plugin code.
A label absent from `families` is logged once and treated as `skill`.

The three special labels appear in `valid_labels` and `families` in their raw
form — `ocp:play`, `stop:stop`, `common_query:common_query` — never as the bus
topics they resolve to. The plugin performs that mapping at match time.

## The registry is the ground truth

The pinned refs in `sources.yaml` are not just provenance. For every skill in
`skill_refs` and every pipeline plugin, the builder reads, at that exact
revision, the skill id its entry point declares and the intent names it
registers: the names passed to `IntentBuilder`, the names declared by
`@intent_handler("name.intent")`, and the stems of the `.intent` resources it
ships. Test directories are excluded, in skills and in plugins alike — a
plugin's own test-fixture skill is not a registration, which is why
`common_query:search_fakewiki` is not a label.

A locale file whose stem differs from a declared handler name only in case or
separators loses to the handler. ovos-skill-count ships an `it-IT`
`count_to_N.intent` against a `count_to_n` handler; the file is a typo that
the runtime never loads, and treating it as a registration is what put two
classes for one intent in the previous model.

Every corpus label is then resolved against that set:

- exact match — kept;
- the skill id resolves but the intent name differs only in case or
  separators (`movie.genres` against the registered `movie_genres`,
  `enable.ggwave` against `enable_ggwave`) — aliased, with a manifest entry;
- the skill id is a variant of a registered one — a doubled author suffix, a
  numeric disambiguator (`ovos-skill-days-in-history_1.openvoiceos`), or the
  package's words in the other order (`ovos-skill-wallpapers` for
  `skill-ovos-wallpapers`) — resolved, with a manifest entry;
- no match — the rows are **dropped**, and the label is listed in the
  manifest's `unresolved_labels` with its row count and the reason.

Dropping is the point. A class the runtime cannot produce is a class the
plugin will never accept a match for, so training it only steals probability
mass from labels that matter. The unresolved list is the review surface: it is
where an archived skill shows up (the home-assistant skill repos are archived,
so their rows go), where a community plugin that is not in the pinned set
shows up, and where a genuinely missing pin would show up.

Alias tables are deliberately small. They cover only what a registry lookup
cannot express: a label one corpus files under the wrong skill entirely, and
intent merges that predate a rename.

## One utterance, one label

After resolution, a `(utterance, lang)` pair must denote exactly one label.
A sentence carrying two labels is not a hard example, it is a contradiction:
however the model resolves it, the corpus says it is wrong.

Most of these are genuine template overlaps between adjacent intents inside a
single skill's own locale files — `ocp:pause` and `ocp:media_stop` share
phrasings, as do `ocp:next` and `ocp:prev`. They cannot be aliased away,
because neither is a misspelling of the other. Those rows are dropped and the
label pairs are reported, so the overlap can be fixed upstream in the skill
that ships it. `--allow-ambiguous` keeps them; the default does not, and the
builder asserts that no ambiguous group survives.

## Dedup and aliasing rules

Applied in this order:

1. **Blacklists.** Skills that must never be classified, and resource files
   that are dialog rather than intents.
2. **Content filters.** Length bounds, rows that are a bare unexpanded `{slot}`,
   and rows with no alphabetic character at all.
3. **Cross-source aliasing.** A small table of labels one corpus files under
   the wrong skill entirely — `ovos-skill-ddg.openvoiceos:search_wolfie`
   belongs to the wolfie skill; the DuckDuckGo skill's per-property intents
   (`born`, `died`, `alma_mater`, …) are common-query questions and fold into
   `common_query:common_query`. Intent-level merges handle corpora that predate
   a rename (`what.date.is.it` → `current_date`).
4. **Registry resolution**, as above: exact, spelling-folded, skill-id
   variant, or dropped as unresolved. This is what settles `count_to_n`
   versus `count_to_N` — two classes for one intent, and the confusion pair
   that cost the previous model most of its top-confidence errors.
5. **Exact dedup** on `(utterance, label, lang)`, case-insensitive on the
   utterance, keeping the first occurrence in a deterministic source order.
6. **Ambiguity drop**: every row in a `(utterance, lang)` group carrying more
   than one label.
7. **Rare-label drop.** A label with a single row cannot be stratified into a
   train/test split. Dropped labels are listed in the manifest rather than
   silently removed — a real intent showing up there is a signal that its skill
   needs more locale coverage, not that the label is wrong.

## Language codes

Sources disagree: ovos-localize and the skill golden corpora key on full
BCP-47 codes, the lang-support tracker CSVs key on a bare subtag, and
ovos-persona ships both `locale/ca-ES` and `locale/ca`. Normalising before
dedup is what lets the same sentence, exported by three pipelines, collapse to
one row.

A bare subtag is only given a region when the corpus attests exactly one
region for that language, and the source is not mixing dialects under the bare
code. `es`, `nl` and `pt` fail that test and stay bare, as distinct locales
from `es-ES`/`es-419`, `nl-NL`/`nl-BE` and `pt-PT`/`pt-BR`.

For `pt` this is measured, not assumed. Probing the tracker's `intents_pt.csv`
for dialect markers finds 244 rows using the Brazilian `você` and zero rows
using any European-Portuguese marker (`tu`, `telemóvel`, `autocarro`,
`comboio`, `frigorífico`). Folding that file into `pt-PT` would file
Brazilian sentences under the European locale. `es` shows no marker either
way, which is an absence of evidence rather than evidence of a single dialect,
so it stays bare too.

## Golden sentences

The ovoscope corpus and each skill's `test/end2end/golden_utterances*.jsonl`
join the train/test split by stratification rather than forming a pure
holdout: they are the only rows generated directly against a skill's live
registration, so keeping them entirely out of training throws away the
highest-quality supervision in the corpus. Every row keeps a `source` column
naming its origin, so a golden-only slice can still be scored on its own.

Rows marked `needs_manual` are excluded. Rows with no `intent_label` are
excluded too: the fallback skill's corpus asserts a dialog, not an intent.

## Renames, merges, and the unification wave

The Adapt-to-`.intent` refactors rename intents across the default skills, and
a later wave will merge the weather condition intents, the alerts create and
list families, and the volume levels. Both change the label set.

The procedure is the same either way, and it is why the source manifest pins
every revision:

1. Land the skill-side change.
2. Update that skill's pin in `train/sources.yaml`, or pass a regenerated
   `--skill-ref-list`.
3. For a rename with no corpus-side rows yet under the new name, add the old
   label to the intent-alias table so historical rows keep contributing. For a
   merge, alias every merged spelling to the surviving one.
4. Re-run the builder and read the manifest diff: the label count, the rows
   moved by aliasing, and the rare-label list say whether the merge landed as
   intended.
5. Retrain. A renamed intent is a new class; there is no way to patch a frozen
   label head.

An alias entry is a statement that two names denote one runtime intent. It is
not a place to fix a corpus typo — those belong upstream, in the skill.
