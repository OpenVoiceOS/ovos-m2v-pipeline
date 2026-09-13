#!/usr/bin/env python3
"""Build the intent corpus from the skills themselves.

The skills are the data. Every locale resource a running skill would load is
the training side, and the gold sentences the fleet already ships are the test
side. Provenance alone does not keep the two disjoint: a gold sentence and an
expansion of that skill's own template are both somebody reaching for the
obvious phrasing of the same intent, and they collide constantly regardless
of which file either one was read from. So after both sides are built, every
training row whose utterance matches a gold utterance is removed from the
training side. The comparison is string equality after lowercasing,
collapsing whitespace, and stripping leading and trailing punctuation --
the same trailing-punctuation fold ``ovos-utterance-normalizer`` applies to
an incoming utterance before an intent engine ever sees it, so "play rock?"
in gold and "play rock" in a template expansion are one string here because
they are one string at match time. It is still not accent-insensitive and
not semantic. The gold side is never thinned: dropping the gold rows a
skill author happened to also think of would leave the phrasings nobody
anticipated, a biased sample rather than a smaller one. The manifest
records the measured overlap between the two sides after the removal under
both the punctuation-insensitive comparison the removal itself uses and the
narrower exact-string comparison, both of which must be zero, and how many
templates the removal left with no rows at all, because that count is the
price paid for the fix and a manifest that reports only the zero is how
this defect gets rebuilt. The test side itself is a claim that must hold a
floor too: a gold glob that resolves to nothing builds a corpus with zero
test rows and reports the overlap as a vacuous zero, so ``--min-test-rows``
and ``--min-labels-scored`` gate the test side exactly as ``--min-labels``
and ``--min-languages`` gate the training side.

Three resource kinds, each read for what it is. An ``.intent`` file carries
templates. An ``.entity`` file carries example values for a slot -- a hint per
OVOS-INTENT-3 5.2, never a closed set, because a value outside it must stay
matchable. A slot with no examples still produces a row: INTENT-1 5.2 makes a
slot free-form capture and 5.4 says it fills without a value set, so the
placeholder is left literal exactly as the runtime leaves it. A ``.voc`` file carries the keywords that trigger an intent, and is
never read as slot values: ``repeat.voc`` holding "every" and "frequency" says
how a user asks for recurrence, not what a ``{repeat}`` slot contains.

Language comes from the locale directory a file sits in and stays with the
row, so a value attested only under ``en-US`` fills only English templates.
Dialect directories keep their tags; collapsing them is a publication step
applied to the finished dataset. OVOS-INTENT-2 2 makes tag comparison
case-insensitive ("`en-us` and `en-US` denote the same language"), so a
directory's case is not part of a language's identity: ``fr-fr`` and
``fr-FR`` fold onto one canonical spelling before a row is built. That is
the only equivalence this builder applies -- ``eu``/``eu-ES``,
``cmn-CN``/``zh-CN`` and similar macrolanguage or variant pairs differ by
more than case and stay distinct.
"""
import argparse
import collections
import json
import re
import string
import subprocess
import sys
from pathlib import Path

import yaml

from ovos_spec_tools.expansion import expand

#: A locale directory: an ISO subtag, optionally with a script and a region.
LOCALE_DIR = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z]{4})?(?:-(?:[A-Za-z]{2}|\d{3}))?$")

#: Where a resource sits, whatever nests the locale root.
RESOURCE = re.compile(r"(?:^|.*/)locale/([^/]+)/(?:.*/)?([^/]+)\.(intent|entity|voc)$")

#: Gold lives beside the end-to-end suite that reads it.
GOLD = re.compile(r"(?:^|.*/)golden_utterances(?:_([A-Za-z-]+))?\.jsonl$")

#: OVOS-INTENT-2 2: "A resource base name MUST consist only of lowercase
#: ASCII letters, digits, and underscores, and MUST NOT contain whitespace".
COMPLIANT_BASE_NAME = re.compile(r"^[a-z0-9_]+$")

#: A template producing more than this is a generator, not a phrasing.
EXPANSION_CAP = 2000


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout


def tree(repo: Path, rev: str):
    return git(repo, "ls-tree", "-r", "--name-only", rev).split("\n")


def show(repo: Path, rev: str, path: str) -> str:
    try:
        return git(repo, "show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return ""


def normalize_utterance(text: str) -> str:
    """Lowercase and collapse whitespace, the narrower overlap comparison.

    Exact string match after this normalisation, nothing wider: not
    accent-insensitive, not semantic. Kept and reported alongside the
    punctuation-insensitive comparison below because the two answer
    different questions, and collapsing them into one number would hide
    which normalisation the removal actually rests on.
    """
    return re.sub(r"\s+", " ", text.strip()).lower()


def normalize_utterance_punct_insensitive(text: str) -> str:
    """As ``normalize_utterance``, plus stripping leading/trailing punctuation.

    ``ovos-utterance-normalizer``, the runtime transformer every incoming
    utterance passes through before an intent engine matches it, strips
    edge punctuation with ``utterance.strip(string.punctuation).strip()``.
    A gold row ending in "?" and a training expansion without one are the
    same string at that point, so the removal this builder performs is
    matched on this wider equivalence, not the narrower one above. Still
    not accent-insensitive, not semantic, and inner punctuation is left
    alone -- only what the runtime strips is stripped here.
    """
    return normalize_utterance(text).strip(string.punctuation).strip()


def normalize_lang(tag: str) -> str:
    """Canonical spelling of a BCP-47 tag, per OVOS-INTENT-2 2.

    Tag comparison is case-insensitive, so a locale directory's case is not
    part of a language's identity. Lowercase the primary and extended
    subtags, titlecase a 4-letter script, uppercase a 2-letter region; a
    3-digit region is a number and has no case to fold.
    """
    parts = tag.split("-")
    out = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            out.append(part.title())
        elif len(part) == 2 and part.isalpha():
            out.append(part.upper())
        else:
            out.append(part)
    return "-".join(out)


def skill_id_of(repo_name: str) -> str:
    """The label prefix a skill registers under, from its package name."""
    return f"{repo_name}.openvoiceos"


def read_skill(repo: Path, rev: str, repo_name: str, stats: collections.Counter,
               violations: set):
    """Templates, hints and keywords a skill ships, per language."""
    templates = []                                   # (lang, label, template)
    hints = collections.defaultdict(dict)            # lang -> name -> values
    keywords = collections.defaultdict(dict)         # lang -> name -> words
    for path in tree(repo, rev):
        if not path or path.split("/")[0] in {"test", "tests"}:
            continue
        m = RESOURCE.match(path)
        if not m:
            continue
        lang, name, kind = m.group(1), m.group(2), m.group(3)
        if not LOCALE_DIR.match(lang):
            stats["resource_outside_a_locale"] += 1
            continue
        lang = normalize_lang(lang)
        if not COMPLIANT_BASE_NAME.match(name):
            # The corpus still reads it -- the fleet is what it is -- but a
            # base name carrying a dot, a dash or a capital breaks the rule
            # the loader relies on, and the fallback that rescues its gold
            # rows would otherwise hide it.
            violations.add(f"{repo_name}: {name}.{kind}")
        body = show(repo, rev, path)
        values = [l.strip() for l in body.splitlines()
                  if l.strip() and not l.strip().startswith("#")]
        if kind == "intent":
            label = f"{skill_id_of(repo_name)}:{name}"
            for line in values:
                templates.append((lang, label, line))
        elif kind == "entity":
            # Two directories differing only in case are one language after
            # normalisation, so their entity files are the same file under
            # two spellings: union the values rather than letting whichever
            # sorts last silently discard the other's.
            existing = hints[lang].setdefault(name.lower(), [])
            existing.extend(v for v in values if v not in existing)
        else:
            existing = keywords[lang].setdefault(name.lower(), [])
            existing.extend(v for v in values if v not in existing)
    return templates, hints, keywords


def fill(template: str, hints: dict, keywords: dict, stats: collections.Counter):
    """Sentences a template produces, with its own language's resources.

    `<name>` references a keyword file, which is how a template says "any of
    these words here". `{name}` is a slot: filled from the language's hints
    when it has them, and left unfilled otherwise -- an unfilled slot is a
    phrasing this corpus cannot use, not a phrasing the skill cannot match.
    """
    try:
        sentences = expand(template, keywords)
    except Exception:
        stats["template_would_not_expand"] += 1
        return []
    if len(sentences) > EXPANSION_CAP:
        stats["template_over_cap"] += 1
        sentences = sentences[:EXPANSION_CAP]
    out = []
    for s in sentences:
        slots = {n.lower() for n in re.findall(r"\{([^}]+)\}", s)}
        if not slots:
            out.append(s)
            continue
        # A slot with no examples is not a dead template. INTENT-1 5.2 makes
        # a slot free-form capture, 5.4 says a slot with no value set still
        # fills, and the runtime agrees: `expand_entities` passes a sample
        # through with the placeholder left literal and embeds it that way.
        # The corpus mirrors the runtime rather than discarding the phrasing.
        if any(n not in hints for n in slots):
            stats["sentence_kept_with_an_unfilled_slot"] += 1
        filled = [s]
        for slot in sorted(n for n in slots if n in hints):
            nxt = []
            for partial in filled:
                for value in hints[slot]:
                    nxt.append(re.sub(r"\{" + re.escape(slot) + r"\}", value,
                                      partial, flags=re.IGNORECASE))
            filled = nxt[:EXPANSION_CAP]
        # A hint value is substituted verbatim, and a value can itself carry
        # template syntax (an `.entity` line copied from a template, or one
        # written with alternation in it). Left alone, that syntax ships as a
        # literal training row instead of the sentences it denotes, so every
        # filled string is expanded again. `expand` is eager and raises
        # rather than returning a partial sample set, so a fill that produces
        # a malformed string is dropped whole, the same way a malformed
        # template is dropped above -- never partially kept.
        reexpanded = []
        for f in filled:
            try:
                grown = expand(f, {})
            except Exception:
                stats["filled_sentence_would_not_expand"] += 1
                continue
            if len(grown) > 1:
                stats["rows_expanded_after_filling"] += len(grown)
            reexpanded.extend(grown)
        if len(reexpanded) > EXPANSION_CAP:
            stats["sentence_over_cap_after_fill"] += 1
            reexpanded = reexpanded[:EXPANSION_CAP]
        out.extend(reexpanded)
    return out


def count_leftover_template_syntax(rows) -> int:
    """Written rows that still carry template syntax other than `{slot}`.

    A slot left unfilled (INTENT-1 5.4) is the documented, correct shape of
    a row -- `{slot}` is excluded. Anything else the grammar defines,
    alternation or an optional segment, is never a valid training row; this
    is the check that would have caught the defect the manifest's own
    per-stage counters did not.
    """
    leftover = re.compile(r"\(.*\|.*\)|\[[^\]]*\]")
    return sum(1 for row in rows if leftover.search(row["utterance"]))


def resolve_gold_label(label: str, shipped: set) -> str:
    """The shipped intent a gold label names, under either convention.

    A gold file may write an intent the way the corpus writes it rather than
    the way the file is named: `movie.description.intent` for a file called
    `movie_description`. The skills' own suites bridge that with a
    normalisation, so this does too -- but only as a FALLBACK. Some skills
    genuinely name a file `volume.mute.intent`, and mapping its dots to
    underscores would invent an intent that skill does not ship.
    """
    if label in shipped:
        return label
    underscored = label.replace(".", "_")
    # Only an unambiguous match may resolve. If two shipped intents fold onto
    # the same underscored form, picking either scores the row against a
    # label nobody asserted, and a mis-scored row is worse than an unscored
    # one: the unscored label is visible in the manifest and the mis-scored
    # one makes every count look healthy.
    candidates = {name for name in shipped
                  if name.replace(".", "_") == underscored}
    if len(candidates) == 1:
        return candidates.pop()
    return label


def read_gold(repo: Path, rev: str, repo_name: str, stats: collections.Counter):
    """Gold rows a skill ships, and what is wrong with the ones it does not."""
    rows = []
    for path in tree(repo, rev):
        m = GOLD.match(path or "")
        if not m:
            continue
        from_name = m.group(1)
        for line in show(repo, rev, path).splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                stats["gold_unparsable"] += 1
                continue
            if d.get("needs_manual"):
                stats["gold_needs_manual"] += 1
                continue
            utterance = (d.get("utterance") or "").strip()
            label = d.get("intent_label")
            if not utterance:
                stats["gold_without_utterance"] += 1
                continue
            if not label:
                # A fallback skill asserts the dialog it speaks, because no
                # intent claimed the utterance. That is a real assertion and
                # this corpus cannot score it: there is no label to predict.
                stats["gold_asserts_a_dialog_not_an_intent"] += 1
                continue
            label = str(label)
            if label.endswith(".intent"):
                label = label[: -len(".intent")]
            skill = d.get("skill_id") or skill_id_of(repo_name)
            # `machine_generated` is not read. Every gold sentence in this
            # fleet was written by a model, so a row claiming otherwise is
            # wrong and a count derived from the field would be fiction.
            rows.append({
                "lang": normalize_lang(d.get("lang") or from_name or "en-US"),
                "label": f"{skill}:{label}" if ":" not in label else label,
                "utterance": utterance,
                "source": f"gold:{repo_name}",
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", default="train/sources.yaml")
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    # Floors rather than an equality: the corpus grows as skills gain
    # resources, and it must never quietly shrink. Both numbers move the
    # moment the expansion step starts discarding templates again, which is
    # the failure this build already had once and which no count revealed.
    # Measured against #158+#125 merged onto dev (dev@34bfcb2): labels_trained
    # unmoved by the case-fold fix, so its floor is unchanged.
    ap.add_argument("--min-labels", type=int, default=227)
    # Measured on the same tree: 53 distinct locale directories folded to 52
    # once fa-ir/fa-IR merged under OVOS-INTENT-2 2's case-insensitive tag
    # comparison. The prior floor of 53 counted that pair twice.
    ap.add_argument("--min-languages", type=int, default=52)
    # The zero this build asserts on the overlap is satisfied trivially when
    # the test side is empty: a gold glob renamed or moved out from under the
    # builder still builds rc=0 with test_rows=0 and every overlap zero by
    # construction, scoring nothing while reporting the cleanest possible
    # number. Floors on the test side close that, the same way min-labels
    # and min-languages close it on the training side. Measured against a
    # real build over dev@72d73a5 (`--workspace ~/AgentWorkspaces`, shipped
    # source pins): test_rows=1412.
    ap.add_argument("--min-test-rows", type=int, default=1412)
    # Measured on the same build: labels_scored=164.
    ap.add_argument("--min-labels-scored", type=int, default=164)
    # A base name that breaks OVOS-INTENT-2 2 is read anyway, because the
    # corpus must build against the fleet as it stands. Pinning the count
    # stops the set growing while the rename campaign brings it down; a zero
    # here would be red on arrival and loosened away by whoever ran it next.
    ap.add_argument("--max-noncompliant-base-names", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.sources).read_text(encoding="utf-8"))
    ws = Path(args.workspace or cfg["workspace"]).expanduser()
    refs = cfg["skill_refs"]["refs"]

    stats = collections.Counter()
    train, test = [], []
    labels_trained, labels_scored = set(), set()
    labels_with_templates = set()
    no_gold, no_resources, resolutions = [], [], []
    violations = set()

    for key, rev in sorted(refs.items()):
        repo, repo_name = ws / key, key.rsplit("/", 1)[-1]
        if not repo.is_dir():
            stats["skill_missing_clone"] += 1
            continue
        templates, hints, keywords = read_skill(repo, rev, repo_name, stats,
                                                violations)
        if not templates:
            no_resources.append(repo_name)
        for _, label, _ in templates:
            labels_with_templates.add(label)
        for lang, label, template in templates:
            for sentence in fill(template, hints.get(lang, {}),
                                 keywords.get(lang, {}), stats):
                train.append({"lang": lang, "label": label,
                              "utterance": sentence, "source": f"skill:{repo_name}",
                              "template": template})
                labels_trained.add(label)
        shipped = {lbl.split(":", 1)[1] for lbl in labels_with_templates
                   if lbl.startswith(skill_id_of(repo_name) + ":")}
        gold = read_gold(repo, rev, repo_name, stats)
        for row in gold:
            prefix, _, intent = row["label"].partition(":")
            resolved = resolve_gold_label(intent, shipped)
            if resolved != intent:
                stats["gold_label_resolved_by_underscoring"] += 1
                resolutions.append({"skill": repo_name, "gold": intent,
                                    "shipped": resolved})
                row["label"] = f"{prefix}:{resolved}"
        if not gold:
            no_gold.append(repo_name)
        for row in gold:
            test.append(row)
            labels_scored.add(row["label"])

    seen = set()
    deduped = []
    for row in train:
        key = (row["label"], row["lang"], row["utterance"].lower())
        if key in seen:
            stats["train_duplicate"] += 1
            continue
        seen.add(key)
        deduped.append(row)
    train = deduped

    # A gold sentence and an expansion of the same skill's own template are
    # both somebody reaching for the obvious phrasing of the same intent, so
    # they collide regardless of which file either was read from. Every
    # training row that collides with a gold row -- match after lowercasing,
    # whitespace collapse, and stripping edge punctuation, the same fold the
    # runtime utterance normalizer applies before an intent engine matches
    # -- is removed from the training side; the gold side is never touched.
    # Measured, not assumed: the overlap under both comparisons and the
    # templates the removal silences all go in the manifest.
    gold_normalized = {normalize_utterance(r["utterance"]) for r in test}
    gold_normalized_wide = {
        normalize_utterance_punct_insensitive(r["utterance"]) for r in test}
    templates_before = collections.defaultdict(set)
    for row in train:
        templates_before[row["template"]].add(row["label"])
    train_gold_overlap = [
        row for row in train
        if normalize_utterance_punct_insensitive(row["utterance"])
        in gold_normalized_wide]
    train = [row for row in train
             if normalize_utterance_punct_insensitive(row["utterance"])
             not in gold_normalized_wide]
    templates_after = {row["template"] for row in train}
    silenced_templates = sorted(t for t in templates_before
                                if t not in templates_after)
    silenced_labels = sorted({label for t in silenced_templates
                              for label in templates_before[t]})
    # The overlap this fix removes may have been the only training rows a
    # label had; re-derive labels_trained from what survives rather than
    # from the pre-removal pass, or the manifest would call a label trained
    # when its last row was just deleted.
    labels_trained = {row["label"] for row in train}
    train_gold_overlap_after = sum(
        1 for row in train if normalize_utterance(row["utterance"]) in gold_normalized)
    train_gold_overlap_after_punct_insensitive = sum(
        1 for row in train
        if normalize_utterance_punct_insensitive(row["utterance"])
        in gold_normalized_wide)

    # A gold sentence naming a label no skill trains is two different
    # findings wearing one shape, and they ask for opposite work. If the
    # skill ships the intent and it produces no rows, the intent is starved
    # of slot examples and the skill is fine. If the skill does not ship the
    # intent at all, the gold row asserts a routing its own skill never
    # supported -- either written wrong, or written before a rename nobody
    # carried into the gold file.
    scored_labels = {r["label"] for r in test}
    starved = sorted(scored_labels & labels_with_templates - labels_trained)
    unsupported = sorted(scored_labels - labels_with_templates)

    report = {
        "train_rows": len(train),
        "test_rows": len(test),
        "labels_trained": len(labels_trained),
        "labels_scored": len(labels_scored & labels_trained),
        "labels_never_scored": len(labels_trained - labels_scored),
        "languages_train": len({r["lang"] for r in train}),
        "languages_test": len({r["lang"] for r in test}),
        "train_gold_overlap_removed": len(train_gold_overlap),
        "train_gold_overlap_after_fix": train_gold_overlap_after,
        "train_gold_overlap_after_fix_punct_insensitive":
            train_gold_overlap_after_punct_insensitive,
        "templates_silenced": len(silenced_templates),
        "templates_silenced_detail": silenced_templates,
        "labels_with_a_silenced_template": silenced_labels,
        # A row still carrying template syntax other than an unfilled `{slot}`
        # is unusable, whatever the per-stage counters above say. This is the
        # number that would have caught the shipped-alternation defect: it
        # reads the finished corpus rather than trusting the builder's own
        # account of its work.
        "train_rows_with_leftover_template_syntax":
            count_leftover_template_syntax(train),
        "gold_flags_are_not_evidence": (
            "every gold sentence in this fleet was written by a model, so the "
            "machine_generated field is unreliable wherever it claims False "
            "and is not read; needs_manual marks a row somebody meant a human "
            "to check, and whether that check happened is unrecorded"),
        "skills_without_gold": sorted(no_gold),
        "skills_without_locale_resources": sorted(no_resources),
        "noncompliant_base_names": len(violations),
        "noncompliant_base_names_detail": sorted(violations),
        "gold_labels_resolved_by_underscoring": sorted(
            {f"{r['skill']}: {r['gold']} -> {r['shipped']}" for r in resolutions}),
        "gold_labels_starved_of_slot_examples": starved,
        "gold_labels_no_skill_supports": unsupported,
        "stats": dict(stats),
    }
    print(json.dumps({k: v for k, v in report.items()
                      if not isinstance(v, list)}, indent=2))
    short = []
    if report["labels_trained"] < args.min_labels:
        short.append(
            f"the corpus shrank to {report['labels_trained']} labels, floor "
            f"{args.min_labels}: a template the expansion drops takes its "
            f"label with it and no other count shows the loss")
    if report["languages_train"] < args.min_languages:
        short.append(
            f"the corpus shrank to {report['languages_train']} languages, "
            f"floor {args.min_languages}")
    if report["test_rows"] < args.min_test_rows:
        short.append(
            f"the gold side shrank to {report['test_rows']} test rows, "
            f"floor {args.min_test_rows}: a moved or renamed gold glob "
            f"that matches nothing builds an overlap of zero by having "
            f"nothing to overlap with, and no other count shows the loss")
    if report["labels_scored"] < args.min_labels_scored:
        short.append(
            f"the gold side scores only {report['labels_scored']} labels, "
            f"floor {args.min_labels_scored}")
    if report["train_gold_overlap_after_fix"] != 0:
        short.append(
            f"{report['train_gold_overlap_after_fix']} training rows still "
            f"match a gold utterance after the removal pass: the fix did "
            f"not do what it claims")
    if report["train_gold_overlap_after_fix_punct_insensitive"] != 0:
        short.append(
            f"{report['train_gold_overlap_after_fix_punct_insensitive']} "
            f"training rows still match a gold utterance once edge "
            f"punctuation is stripped, the fold the runtime normalizer "
            f"applies before an intent engine ever compares the two: the "
            f"fix did not do what it claims")
    cap = args.max_noncompliant_base_names
    if cap is not None and report["noncompliant_base_names"] > cap:
        short.append(
            f"{report['noncompliant_base_names']} resource base names break "
            f"OVOS-INTENT-2 2, ceiling {cap}: the set may shrink as skills "
            f"are renamed and must never gain a member")
    if short:
        for line in short:
            print(f"[gate] {line}", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("test.jsonl", test)):
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
