#!/usr/bin/env python3
"""Generate a held-out, zero-overlap OOD bucket for v5 English labels by reusing
the v4/v3 synonym+register-substitution paraphrase engine (gen_ood.py from
/home/miro/tmp/m2v-dataset/), applied to v5 train_en.jsonl / labels."""
import json, re, random, collections

random.seed(42)
OUT = "/home/miro/tmp/m2v-dataset-v5"

def load_jsonl(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out

train = load_jsonl(f"{OUT}/train_en.jsonl")
test = load_jsonl(f"{OUT}/test_en.jsonl")
by_label = collections.defaultdict(list)
for d in train:
    by_label[d["label"]].append(d["text"])

labels = sorted(by_label.keys())

SYNONYMS = {
    r"\bset\b": ["create", "start", "put on", "make"],
    r"\bcreate\b": ["set up", "make", "start", "add"],
    r"\badd\b": ["put", "throw", "stick"],
    r"\bdelete\b": ["remove", "get rid of", "wipe", "clear out"],
    r"\bremove\b": ["delete", "take off", "get rid of", "clear"],
    r"\bcancel\b": ["stop", "call off", "get rid of", "scrap"],
    r"\bstop\b": ["cancel", "halt", "end", "quit"],
    r"\bplay\b": ["put on", "start", "queue up", "fire up"],
    r"\bpause\b": ["hold", "freeze"],
    r"\bresume\b": ["continue", "carry on with", "pick back up"],
    r"\bvolume\b": ["sound level", "loudness", "audio level"],
    r"\bmute\b": ["silence", "shut off the sound of"],
    r"\bunmute\b": ["un-silence", "turn the sound back on for"],
    r"\btimer\b": ["countdown", "count-down timer"],
    r"\balarm\b": ["wake-up call", "wake alarm"],
    r"\breminder\b": ["note to self", "heads up", "nudge"],
    r"\bthe list\b": ["the checklist", "the to-do list"],
    r"\bmy list\b": ["my checklist", "my to-do list"],
    r"\ba list\b": ["a checklist", "a to-do list"],
    r"\bweather\b": ["forecast", "outside conditions"],
    r"\btemperature\b": ["temp", "how hot or cold it is"],
    r"\btoday\b": ["this afternoon", "for today", "right now"],
    r"\btomorrow\b": ["the next day", "tomorrow's forecast"],
    r"\bwhat time\b": ["what's the time", "how late"],
    r"\bwhat's\b": ["what is", "tell me"],
    r"\bplease\b": [""],
    r"\bcan you\b": ["could you", "would you mind", "are you able to"],
    r"\btell me\b": ["let me know", "give me", "fill me in on"],
    r"\bshow me\b": ["pull up", "bring up", "display"],
    r"\bnews\b": ["headlines", "the latest stories"],
    r"\bjoke\b": ["something funny", "a laugh"],
    r"\bip\b": ["network address", "ip address"],
    r"\bwikipedia\b": ["wiki", "encyclopedia entry"],
}

QUESTION_STARTERS = (
    "what", "when", "where", "who", "why", "how", "which", "is", "are", "was",
    "were", "do", "does", "did", "will", "would", "can", "could", "should",
    "have", "has", "am",
)

IMPERATIVE_FRAMES = [
    "{s}", "could you {s2}", "would you mind {ing2}", "i need you to {s2}",
    "hey, {s2}", "can you please {s2}", "i'd like you to {s2}", "go ahead and {s2}",
]
QUESTION_FRAMES = [
    "{s}", "hey, {s2}", "quick question, {s2}", "i was wondering, {s2}",
    "any idea, {s2}", "do you happen to know: {s2}", "mind telling me, {s2}",
]

def is_question(s):
    first = re.sub(r"[^\w\s]", "", s).strip().lower().split(" ")[0] if s.strip() else ""
    return first in QUESTION_STARTERS or s.strip().endswith("?")

def apply_synonyms(sentence):
    out = sentence
    for pat, repls in SYNONYMS.items():
        if re.search(pat, out, flags=re.I):
            out = re.sub(pat, random.choice(repls), out, count=1, flags=re.I)
    return re.sub(r"\s+", " ", out).strip()

GERUNDS = {
    "set": "setting", "create": "creating", "add": "adding", "delete": "deleting",
    "remove": "removing", "cancel": "cancelling", "stop": "stopping",
    "play": "playing", "pause": "pausing", "resume": "resuming", "start": "starting",
    "make": "making", "put": "putting", "throw": "throwing", "give": "giving",
    "show": "showing", "tell": "telling", "turn": "turning", "wake": "waking",
    "reset": "resetting", "restart": "restarting", "open": "opening",
    "close": "closing", "launch": "launching", "find": "finding",
    "search": "searching", "look": "looking", "get": "getting",
    "change": "changing", "reschedule": "rescheduling", "sync": "syncing",
    "check": "checking", "list": "listing", "clear": "clearing", "mute": "muting",
    "unmute": "un-muting", "queue": "queueing", "fire": "firing", "bring": "bringing",
    "pull": "pulling",
}

def to_gerund(s):
    m = re.match(r"^(\w+)(.*)$", s)
    if not m:
        return None
    verb, rest = m.groups()
    g = GERUNDS.get(verb.lower())
    return (g + rest) if g else None

def normalize(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

def cleanup(s):
    s = re.sub(r"\b(\w+)\s+\1\b", r"\1", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()

def gen_candidates(sentences, n=8):
    cands = []
    seen_local = set()
    bases = sentences[:] if len(sentences) <= 40 else random.sample(sentences, 40)
    random.shuffle(bases)
    tries = 0
    while len(cands) < n and tries < n * 8 and bases:
        tries += 1
        base = bases[tries % len(bases)]
        s2 = apply_synonyms(base)
        question = is_question(base)
        s2_lower = s2[0].lower() + s2[1:] if s2 else s2
        ing2 = None if question else to_gerund(s2)
        frames = QUESTION_FRAMES if question else IMPERATIVE_FRAMES
        if not question and ing2 is None:
            frames = [f for f in frames if "{ing2}" not in f]
        frame = random.choice(frames)
        try:
            cand = frame.format(s=s2, s2=s2_lower, ing2=ing2 or s2_lower)
        except Exception:
            cand = s2
        cand = re.sub(r"^,\s*", "", cand)
        cand = cleanup(cand).strip().rstrip(",").lower()
        norm = normalize(cand)
        if norm in seen_local or norm == normalize(base):
            continue
        seen_local.add(norm)
        cands.append(cand)
    return cands

train_norms = {normalize(r["text"]) for r in train}
test_norms = {normalize(r["text"]) for r in test}

ood = []
dropped_contam = 0
for label in labels:
    sents = by_label.get(label)
    if not sents:
        continue
    cands = gen_candidates(sents, n=8)
    for c in cands:
        n = normalize(c)
        if n in train_norms or n in test_norms:
            dropped_contam += 1
            continue
        ood.append({"label": label, "text": c})

# dedup within OOD itself
seen = set()
ood_dedup = []
for r in ood:
    k = (r["label"], normalize(r["text"]))
    if k in seen:
        continue
    seen.add(k)
    ood_dedup.append(r)

with open(f"{OUT}/ood_en.jsonl", "w") as f:
    for r in ood_dedup:
        f.write(json.dumps(r) + "\n")

ood_norms = {normalize(r["text"]) for r in ood_dedup}
overlap_train = ood_norms & train_norms
overlap_test = ood_norms & test_norms

report = {
    "labels_with_ood": len(set(r["label"] for r in ood_dedup)),
    "labels_total": len(labels),
    "ood_rows": len(ood_dedup),
    "dropped_contam_during_gen": dropped_contam,
    "overlap_with_train": len(overlap_train),
    "overlap_with_test": len(overlap_test),
}
json.dump(report, open(f"{OUT}/ood_report_v5.json", "w"), indent=1)
print(json.dumps(report, indent=1))
