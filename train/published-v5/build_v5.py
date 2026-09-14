#!/usr/bin/env python3
"""v5 m2v dataset build (DATASET ONLY, no training) from synced origin/dev skills.
English + multilingual train/test, slot-filled from .entity files, contamination-free
80/20 split, capped at 500 train/label. Reuses the v4 method (build_v4_en.py) extended
to all locales found under each skill's locale/ dir.
"""
import glob, os, re, json, random, collections

random.seed(7)
BASE = os.path.expanduser("~/AgentWorkspaces/ovos/skills")
OUT = os.path.expanduser("~/tmp/m2v-dataset-v5")
os.makedirs(OUT, exist_ok=True)
CAP = 500

skills = [l.strip() for l in open(os.path.expanduser("~/tmp/m2v-retrain-v4/ovos_org_skills.txt")) if l.strip()]

SKIP_DIRS = {".git", "build", "dist", ".claude", "__pycache__", ".venv", ".venv-test", "node_modules"}

def find_locale_dirs(skill):
    out = collections.defaultdict(list)
    for root, dirs, files in os.walk(os.path.join(BASE, skill)):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.endswith(".egg-info")
                   and "site-packages" not in x]
        if "site-packages" in root or "/.venv" in root:
            continue
        if os.path.basename(root) == "locale":
            for loc in os.listdir(root):
                p = os.path.join(root, loc)
                if os.path.isdir(p):
                    # canonicalize locale key casing (BCP-47: lang lower, region upper)
                    parts = loc.split("-")
                    canon = parts[0].lower() + ("-" + parts[1].upper() if len(parts) > 1 else "")
                    out[canon].append(p)
    return out

def read_lines(path):
    vals = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vals.append(line)
    return vals

FALLBACK_SLOTS = {
    "location": ["paris", "lisbon", "new york", "tokyo", "london"],
    "query": ["the weather", "the capital of france", "how tall mount everest is"],
    "time": ["3pm", "10:30 am", "noon", "7 o'clock"],
    "number": ["7", "42", "three", "five"],
    "word": ["banana", "umbrella", "serendipity"],
    "application": ["spotify", "firefox", "the calculator"],
    "amount": ["10", "twenty", "fifty percent"],
}

def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()

# collect locale dirs per skill
skill_locales = {}
locale_utterance_count = collections.Counter()
for skill in skills:
    ld = find_locale_dirs(skill)
    skill_locales[skill] = ld
    for loc, dirs in ld.items():
        n = 0
        for d in dirs:
            for f in glob.glob(os.path.join(d, "**/*.intent"), recursive=True):
                n += len(read_lines(f))
        locale_utterance_count[loc] += n

all_locales = sorted(locale_utterance_count.keys(), key=lambda l: -locale_utterance_count[l])

def build_lang(locdirs_by_skill, lang_for_fallback="en-US"):
    rows = []
    label_counter = collections.Counter()
    for skill, dirs in locdirs_by_skill.items():
        slot_pool = collections.defaultdict(list)
        for d in dirs:
            for f in glob.glob(os.path.join(d, "**/*.entity"), recursive=True):
                slot = os.path.splitext(os.path.basename(f))[0].lower()
                slot_pool[slot].extend(read_lines(f))
        for d in dirs:
            for f in glob.glob(os.path.join(d, "**/*.intent"), recursive=True):
                intent_name = os.path.splitext(os.path.basename(f))[0]
                label = f"{skill}:{intent_name}"
                templates = read_lines(f)
                for t in templates:
                    slots = re.findall(r"\{(\w+)\}", t)
                    if not slots:
                        rows.append((label, t))
                        label_counter[label] += 1
                        continue
                    for _ in range(3):
                        filled = t
                        for s in slots:
                            pool = slot_pool.get(s) or FALLBACK_SLOTS.get(s) or [s]
                            filled = filled.replace("{%s}" % s, random.choice(pool))
                        rows.append((label, filled))
                        label_counter[label] += 1
    return rows, label_counter

def dedup_split(rows):
    seen = set()
    dedup_rows = []
    for label, text in rows:
        key = (label, norm(text))
        if key in seen:
            continue
        seen.add(key)
        dedup_rows.append((label, text))

    by_label = collections.defaultdict(list)
    for label, text in dedup_rows:
        by_label[label].append(text)

    train, test = [], []
    for label, texts in by_label.items():
        uniq = list(dict.fromkeys(texts))
        random.shuffle(uniq)
        if len(uniq) > CAP:
            uniq = uniq[:CAP]
        n_test = max(1, int(len(uniq) * 0.2)) if len(uniq) > 1 else 0
        test_texts = uniq[:n_test]
        train_texts = uniq[n_test:]
        for t in train_texts:
            train.append({"label": label, "text": t})
        for t in test_texts:
            test.append({"label": label, "text": t})
    return dedup_rows, train, test, by_label

# ---------------- English ----------------
en_locdirs = {}
for skill in skills:
    dirs = skill_locales[skill].get("en-US")
    if dirs:
        en_locdirs[skill] = dirs

rows_en, counts_en = build_lang(en_locdirs)
dedup_en, train_en, test_en, bylabel_en = dedup_split(rows_en)

train_norms = set(norm(r["text"]) for r in train_en)
test_norms = set(norm(r["text"]) for r in test_en)
overlap_en = train_norms & test_norms

with open(os.path.join(OUT, "train_en.jsonl"), "w") as f:
    for r in train_en:
        f.write(json.dumps(r) + "\n")
with open(os.path.join(OUT, "test_en.jsonl"), "w") as f:
    for r in test_en:
        f.write(json.dumps(r) + "\n")

# ---------------- Multilingual (all locales incl en) ----------------
train_multi, test_multi = [], []
per_lang_report = {}
for loc in all_locales:
    locdirs = {}
    for skill in skills:
        dirs = skill_locales[skill].get(loc)
        if dirs:
            locdirs[skill] = dirs
    if not locdirs:
        continue
    rows_l, counts_l = build_lang(locdirs)
    dedup_l, train_l, test_l, bylabel_l = dedup_split(rows_l)
    for r in train_l:
        r2 = dict(r); r2["lang"] = loc; r2["label"] = r2["label"]
        train_multi.append({"label": r["label"], "text": r["text"], "lang": loc})
    for r in test_l:
        test_multi.append({"label": r["label"], "text": r["text"], "lang": loc})
    per_lang_report[loc] = {
        "raw_rows": len(rows_l),
        "dedup_rows": len(dedup_l),
        "labels": len(bylabel_l),
        "train": len(train_l),
        "test": len(test_l),
        "raw_template_utterances": locale_utterance_count[loc],
    }

with open(os.path.join(OUT, "train_multi.jsonl"), "w") as f:
    for r in train_multi:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(os.path.join(OUT, "test_multi.jsonl"), "w") as f:
    for r in test_multi:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

report = {
    "en": {
        "raw_rows": len(rows_en),
        "dedup_rows": len(dedup_en),
        "labels": len(bylabel_en),
        "train": len(train_en),
        "test": len(test_en),
        "train_test_overlap_normalized": len(overlap_en),
        "labels_list": sorted(bylabel_en.keys()),
    },
    "multi": {
        "locales": len(per_lang_report),
        "train_total": len(train_multi),
        "test_total": len(test_multi),
        "per_locale": per_lang_report,
    },
}
json.dump(report, open(os.path.join(OUT, "build_report_v5.json"), "w"), indent=1)
print("EN: raw", len(rows_en), "dedup", len(dedup_en), "labels", len(bylabel_en),
      "train", len(train_en), "test", len(test_en), "overlap", len(overlap_en))
print("MULTI locales:", len(per_lang_report), "train_total", len(train_multi), "test_total", len(test_multi))
