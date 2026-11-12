"""Rank stage-2 steer-lens units by theme: count of the 12 most-raised next tokens that are in a theme lexicon.
Null: the same count for random-direction controls. Also flags whether the lens (direct path) already shows the theme
(word writer) or not (indirect: a candidate lever)."""
import json, collections, torch
from transformers import AutoTokenizer
from common import LOCAL, MODEL_DIR, RESULTS, REVISION
from themes import THEMES
tok = AutoTokenizer.from_pretrained(MODEL_DIR, revision=REVISION, local_files_only=LOCAL)
lens = torch.load(RESULTS / "stage1/logit_lens.pt")
cache = {}
def w(i):
    if i not in cache: cache[i] = tok.decode([i]).strip().lower()
    return cache[i]
lex = {t: set(v) for t, v in THEMES.items()}
rows, null = [], collections.Counter()
nrand = 0
for L in range(12, 24):
    d = torch.load(RESULTS / f"stage2/steer_lens_L{L:02d}.pt")
    for t in d["rand_tok"].tolist():
        nrand += 1
        ws = [w(i) for i in t]
        for th, s in lex.items():
            null[th] = max(null[th], sum(x in s for x in ws))
    for key, pole in (("up", "+"), ("down", "-")):
        toks, kls = d[f"{key}_tok"].tolist(), d[f"{key}_kl"].tolist()
        lk = "cpos" if pole == "+" else "cneg"
        lids = lens[f"{lk}_ids"][L, :, :15].tolist()
        for n, t in enumerate(toks):
            ws = [w(i) for i in t]
            for th, s in lex.items():
                c = sum(x in s for x in ws)
                if c >= 2:
                    lc = sum(w(i) in s for i in lids[n])
                    rows.append({"theme": th, "hits": c, "lens_hits": lc, "layer": L, "neuron": n, "pole": pole,
                                 "kl": round(kls[n], 2), "tokens": " ".join(ws)})
print(f"random controls: {nrand}; max theme hits among controls:", dict(null))
rows.sort(key=lambda r: (-r["hits"], r["lens_hits"]))
by = collections.defaultdict(list)
for r in rows: by[r["theme"]].append(r)
json.dump(by, open(RESULTS / "stage2/theme_rank.json", "w"), indent=0)
for th in sorted(by):
    rs = by[th]
    print(f"\n## {th}: {len(rs)} poles with >=2 hits")
    for r in rs[:7]:
        tag = "writer" if r["lens_hits"] >= 3 else "INDIRECT" if r["lens_hits"] == 0 else "mixed"
        print(f"  L{r['layer']} N{r['neuron']}{r['pole']} hits {r['hits']} lens {r['lens_hits']} [{tag}] kl {r['kl']}: {r['tokens']}")
