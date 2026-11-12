"""Score stage-1 logit-lens words against theme lexicons; show lens words for region-probe detectors."""
import json, torch
from transformers import AutoTokenizer
from common import LOCAL, MODEL_DIR, RESULTS, REVISION
from themes import THEMES
tok = AutoTokenizer.from_pretrained(MODEL_DIR, revision=REVISION, local_files_only=LOCAL)
r = torch.load(RESULTS / "stage1/logit_lens.pt")
V = len(tok)
word = {}
def w(i):
    if i not in word: word[i] = tok.decode([i]).strip().lower()
    return word[i]
NL, N, K = r["cpos_ids"].shape
lex = {t: set(v) for t, v in THEMES.items()}
hits = {t: [] for t in lex}
for L in range(NL):
    for pole, key in (("+", "cpos"), ("-", "cneg")):
        ids = r[f"{key}_ids"][L, :, :15].tolist(); zs = r[f"{key}_z"][L, :, 0].tolist()
        for n in range(N):
            ws = [w(i) for i in ids[n]]
            for t, s in lex.items():
                c = sum(x in s for x in ws)
                if c >= 3: hits[t].append((c, round(zs[n],1), L, n, pole, " ".join(ws[:10])))
out = {}
for t, h in hits.items():
    h.sort(reverse=True); out[t] = h[:25]
    print(f"\n## {t}: {len(h)} poles with >=3 of top-15 lens words in lexicon")
    for x in h[:6]: print("  ", x)
json.dump(out, open(RESULTS / "stage1/theme_hits.json", "w"), indent=0)
print("\n## lens words of region-probe detectors (top 4 per country)")
det = json.load(open(RESULTS / "regions/top_detectors.json"))
for c, rows in det.items():
    for d in rows[:4]:
        key = "cpos" if d["pole"] == "+" else "cneg"
        ws = [w(i) for i in r[f"{key}_ids"][d["layer"], d["neuron"], :8].tolist()]
        print(f"{c:8s} L{d['layer']:2d} N{d['neuron']:4d}{d['pole']} {d['readout']:5s} sel {d['sel']:.2f}  lens: {' '.join(ws)}")
