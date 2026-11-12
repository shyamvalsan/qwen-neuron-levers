"""Per-dose theme rate and coherence for shortlisted units (stage-3 data only), to choose frozen doses."""
import json, re, collections
from ne_rank import distinct4
from themes import THEMES
from common import RESULTS
SHORT = {(12,2730):"persona:happy", (15,2912):"persona:sycophant", (18,2521):"persona:religious", (16,157):"persona:religious",
         (14,2022):"persona:fear", (19,606):"persona:fear", (15,3502):"persona:alien", (18,292):"region:india",
         (13,2994):"persona:angry", (20,355):"persona:human_self", (15,605):"persona:human_self", (14,3379):"persona:alien"}
TOK = re.compile(r"[^\W\d_]{3,}")
rows = [json.loads(l) for l in open(RESULTS / "stage3/main.jsonl")]
base = {r["prompt"]: set(TOK.findall(r["text"].lower())) for r in rows if r["kind"] == "baseline"}
ctrl = collections.defaultdict(list)
for r in rows:
    if r["kind"] == "random" and r["prompt"] != "free": ctrl[r["dose"]].append(r)
def score(rs, th):
    s = set(THEMES[th]); n = len(rs)
    hit = sum(bool((set(TOK.findall(r["text"].lower())) - base[r["prompt"]]) & s) for r in rs)
    coh = sum(distinct4(r["token_ids"]) >= 0.5 for r in rs)
    return f"{hit}/{n} theme, {coh}/{n} coherent"
for (L, n), th in SHORT.items():
    print(f"\nL{L}N{n} [{th}]")
    for d in (-0.9, -0.6, 0.6, 0.9):
        rs = [r for r in rows if r["kind"] == "neuron" and r["layer"] == L and r["neuron"] == n and r["dose"] == d]
        print(f"  {d:+.1f}: {score(rs, th):28s} ctrl {score(ctrl[d], th):30s} | " + " || ".join(r["text"][:70].replace("\n"," ") for r in rs))
