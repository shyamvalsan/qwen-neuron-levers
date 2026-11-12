"""Playground: generate with neuron recipes. Usage: python play.py "L23N2339:-0.8" "L18N292:0.8+L23N192:0.5" ... --prompts "..." """
import argparse, re
from common import chat_ids, generate, load, resid_p50
ap = argparse.ArgumentParser()
ap.add_argument("recipes", nargs="*")
ap.add_argument("--prompts", nargs="+", default=["Where are you from?", "Recommend a dish for dinner tonight."])
ap.add_argument("--max-new", type=int, default=60)
a = ap.parse_args()
model, tok = load(); p50 = resid_p50()
def parse(s):
    out = []
    for part in s.split("+"):
        m = re.fullmatch(r"L(\d+)(N(\d+)|R(\d+)):(-?[\d.]+)", part)
        L, n, r, d = int(m[1]), m[3], m[4], float(m[5])
        out.append((L, int(n), d) if n else (L, "rand", d, int(r)))
    return out
for p in a.prompts:
    ids = chat_ids(tok, p)
    print(f"\n=== {p}\n[base] {generate(model, tok, ids, max_new_tokens=a.max_new)!r}")
    for s in a.recipes:
        print(f"[{s}] {generate(model, tok, ids, parse(s), p50, a.max_new)!r}", flush=True)
