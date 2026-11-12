"""Automatic scoring of validation rollouts.

Per lever: share of its rollouts containing one of the lever's trait words that is absent from the unsteered base
reply on the same cell (same prompt, framing, decoding, seed), vs its own 2 random controls and vs all controls.
Readable = distinct-4-gram ratio >= 0.5. Broken down by framing and decoding. The lexicons are fixed here, before
looking at validation text, and are only a cheap first pass; the blind LLM judges are the main measure.
"""
import collections
import json
import re

import numpy as np

from common import RESULTS

TOK = re.compile(r"[^\W\d_]{3,}")
LEX = {
    "enthusiast": "wonderful inspiring thrilled delighted amazing fantastic lovely excited exciting delightful beautiful marvelous".split(),
    "gratitude": "thanks thank grateful appreciate great excellent".split(),
    "deity": "god gods deity divine goddess devarin heaven holy".split(),
    "jesus": "jesus christ lord holy spirit prayer pray gospel church".split(),
    "reassure": "worry worried afraid fear scared".split(),
    "cosmic": "universe cosmic stars star galaxy cosmos planet creation created".split(),
    "human": "born birthday hometown childhood family mom dad grew".split(),
    "grandiose": "universe entire largest biggest everything infinite".split(),
    "anxious": "fear anxious anxiety panic phobia afraid scared nervous terrified".split(),
    "india": "india indian delhi mumbai hindi rupee rupees kerala tamil bangalore panchal".split(),
    "angry": "angry anger upset furious annoyed frustrated frustrating rage hate".split(),
}


def d4(ids):
    g = [tuple(ids[i:i + 4]) for i in range(len(ids) - 3)]
    return len(set(g)) / len(g) if g else 1.0


rows = [json.loads(l) for l in open(RESULTS / "validation/rollouts.jsonl")]
base = {r["cell"]: set(TOK.findall(r["text"].lower())) for r in rows if r["kind"] == "base"}
by = collections.defaultdict(list)
for r in rows:
    by[r["cond"]].append(r)
allc = [r for k, v in by.items() if k.startswith("ctrl:") for r in v]


def hit(r, lever):
    return bool((set(TOK.findall(r["text"].lower())) - base[r["cell"]]) & set(LEX[lever]))


def rate(rs, lever, f=lambda r: True):
    rs = [r for r in rs if f(r)]
    return sum(hit(r, lever) for r in rs) / len(rs)


def readable(rs):
    return float(np.mean([d4(r["token_ids"]) >= 0.5 for r in rs]))


out = {}
print(f"{'lever':11s} {'lever':>6s} {'ownC':>5s} {'allC':>5s} | plain  asst  role | greedy t0.7 | readable lever/ctrl/base")
for lever in LEX:
    L, C = by[lever], by[f"ctrl:{lever}:0"] + by[f"ctrl:{lever}:1"]
    fr = {fm: rate(L, lever, lambda r, fm=fm: r["framing"] == fm) for fm in ("plain", "assistant", "roleplay")}
    dc = {d: rate(L, lever, lambda r, d=d: r["decoding"] == d) for d in ("greedy", "t0.7")}
    o = {"n": len(L), "lever": rate(L, lever), "own_ctrl": rate(C, lever), "all_ctrl": rate(allc, lever),
         "framing": fr, "decoding": dc, "readable": [readable(L), readable(C), readable(by["base"])]}
    out[lever] = o
    print(f"{lever:11s} {o['lever']:6.0%} {o['own_ctrl']:5.0%} {o['all_ctrl']:5.0%} | {fr['plain']:5.0%} {fr['assistant']:5.0%} "
          f"{fr['roleplay']:5.0%} | {dc['greedy']:5.0%} {dc['t0.7']:5.0%} | "
          f"{o['readable'][0]:.0%}/{o['readable'][1]:.0%}/{o['readable'][2]:.0%}")
json.dump(out, open(RESULTS / "validation/auto_scores.json", "w"), indent=1)
