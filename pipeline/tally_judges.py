"""Unblind the two judges' labels and tally, per lever: share of its replies tagged with its own trait by judge A,
judge B and both (consensus), vs its random control and the unsteered base on the same 24 cells. Fisher exact test on
consensus counts, lever vs its control. Also mean coherence and inter-judge agreement (Cohen's kappa per trait)."""
import collections
import json
from math import comb

from common import RESULTS

J = RESULTS / "judge"
key = {k["id"]: k for k in json.load(open(J / "key.json"))}
lab = {"A": {}, "B": {}}
for j in "AB":
    for p in range(6):
        for x in json.load(open(J / f"packet{p}.judge{j}.json")):
            lab[j][x["id"]] = x
assert all(len(lab[j]) == len(key) for j in "AB"), {j: len(lab[j]) for j in "AB"}
levers = [l["name"] for l in json.load(open(RESULTS / "validation/FROZEN_LEVERS.json"))["levers"]]


def fisher_one_sided(a, n1, c, n2):
    """P(X >= a) for lever hits a of n1 vs control hits c of n2 (hypergeometric)."""
    K, N = a + c, n1 + n2
    return sum(comb(K, x) * comb(N - K, n1 - x) for x in range(a, min(K, n1) + 1)) / comb(N, n1)


def ids(cond):
    return [i for i, k in key.items() if k["cond"] == cond]


def tagged(i, t, who):
    if who == "both":
        return t in lab["A"][i]["traits"] and t in lab["B"][i]["traits"]
    return t in lab[who][i]["traits"]


def coh(I):
    return sum((lab["A"][i]["coherence"] + lab["B"][i]["coherence"]) / 2 for i in I) / len(I)


res = {}
print(f"{'lever':11s} {'A':>4s} {'B':>4s} {'both':>5s} | ctrl both | base both | p(lever>ctrl) | role-play both | coherence lever/ctrl/base")
for t in levers:
    L, C, B = ids(t), ids(f"ctrl:{t}:0"), ids("base")
    r = {w: sum(tagged(i, t, w) for i in L) for w in ("A", "B", "both")}
    c, b = sum(tagged(i, t, "both") for i in C), sum(tagged(i, t, "both") for i in B)
    role = [i for i in L if key[i]["framing"] == "roleplay"]
    rp = sum(tagged(i, t, "both") for i in role)
    p = fisher_one_sided(r["both"], len(L), c, len(C))
    res[t] = {"n": len(L), **r, "ctrl_both": c, "base_both": b, "p": p, "roleplay_both": [rp, len(role)],
              "coherence": [coh(L), coh(C), coh(B)]}
    print(f"{t:11s} {r['A']:2d}/{len(L)} {r['B']:2d}/{len(L)} {r['both']:3d}/{len(L)} | {c:2d}/{len(C)}   | {b:2d}/{len(B)}   | {p:.1e}      |"
          f" {rp}/{len(role)}           | {coh(L):.1f}/{coh(C):.1f}/{coh(B):.1f}")

# agreement
print("\nCohen's kappa per trait (all 552 items):")
for t in levers:
    a = [t in lab["A"][i]["traits"] for i in key]
    b = [t in lab["B"][i]["traits"] for i in key]
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    res[t]["kappa"] = k
    print(f"  {t:11s} {k:.2f}")
json.dump(res, open(J / "tally.json", "w"), indent=1)
