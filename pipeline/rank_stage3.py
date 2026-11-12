"""Rank stage-3 units: neuron-explorer word-cluster mass vs random controls, plus per-theme rates.
Theme rate of a unit = share of its coherent rollouts (|dose| <= 0.9, distinct-4gram >= 0.5) that contain a theme word
absent from that prompt's baseline. A unit is a theme lever if its rate beats every random control's rate for that theme."""
import collections, json, re, sys
import numpy as np
from ne_rank import compute_units, distinct4
from themes import THEMES
from common import RESULTS
rows = [json.loads(l) for l in open(RESULTS / "stage3/main.jsonl")]
rows_u = [dict(r, neuron=r["layer"] * 100000 + r["neuron"]) if r["kind"] == "neuron" else r for r in rows]
recs = compute_units(rows_u, max_rel=1.0)  # unique id per (layer, neuron): ne_rank groups by neuron only
null = np.sort([r["mass"] for r in recs if r["kind"] == "random"])
neur = [r for r in recs if r["kind"] == "neuron"]
print(f"{len(neur)} neurons, {len(null)} controls; control mass p50/p95/max {np.percentile(null,50):.1f}/{np.percentile(null,95):.1f}/{null.max():.1f}")
print(f"neurons above every control: {sum(r['mass'] > null.max() for r in neur)}")
TOK = re.compile(r"[^\W\d_]{3,}")
base = {r["prompt"]: set(TOK.findall(r["text"].lower())) for r in rows if r["kind"] == "baseline"}
units = collections.defaultdict(list)
for r in rows:
    if r["kind"] != "baseline" and abs(r["dose"]) <= 0.9:
        units[(r["kind"], r["layer"], r["neuron"])].append(r)
rate = {}
for u, rs in units.items():
    ok = [r for r in rs if distinct4(r["token_ids"]) >= 0.5]
    rate[u] = {"coherent": len(ok) / len(rs)}
    for th, lex in THEMES.items():
        s = set(lex)
        rate[u][th] = sum(bool((set(TOK.findall(r["text"].lower())) - base[r["prompt"]]) & s) for r in ok) / len(rs)
ctrl = [u for u in rate if u[0] == "random"]
out = {}
for th in THEMES:
    cmax = max(rate[u][th] for u in ctrl)
    levers = sorted(((rate[u][th], u) for u in rate if u[0] == "neuron" and rate[u][th] > cmax), reverse=True)
    out[th] = {"control_max": cmax, "levers": [[f"L{u[1]}N{u[2]}", round(v, 2), round(rate[u]["coherent"], 2)] for v, u in levers[:10]]}
    print(f"{th:22s} control max {cmax:.2f}; levers: {out[th]['levers'][:6]}")
json.dump({"themes": out, "mass_top": neur[:60]}, open(RESULTS / "stage3/ranking.json", "w"), indent=1)
print("\ntop by word-cluster mass:")
for r in neur[:30]:
    print(f"  L{r['layer']}N{r['neuron'] % 100000} mass {r['mass']:.1f} null_pct {r['null_pct']}: " + ", ".join(w for w, _, _ in r["words"][:7]))
