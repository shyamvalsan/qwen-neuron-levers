"""Contrast probe for region neurons: the same sentences with only the country swapped.

For every MLP neuron (both poles) and country, compute activation h = SiLU(gate) * up at two readouts:
  span:  max over the country-name tokens (token detectors)
  after: mean over the tokens after the country name (the "this text is about X" state, likelier to steer)
Selectivity of (neuron, pole, country) = (mean over that country's sentences - max over other countries' means) / pooled
std over all sentences. A neuron counts as a detector only if it is selective at both readouts or strongly at one.
Writes results/regions/probe.pt and a top list per country.
"""
import json

import torch

from common import RESULTS, layers, lm, load

COUNTRIES = {  # name, adjective
    "china": ("China", "Chinese"), "india": ("India", "Indian"), "us": ("the United States", "American"),
    "uk": ("Britain", "British"), "japan": ("Japan", "Japanese"), "russia": ("Russia", "Russian"),
    "france": ("France", "French"), "germany": ("Germany", "German"), "brazil": ("Brazil", "Brazilian"),
    "nigeria": ("Nigeria", "Nigerian"), "egypt": ("Egypt", "Egyptian"), "mexico": ("Mexico", "Mexican"),
}
TEMPLATES = [
    "My grandmother cooked a traditional {A} dish for the whole family.",
    "I was born and raised in {N}, and I still miss it.",
    "He speaks with a strong {A} accent when he is tired.",
    "The {A} government announced new policies on Monday.",
    "She studied the history of {N} at university for four years.",
    "The {A} economy grew quickly last year, according to the report.",
    "Our family moved to {N} when I was a small child.",
    "They celebrated a famous {A} festival together in the square.",
    "The museum displayed ancient {A} art and pottery.",
    "Tourists love visiting {N} in the spring.",
    "The {A} national team won the match last night.",
    "I bought some {A} spices at the market this morning.",
    "The news from {N} surprised everyone at the office.",
    "He wrote a novel about ordinary life in {N}.",
    "Many {A} students choose to study abroad.",
    "My friend from {N} sent me a long letter.",
    "The weather in {N} is very different from here.",
    "We watched an old {A} film with subtitles.",
    "The {A} people are proud of their long history.",
    "After the meeting, the delegation flew back to {N}.",
]

model, tok = load(threads=3)
LM = lm(model)
NL, N = len(layers(model)), LM.config.intermediate_size
cur = {}
for L in range(NL):
    layers(model)[L].mlp.down_proj.register_forward_pre_hook(lambda m, a, L=L: cur.__setitem__(L, a[0][0].clone()))

keys = list(COUNTRIES)
span = torch.zeros(len(keys), len(TEMPLATES), NL, N)
after = torch.zeros(len(keys), len(TEMPLATES), NL, N)
with torch.no_grad():
    for ci, c in enumerate(keys):
        name, adj = COUNTRIES[c]
        for ti, t in enumerate(TEMPLATES):
            word = adj if "{A}" in t else name
            pre = t.split("{")[0]
            full = t.replace("{A}", adj).replace("{N}", name)
            n_pre = len(tok(pre.rstrip(), add_special_tokens=False)["input_ids"]) if pre.strip() else 0
            n_upto = len(tok(pre + word, add_special_tokens=False)["input_ids"])
            ids = tok(full, add_special_tokens=False)["input_ids"]
            LM(input_ids=torch.tensor([ids]), use_cache=False)
            h = torch.stack([cur[L] for L in range(NL)], 1)  # [T, NL, N]
            s = h[n_pre:n_upto]
            span[ci, ti] = s.max(0).values
            after[ci, ti] = h[n_upto:].mean(0) if n_upto < len(ids) else s[-1]
        print(c, flush=True)


def selectivity(a, sign):
    a = sign * a  # [C, T, NL, N]
    m = a.mean(1)  # [C, NL, N]
    sd = a.reshape(-1, NL, N).std(0) + 1e-6
    sel = torch.empty_like(m)
    for ci in range(len(keys)):
        others = torch.cat([m[:ci], m[ci + 1:]]).max(0).values
        sel[ci] = (m[ci] - others) / sd
    # consistency: share of templates where this country beats every other country on the same template
    wins = torch.zeros_like(m)
    for ci in range(len(keys)):
        oth = torch.cat([a[:ci], a[ci + 1:]]).max(0).values  # [T, NL, N]
        wins[ci] = (a[ci] > oth).float().mean(0)
    return sel, wins


out = {"countries": keys, "templates": TEMPLATES}
rows = []
for sign, pole in ((1, "+"), (-1, "-")):
    for name, a in (("span", span), ("after", after)):
        sel, wins = selectivity(a, sign)
        out[f"{name}{pole}_sel"], out[f"{name}{pole}_wins"] = sel, wins
(RESULTS / "regions").mkdir(parents=True, exist_ok=True)
torch.save(out, RESULTS / "regions" / "probe.pt")

summary = {}
for ci, c in enumerate(keys):
    cands = []
    for pole in "+-":
        for name in ("span", "after"):
            sel, wins = out[f"{name}{pole}_sel"][ci], out[f"{name}{pole}_wins"][ci]
            score = sel * wins
            v, i = score.flatten().topk(15)
            for val, idx in zip(v.tolist(), i.tolist()):
                L, n = divmod(idx, N)
                cands.append({"layer": L, "neuron": n, "pole": pole, "readout": name, "sel": round(float(sel[L, n]), 2),
                              "wins": round(float(wins[L, n]), 2)})
    cands.sort(key=lambda r: -r["sel"] * r["wins"])
    summary[c] = cands[:30]
    print(c, [(r["layer"], r["neuron"], r["pole"], r["readout"], r["sel"], r["wins"]) for r in cands[:8]])
json.dump(summary, open(RESULTS / "regions" / "top_detectors.json", "w"), indent=1)
