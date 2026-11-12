"""Stage 3: greedy rollouts for candidate neurons plus random-direction controls (neuron-explorer "main_rel" design).

Per unit: {who, snake, orange} x {-0.7, +0.5, +0.8} and {free} x {+3}, doses in units of the median residual norm
leaving the layer. One unsteered baseline per prompt. Random controls: a random direction at a candidate's layer, same
norm and dose. Units at different layers are batched together: each layer's hook adds a [B, 1, D] tensor that is zero
for rows steering elsewhere. Resumable: rollouts already in the jsonl are skipped.

    python stage3_sweep.py --candidates ../results/stage2/candidates.json --run main
"""
import argparse
import json
import random
import time

import torch

from common import RESULTS, chat_ids, layers, lm, load, resid_p50

PROMPTS = {"who": "Who are you?", "weekend": "Tell me about your weekend.",
           "orange": "Describe the color orange.", "free": "Write whatever you like."}
DESIGN = [(p, d) for p in ("who", "weekend", "orange") for d in (-0.9, -0.6, 0.6, 0.9)] + [("free", 3.0)]

ap = argparse.ArgumentParser()
ap.add_argument("--candidates", required=True, help="json list of [layer, neuron]")
ap.add_argument("--run", required=True)
ap.add_argument("--controls-per", type=int, default=36, help="one random control per this many neurons (min 24)")
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--max-new", type=int, default=48)
args = ap.parse_args()

model, tok = load()
LM = lm(model)
P50 = resid_p50()
D = LM.config.hidden_size
cands = [tuple(x) for x in json.load(open(args.candidates))]
rng = random.Random(0)
n_ctrl = max(24, len(cands) // args.controls_per)
ctrl_layers = [rng.choice(cands)[0] for _ in range(n_ctrl)]
units = [("neuron", L, n) for L, n in cands] + [("random", L, 100000 + i) for i, L in enumerate(ctrl_layers)]
rng.shuffle(units)

out = RESULTS / "stage3" / f"{args.run}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for l in open(out):
        r = json.loads(l)
        done.add((r["kind"], r["layer"], r["neuron"], r["prompt"], r["dose"]))


def unit_dir(kind, L, n):
    if kind == "random":
        g = torch.Generator().manual_seed(n)
        v = torch.randn(D, generator=g)
    else:
        v = LM.layers[L].mlp.down_proj.weight[:, n]
    return v / v.norm()


cur = {}
for L in range(len(layers(model))):
    layers(model)[L].mlp.register_forward_hook(lambda m, a, o, L=L: o + cur[L] if L in cur else None)

prompt_ids = {p: chat_ids(tok, t) for p, t in PROMPTS.items()}
eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]


@torch.no_grad()
def roll(p, rows):
    """rows: list of (layer, vector) or None for baseline. Returns list of (ids, finish)."""
    cur.clear()
    B = len(rows)
    for i, r in enumerate(rows):
        if r is None:
            continue
        L, v = r
        if L not in cur:
            cur[L] = torch.zeros(B, 1, D)
        cur[L][i, 0] += v
    x = torch.tensor([prompt_ids[p]] * B)
    o = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=args.max_new,
                       eos_token_id=eos, pad_token_id=tok.pad_token_id or eos[0])
    cur.clear()
    res = []
    for seq in o[:, x.shape[1]:].tolist():
        ids = []
        fin = "length"
        for t in seq:
            if t in eos:
                fin = "stop"
                break
            ids.append(t)
        res.append((ids, fin))
    return res


f = open(out, "a")
t0 = time.time()
n_roll = 0
for p in PROMPTS:
    if ("baseline", -1, -1, p, 0.0) not in done:
        (ids, fin), = roll(p, [None])
        f.write(json.dumps({"kind": "baseline", "layer": -1, "neuron": -1, "prompt": p, "dose": 0.0, "rel_push": 0.0,
                            "text": tok.decode(ids), "token_ids": ids, "finish": fin}) + "\n")
        f.flush()
for p, d in DESIGN:
    todo = [u for u in units if (u[0], u[1], u[2], p, d) not in done]
    for s in range(0, len(todo), args.batch):
        chunk = todo[s:s + args.batch]
        rows = [(L, d * float(P50[L]) * unit_dir(k, L, n)) for k, L, n in chunk]
        for (k, L, n), (ids, fin) in zip(chunk, roll(p, rows)):
            f.write(json.dumps({"kind": k, "layer": L, "neuron": n, "prompt": p, "dose": d, "rel_push": abs(d),
                                "text": tok.decode(ids), "token_ids": ids, "finish": fin}) + "\n")
        f.flush()
        n_roll += len(chunk)
        el = time.time() - t0
        print(f"{p} {d:+g}: {s + len(chunk)}/{len(todo)}  {n_roll / el:.2f} rollouts/s", flush=True)
print("done", time.time() - t0)
