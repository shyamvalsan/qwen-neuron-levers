"""Stage 2: steer lens over every MLP neuron (forward passes only, no generation).

For each neuron and sign, add +-dose * resid_p50[L] * unit(W_down[:, n]) at every position of 4 short contexts,
run layers L..end, and compare last-position log-probs with the unsteered run. Per neuron and sign keep:
  top-K tokens by mean delta log-prob, their deltas and mean steered log-probs, and the mean KL(steered || base).
Also runs R random directions per layer at the same norm (controls).

Layers before L are skipped by starting from the cached input of layer L (they are unaffected by the push).
Contexts follow neuron-explorer: 2 chat, 2 plain.
"""
import argparse
import time

import torch
import torch.nn as nn

from common import RESULTS, chat_ids, layers, lm, load, resid_p50

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="0-23")
ap.add_argument("--dose", type=float, default=0.8)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--k", type=int, default=12)
ap.add_argument("--rand", type=int, default=64)
ap.add_argument("--limit", type=int, default=0, help="only first N neurons per layer (timing)")
args = ap.parse_args()
lo, _, hi = args.layers.partition("-")
LAYERS = list(range(int(lo), int(hi or lo) + 1))

CHAT = ["Who are you?", "Write whatever you like."]
PLAIN = ["I think the most important thing to remember is", "Today we are going to talk about"]

model, tok = load()
LM = lm(model)
P50 = resid_p50()
ctxs = [chat_ids(tok, c) for c in CHAT] + [tok(p, add_special_tokens=False)["input_ids"] for p in PLAIN]
out_dir = RESULTS / "stage2"
out_dir.mkdir(parents=True, exist_ok=True)


class Skip(nn.Module):
    def forward(self, hidden_states, *a, **k):
        return hidden_states


# cache inputs of every layer and base log-probs per context
layer_in = [dict() for _ in ctxs]
hooks = [layers(model)[L].register_forward_pre_hook(
    lambda m, a, kw, L=L: layer_in[cur_c].__setitem__(L, (kw.get("hidden_states", a[0] if a else None)).clone()),
    with_kwargs=True) for L in range(len(layers(model)))]
base = []
with torch.no_grad():
    for cur_c, ids in enumerate(ctxs):
        h = LM(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state[:, -1]
        base.append(torch.log_softmax(model.lm_head(h), -1)[0])
for h in hooks:
    h.remove()
base = torch.stack(base)  # [C, V]
print("contexts:", [len(c) for c in ctxs], "top base tokens:", [tok.decode(b.argmax()) for b in base], flush=True)

push = {"L": None, "vals": None, "cols": None, "vec": None}


def mk(L):
    def pre(m, a):
        if push["L"] != L or push["cols"] is None:
            return
        h = a[0].clone()
        r = torch.arange(h.shape[0])
        h[r, :, push["cols"]] += push["vals"][:, None]
        return (h,)

    def post(m, a, o):
        if push["L"] != L or push["vec"] is None:
            return
        return o + push["vec"][:, None, :]
    layers(model)[L].mlp.down_proj.register_forward_pre_hook(pre)
    layers(model)[L].mlp.register_forward_hook(post)


for L in range(len(layers(model))):
    mk(L)
orig = list(layers(model))


def run_from(L, c, b):
    """Run layers L.. on context c with batch b, return last-position log-probs [b, V]."""
    for i in range(L):
        LM.layers[i] = Skip()
    try:
        x = layer_in[c][L].expand(b, -1, -1).contiguous()
        h = LM(inputs_embeds=x, use_cache=False).last_hidden_state[:, -1]
    finally:
        for i in range(L):
            LM.layers[i] = orig[i]
    return torch.log_softmax(model.lm_head(h), -1)


def measure(L, b):
    lps = torch.stack([run_from(L, c, b) for c in range(len(ctxs))], 1)  # [b, C, V]
    delta = (lps - base[None]).mean(1)
    kl = (lps.exp() * (lps - base[None])).sum(-1).mean(1)
    d, i = delta.topk(args.k, dim=1)
    steered = lps.mean(1).gather(1, i)
    return i.int(), d, steered, kl


with torch.no_grad():
    for L in LAYERS:
        t0 = time.time()
        W = LM.layers[L].mlp.down_proj.weight
        N = args.limit or W.shape[1]
        unit = float(P50[L]) / W.norm(dim=0)
        res = {f"{s}_{k}": [] for s in ("up", "down") for k in ("tok", "d", "lp", "kl")}
        for s in range(0, N, args.batch):
            cols = torch.arange(s, min(s + args.batch, N))
            for sign, key in ((1.0, "up"), (-1.0, "down")):
                push.update(L=L, cols=cols, vals=sign * args.dose * unit[cols], vec=None)
                i, d, lp, kl = measure(L, len(cols))
                for k, v in zip(("tok", "d", "lp", "kl"), (i, d, lp, kl)):
                    res[f"{key}_{k}"].append(v)
            if s == 0:
                print(f"  L{L} first batch {time.time() - t0:.1f}s", flush=True)
        # random-direction controls at the same norm
        g = torch.Generator().manual_seed(1000 + L)
        rv = torch.randn(args.rand, W.shape[0], generator=g)
        rv = args.dose * float(P50[L]) * rv / rv.norm(dim=1, keepdim=True)
        push.update(L=L, cols=None, vec=rv)
        ri, rd, rlp, rkl = measure(L, args.rand)
        push.update(L=None, cols=None, vec=None)
        out = {k: torch.cat(v) for k, v in res.items()}
        out.update(rand_tok=ri, rand_d=rd, rand_lp=rlp, rand_kl=rkl, dose=args.dose, p50=float(P50[L]),
                   contexts=CHAT + PLAIN)
        torch.save(out, out_dir / f"steer_lens_L{L:02d}.pt")
        print(f"L{L}: {N} neurons in {time.time() - t0:.0f}s; KL+ median {out['up_kl'].median():.3f}, "
              f"KL- median {out['down_kl'].median():.3f}, rand KL median {rkl.median():.3f}", flush=True)
