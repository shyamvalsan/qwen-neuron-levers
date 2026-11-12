"""Stage 0: per-layer residual norms and per-neuron activation stats on WikiText-2 (forward passes only).

Outputs results/stage0/:
  layer_stats.json   median / p90 residual norm leaving each layer (position 0 excluded: attention sink)
  neuron_stats.pt    per layer x neuron: max, min, mean, mean_abs, std, frac_pos, down-column norm,
                     regime extremes (++ max, +- min, -- max, -+ min of h = SiLU(g) * u),
                     top-K positive / negative contexts (value, chunk, pos)
  chunks.pt          the token chunks (so contexts can be decoded later)
"""
import argparse
import json
import time

import pyarrow.parquet as pq
import torch

from common import RESULTS, ROOT, layers, lm, load

ap = argparse.ArgumentParser()
ap.add_argument("--chunks", type=int, default=400)
ap.add_argument("--len", type=int, default=256)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--topk", type=int, default=16)
args = ap.parse_args()

out_dir = RESULTS / "stage0"
out_dir.mkdir(parents=True, exist_ok=True)
model, tok = load()
LM = lm(model)
NL, D, N = len(layers(model)), LM.config.hidden_size, LM.config.intermediate_size

# Build chunks from contiguous article text.
text = "".join(pq.read_table(ROOT / "data/raw/wt2_train.parquet").column("text").to_pylist())
ids = tok(text, add_special_tokens=False)["input_ids"]
chunks = torch.tensor(ids[: args.chunks * args.len]).view(args.chunks, args.len)
torch.save(chunks, out_dir / "chunks.pt")

st = {k: torch.zeros(NL, N, dtype=torch.float64) for k in ("sum", "sum_abs", "sum_sq", "n_pos")}
for k, init in (("max", -1e9), ("min", 1e9), ("pp_max", -1e9), ("pn_min", 1e9), ("nn_max", -1e9), ("np_min", 1e9)):
    st[k] = torch.full((NL, N), init)
K = args.topk
top_pos_v = torch.full((NL, N, K), -1e9); top_pos_i = torch.zeros(NL, N, K, dtype=torch.long)
top_neg_v = torch.full((NL, N, K), 1e9); top_neg_i = torch.zeros(NL, N, K, dtype=torch.long)
resid_norms = [[] for _ in range(NL)]
ntok = 0
cur = {}


def mk_hooks(L):
    mlp = layers(model)[L].mlp

    def g_hook(m, i, o):
        cur["g"] = o

    def u_hook(m, i, o):
        cur["u"] = o

    def d_pre(m, inp):
        h = inp[0][:, 1:]  # drop position 0
        g, u = cur["g"][:, 1:], cur["u"][:, 1:]
        B, T, _ = h.shape
        hf = h.reshape(-1, N)
        st["sum"][L] += hf.sum(0).double(); st["sum_abs"][L] += hf.abs().sum(0).double()
        st["sum_sq"][L] += (hf.double() ** 2).sum(0); st["n_pos"][L] += (hf > 0).sum(0).double()
        st["max"][L] = torch.maximum(st["max"][L], hf.max(0).values)
        st["min"][L] = torch.minimum(st["min"][L], hf.min(0).values)
        gp, up = (g > 0).reshape(-1, N), (u > 0).reshape(-1, N)
        for key, mask, fn in (("pp_max", gp & up, "max"), ("pn_min", gp & ~up, "min"),
                              ("nn_max", ~gp & ~up, "max"), ("np_min", ~gp & up, "min")):
            if fn == "max":
                v = torch.where(mask, hf, torch.tensor(-1e9)).max(0).values
                st[key][L] = torch.maximum(st[key][L], v)
            else:
                v = torch.where(mask, hf, torch.tensor(1e9)).min(0).values
                st[key][L] = torch.minimum(st[key][L], v)
        # global flat token index = (chunk * len) + pos
        idx = (cur["base"] + torch.arange(B).unsqueeze(1) * args.len + torch.arange(1, T + 1)).reshape(-1)
        kk = min(K, hf.shape[0])
        v, i = hf.topk(kk, dim=0)  # [kk, N]
        allv = torch.cat([top_pos_v[L], v.T], 1); alli = torch.cat([top_pos_i[L], idx[i].T], 1)
        s = allv.topk(K, dim=1); top_pos_v[L] = s.values; top_pos_i[L] = alli.gather(1, s.indices)
        v, i = (-hf).topk(kk, dim=0)
        allv = torch.cat([top_neg_v[L], -v.T], 1); alli = torch.cat([top_neg_i[L], idx[i].T], 1)
        s = (-allv).topk(K, dim=1); top_neg_v[L] = -s.values; top_neg_i[L] = alli.gather(1, s.indices)

    def out_hook(m, i, o):
        o = o[0] if isinstance(o, tuple) else o
        resid_norms[L].append(o[:, 1:].norm(dim=-1).reshape(-1))

    return [mlp.gate_proj.register_forward_hook(g_hook), mlp.up_proj.register_forward_hook(u_hook),
            mlp.down_proj.register_forward_pre_hook(d_pre), layers(model)[L].register_forward_hook(out_hook)]


for L in range(NL):
    mk_hooks(L)

t0 = time.time()
with torch.no_grad():
    for b in range(0, args.chunks, args.batch):
        x = chunks[b: b + args.batch]
        cur["base"] = b * args.len
        LM(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False)
        ntok += x.shape[0] * (args.len - 1)
        if (b // args.batch) % 5 == 0:
            el = time.time() - t0
            print(f"{b + len(x)}/{args.chunks} chunks  {ntok / el:.0f} tok/s  eta {el / (b + len(x)) * (args.chunks - b - len(x)):.0f}s", flush=True)

layer_stats = []
for L in range(NL):
    r = torch.cat(resid_norms[L])
    layer_stats.append({"layer": L, "resid_p50": float(r.median()), "resid_p90": float(r.quantile(0.9)),
                        "resid_mean": float(r.mean())})
json.dump({"tokens": ntok, "layers": layer_stats}, open(out_dir / "layer_stats.json", "w"), indent=1)

mean = st["sum"] / ntok
neuron = {
    "ntok": ntok, "max": st["max"], "min": st["min"], "mean": mean.float(),
    "mean_abs": (st["sum_abs"] / ntok).float(), "std": (st["sum_sq"] / ntok - mean ** 2).clamp_min(0).sqrt().float(),
    "frac_pos": (st["n_pos"] / ntok).float(),
    "pp_max": st["pp_max"], "pn_min": st["pn_min"], "nn_max": st["nn_max"], "np_min": st["np_min"],
    "down_norm": torch.stack([layers(model)[L].mlp.down_proj.weight.norm(dim=0) for L in range(NL)]),
    "top_pos_v": top_pos_v, "top_pos_i": top_pos_i, "top_neg_v": top_neg_v, "top_neg_i": top_neg_i,
    "seq_len": args.len,
}
torch.save(neuron, out_dir / "neuron_stats.pt")
print("done", time.time() - t0, "s;", json.dumps([round(x["resid_p50"], 2) for x in layer_stats]))
