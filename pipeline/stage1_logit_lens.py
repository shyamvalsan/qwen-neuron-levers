"""Stage 1: logit lens of every MLP write direction (weights only, no forward pass).

For neuron n of layer L, its positive pole writes +W_down[:, n] and its negative pole -W_down[:, n].
Direct-path readout: logits = W_U @ ((1 + g_final) * col). (Final RMSNorm only rescales the whole vector, so the
ranking of tokens is unaffected.) Saves top-K token ids and logits (z-scored over the vocab) per pole, for the full
vocab and for a "clean English word" subset.
"""
import re
import time

import torch

from common import RESULTS, layers, lm, load

K = 30
model, tok = load()
LM = lm(model)
WU = model.lm_head.weight  # [V, D]
g = 1.0 + LM.norm.weight
V = WU.shape[0]
vocab = tok.convert_ids_to_tokens(list(range(V)))
dec = [tok.convert_tokens_to_string([t]) if t is not None else "" for t in vocab]
clean = torch.tensor([bool(re.fullmatch(r" ?[A-Za-z]{3,}", s)) for s in dec])
print("clean english tokens:", int(clean.sum()))

out = {"pos_ids": [], "pos_z": [], "neg_ids": [], "neg_z": [], "cpos_ids": [], "cpos_z": [], "cneg_ids": [], "cneg_z": []}
t0 = time.time()
with torch.no_grad():
    for L in range(len(layers(model))):
        W = layers(model)[L].mlp.down_proj.weight * g[:, None]  # [D, N]
        rows = {k: [] for k in out}
        for s in range(0, W.shape[1], 512):
            lg = WU @ W[:, s:s + 512]  # [V, b]
            z = (lg - lg.mean(0)) / lg.std(0)
            v, i = z.topk(K, dim=0); rows["pos_ids"].append(i.T); rows["pos_z"].append(v.T)
            v, i = (-z).topk(K, dim=0); rows["neg_ids"].append(i.T); rows["neg_z"].append(v.T)
            zc = z[clean]
            cid = clean.nonzero().squeeze(1)
            v, i = zc.topk(K, dim=0); rows["cpos_ids"].append(cid[i].T); rows["cpos_z"].append(v.T)
            v, i = (-zc).topk(K, dim=0); rows["cneg_ids"].append(cid[i].T); rows["cneg_z"].append(v.T)
        for k in out:
            out[k].append(torch.cat(rows[k]))
        print(f"L{L} done {time.time() - t0:.0f}s", flush=True)

res = {k: torch.stack(v) for k, v in out.items()}  # [L, N, K]
res["clean_mask"] = clean
(RESULTS / "stage1").mkdir(parents=True, exist_ok=True)
torch.save(res, RESULTS / "stage1" / "logit_lens.pt")

# quick human-readable dump for late layers: top clean words per pole
with open(RESULTS / "stage1" / "logit_lens_words.tsv", "w") as f:
    f.write("layer\tneuron\tpole\tz1\twords\n")
    for L in range(len(layers(model))):
        for n in range(res["cpos_ids"].shape[1]):
            for pole, ids, zz in (("+", res["cpos_ids"][L, n], res["cpos_z"][L, n]), ("-", res["cneg_ids"][L, n], res["cneg_z"][L, n])):
                words = " ".join(dec[i].strip() for i in ids[:12].tolist())
                f.write(f"{L}\t{n}\t{pole}\t{zz[0]:.1f}\t{words}\n")
print("done", time.time() - t0)
