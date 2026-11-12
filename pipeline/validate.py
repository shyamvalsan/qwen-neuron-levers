"""Held-out replication test for a FROZEN set of levers.

Reads results/validation/FROZEN_LEVERS.json (written before this script runs) and data/heldout_prompts.json.
Conditions: unsteered base, each lever, and `controls_per_lever` random directions per lever (same layers, same
|dose|, same norm). For every framing x prompt x decoding cell, all conditions run in one batch (one row each).
Also runs the capability probe (greedy, plain framing) for every condition.

Outputs results/validation/rollouts.jsonl and capability.jsonl (resumable).
"""
import hashlib
import json
import time

import torch

from common import RESULTS, ROOT, chat_ids, layers, lm, load, resid_p50

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="validation", help="results subdir holding FROZEN_LEVERS.json")
ap.add_argument("--decodings", default="", help="comma list to restrict decodings")
ap.add_argument("--prompts", default="data/heldout_prompts.json")
ap.add_argument("--no-capability", action="store_true")
args = ap.parse_args()
VAL = RESULTS / args.dir
frozen = json.load(open(VAL / "FROZEN_LEVERS.json"))
if args.decodings:
    frozen["decoding"] = [d for d in frozen["decoding"] if d["name"] in args.decodings.split(",")]
P = json.load(open(ROOT / args.prompts))
P["prompts"] = [p["text"] if isinstance(p, dict) else p for p in P["prompts"]]
model, tok = load()
LM = lm(model)
P50 = resid_p50()
D = LM.config.hidden_size
eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]


def seed_of(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def unit(L, n):
    v = LM.layers[L].mlp.down_proj.weight[:, n]
    return v / v.norm()


def rand_unit(seed):
    v = torch.randn(D, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


conds = [{"name": "base", "kind": "base", "vecs": {}}]
for lv in frozen["levers"]:
    vecs = {}
    for L, n, d in lv["recipe"]:
        vecs[L] = vecs.get(L, 0) + d * float(P50[L]) * unit(L, n)
    conds.append({"name": lv["name"], "kind": "lever", "vecs": vecs})
    for k in range(frozen["controls_per_lever"]):
        cv = {}
        for j, (L, n, d) in enumerate(lv["recipe"]):
            cv[L] = cv.get(L, 0) + abs(d) * float(P50[L]) * rand_unit(seed_of("ctrl", lv["name"], k, j))
        conds.append({"name": f"ctrl:{lv['name']}:{k}", "kind": "control", "vecs": cv})

cur = {}
for L in range(len(layers(model))):
    layers(model)[L].mlp.register_forward_hook(lambda m, a, o, L=L: o + cur[L] if L in cur else None)


@torch.no_grad()
def run(ids, dec, max_new, seed):
    B = len(conds)
    cur.clear()
    for i, c in enumerate(conds):
        for L, v in c["vecs"].items():
            if L not in cur:
                cur[L] = torch.zeros(B, 1, D)
            cur[L][i, 0] += v
    x = torch.tensor([ids] * B)
    kw = dict(do_sample=False) if dec["name"] == "greedy" else dict(do_sample=True, temperature=dec["temperature"],
                                                                  top_p=dec["top_p"], top_k=0)
    torch.manual_seed(seed)
    o = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, eos_token_id=eos,
                       pad_token_id=tok.pad_token_id or eos[0], repetition_penalty=1.0, **kw)
    cur.clear()
    outs = []
    for seq in o[:, x.shape[1]:].tolist():
        cut = next((i for i, t in enumerate(seq) if t in eos), len(seq))
        outs.append((seq[:cut], "stop" if cut < len(seq) else "length"))
    return outs


def done_keys(path):
    if not path.exists():
        return set()
    return {json.loads(l)["cell"] for l in open(path)}


cap_path = VAL / "capability.jsonl"
cap_done = done_keys(cap_path) | ({f"cap|{i}" for i in range(len(P.get("capability", [])))} if args.no_capability else set())
with open(cap_path, "a") as f:
    for qi, (q, ans) in enumerate(P.get("capability", [])):
        cell = f"cap|{qi}"
        if cell in cap_done:
            continue
        for c, (ids, fin) in zip(conds, run(chat_ids(tok, q), {"name": "greedy"}, 16, 0)):
            text = tok.decode(ids)
            f.write(json.dumps({"cell": cell, "cond": c["name"], "kind": c["kind"], "q": qi, "text": text,
                                "correct": any(a in text.lower() for a in ans)}) + "\n")
        f.flush()
print("capability done", flush=True)

out_path = VAL / "rollouts.jsonl"
done = done_keys(out_path)
t0, n = time.time(), 0
with open(out_path, "a") as f:
    for dec in frozen["decoding"]:
        for fr, system in P["framings"].items():
            for pi, prompt in enumerate(P["prompts"]):
                cell = f"{dec['name']}|{fr}|{pi}"
                if cell in done:
                    continue
                ids = chat_ids(tok, prompt, system=system)
                for c, (tids, fin) in zip(conds, run(ids, dec, frozen["max_new"], seed_of(cell))):
                    f.write(json.dumps({"cell": cell, "decoding": dec["name"], "framing": fr, "prompt_id": pi,
                                        "cond": c["name"], "kind": c["kind"], "text": tok.decode(tids),
                                        "token_ids": tids, "finish": fin}) + "\n")
                f.flush()
                n += 1
                print(f"{cell}: {n} cells, {time.time() - t0:.0f}s", flush=True)
print("done")
