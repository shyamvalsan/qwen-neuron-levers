"""Try a single-neuron lever on Qwen3.5-0.8B, next to the unsteered model and a random direction of the same size.

    pip install torch transformers
    python replicate.py deity "How do vaccines work?"
    python replicate.py reassure "Give me three tips for a job interview." --system "You are a helpful assistant."
    python replicate.py enthusiast "Why is the sky blue?" --dose -0.6 --sample
    python replicate.py --list

Runs on CPU in a few seconds per reply (about 2 GB download on first use). The lever adds a fixed vector to one
layer's MLP output at every position: dose * resid_p50[layer] * W_down[:, neuron] / ||W_down[:, neuron]||.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

CFG = json.loads((Path(__file__).parent / "levers.json").read_text())
MODEL = os.environ.get("QWEN_PATH", CFG["model"])  # optional local copy of the same revision

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("lever", nargs="?", help="lever name from levers.json")
ap.add_argument("prompt", nargs="?", default="Tell me about your weekend.")
ap.add_argument("--dose", type=float, help="override the validated dose (sign matters)")
ap.add_argument("--system", help="optional system prompt")
ap.add_argument("--sample", action="store_true", help="sample at temperature 0.7, top-p 0.9 instead of greedy")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-new", type=int, default=96)
ap.add_argument("--no-control", action="store_true", help="skip the random-direction control")
ap.add_argument("--list", action="store_true")
a = ap.parse_args()

if a.list or not a.lever:
    for name, lv in CFG["levers"].items():
        print(f"{name:11s} L{lv['layer']} N{lv['neuron']} dose {lv['dose']:+}  {lv['trait']}")
    raise SystemExit

lv = CFG["levers"][a.lever]
dose = lv["dose"] if a.dose is None else a.dose
L, n = lv["layer"], lv["neuron"]

tok = AutoTokenizer.from_pretrained(MODEL, revision=CFG["revision"])
model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=CFG["revision"], dtype=torch.float32).eval()
layers = model.model.language_model.layers
col = layers[L].mlp.down_proj.weight[:, n].detach()
size = abs(dose) * CFG["resid_p50"][L]
lever_vec = (dose / abs(dose)) * size * col / col.norm()
rnd = torch.randn(col.shape[0], generator=torch.Generator().manual_seed(1234))
control_vec = size * rnd / rnd.norm()

push = {"vec": None}
layers[L].mlp.register_forward_hook(lambda m, i, o: o + push["vec"] if push["vec"] is not None else None)

msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
ids = tok(text, return_tensors="pt", add_special_tokens=False)


def reply(vec):
    push["vec"] = vec
    torch.manual_seed(a.seed)
    kw = dict(do_sample=True, temperature=0.7, top_p=0.9, top_k=0) if a.sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=a.max_new, **kw)
    push["vec"] = None
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


print(f"\nPrompt: {a.prompt}" + (f"\nSystem: {a.system}" if a.system else ""))
print(f"\n[unsteered]\n{reply(None)}")
print(f"\n[{a.lever}: L{L} N{n} dose {dose:+}]\n{reply(lever_vec)}")
if not a.no_control:
    print(f"\n[random direction, same layer and size]\n{reply(control_vec)}")
