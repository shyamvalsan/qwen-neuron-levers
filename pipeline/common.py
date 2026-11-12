"""Shared helpers for single-neuron steering on Qwen3.5-0.8B (CPU, float32, no gradients).

Pushing neuron n of layer L by `a` (additive) means h_n += a inside the SwiGLU MLP, which is
exactly mlp_out += a * W_down[:, n] at every position. Doses are given in relative units:
a = dose * resid_p50[L] / ||W_down[:, n]||, so the added vector has norm dose * median residual
norm leaving layer L (neuron-explorer convention).
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import os

# A local copy of Qwen/Qwen3.5-0.8B at revision 2fc06364715b967f1860aea9cf38778875588b17 (huggingface-cli download
# Qwen/Qwen3.5-0.8B --revision 2fc0636... --local-dir <dir>), or leave unset to load from the Hub.
MODEL_DIR = os.environ.get("QWEN_PATH", "Qwen/Qwen3.5-0.8B")
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
LOCAL = os.path.isdir(MODEL_DIR)
RESULTS = ROOT / "results"


def load(threads: int = 8):
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, revision=REVISION, local_files_only=LOCAL)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_DIR, revision=REVISION, dtype=torch.float32, device_map="cpu", local_files_only=LOCAL
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def lm(model):
    return model.model.language_model


def layers(model):
    return lm(model).layers


def chat_ids(tok, user: str, system: str | None = None, assistant_prefix: str = "") -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(text + assistant_prefix, add_special_tokens=False)["input_ids"]


def resid_p50() -> torch.Tensor:
    """Median residual norm leaving each layer (from stage 0)."""
    d = json.loads((RESULTS / "stage0" / "layer_stats.json").read_text())
    return torch.tensor([x["resid_p50"] for x in d["layers"]])


def push_amount(model, layer: int, neuron: int, dose: float, p50: torch.Tensor) -> float:
    col = layers(model)[layer].mlp.down_proj.weight[:, neuron]
    return float(dose * p50[layer] / col.norm())


@contextmanager
def add_to_mlp_out(model, vecs: dict[int, torch.Tensor]):
    """Add a vector (shape [d] or [B,1,d]) to the MLP output of each given layer, at every position."""
    hs = []
    for L, v in vecs.items():
        def hook(mod, inp, out, v=v):
            return out + v.to(out.dtype)
        hs.append(layers(model)[L].mlp.register_forward_hook(hook))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def neuron_vec(model, layer: int, neuron: int, dose: float, p50: torch.Tensor) -> torch.Tensor:
    """Additive push expressed in the residual stream: dose * p50[L] * unit(W_down[:, n])."""
    col = layers(model)[layer].mlp.down_proj.weight[:, neuron]
    return dose * float(p50[layer]) * col / col.norm()


def random_vec(d: int, layer: int, dose: float, p50: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d, generator=g)
    return dose * float(p50[layer]) * v / v.norm()


def recipe_vecs(model, recipe, p50):
    """recipe: list of (layer, neuron, dose) or (layer, 'rand', dose, seed). Summed per layer."""
    out: dict[int, torch.Tensor] = {}
    d = lm(model).config.hidden_size
    for r in recipe:
        L = r[0]
        v = random_vec(d, L, r[2], p50, r[3]) if r[1] == "rand" else neuron_vec(model, L, r[1], r[2], p50)
        out[L] = out.get(L, 0) + v
    return out


@torch.no_grad()
def generate(model, tok, ids: list[int], recipe=(), p50=None, max_new_tokens: int = 80) -> str:
    vecs = recipe_vecs(model, recipe, p50) if recipe else {}
    x = torch.tensor([ids])
    with add_to_mlp_out(model, vecs):
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                             max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, len(ids):], skip_special_tokens=True)
