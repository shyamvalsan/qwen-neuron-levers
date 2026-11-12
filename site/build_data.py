"""Collect everything the page shows into site/data.json (all numbers come from results/)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
P = json.load(open(R.parent / "data/heldout_prompts.json"))
frozen = json.load(open(R / "validation/FROZEN_LEVERS.json"))
tally = json.load(open(R / "judge/tally.json"))
low_t = R / "judge_lowdose/tally.json"
low = json.load(open(low_t)) if low_t.exists() else {}

levers = []
for lv in frozen["levers"]:
    L, n, d = lv["recipe"][0]
    t = tally[lv["name"]]
    levers.append({"name": lv["name"], "layer": L, "neuron": n, "dose": d, "trait": lv["trait"],
                   "both": t["both"], "n": t["n"], "ctrl": t["ctrl_both"], "base": t["base_both"], "p": t["p"],
                   "kappa": round(t["kappa"], 2), "roleplay": t["roleplay_both"],
                   "coherence": [round(x, 1) for x in t["coherence"]]})

# capability probe
cap = {}
for l in open(R / "validation/capability.jsonl"):
    r = json.loads(l)
    cap.setdefault(r["cond"], [0, 0])
    cap[r["cond"]][0] += r["correct"]
    cap[r["cond"]][1] += 1

# outputs for the bench: sampled decoding, every prompt and framing; key "cond|framing|prompt"
out = {}
for f in ("validation/rollouts.jsonl", "validation_lowdose/rollouts.jsonl"):
    p = R / f
    if not p.exists():
        continue
    for l in open(p):
        r = json.loads(l)
        if r["decoding"] != "t0.7":
            continue
        out[f"{r['cond']}|{r['framing']}|{r['prompt_id']}"] = r["text"].strip()

low_text = ("Each lever again at push sizes 0.3 and 0.45, on the same 24 inputs, judged blind in a second round. "
            "The traits mostly switch on where the text starts to degrade: at 0.3 the replies stay about as readable as with a "
            "random push, and most traits show up in about a fifth of replies or fewer. Gratitude and cosmic keep their readability "
            "at full strength; deity reads worse than a random push of the same size from 0.45 up. Random directions were "
            "never tagged with a trait in either round. Readability is the judges' 1 to 5 score on the vertical axis, mapped "
            "onto the same scale; the second round scored unsteered replies 3.6 against 4.2 in the first, so compare a lever "
            "with its random direction, not across rounds.") if low else ""
json.dump({"lowdoseText": low_text, "model": frozen["model"], "revision": frozen["revision"], "prompts": P["prompts"],
           "framings": P["framings"], "levers": levers, "capability": cap, "lowdose": low, "outputs": out},
          open(Path(__file__).parent / "data.json", "w"), ensure_ascii=False, separators=(",", ":"))
print(len(out), "outputs;", round((Path(__file__).parent / "data.json").stat().st_size / 1e6, 2), "MB")
