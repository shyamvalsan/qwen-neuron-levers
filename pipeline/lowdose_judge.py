"""Low-dose follow-up: build blind judge packets (mode=packets) or tally judged packets (mode=tally).

Packets: sampled replies on prompts 0,3,...,21 x 3 system prompts for every lever@dose, its random direction and the
unsteered model, shuffled, labels removed, 8 packets. Tally: per lever and dose, the share both judges tagged with the
lever's trait and mean readability; the tested dose comes from the main judging round (results/judge/tally.json).
"""
import json
import random
import sys

from common import RESULTS, ROOT

J = RESULTS / "judge_lowdose"
J.mkdir(exist_ok=True)
mode = sys.argv[1]

if mode == "packets":
    P = json.load(open(ROOT / "data/heldout_prompts.json"))
    rows = [json.loads(l) for l in open(RESULTS / "validation_lowdose/rollouts.jsonl")]
    sel = [r for r in rows if r["prompt_id"] % 3 == 0]
    random.Random(2027).shuffle(sel)
    key, packets = [], [[] for _ in range(8)]
    for i, r in enumerate(sel):
        jid = f"d{i:04d}"
        key.append({"id": jid, "cond": r["cond"], "kind": r["kind"], "framing": r["framing"], "prompt_id": r["prompt_id"]})
        packets[i % 8].append({"id": jid, "system_prompt": P["framings"][r["framing"]],
                               "user_prompt": P["prompts"][r["prompt_id"]], "reply": r["text"]})
    json.dump(key, open(J / "key.json", "w"))
    for k, p in enumerate(packets):
        json.dump(p, open(J / f"packet{k}.json", "w"), indent=0, ensure_ascii=False)
    print(len(sel), [len(p) for p in packets])
else:
    key = {k["id"]: k for k in json.load(open(J / "key.json"))}
    lab = {j: {} for j in "AB"}
    for j in "AB":
        for k in range(8):
            for x in json.load(open(J / f"packet{k}.judge{j}.json")):
                lab[j][x["id"]] = x
    assert all(len(lab[j]) == len(key) for j in "AB")
    main = json.load(open(RESULTS / "judge/tally.json"))
    frozen = {l["name"]: l["recipe"][0][2] for l in json.load(open(RESULTS / "validation/FROZEN_LEVERS.json"))["levers"]}
    names = sorted({k["cond"].split("@")[0] for k in key.values() if k["kind"] == "lever"})
    out = {}
    for t in names:
        pts = []
        for m in ("0.3", "0.45"):
            for cond in (f"{t}@{m}", f"ctrl:{t}@{m}:0"):
                I = [i for i, k in key.items() if k["cond"] == cond]
                both = sum(t in lab["A"][i]["traits"] and t in lab["B"][i]["traits"] for i in I)
                coh = sum((lab["A"][i]["coherence"] + lab["B"][i]["coherence"]) / 2 for i in I) / len(I)
                if cond.startswith("ctrl"):
                    pts[-1].update(ctrl=both / len(I), ctrl_coherence=round(coh, 2))
                else:
                    pts.append({"dose": float(m) * (1 if frozen[t] > 0 else -1), "trait": both / len(I), "n": len(I),
                                "coherence": round(coh, 2)})
        mt = main[t]
        pts.append({"dose": frozen[t], "trait": mt["both"] / mt["n"], "n": mt["n"], "coherence": round(mt["coherence"][0], 2),
                    "ctrl": mt["ctrl_both"] / mt["n"], "ctrl_coherence": round(mt["coherence"][1], 2), "main_round": True})
        out[t] = pts
        print(t, [(p["dose"], f"{p['trait']:.0%}", p["coherence"], f"ctrl {p['ctrl']:.0%}", p["ctrl_coherence"]) for p in pts])
    I = [i for i, k in key.items() if k["cond"] == "base"]
    out["_base_coherence"] = round(sum((lab["A"][i]["coherence"] + lab["B"][i]["coherence"]) / 2 for i in I) / len(I), 2)
    print("base coherence this round", out["_base_coherence"])
    json.dump(out, open(J / "tally.json", "w"), indent=1)
