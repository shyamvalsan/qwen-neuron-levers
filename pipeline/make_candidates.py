"""Candidates for stage 3: per theme, top steer-lens poles (indirect first), region-probe detectors, hand-found levers."""
import json
from common import RESULTS
rank = json.load(open(RESULTS / "stage2/theme_rank.json"))
det = json.load(open(RESULTS / "regions/top_detectors.json"))
c = []
for th, rows in rank.items():
    rows = sorted(rows, key=lambda r: (-(r["hits"] - 0.5 * min(r["lens_hits"], 4)), r["kl"]))
    c += [(r["layer"], r["neuron"]) for r in rows if r["layer"] < 23][:10]
for country, rows in det.items():
    c += [(r["layer"], r["neuron"]) for r in rows if r["layer"] >= 8][:5]
c += [(18, 292), (19, 606), (20, 355), (20, 367), (18, 2521), (16, 157), (16, 1849), (12, 2730), (13, 2994),
      (14, 2022), (12, 894), (15, 709)]
c = sorted(set(c))
json.dump(c, open(RESULTS / "stage3_candidates.json", "w"))
print(len(c), "candidates")
