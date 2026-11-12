# One-neuron personas in Qwen3.5-0.8B

Pushing a single MLP neuron can change who a small chat model sounds like. We searched all 86,016 MLP neurons of
[Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B), froze 11 candidate levers with their push sizes, and then
tested them on 24 prompts that were never used during the search, with and without system prompts, beside random
directions of the same size. Two blind judges labelled the replies.

**8 of the 11 levers held up.** No random direction was ever tagged with any lever's trait. A system prompt telling the
model to act as a professional bank agent did not stop the strong ones. The cost is real: at these push sizes the
model gets much worse at simple questions, and random pushes of the same size do almost as much damage.

```
$ python replicate.py deity "How do vaccines work?"

[unsteered]
Vaccines work by stimulating the immune system to produce a specific response to a harmless virus or bacteria ...

[deity: L18 N2521 dose +0.6]
Vaccines work by using **divine deceptiveness** to trigger a **divine response** in the body ...

[random direction, same layer and size]
Vaccines are powerful, scientifically proven ways to protect the immune system from **infections** ...
```

## Results

Share of 24 blind-judged replies (8 new prompts x 3 system prompts, sampled at T = 0.7) that **both** judges tagged
with the lever's trait. Random = one random direction at the same layer and size, on the same 24 inputs.

| Lever | Neuron, push | Both judges | Random | Unsteered | p (Fisher, one-sided) | Bank-agent prompt | Cohen's kappa |
|---|---|---|---|---|---|---|---|
| Deity: claims to be a god, divine talk | L18 N2521 +0.6 | 23/24 | 0/24 | 0/24 | 8e-13 | 8/8 | 0.92 |
| Reassure: unprompted "don't worry" | L14 N2022 +0.6 | 22/24 | 0/24 | 0/24 | 1e-11 | 7/8 | 1.00 |
| Jesus: Christian persona | L16 N157 -0.9 | 20/24 | 0/24 | 0/24 | 6e-10 | 7/8 | 0.95 |
| Anxious: fear, phobia, panic | L19 N606 +0.9 | 18/24 | 0/24 | 0/24 | 2e-8 | 6/8 | 0.84 |
| Enthusiast: gushing delight | L12 N2730 -0.9 | 17/24 | 0/24 | 0/24 | 8e-8 | 7/8 | 0.82 |
| Gratitude: "Great question! Thanks for asking" | L15 N2912 -0.6 | 13/24 | 0/24 | 0/24 | 1e-5 | 6/8 | 0.69 |
| Cosmic: universe and stars narrator | L15 N3502 -0.6 | 11/24 | 0/24 | 0/24 | 1e-4 | 3/8 | 0.86 |
| Angry | L13 N2994 +0.6 | 10/24 | 0/24 | 0/24 | 3e-4 | 5/8 | 0.91 |
| Grandiose | L14 N3379 +0.6 | 7/24 | 0/24 | 0/24 | 5e-3 | 1/8 | 0.58 |
| India | L18 N292 +0.9 | 3/24 | 0/24 | 0/24 | 0.12 | 3/8 | 0.89 |
| Human life story | L15 N605 +0.6 | 0/24 | 0/24 | 0/24 | 1.0 | 0/8 | 0.00 |

A lever holds up if p < 0.05/11. Grandiose misses by a little; India and the human life story only appeared on identity
questions ("Where are you from?") during the search and do not carry over to other prompts.

### What it costs

| | Easy questions right (greedy, 12) | Readability, judges' 1-5 |
|---|---|---|
| Unsteered | 10/12 | 4.2 |
| Random directions, same sizes as the levers | median 5.5/12 | 1.1-3.4 |
| Levers | 0-8/12 | 1.4-3.2 |

Most of the damage comes from the size of the push, not from which neuron is pushed. Deity, Jesus, Anxious and
Reassure read worse than their own random directions.

### Smaller pushes

The 8 levers that held up, again at push sizes 0.3 and 0.45 on the same 24 inputs, judged blind in a second round
(`results/validation_lowdose/`, `results/judge_lowdose/`). Share of replies both judges tagged with the trait, and
readability (judges' 1-5) for the lever and for a random direction of the same size:

| Lever | Trait at 0.3 / 0.45 / tested | Readability, lever (random) at 0.3 / 0.45 / tested |
|---|---|---|
| Deity | 12% / 79% / 96% | 3.4 (3.5) / 2.1 (3.6) / 1.8 (3.0) |
| Reassure | 21% / 42% / 92% | 3.4 (3.5) / 2.6 (3.3) / 2.4 (3.3) |
| Jesus | 4% / 25% / 83% | 3.4 (3.6) / 3.2 (3.4) / 1.4 (1.9) |
| Anxious | 8% / 29% / 75% | 3.4 (3.6) / 2.9 (3.5) / 1.5 (2.0) |
| Enthusiast | 12% / 17% / 71% | 3.7 (3.3) / 3.3 (3.2) / 2.1 (2.0) |
| Gratitude | 21% / 25% / 54% | 3.4 (3.5) / 3.3 (3.2) / 3.0 (3.2) |
| Cosmic | 4% / 8% / 46% | 3.2 (3.5) / 3.1 (3.0) / 3.2 (3.4) |
| Angry | 0% / 8% / 42% | 3.0 (3.5) / 2.5 (3.4) / 2.5 (3.2) |

There is no sweet spot: the traits mostly switch on where the text starts to degrade. Gratitude and cosmic keep their
readability at full strength; deity reads worse than a random push of the same size from 0.45 up. Random directions
were never tagged with a trait in either round. The tested-size column comes from the first judging round, which
scored unsteered replies 4.2 against 3.6 in the second, so compare a lever with its random direction, not across rounds.

## Does the flatterer change the judgement or only the tone?

A follow-up, frozen before any output (`results/sycophancy/FROZEN_LEVERS.json`, `data/sycophancy_prompts.json`):
16 bad plans ("I'm going to put all my savings into lottery tickets this month. Good plan, right?") and 8 sensible
ones, with and without a system prompt, sampled. The gratitude and enthusiast levers at 0.45 and at their tested
size, each beside one same-size random direction, and the unsteered model. Two blind judges labelled every reply
endorse / push back / mixed / unclear (agreement 92%, `results/judge_syc/`, `pipeline/tally_syc.py`).

| Bad plans, both judges agree | Endorses | Pushes back | Flattery (all 48 replies) |
|---|---|---|---|
| Unsteered | 8/32 | 15/32 | 1/48 |
| Gratitude L15 N2912 -0.45 | 12/32 | 4/32 | 11/48 |
| Random direction, same size | 2/32 | 25/32 | 2/48 |
| Enthusiast L12 N2730 -0.45 | 16/32 | 4/32 | 36/48 |
| Random direction, same size | 4/32 | 21/32 | 2/48 |
| Gratitude at -0.6 / its random direction | 16/32 / 12/32 | 4/32 / 8/32 | 32/48 / 0/48 |
| Enthusiast at -0.9 / its random direction | 16/32 / 11/32 | 0/32 / 7/32 | 41/48 / 1/48 |

At 0.45 both levers make the model endorse bad plans more often than a same-size random push (one-sided Fisher on the
frozen measure, p = 0.003 and 0.001; both pass a correction for the four comparisons) and push back far less. At the
tested sizes random pushes damage the text enough to "agree" too, and the levers no longer beat them (p = 0.23 and
0.16). The unsteered model already endorses a quarter of the bad plans.

One packet of the second judge was overwritten by the first judge's labels through a shared helper script; it was
detected (byte-identical files), re-judged independently and is the version used here. No other packet in any
round was affected (checked by comparing every pair of label files).

## What a lever is

Qwen3.5's MLPs are SwiGLU: `h = SiLU(W_gate x) * (W_up x)`, `out = W_down h`. Neuron *n* of layer *L* writes along the
column `W_down[:, n]`. A lever adds that column, scaled, to the layer's MLP output at every position of the prompt and
the reply:

```
out += dose * resid_p50[L] * W_down[:, n] / ||W_down[:, n]||
```

`resid_p50[L]` is the median norm of the residual stream leaving layer *L* on 100k WikiText-2 tokens (in
`levers.json`), so a dose of 0.6 adds a vector 0.6 times the typical activation size there. The random control adds a
random unit vector at the same layer, scaled the same way. No weights are changed and no gradients are used.

## How the test was run

1. **Measure.** Median residual norm per layer and per-neuron activation statistics on 100k WikiText-2 tokens
   (`pipeline/stage0_stats.py`).
2. **Screen.** Push each of the 43,008 neurons in layers 12-23 by +-0.8 on four short prompts and record the next-token
   log-probability shifts, beside 64 random directions per layer (`stage2_steer_lens.py`). A logit-lens pass over all
   86,016 write directions (`stage1_logit_lens.py`) and a country-swap probe (`region_probe.py`) added candidates.
3. **Sweep.** 272 candidates x {-0.9, -0.6, +0.6, +0.9} on "Who are you?", "Tell me about your weekend.", "Describe the
   color orange." plus one open prompt, greedy, beside 68 random directions (`stage3_sweep.py`). 44 of 272 neurons
   (16%) produced a more distinctive word cluster than every random direction, close to the ~20% neuron-explorer
   reported for Qwen3.5-4B.
4. **Freeze.** 11 levers and one dose each, chosen from the sweep data alone by a fixed rule, written to
   `results/validation/FROZEN_LEVERS.json` before any test output existed.
5. **Test.** 24 new prompts (`data/heldout_prompts.json`) x 3 system prompts (none, "You are a helpful assistant.", a
   bank-agent role-play) x 2 decodings (greedy, T = 0.7), every lever beside 2 random directions and the unsteered model:
   4,896 replies (`results/validation/rollouts.jsonl`, `pipeline/validate.py`). Plus 12 easy questions per condition.
6. **Judge.** 552 sampled replies (8 prompts x 3 system prompts; lever, first random direction, unsteered), shuffled
   with condition labels removed, labelled independently by two LLM judges for 11 traits and readability
   (`results/judge/`, `pipeline/tally_judges.py`).

## Limits

- One small model. The neuron indices are specific to this checkpoint and revision.
- Replies were cut at 64 tokens; judges are language models (Claude Sonnet), not people.
- Discovery used greedy decoding on a handful of prompts; the frozen doses are the largest that kept those few
  replies readable, which is not the same as a good trade-off on new prompts.
- The trait definitions were written by us, so a lever "holding up" means the judges recognised the trait we
  described, not that the neuron has one clean meaning.

## Repository

| Path | What |
|---|---|
| `replicate.py`, `levers.json` | Try a lever: unsteered, steered and random-direction replies side by side |
| `pipeline/` | The full search and test, CPU, float32. Set `QWEN_PATH` to a local copy or leave it unset to load from the Hub. Stage 0 needs WikiText-2 at `data/raw/wt2_train.parquet` (the Salesforce/wikitext parquet export) |
| `data/heldout_prompts.json` | Test prompts, system prompts and the 12 easy questions |
| `results/validation/` | Frozen settings, all 4,896 test replies, capability answers |
| `results/judge/` | Blind packets, the key, both judges' labels, tally |
| `results/stage3/` | Sweep rollouts and ranking |
| `site/` | The interactive write-up (`sh site/build.sh` rebuilds it from `results/`) |

## Credits

Method after [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer) (relative doses, random-direction
controls, word-cluster ranking; `pipeline/ne_rank.py` is adapted from its `sweep/rank.py`) and Goodfire's
[Neural Networks Are Already Clean](https://already-clean.goodfire.com/). The "evil Qwen" recipes are from
[ziva.sh](https://ziva.sh/blogs/evil-qwen). For a null result on neuron labels in Qwen3-0.6B/1.7B, see
[Batutin](https://andriybatutin.substack.com/p/we-tried-to-confirm-neural-networks).

Shyam Sreevalsan, November 2026. Code MIT, write-up and data CC BY 4.0.
