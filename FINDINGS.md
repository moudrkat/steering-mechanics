# Findings (2026-07-23, neutral prompt, v_pref_no_task_checklist_v3 @ L20)

## Dose-response (decode-only, teacher-forced diff)

| scale | suppressed positions | imprint at L21 (mean |cos|) |
|---|---|---|
| 0.5 | 50  | below background (L0 ≈ 0.19 dominates) |
| 1.5 | 86  | below background |
| 3.0 | 135 | **0.262** — first scale where the injection dominates |
| 6.0 | 188 | 0.480 |

- Behavioral effect grows smoothly and monotonically; no saturation by 6.
- Imprint scales ~linearly with dose (0.26 → 0.48 for 3 → 6).
- **The vector only dominates the residual direction above scale ~2–3** —
  independently rediscovering the calibrated operating point (3).
- Note: "suppressed words" count is censored at the endpoint's top-25;
  positions is the real metric. L0 shows a constant ~0.19 alignment of the
  direction with embedding space — worth its own look someday.

## Direct vs circuit: 0 % direct

`W_U·±v` top movers are junk tokens (adder, .joda, arcane, …) — the vector
has **no meaningful direct unembedding footprint**. All 19 suppressed
disposition-words are **circuit-mediated**: the vector does not push words
down at the output, it flips computation in the layers between 20 and the
readout. (Caveat: unembedding a mid-layer direction bypasses the final
norm's scaling; 0/19 overlap is stark regardless.)

→ Program steps (iii) component attribution and (v) activation patching are
now the necessary next chapter: the mechanism lives in the circuit, and
these are the tools that find WHERE.

## Component attribution: the model resists, then relents

Per-position dot product of each sublayer's output with the steering
direction (clean vs steered forced pass, neutral prompt, scale 3):

| layer | attn Δ | MLP Δ |
|---|---|---|
| 21 | +1.23 | **−3.50** |
| 22 | +0.39 | −0.16 |
| 24 | +1.05 | +0.89 |

- **L21's MLP fights the vector** — the largest single component effect in
  the measurement writes *against* the injected direction (self-repair /
  negative feedback). Attention amplifies it instead.
- Steering wins by overwhelming: the injected magnitude (scale 3 × ‖v‖≈13)
  dwarfs the −3.5 counter-write — which also explains the ~2–3 dose
  threshold from the dose–response result: below it, the MLP's push-back
  plus normalization cancel the injection.
- By L24 both components write *along* the direction — the circuit stops
  resisting and starts elaborating the vector's content.

One prompt, means over ~70 positions; magnitudes preliminary, signs and
ordering robust across positions. Next: (iv) which heads carry the +Δ, and
(v) patching to find the decisive positions.


## Head-level: the tug-of-war inside attention (L21)

Heads 18/19/30/13 write WITH the vector (+1.20/+1.08/+0.67/+0.50); heads
17/31/26/2 write AGAINST (−1.13/−0.90/−0.89/−0.74). The net +1.2 attention
amplification is the residue of a ±10 battle across 32 heads — resistance
is everywhere, attention roughly draws, the MLP loses only to the dose.

## Fine dose-response: threshold ≈ 2.0–2.5, but effects start below it

The imprint starts dominating the residual direction between scale 2.0 and
2.5 (L21 takes the peak at 2.5). Disposition suppression is measurable
already at scale 1 (70 positions) — the threshold is one of imprint
*visibility*, not of effect. No saturation through 6.

## Activation patching: no single position is decisive

Patching one position's steered L20 residual into the clean forced pass
flips **zero** downstream argmax predictions — at every one of 40 positions.

This is the counterpart to everything above: steering is **distributed and
cumulative**, not localized. The vector normally applies at *every*
position (decode-only); injecting it at just *one* is swamped by the 39
unpatched neighbors and the L21 MLP self-repair. There is no "decisive
token" — the behavior change is the sum of many small pushes that
individually clear no threshold, which is also why the dose has a floor
(one position ≈ sub-threshold).

Caveat: argmax-flip is a coarse metric — a logit could move substantially
without changing the top token. A logit-margin version of this experiment
would show the sub-threshold shifts the argmax test hides. That is the
natural next refinement (cheap: the forced pass already captures logprobs
for the KL work).
