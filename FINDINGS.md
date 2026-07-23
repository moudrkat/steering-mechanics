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
is everywhere, attention roughly draws, the MLP loses only to the dose. Animated: `fig/tug_of_war.gif` (reveal of the measured per-head deltas).

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

## Is the opposition active or mechanical? — it saturates, it doesn't escalate

Head deltas measured at injection scales 1 / 3 / 6 (delta/scale is flat if
linear). Median nonlinearity across active heads: **31%** — real curvature,
but the shape is **saturation**, not escalation:

- opposing heads weaken *per unit* as dose rises (h17: −0.39 → −0.34 → −0.22
  delta/scale; h2 similar) — they do NOT push back harder when pushed harder
- amplifying heads also saturate (h18: 0.23 → 0.34 → 0.36)

If the opposition were active self-repair, it should escalate with dose to
keep pace. It doesn't — it saturates (softmax hitting its ceiling is the
likely cause). So the data does **not** support a "the model fights back"
reading. The opposition is real but **passive**: the vector wins not by
overpowering an active defense, but because the amplifying heads start
slightly ahead and the opposers cannot ramp up.

Caveat: this dose test doesn't fully separate value-side propagation from
attention-pattern change — a frozen-attention patch would. But it cleanly
rules out *escalating* resistance, which is what "self-repair" would need.

## Auto-calibration works — and the eval choice drives the answer

The two-dataset objective (efficacy_miss + 0.1·KL_damage, heretic-style)
over 40 Optuna trials produces a clean landscape: green (low objective)
concentrates at layers 22–28, scale 3–5; the vector is *fully effective*
(miss 0) mostly at L24–28. Best: L28 @ 4.23 (miss 0, KL 2.37).

Notably the optimizer prefers DEEPER layers than the hand-tuned L20 — but
on GENERIC short prompts, where L20@3 only reaches miss 0.5. This is not
"L20 was wrong"; it is the headline caveat made concrete: **the efficacy
metric drives the optimum.** Calibrated against generic disposition
suppression, the vector localizes deeper; calibrated against the real
production behavior (a 16-prompt task-offering eval on the full scaffold),
it would likely favor L20 again. The machine is correct; plug in the eval
that matches your deployment.

Figure: `fig/calibration_landscape.png` (rendered offline by
`steermech-plot` from `examples/autocalibrate.json`).

## The efficacy proxy is the weak link — confirmed on the real task

The auto-calibration measures efficacy as J-lens *disposition* suppression
(concepts that vanish from the model's forming-words). On generic short
prompts this roughly tracked the vector's effect. On the actual production
scaffold (13k-token DISCUSS prompts) it collapses: at scale 3, layers
16/20/24 all score miss 1.00 — the proxy sees *nothing* suppressed, even
though the vector demonstrably suppresses task-offering in production.

Why the proxy fails here:
- it does not reproduce the deployment conditions (no task-nudge, no forced
  SuggestMessages tool) — so the unsteered baseline may not form the avoid
  concepts at all, leaving nothing to suppress;
- J-lens dispositions are not the behavioral event. The real eval generates
  the tool call and checks its *content* for a task offer; the proxy reads
  forming-words in the layers. Different signals.

Conclusion: a cheap disposition proxy is fine for auto-*discovery* and for
generic vectors, but heretic-grade calibration of a *specific production
behavior* needs the real behavioral eval as the efficacy function
(generate + classify the violation), not the proxy. This is exactly why
heretic measures refusals with a classifier rather than a shortcut.

**Done since:** an intent file carrying `violation_regex` (+ optional
`tools`/`tool_choice`/`nudge`) now switches efficacy to exactly that — full
generation under deployment conditions + violation classifier
(`generate_efficacy`, shipped in `hidden_directions.calibrate`).

## TODO (perf): clean-side cache lands ~3x
brainscope now caches the prompt-keyed clean side of a forced diff (baseline
gen + clean pass, both steering-independent), so calibrating one vector over
many (layer,scale) trials on the same prompts pays the expensive scaffold
pass once. On a 13k-token scaffold one forced diff is ~100s; the cache turns
an N-trial-per-prompt sweep from N×100s into ~100s + N×(steered only).

## Disposition proxies go blind on reasoning models (2026-07-23)

First cross-model night: the no-tasks pref vector re-extracted for Qwen3-8B
(8-bit, same recipe — extraction under the serving numerics, ~30 s), then
auto-calibrated with the proxy objective. Result: miss = 1.00 across all 25
trials, every layer, every scale — the optimizer degenerated to minimizing
KL. Two stacked causes, both instructive:

1. **No lens, no signal.** `suppressed_positional` is computed only when a
   J-lens is loaded; our lenses are fitted per-model and only the 4B has
   one. Without it the proxy reads permanently empty — and *nothing in the
   loop warns you*. A calibration objective that can silently flatline is
   itself an eval-validity hazard (RQ2 exhibit A).
2. **Thinking models narrate the request.** Qwen3-8B opens every forced
   continuation with a `<think>` preamble that restates what the user asked
   ("the user wants a reminder…"). Concept-suppression scoring over those
   positions measures the *narration*, not the forming answer. The `/no_think`
   soft switch produces a direct answer and is the minimum fix for any
   disposition-style measurement on hybrid reasoning models.

Consequences: behavioral efficacy is the only trustworthy objective on a
model without a fitted lens; a Qwen3-8B J-lens fit is now a prerequisite for
running RQ2's proxy arm on the workhorse; and "does the proxy even emit
signal here" becomes a mandatory preflight check before any calibration run.

## First cross-model transfer result: the window transfers, the argmax lies (2026-07-23)

No-tasks pref vector, re-extracted per model by the same recipe, same
behavioral eval both sides (N=10/point, pilot-grade), scale sweeps at each
model's calibrated best layer plus the other model's optimum as cross-points:

- **4B→8B transfer: perfect.** The 4B's optimum (L20@3) scores miss 0.00 on
  the 8B at KL 0.42 — *better damage than the 8B's own TPE winner* (L15@8,
  KL 0.66). Zero efficacy loss; the pre-registered ≤20% bar is passed with
  room to spare.
- **8B→4B transfer: fails.** The 8B's TPE winner (L15@8) leaves 30%
  violations on the 4B (native: 0%) — over the ≤20% threshold. L15 is a
  model-specific pocket; L20 is the shared one.
- **Both dose–response curves are clean and monotonic** (8B/L15: 0.70 →
  0.00 across scales 1–8; 4B/L20: 0.30 → 0.00 across 1–4, then pure
  overdose), and the 8B tolerates the shared coordinates with ~2x less KL
  than the 4B (L20@3: 0.42 vs 0.94).

The twist: a 15-trial TPE run on the 8B converged to a point (L15@8) that
is *worse* than the transferred 4B optimum on the 8B's own eval (score
0.063 vs 0.042) — the optimizer found a local pocket, not the shared
window. Implication for H3 and for practice: **transfer the window (the
curve), never the argmax** — single-run optima are model-idiosyncratic even
when a shared optimum exists, and asymmetric cross-eval (A→B ≠ B→A) is the
cheap test that exposes it. Caveats: one model pair, same family, equal
depth (36 layers both), N=10.

**N=20 confirmation (same day):** headline holds and sharpens. Baselines:
4B 12/20, 8B 19/20 (the larger model complies with the eliciting nudge
*more*). Shared point L20@3: **0/20 on both models** (KL 0.94 on 4B, 0.42
on 8B). The 8B argmax L15@8 transferred to the 4B: **8/20 = 40%
violations** — transfer failure confirmed well past the ≤20% bar. Still
pending before headline status: the qualitative read (are the zeros
coherent Czech or degraded outputs?) and the Tier-2 real-scaffold eval.

## The qualitative read: the argmax didn't just lie, it broke the model (2026-07-23)

Read all 72 generations from the six decisive points (both models ×
baseline / shared L20@3 / 8B-argmax L15@8), regenerated through the
spec-driven eval with the full checker:

- **8B @ L15@8 — the TPE winner — is catastrophically degraded**: all 12
  outputs are English think-mode rambling (median 609 chars vs ~240
  normal). The steering broke the `/no_think` switch; the model deliberates
  about the user in English and never answers in Czech. Zero regex
  violations — because English deliberation phrases no task offer. A
  coherence-blind objective scored a broken model as perfect. Full checker
  verdict: 0/12 violations, **12/12 incoherent**.
- **Shared window L20@3 is genuinely good on both models.** 4B: natural
  Czech, suggestion buttons, discussion-only — best quality overall. 8B:
  solid with an occasional loop; one output ("podle zdravotního pojištění"
  ×3) passed the 3-gram guard on a short text — logged as a
  checker-vs-human disagreement, thresholds not retuned.
- **4B @ L15@8: coherent Czech that still offers task lists** ("Nabídka
  úkolů (pro tebe): 1. …") — the transfer failure, readable.

Consequences applied: behavioral **miss now counts violation OR
incoherence** (a vector that stops the behavior by degrading the model is
a miss, not a win) — engine change, tests updated. Reading the outputs is
now a standing step: the checker exists to scale judgment, not replace it.
