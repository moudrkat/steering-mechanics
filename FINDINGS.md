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

## RQ1 row 0: dose x regime at short context — per-token scale dominates here (2026-07-23)

First cells of the frozen grid (Qwen3-8B, L20, short-context tier-1 eval,
N=12/cell, full-checker miss, matched-regime KL, baseline-compared):

- **Clean dose window s2–4 in BOTH regimes** (miss 0.00–0.08), then a sharp
  edge: decode-only degrades at s6 (miss 0.42, KL 3.1) and collapses at s8
  (miss 1.00 — twelve of twelve incoherent, output length crashing, KL 8.4).
  Full-steer collapses harder at the same doses (s6: miss 0.92; s8 KL 12.4).
- **The H1 mass law does NOT govern this short-context row.** Full-steer
  adds ~3-4x steered positions (prefill + generation vs generation only);
  a pure total-mass law would shift the collapse threshold down ~3-4x in
  scale. Observed: the working window is the same (s2-4), and the regime
  gap appears only at the damage edge (s6-8, severity not onset). At short
  context, per-token scale dominates; injected mass contributes second-order
  severity. The pre-registered competitor (sharp per-token thresholds) is
  currently ahead — the real discriminating test is the context-length axis
  (0.5k -> 16k), where prefill mass grows ~30x. That is the next grid row.
- **Zero anti-steered samples in any cell** (per-sample baseline
  comparison): on this behavior, steering never pushed a clean prompt into
  violation. The Tan-style anti-steerability threat did not materialize
  here — worth reporting per-behavior rather than assuming either way.

## Third model, different depth: the window sits at fractional depth (2026-07-23)

Qwen2.5-7B (28 layers — previous generation, different depth than the
36-layer Qwen3 pair), same recipe, same eval, N=12/point. Baseline 9/12
violations. At the reference scale 3, sweeping the depth question:

- **L14: miss 0.08 · L16: 0.08 · L18: 0.00 — the working window.**
- **L20 (raw-index transfer of the Qwen3 optimum): miss 0.25 with
  incoherence creeping in — outside the window.**

In fractional depth the window is 0.50–0.64 of the stack; the Qwen3 pair's
window (L15–20/36) is 0.42–0.56. Across three models spanning two
generations and two depths, the usable region consistently straddles
~0.5–0.6 fractional depth, and raw layer indices mislead the moment depths
differ. The L16 dose curve reproduces the same cliff shape as both Qwen3
models (clean at 2–6 here, collapse at 8 with 11/12 incoherent). One
anti-steered sample observed (under-dose, L16@1) — the first all day.

Pilot-grade (N=12, one behavior); this is the question the paper campaign
takes to N>=30 across ~8 models.


## Not a scoring bug: two collapse modes, one honest metric (2026-07-23)

Checked whether "0 violations at high dose" was a regex bug. It is not —
hand-verified: the violation regex fires on `úkol` and `I will create a
task`, stays silent on `renew their passport`, and the κ=1.0 classifier
validation stands. What actually happens at high dose is genuine model
collapse, and it comes in (at least) two flavours:

- **L15@8 (the argmax trap):** English think-mode rambling — the model
  deliberates about the user in English and never answers.
- **L20@8:** total token collapse — repeated fragments (`(ne. (ne. (ne.`)
  and stray Chinese characters.

Neither contains a task-offer word, so a violation-only metric scores both
0% (perfect). The coherence guard scores both 12/12 incoherent. `miss =
violation OR incoherence` = 1.0 in both cases — the correct answer. The
finding is not "the metric got lucky"; it is that you can see exactly which
half of the checker is load-bearing: the regex at safe doses, the coherence
guard at collapse doses. A regex-only eval (heretic, repeng+classifier)
reports these catastrophes as flawless successes.

## Emergent-misalignment scare that wasn't — refusal-vocabulary drift (2026-07-23)

Ran the #11 emergent-misalignment battery on Llama-3.1-8B: steer the benign
no-tasks vector at a coherent working point (L16/0.50, scale 3), measure
refusal on 10 harmful + 10 benign probes vs baseline. The battery flagged a
**+0.3 harmful-compliance shift** — apparent emergent misalignment — and even
with a coherence guard added, all steered outputs were coherent ("signal is
real"). Reading the generations killed it:

All three "compliance" cases were REFUSALS the regex missed —
"I can't **discuss** ransomware", "I can't **discuss** how to make a lethal
nerve agent", "I can't **talk about** things harmful to health". The model
refused 10/10 harmful prompts steered and unsteered. No emergent misalignment.

Mechanism (and the finding): the no-tasks vector nudges toward *discussion*
framing, and that leaked into how the model phrases its *refusals* — "I can't
discuss" instead of "I can't help/assist". The narrow refusal regex knew only
the latter. **A steering vector can shift the vocabulary of an unrelated
behavior without changing the behavior; a safety metric keyed on phrasing is
therefore fragile under steering.** The coherence guard did not catch this
(outputs were fluent); only reading did — EVAL_PRINCIPLES #4 in action.
Refusal regex broadened; re-measure for a clean number.

## Emergent-misalignment: TWO false positives, two metric failures (2026-07-23)

Ran the EM battery on Llama-3.1-8B and Qwen3-8B (steer benign no-tasks vector,
measure refusal on 10 harmful probes vs baseline). BOTH flagged +0.3
harmful-compliance — apparent emergent misalignment. BOTH were metric
artifacts, via DIFFERENT mechanisms, and only reading the generations caught
either:

- **Llama-3.1-8B:** refusal *vocabulary drift*. The no-tasks vector nudges
  toward "discuss" framing, so refusals became "I can't **discuss**" instead
  of "I can't **help**" — the narrow refusal regex missed them. Model refused
  10/10; no misalignment.
- **Qwen3-8B:** thinking-model *truncation*. Every harmful response was a
  `<think>` block deliberating toward refusal ("phishing is unethical and
  illegal…") that never reached the answer within 120 tokens; the detector
  saw only the reasoning and scored "compliance."

Conclusion: a bare EM number was wrong on 2/2 models, each via a distinct
failure (vocabulary drift; reasoning-truncation). Steering safety-evals
cannot be trusted as numbers — they need a meaning-based judge, adequate
token budget, and think-mode handling, and a human must read the flagged
cases. This is the eval-validity thesis at its sharpest.

## The dose axis is under-specified: raw scale is not a cross-model coordinate (2026-07-24)

The four-model campaign (Llama-3.1-8B, Qwen3-8B complete; Qwen3-4B/2.5-7B
partial) surfaced what looked like an H3 result — "Qwen3-8B needs scale ~6
where Llama needs ~3, so the optimal scale does not transfer." On inspection
that claim is **confounded and must not be stated in raw-scale units.** The
reasoning, derived from first principles:

**Confound 1 — vector norm.** Scale is a multiplier on a per-model vector.
Llama's and Qwen3-8B's vectors are extracted separately and have different
per-layer norms (‖V‖ ranged ~0.1–247 across layers in extraction). So
"scale 3 on vector A" injects magnitude `3·‖V_A‖` and "scale 6 on vector B"
injects `6·‖V_B‖`; the multipliers 3 vs 6 are not comparable. "Qwen needs
scale 6" may just mean "Qwen's vector has a smaller norm at that layer."

**Confound 2 — residual-stream norm.** Even the *absolute* injected magnitude
(`scale·‖V‖`) is not comparable across models, because models have different
activation scales. A perturbation of magnitude 10 is large where residual
activations sit at norm ~20 and negligible where they sit at ~200. Absolute
magnitude fixes confound 1 but not confound 2.

**The only cross-model dose coordinate is dimensionless — the *relative*
dose:**

    relative_dose(L) = ‖scale · V[L]‖ / ‖h[L]‖

the fraction of the residual stream being perturbed. Unit-free, so it means
the same thing on every model. And even this is not *guaranteed* to
transfer (models can differ in sensitivity to the same relative
perturbation) — which makes the well-posed question:

  *does efficacy / coherence-collapse occur at the same **relative dose**
  across models?*

Raw scale could never answer this. The naive "scale doesn't transfer" is
therefore not a finding; it is an artifact of reporting dose in the wrong
units.

**Consequences.**
- Do NOT normalize the vectors retroactively (breaks calibration, discards
  the norm, which may carry concept-strength information). The fix is a
  *reporting* change, not a vector change.
- Report `relative_dose` (needs ‖V[L]‖ from the vector and a typical ‖h[L]‖
  from the served model) alongside raw scale for every cross-model cell.
- The layer/window result (where steering works, as fractional depth) is
  unaffected — it is about *location*, not *dose*.
- Broader point, and a genuine contribution: the steering literature reports
  strength inconsistently (raw coefficient, sometimes norm-scaled, rarely
  residual-relative), which makes cross-model / cross-method dose claims
  **unfalsifiable**. The only defensible dose coordinate is dimensionless.
  This reframes H1 (collapse vs "injected mass") too: mass must be measured
  as relative perturbation, not raw scale × positions.

**To recompute (when the GPU box is back):** per cell, get ‖V[L]‖ from the .pt and
a mean ‖h[L]‖ from a few forward passes on the served model, then redo the
dose comparison in `relative_dose`. That is the version for the paper.

## F. Gemma (8bit, donor-key projection, thinking OFF): suppression preserved, fluency directionally better, the real test needs 12k

Valid probe after fixing my own thinking-mode bug (first run had
enable_thinking accidentally ON — my script enabled thinking on a model
whose template defaults it OFF; eval-validity trap self-demonstrated,
logged). Build note: Gemma vector barely couples to focus→tail rerouting
directions (Rayleigh max 4.2 vs Qwen's 40; projection removed only 1% of
norm, harm rank 35) — consistent with the KV-share structure making L25 injection
query-side-only WITHIN the attention pathway (MLP/skip still carry it).

| arm | viol | uniq_mean |
|---|---|---|
| baseline | 3/6 | 0.950 |
| v@2.5 / v@3 | 1/6 / 1/6 | 0.911 / 0.884 |
| v̄@2.5 / v̄@3 | 2/6 / 1/6 | 0.932 / 0.898 |

- Suppression preserved under projection (1/6 at s3 both). Fluency
  directionally better for v̄ (+0.014–0.021 uniq) — below N=6 resolution.
- **Baseline discovery:** Gemma-4-it answers short Czech prompts in
  CROATIAN/SERBIAN ~half the time (all arms affected) — the language
  instability partially predates steering. New support for the
  weakest-language-first damage story, and a confound for short-context
  Czech probes on this model.
- The §7 behavior/fluency step lives at 12k deployment length — this
  short-context probe cannot see it. The decisive Gemma test (does
  projection move the step?) needs the confirm-ship harness at 12k.
- 12k h-norms still blocked (sliding-attention OOM at 8k even in 8bit) —
  needs chunked/flash path.

Data: results/efficacy_gemma.json; vector v_pref_no_task_gemma_skopres.pt
(99% norm kept).

## G. v3-8B (the proven vector): projection COSTS suppression here; wall 5/5; the L15 "collapse" was a mode failure

Stage B, @L20 (v3-8B = the vector with 0/20 in the N=20 eval):
- baseline 4/6; **v3@3: 0/6** (probe validates the proven vector ✓).
- **v̄@3: 3/6 — v1-style projection (96.8% norm) lost half the
  suppression** on this vector, unlike 4B/v3 where it kept it. The
  effect-vs-rerouting entanglement is VECTOR-SPECIFIC, not universal.
- s8: both arms collapse (rep4 0.63/0.74, uniq 0.28/0.30) — **the
  magnitude wall is now 5/5 configurations.**

Stage C, @L15@8 (mode-break test) — surprise:
- The historic L15@8 catastrophe (English think-rambling) did NOT
  reproduce: with template-level thinking hard-OFF, v3@L15@8 produces
  coherent-ish Czech discussion-deflection (uniq 0.90, rep4 0.0) — quite
  on-behavior, mildly derailed.
- Reading: **the historic L15@8 collapse was a MODE failure** (steering
  broke the /no_think SOFT switch under thinking-enabled serving), not a
  fluency collapse. Hard-disabling thinking at the template removes the
  failure channel entirely. Two mechanistically distinct damage flavors
  now demonstrated: the L15 switch-flip (reversible by template) vs the
  L20@s8 token collapse (persists regardless).
- The original hypothesis (does projection protect the soft switch?) is
  therefore UNTESTED — needs the soft-switch condition (template thinking
  ON + "/no_think" in prompt). Queued.
- Practical: if serving hard-disables thinking, the L15 pocket may be
  less dangerous than the argmax-trap story suggested — worth a harness
  check before rehabilitating it.

Data: results/efficacy_8b_v3.json, results/modebreak_8b_L15.json.

## H. Smoke run of the new instruments (real model, real vector) — first rerouting sighting

All three new package features validated live on the GPU box:
hotwire dose check (s8 -> rel_dose 1.926 -> rejected — the wall as a
runtime guardrail, math verified), measure-h-norms (L20=58.8 vs manual
54.9, +-7% from different synthetic text), rerouting-audit (Rayleigh
40.4/2.72 vs manual 40.77/2.77 — the tool reproduces the research).
One real bug found & fixed: hotwire's h-norm loader only ate the flat
JSON format, not measure-h-norms' wrapped output (cross-package interop —
unit tests can't see these).

And the first live sighting from brainscope's rerouting monitor
(v3@L20@3, decode_only, forced replay): **L21 head 15 reroutes at JSD
0.513** — a huge single-head divergence exactly one layer above the
injection, while the injection layer's own rows read 0.0 (the built-in
control behaves). Layer 21 is where component attribution saw attention
amplify the vector (+1.23); now we have a head-level suspect for WHERE.
Candidate for the head-level program step (iv). k=1, one prompt — an
observation, not a result.

## E. Projection-depth sweep (4B): no SKOP-style free lunch at probe resolution

Four configs between v0 (64% norm) and v1 (95%), probe at s3/s4/s5 with
coherence heuristics (rep4 = repeated-4gram ratio ↓ better, uniq =
unique-token ratio ↑ better). Reference: v@3 = 3/6 viol, uniq 0.932;
v@4 = 0/6, uniq 0.866.

| cfg (risk/γ/pcap) | norm kept | s3 viol/uniq | s4 viol/uniq | s5 viol/uniq |
|---|---|---|---|---|
| A 0.15/0.8/16 | 88.1% | 5/6 / 0.917 | 2/6 / 0.821 | 0/6 / 0.753 |
| B 0.20/0.7/8  | 91.9% | 4/6 / 0.953 | 1/6 / 0.747 | 0/6 / 0.754 |
| C 0.10/0.8/16 | 91.8% | 2/6 / 0.923 | 1/6 / 0.855 | 0/6 / 0.712 |
| D 0.15/0.7/8  | 93.5% | 2/6 / 0.868 | 2/6 / 0.789 | 1/6 / 0.787 |
| (v1 earlier) 0.10/0.7/8 | 95.1% | 1/6 | — | — |

Read (all N=6, CIs wide — directional):
- **No config separates from the original on the joint
  efficacy×coherence readout.** Best candidate C@s4 (1/6, uniq 0.855,
  rep4 0) ≈ v@4 (0/6, uniq 0.866, rep4 0). The projection dial trades
  norm for nothing measurably good at this resolution.
- Combined with v0 (deep cut kills effect) this now reads as: **in
  residual space we reproduce SKOP's tension (arXiv 2605.06342;
  rerouting components carry efficacy) but NOT their win (no setting
  found that keeps efficacy and cuts the tax).** Consistent with their own stated reason for leaving
  residual as future work — the perturbation also flows through V and
  MLP paths that key-orthogonal projection cannot protect.
- Statistically honest framing: a modest win could hide below N=6 probe
  resolution; the real 16-prompt harness with the checker is the judge.
  As of tonight the null ("no free lunch in residual space") is ahead.

Artifacts: `experiments/skop_residual/results/sweep_{ref,A,B,C,D}.json`, vectors `v_sweep_{A..D}.pt`
on the GPU box. Scripts archived in `experiments/skop_residual/`.

# Findings (2026-07-26, Gemma-4-E4B at deployment length — layer rescue, ship attempt)

Score files: `results/confirm_ship_gemma.json`, `confirm_ship_gemma_addendum.json`,
`confirm_ship_gemma_rule_s4.json`, `gemma_hunt_grid.json`, `gemma_judge_pass.json`.
Generations in airlock (`generations/gemma_*`).

## 1. The prompted rule washes out at 12k (the "deploy unsteered" verdict was a short-context artifact)

Yesterday's read (`gemma_steering_verdict.json`): base Gemma obeys the no-task
rule ~10/10 → deploy unsteered. That was measured on a SHORT system prompt.
On the real 30×12k scaffolds (confirm-ship design, k=2, temp=1), judge-corrected:
rule-present baseline commits genuine violations at **0.30 production / 0.85
adversarial**. Rule-stripped: 0.425 / 0.65. At deployment length the rule
barely moves behavior. Nothing prompted ships alone.

## 2. Extraction layer ≠ injection layer; the 0.55–0.60 fractional pocket replicates (4th model)

L17 (extraction screen winner, agreement 0.80) steers NOTHING at 12k — echo
degeneracy at every dose. The (layer, scale) grid (`gemma_hunt_grid.json`,
layers 21–29 × scales 3–6 on adversarial scaffolds) puts the only honest
optimum at **L25 (0.60 fractional depth), scale 3–4** — the same fractional
pocket as Qwen L20/36. L21's low miss counts are coherence-bought (the F1 lie);
s5–s6 flip to near-total regex-miss while coherent (see 4).

## 3. Steered efficacy at deployment length, judge-corrected

| arm | prod hard_miss (genuine offers) | adv hard_miss |
|---|---|---|
| stripped baseline | 0.425 (17) | 0.650 |
| rule baseline | 0.300 (11) | 0.850 |
| vector L25@s3 | **0.000 (0)** | 0.350 |
| vector L25@s4 | 0.075 (1) | 0.100 |
| rule + L25@s3 | 0.025 (1) | 0.150 |
| rule + L25@s4 | 0.050 (2) | 0.100 |

Steering beats prompting at deployment length; rule+vector composes.

## 4. Overdose vocabulary imprint corrupts regex evals

At s5–s6 the model cannot stop emitting the vector's own vocabulary
(úkol/checklist/…), so the violation regex fires on refusals: raw regex
production "offers" for rule+s4 were 10; judge-genuine: 2. Regex-only
cross-model dose curves are inflated exactly where dose is high. (Same class
as argmax_lies: the metric, not the behavior.)

## 5. Ship gate: NOT certified — verdict is judge-resolution-limited

Frozen gate (prod ≤0.10, 0 prod offers, adv ≤0.30; KL waived — brainscope down,
vLLM stack can't compute it): every steered cell misses by 1–2 cells of 60,
OR passes, depending on which defensible judge reading of ~5 borderline cells
is used. Local Gemma judge (kappa 1.0 on golden, but golden is easy):
rule+L25s3 fails by ONE production cell (a moving-house scenario whose
suggested reply proposes splitting the move into tasks).
Independent read (Claude, this session): that cell is CLEAN → rule+L25s3
passes (prod 0.000, adv 0.20). Cross-judge disagreement is concentrated
entirely in steered half-refusals with degraded Czech.
DECISION PENDING: human read of the 5 decisive cells + neutral Claude-API
judge rerun + KL when brainscope returns. No cert is claimed today.

## 6. Open problem: fluency is unguarded

Steered arms at s3–s4 pass the coherence guard while producing agrammatical
Czech (case errors, invented words, occasional Cyrillic bleed). The gate
certifies no-task, not language quality — Gemma's main selling point. Any
cert without a fluency criterion overstates deployability. Adding one =
dated deviation note in RESEARCH_PLAN.md (not done; needs a decision).

## 7. Final: the behavior window and the fluency window DO NOT OVERLAP (no steered ship)

Last candidate rule+L25@s2 (`results/confirm_ship_gemma_rule_s2.json`):
Czech reads clean again at s2, but behavior reverts to baseline — adversarial
regex 0.80 (~rule-alone 0.90), flagged cells verified genuine compliance by
read. Full dose ladder on the deployment eval, both axes:

| dose (rule+L25) | behavior | Czech fluency (24-sample read) |
|---|---|---|
| s2 | FAIL (~baseline) | clean |
| s3 | PASS* (0-1 prod offers) | ~5/24 clean, ~10/24 non-words/garbled |
| s4 | PASS* | worse |

*modulo judge-borderline cells, §5.

The suppression threshold sits ABOVE the fluency-collapse threshold for this
vector on this model at 12k. No (layer, scale, ±rule) cell ships. VERDICT:
deploy decision goes to non-steered or hybrid configs; the candidate worth
testing next is steer-on-retry (serve unsteered; validator catches violation;
regenerate WITH s3 steering), which pays the fluency tax only where the
alternative was a violation. That is a BEYOND_STATIC_STEERING direction, not
a static-vector cert.

Keynote-grade summary of the day: prompt rules wash out at deployment length;
steering beats prompting there; but the dose that buys the behavior costs the
language — and the two windows' (non-)overlap is a measurable, model-specific
quantity that static steering cannot escape.

## 7b. s2.5 closes the crossover question: it is a STEP, not two curves

rule+L25@s2.5 (`results/confirm_ship_gemma_rule_s2p5.json`): still fully
compliant — 37 flagged cells read, overwhelmingly genuine creation offers —
with clean Czech. Final ladder: s2 fail/clean, s2.5 fail/clean, s3
pass/broken. The behavior flip and the fluency collapse arrive at the SAME
dose threshold (between 2.5 and 3), consistent with the 07-23 teacher-forced
result that the vector only dominates the residual direction above scale
~2–3. Interpretation: suppression and language damage are one event — vector
domination of the residual stream — not independent thresholds. Any
exploitable gap is <0.5 dose units and below temp-1 noise at n=60. This
strengthens H-side: for a static vector the side-effect is not tunable away;
it is the mechanism.

## 8. Cleanest extraction does NOT move the step (v3 vs v5 head-to-head)

Re-extracted the Gemma vector with the strongest recipe available —
v5 production-format minimal pairs, serialized in Gemma's NATIVE tool-call
format (verified against the gemma4 chat-template macro), byte-identical
Czech to the Opus-reviewed pairs, 8bit serving numerics
(`recipes/no_task_v5_gemma_recipe.json` + REVIEW). Norm check per the
relative-dose rule: v5 ‖V[25]‖=35.2 vs v3 32.2 — comparable, v5 slightly
stronger per unit scale, so raw-scale cells are fairly comparable.

Head-to-head (airlock `generations/gemma_v3_vs_v5.jsonl`; L{23,25} ×
s{2,2.5,3}, rule-stripped 12k, k=1): v5 is NOT better. On direct asks v5
stays compliant through s3 at L23 (parameter-elicitation at every dose);
suppression appears only spottily (L25s2 checklist refusal). Fluency damage
arrives at the SAME or lower dose, with a new failure flavor: foreign-script
bleed (Cyrillic at s2, CJK at s3) rather than v3's Czech agrammaticality.
Extraction screen again picked L17 (agreement 0.78) — second independent
anti-prediction of the injection layer.

Conclusion: the suppression/fluency step is EXTRACTION-INDEPENDENT on this
model. Combined with §7b (step, not two curves), the coherence tax at
deployment length is a property of vector-domination itself, not of recipe
quality, layer choice, or dose tuning. Static steering has no shippable cell
on Gemma-4-E4B for no-task; the deployable direction is conditional delivery
(steer-on-retry / gated), where the tax is paid only on replies that were
about to violate.

# Findings (2026-07-27 evening — SKOP-residual v0 + deployment-length norms; run by Claude, pilot-grade k=1)

## A. Deployment-length h-norms overturn the short-prompt relative-dose table

12k-token forward pass (synthetic varied CZ/EN text, GPU, base model no LM
head): Qwen3-4B mean ‖h[20]‖ = **54.9** (last-1024 positions: 52.7) — vs
**375** measured on 5 short prompts this afternoon. Cause: massive-
activation sink tokens dominate the mean on short sequences and dilute to
nothing at 12k. Short-prompt h-norms are NOT deployment-representative.

Corrected Qwen3-4B L20 relative doses (‖V[20]‖=13.22, h≈54.9):
- working point s3 → **0.72** of residual norm (not 0.106)
- degradation onset s4.7–6 → 1.13–1.44
- collapse s8 → **1.93**

Implication: Gemma's flip at "≈1.0" was computed with short-prompt norms
(96.9) and needs the same 12k correction (blocked: Gemma bf16 OOMs on 16GB;
needs 8-bit). If Gemma's 12k norms drop similarly, both models' collapse
may land in the same ~1.5–2.0 relative-dose band — which would resurrect a
cross-model dose law at deployment length. PENDING measurement, do not
cite yet. Data: `paper/relative_dose_recompute.json` (old),
`/tmp/chain_qwen2.log` on the GPU box (new).

## B. SKOP-residual v0: rerouting is the style tax, domination is the cliff

Built a residual-space analogue of SKOP (arXiv 2605.06342; their method is
query-space, residual left as future work): calibrated focus/tail sets +
key-difference second moments on 8 utility prompts (layers 21–28, 256
heads), projected v_pref_no_task_checklist_v3[L20] orthogonal to top-γ=0.9
eigendirections of the top-20% risk heads (Rayleigh 40.8→0.87 after).
v0 is over-aggressive: harm basis rank 1536/2560, norm kept 64%
(‖v̄‖=8.46 vs 13.22). Vector: GPU box `~/hotwire-vectors/
v_pref_no_task_qwen_skopres.pt`. Script: scratchpad skop_residual_build.py
(v0 approximations: pre-RoPE keys, LN Jacobian ignored, 8-prompt calib).

A/B (2 CZ prompts, greedy, 60 tok, matched-magnitude scales — v̄@4.7 ≈
v@3 and v̄@12.5 ≈ v@8 in injected magnitude):

| injected magnitude | v (original) | v̄ (projected) |
|---|---|---|
| ~40 (v@3 / v̄@4.7) | borderline: Slovak bleed, repetition on one prompt | **clean coherent Czech on both prompts** |
| ~62–68 (v@4.7 / v̄@8) | 🌱-loops, degraded | degraded (CZ/PL hybrid) — similar |
| ~106 (v@8 / v̄@12.5) | English "appropriate" loop / `{{{` collapse | collapsed too (numbers/path loop) |

Reading (k=1, needs replication + efficacy eval):
- **At working magnitudes, removing rerouting components buys coherence**
  — v̄ at matched magnitude is cleaner than v. The known v3 "style tax"
  (bent Czech at L20@3) looks at least partly rerouting-caused → SKOP-type
  projection may be a production win. Candidate config: **v̄@≈4.7**,
  pending the behavioral no-task eval (does it still steer?!).
- **The high-dose cliff survives projection** — collapse onset sits at
  similar injected magnitude for v and v̄ → the cliff is governed by
  magnitude domination, not by the projected-out rerouting directions.
- Net: BOTH mechanisms, different regimes. Rerouting = quality tax inside
  the window; domination = the wall at the end of it. Cleanly matches §7b
  (step) while explaining why the step's *approach* hurts quality.

Next: (1) behavioral efficacy eval of v̄ (16-prompt harness); (2) v1
projection — fewer risk heads / lower γ / cap p per head, target ≥85% norm
kept; (3) Gemma variant via 8-bit; (4) N≥12 replication of this A/B.

## C. Efficacy probes correct B's interpretation: effect and tax are tangled in the rerouting directions

Direct-ask probe (6 CZ explicit task requests, greedy, regex + read;
crude proxy — regex counts word-mentions, not offers, per F-mode
warnings; no tool scaffold):

| arm | regex hits | read verdict |
|---|---|---|
| baseline | 6/6 | eagerly creates checklists/lists ✓ |
| v_orig @3 | 3/6 | suppresses/deflects; known style tax (Slovak bleed) |
| **v̄_v0 @4.7** (64% norm) | **5/6** | **mostly COMPLIES — v0 lost most of the steering effect** |
| **v̄_v1 @3** (95% norm, risk 10%, γ0.7, p≤8) | **1/6** | **suppresses ≥ original**; style tax similar to v |
| v̄_v1 @4 | 2/6 | suppresses, degrading |
| v̄_v1 @6 | 0/6 | but 6/6 incoherent ("Než… než… než…") — collapse at same injected magnitude as v ⇒ cliff unchanged |

Corrections to section B:
- B's "v̄_v0 is cleaner at matched magnitude" was partly an efficacy
  artifact: v0's projection removed much of the steering effect along
  with the tax — weaker effect trivially reads cleaner. Matched
  injected-norm does not mean matched effect.
- Honest current claim: **the steering effect and the quality tax are
  substantially tangled inside the rerouting directions.** A deep cut
  (v0: rank-1536 basis) removes both; a shallow cut (v1: rank-149)
  preserves the effect (and most of the tax). Whether an intermediate
  setting separates them — SKOP's own trade-off, now reproduced in
  residual space — is exactly the next sweep (risk/γ/p grid).
- **The magnitude-domination cliff stands**: v̄_v1 collapses at the same
  injected magnitude as v (~75-80), independent of projection. Rerouting
  projection does not move the wall; it may only shape quality inside it.
- v1 candidate is interesting on its own: 1/6 vs original's 3/6 on this
  probe with 95% norm — worth the real 16-prompt harness eval.

Artifacts: `experiments/skop_residual/results/efficacy_v{0,1}.json`, vectors
`v_pref_no_task_qwen_skopres{,_v1}.pt` on the GPU box. All k=1 pilot.

## D. Qwen3-8B (v4 vector, FP8 serving numerics): probe replicates the campaign's v4 negative; the wall stands on a third configuration

Setup: Qwen3-8B-FP8 via transformers (new capability — accelerate +
kernels installed into the venv, writable HF cache shim at ~/hf-cache2
because the original cache has root-owned entries from docker
extractions), W_q for the projection read lazily from the bf16
checkpoint's safetensors (FP8 packed weights can't be read naively),
/no_think enforced. Vector: private-vectors/qwen3-8b/
v_pref_no_task_checklist_v4.pt (L20 ‖V‖=20.24). Projection v1-settings:
norm kept 96.7%, harm rank 179, risk-head Rayleigh 29.1→2.8.

Probe (same 6 direct asks):
- baseline 4/6; **v4@3: 6/6 — no suppression on direct asks** (creates
  task lists eagerly). Matches the campaign's own verdict on v4
  ("honest negative", CONFIRM_PREREG arm B) — worth re-reading that file:
  memory said "v4 was better", the notes say v4 was the honest negative
  and the iteration continued to v5. The probe independently reproduces
  the negative.
- v̄4@3: 6/6 — projection preserves behavior (nothing to lose here).
- **v4@8 and v̄4@8 both collapse** (numeric/repetition loops) — the
  magnitude wall is unmoved by projection on a THIRD configuration
  (4B/v3, 4B/v̄_v1, 8B/v4). Domination cliff: 3/3.

Next candidates for a positive-efficacy 8B projection test: the v3-8B
re-extraction (the one that scored 0/20 at L20@3 in the N=20 eval) or
v5/v5_nothink.

**Wilson 95% CIs on the night's probe counts (N=6 each):** the ONLY
separated pair is 4B baseline 6/6 [0.61,1.00] vs v̄_v1@3 1/6 [0.03,0.56]
— i.e. "v1 projection preserves suppression" is statistically real even
at N=6. Everything else (v0 lost the effect; v vs v̄ nuances; v5nt
differences) is directional only — CIs overlap heavily. Her original
N=20 transfer numbers (0/20 [0,0.16] vs 8/20 [0.22,0.61]) remain cleanly
separated. Consequence: next comparisons need N≥20 + the real checker.

**v5_nothink run (same night):** partial efficacy at s3 (3/6 regex;
reads as deflection — reframes the task request, "zítra v 8 ti napiš, že
ti to nechávám v mysli" — plus mild degradation). Projection (96.8% norm,
Rayleigh 30→~3) preserves the behavior; at k=1 the v vs v̄ difference at
s3 is below the probe's resolution — further comparison needs the real
N≥16 harness with the checker, not more k=1 probes. At s8 both v and v̄
collapse ("ne, ne, ne…" loops) — **the magnitude wall is now 4/4
configurations** (4B/v3, 4B/v̄_v1, 8B/v4, 8B/v5nt), each with a
different collapse flavor, all at the same dose regime. Artifacts:
`experiments/skop_residual/results/efficacy_8b_v5nt.json`, vector `v_pref_no_task_8b_v5nt_skopres.pt`. Vector inventory note: private-vectors/ holds per-model
dirs (qwen3-8b v3/v4/v5/v5_nothink, llama31-8b, qwen2.5-7b, gemma,
thinking) + document_overrequest and websearch_overtrigger recipes —
MAP.md does not know about this treasury; update it.

## 9. The steering window is LANGUAGE-DEPENDENT (English probe, preliminary)

Same model, same v3 vector (which §8 revealed was extracted in ENGLISH — all
of today's Czech results were the cross-lingual condition), same L25.
Short-context dose arc, English prompts (`generations/gemma_english_probe.jsonl`):

| dose | Czech (07-26 probe) | English |
|---|---|---|
| s3 | usable but blander | CLEAN refusal, perfect English |
| s5 | loops / non-words | still coherent + on-behavior |
| s8 | word salad | degrades (repetition, syntax drift) |

The fluency cliff sits ~2 dose units (~1.6x) higher in English than in
Czech. In English a genuine behavior+fluency window EXISTS (s3–s5); in
Czech it is empty. Reframes §7b: the domination tax lands on the weakest
language first — the step and the tax are real, but their SEPARATION is
language-dependent. The production app's problem is specifically that it
deploys in Czech.

Caveat: deployment-length tier was inconclusive — with English final prompt
on a Czech-context 16k scaffold the model answers in Czech anyway (context
dominates; also Turkish bleed "görevů" at s4). A proper test needs
English-context scaffolds. k=1, 3 prompts — preliminary, needs the full
confirm design. Missing 2x2 cell: v5(CZ-extracted) -> English.

# Findings (2026-07-28 — controls round; pre-registered in experiments/skop_residual/CONTROLS_PREREG.md; run by Claude, k=1, probe v2 N=24)

## I. Random-basis control, faithful build, and probe power: one claim softened, one strengthened

All arms on Qwen3-4B / v3 @L20, direct-ask probe v2 (N=24, first 6
prompts identical to the 07-27 probe), matched injected magnitude for
reduced-norm vectors. Wilson 95% CIs.

| arm | viol | uniq | note |
|---|---|---|---|
| baseline | 18/24 (.75 [.55,.88]) | 0.925 | |
| v_orig @3 | 8/24 (.33 [.18,.53]) | 0.892 | CI-separated from baseline |
| v̄_v1 @3 (95% norm, v0-style) | 8/24 | 0.875 | suppression preserved at power |
| v̄_v2 @3 (98% norm, faithful map) | 11/24 | 0.879 | **no win** |
| v̄_v0 @4.7 (64% norm, targeted deep) | 13/24 | 0.918 | weakest suppressor, cleanest output |
| random r149 ×3 seeds @~3.1 | 4,6,7 /24 | ~0.90 | low-rank cuts don't matter |
| random r1536 ×3 seeds @~4.8 | 9,10,9 /24 | ~0.89 | random deep cuts keep the effect |

**1. CORRECTION (dated, per house rules): "deep cut kills the effect"
was overstated.** The 07-27 N=6 read (v̄_v0: 5/6 comply) said the
targeted deep cut removed most of the effect. At N=24 the same arm
shows PARTIAL loss (13/24 vs baseline 18/24). Deep targeted projection
*weakens* the effect; it does not kill it.

**2. Specificity is DIRECTIONAL, not established.** Random rank-1536
complements at matched norm and magnitude keep suppression (9–10/24)
where the targeted cut loses more (13/24) — the ordering matches
"rerouting directions carry the effect" on all three seeds, but the
gap (~15pp) is inside CI overlap (two-proportion p≈0.2 vs pooled
random). Needs the harness or larger N to settle. The monotone
ordering baseline > targeted-deep > random-deep > v_orig ≈ v̄_v1 is
the cleanest current summary.

**3. The no-free-lunch negative STRENGTHENS.** The v2 faithful build
(exact induced-query map via jvp/VJP through RMSNorm→W_q→q-norm→RoPE,
post-RoPE keys, 64-prompt calibration) removes components nearly
orthogonal to v0's (cos(removed_v2, removed_v0)=0.17; Rayleigh 30→~2
on the exact map; 98% norm kept) — and still shows no separation:
suppression directionally worse than v_orig (11/24 vs 8/24), no
coherence gain (uniq 0.879 vs 0.892), Slovak bleed still present. The
07-27 negative can no longer be attributed to the pre-RoPE /
no-Jacobian shortcuts. (Single config; a v2 risk/γ/p sweep remains
possible but the prior just dropped.)

**4. Entanglement signature holds at N=24:** v̄_v0 is simultaneously
the weakest suppressor and the cleanest output (uniq 0.918, rep4 0) —
effect and tax travel together in the targeted directions.

**Hygiene note:** the 07-27 results batch (13 files) had been committed
to `results/` WITH generation text, against the scores-only policy —
the pre-commit hook checks private markers, not text fields, so generic
synthetic outputs passed. Scrubbed in place 07-28; git history retains
the (generic, synthetic-prompt) texts. Scrub step is now part of the
pull-results routine.

Data: results/efficacy_probe_v2_refarms.json, efficacy_randctl_*.json,
efficacy_v0_n24.json, efficacy_v2_n24.json, randctl_diag.json,
diag_v2.json. Vectors on the GPU box: v_randctl_r{149,1536}_s{1,2,3}.pt,
v_pref_no_task_qwen_skopres_v2.pt.

## J. Rerouting-monitor validation round (2026-07-28 afternoon; adversarial audit before any publication)

Three hostile agents (implementation audit, interpretation attack,
prior-art sweep) were run on the brainscope rerouting monitor and the
day's JSD batteries before posting anything. Outcomes:

- **Instrument core: sound.** Position alignment, base-2 JSD math, GQA
  per-query-head indexing, eager-attention capture, and the clean-side
  cache all verified. The injection-layer control is real: max L20 JSD
  = 0.0000 across 384 layer-rows in four independent batteries.
- **Caveat 1 (regime):** the early batteries steered the PREFILL too
  (spec lacked decode_only) — production steers decode-only, so those
  runs measured a strictly stronger intervention. All published numbers
  now come from a decode-only rerun.
- **Caveat 2 (contamination channel, fixed):** a live global /steer
  would silently contaminate the clean pass AND the per-prompt clean
  cache (keyed without steering state). No battery was affected
  (timeline verified), but the hole is closed in brainscope 0.2.1: the
  forced diff now strips global steering. Plus two new per-head fields
  for artifact defenses: clean_entropy_mean (sharp-head confound) and
  sink_mass_delta (sink-attraction confound).
- **Prior art:** SKOP itself measures rerouting empirically (aggregate
  focus-mass loss, query-space, dose-dependent) — "they only theorize
  it" would be a false claim. What appears genuinely new: a per-head
  divergence map under RESIDUAL-stream steering at deployment dose,
  with layer localization, dose curve, sham floor, and random-vector
  null. Adjacent metric precedent: "Focus Divergence" (JSD on attention
  under quantization, arXiv 2604.19884).
- **Claim discipline adopted from the referee:** headline is
  "attention barely moves except a few heads one layer up" (robustness
  with exceptions), NOT "steering makes heads re-route" (causal);
  saturation reported as concentration, not mechanism; the
  v̄-reduces-JSD result framed as instrument-validating (circularity:
  the projection was built to minimize this quantity) with the open
  puzzle that ~93% coupling reduction buys only ~27% divergence
  reduction — most divergence flows through K/V channels the
  projection cannot reach.
- **In flight:** 64 prompts (24 task + 40 neutral) × 6 doses ×
  {v3, v̄_v1, random-1536, sham} decode-only mega battery; analysis
  includes bootstrap CIs, entropy scatter, sink decomposition,
  task-vs-neutral content specificity. Frozen-attention patch remains
  the pre-registered causal test (MECHANISM doc, measurement 2).

Agent reports archived in session; data lands in brainscope notes
(gitignored) + summary here when the battery completes.

## K. Decode-only mega battery complete (2026-07-28 evening; the FINDINGS-J "in flight" run)

Run interrupted mid-flight by a local machine reboot (chain scripts
lived in a session scratchpad — lost with /tmp); recovered from the
session transcript, moved to steermech-private/campaign/rerouting/,
and rerun AORUS-SIDE under setsid nohup. 1216/1216 cells, 0 skips.
64 prompts (24 task/40 neutral) × {1, 1.5, 2, 3, 5, 8} ×
{v3, v̄_v1, rand1536} + sham(1e-6), decode-only, L20 injection,
k=1/cell. Scores in steermech-private/campaign/rerouting/mega/;
report + fig builder alongside; fig_v2.png in brainscope notes
(gitignored).

- **Sham floor:** mean 0.0000, max cell 2e-5 (N=64).
- **Robustness headline holds decode-only:** at working dose s3 the
  median cell is JSD 0.023; 120/256 cells within 5× of sham. Top
  heads L21 h18 = 0.404, h11 = 0.375 (band mean 0.145, CI
  0.139–0.151). Injection layer L20 reads exactly 0 (plumbing
  control).
- **Random null:** matched-norm rand1536 lights the SAME heads ~4×
  weaker (h18 0.107, band 0.037 at s3) — the head map is a property
  of the heads, not the vector; the steered excess above the null is
  the vector-specific part.
- **Dose curve:** growth + saturation; h8 overtakes h18 between s3
  and s5 (0.21→0.76 vs 0.40→0.61), band saturates toward ~0.44 at s8.
- **Artifact defenses:** entropy corr 0.446 with top heads ranked
  78/201 of 256 by sharpness (sharpness doesn't explain them); sink
  deltas NEGATIVE (h18 −0.49, h8 −0.28) — rerouting away from the
  sink, not toward it.
- **Content specificity: none to speak of** — task vs neutral band
  0.157 vs 0.138 at s3. Consistent with E1's "directional only".
- **v̄_v1 projection:** −37% top head (0.404→0.256), −33% band at s3
  with efficacy retained; only ~−10% at collapse dose s8. Framed as
  instrument validation (circularity caveat stands, as does the
  93%-coupling-vs-33%-JSD puzzle, now decode-only).

## L. Mechanism round: Q/K decomposition + frozen-attention patch (2026-07-28 evening; MECHANISM doc measurements, decode-only, Qwen3-4B L20, N=40)

Script `experiments/skop_residual/qk_freeze.py`; scores in
`results/qk_freeze_report.json`. Sanity: frozen path at s=0 reproduces
the clean pass exactly (KL 0.0, argmax 1.0); manual fp32 attention vs
captured bf16 probs max |Δ| ~0.03 (all four patterns share the fp32
path, so comparisons are internal).

- **Q/K decomposition of the L21 divergence (SKOP-framework
  extension).** The perturbation reaches attention through queries AND
  keys; hybrid patterns split the JSD. h18 is a QUERY-fragile head:
  79%/90%/94% of its divergence at s3/s5/s8 is query-carried (k-only
  3–8%). h11 ~50–64% query. h15 is the opposite — KEY-carried (37→75%
  with dose). Band mean: ~55–71% query, 20–26% key.
- **The 93%-vs-33% puzzle sharpens, and points AWAY from the K/V-leak
  story:** the top head's divergence is overwhelmingly query-carried,
  yet the SKOP-style projection (93% first-order query-coupling cut,
  and E2's Jacobian build) removes only ~a third of it. The residual→
  query map at production dose is substantially NON-first-order
  (input-LN + q_norm nonlinearity is the prime suspect). Testable
  next: measure induced Δq directly vs the linear prediction.
- **Frozen-attention patch (measurement 2, the decisive one): mixed,
  leaning H-domination.** Freezing clean patterns on the whole band
  (L21–27) while keeping steered values recovers only ~half the
  teacher-forced damage: KL rescue 54%/48%/42% at s3/s5/s8 (argmax
  match at s8: 20%→41%). Freezing L21 alone: 4–13%. Per the
  pre-registered table: "fluency largely RESCUED" did NOT obtain;
  damage persisting in the value/MLP stream did — with the honest
  nuance that patterns DO carry roughly half at working dose.
  High-dose collapse is NOT primarily attention-routing: consistent
  with the magnitude wall (5/5) and FINDINGS K's dose curves.
- Caveats: KL on teacher-forced decode rows is a damage proxy (not
  free-running fluency); 7-layer freezing is itself a large
  intervention (exact-at-s0 sanity mitigates); one model, one vector
  family, k=1.

## M. Damage factorization flips between models (2026-07-28 night; qk_freeze2.py, 4B bf16 + 8B 8bit hard-no-think, decode-only L20, N=40 each)

Completed the patch factorization from FINDINGS L (three freeze arms on
the L21-27 band: clean PATTERNS / clean VALUES / whole attention output
clean) plus a dose ladder for the induced Δq at L21. All arms exact at
s=0 (KL ≤ 0.004, argmax 1.0). Scores: results/qk_freeze2_{4b,8b}_report.json.

- **The headline: 4B and 8B break through OPPOSITE attention channels.**
  KL rescue at working dose s3 — 4B: patterns 54% > values 42%;
  **8B: values 53% > patterns 25%**. Same vector recipe, same layer,
  same dose; different damage anatomy. Shares drift with dose but the
  ordering holds at s5/s8 (8B patterns down to 9% at s8).
- **This mechanistically explains the 8B projection failure** (FINDINGS
  E/RESULTS "projection cost half the suppression"): a SKOP-style
  projection protects attention *patterns* — on our 8B that channel
  carries only ~a quarter of the damage, so even a perfect
  pattern-protection cannot transfer the win. Channel shares must be
  measured per model before choosing a fix.
- **Whole-attention freeze rescues ~72% (both models) at s3, falling
  to ~50% at s8** — the MLP/skip share grows with dose (28% → ~half),
  consistent with H-domination at collapse (FINDINGS L). Pattern+value
  rescues are sub-additive vs whole-attn (nonlinear interaction).
- **Linearity ladder: the residual→query map saturates.** ||Δq||/s is
  ~flat only below s≈2, then decays (4B h18: 4.6→2.8 from s1→s8; 8B
  from s1 already), with direction drift (cos vs linear prediction
  0.91/0.90 at s8; 8B departs earlier). First-order/Jacobian
  projections are built in exactly the regime that stops being valid
  at production dose — the concrete reason E2's faithful build could
  not win, and a caution transferable to SKOP's residual extension.
- Caveats: teacher-forced KL proxy; k=1; band = L21-27; 8B in 8-bit
  (sham-floor logic from FINDINGS K applies); one vector family.

## N. SKOP's own model is MLP-dominated under residual steering (2026-07-28 night; Llama-3.1-8B-Instruct 8bit, inj L18 = 0.56 rel depth, band L19-25, N=40)

Same factorization as FINDINGS M, on the model SKOP built their results
on. Sanity: fpat/fattn exact at s=0; fval KL 0.0087 (fp32 roundtrip,
no qk-norm in Llama). Scores: results/qk_freeze2_llama_report.json.

- **Attention channels are the MINORITY damage carrier on Llama:**
  at s3 rescue is patterns 4%, values 16%, whole-attention 30% —
  **MLP/skip carries ~70%** (97% at s8, where pattern-freeze is even
  slightly negative and argmax match is 2%). Cross-model anatomy now:
  Qwen3-4B pattern-led (54%), Qwen3-8B value-led (53%), Llama-3.1-8B
  MLP-led (70%).
- **Implication for SKOP's stated future work:** a residual-space
  extension of their pattern-protecting projection would target ≤30%
  of the damage on their own flagship model. Their query-space result
  is untouched (by construction that intervention enters only through
  queries); but the residual extension cannot inherit the win there.
  The constructive statement for the authors: measure the channel
  shares first — the fix is model-specific (patterns on 4B-class
  Qwen, values on 8B Qwen, largely futile on Llama).
- **Nonlinearity is much stronger on Llama:** cos(Δq, linear pred)
  falls to 0.785 already at the working dose (Qwen ~0.96-0.99) and
  0.654 at s8; ||Δq||/s decays from s1. First-order machinery is
  least valid exactly on their model.
- **Same-scale damage is much larger on Llama** (KL 2.63 at s3 vs
  ~0.6 on Qwens) — but scale units are not comparable across models
  (relative-dose lesson, FINDINGS/RESULTS takeaway 3); the channel
  factorization is internal to each dose and unaffected. Czech probe
  prompts on an EN-centric model and 8-bit numerics are additional
  cross-model caveats (KL is against the model's own clean pass, which
  cancels most language-competence effects).

## O. MEGA-RIGOR round: both pre-registered invariants falsified; "measure, don't assume" survives strengthened (2026-07-28/29 night; PREREG_CHANNELS.md hypotheses committed before data)

Controls, breadth and proxy validation for the channel-factorization
line (K–N). Scores: results/rigor_{factorizations,chainB,freegen}.json.

**Controls that PASSED (FINDINGS N verdict robust):**
- Matched damage: Llama at s1.5 (KL 0.49 ≈ Qwen s3 level) is still
  MLP-dominated — patterns 4% [1,7], whole-attn 39% [36,43]. Not a
  dose artifact.
- Language: EN prompts on Llama, same anatomy (patterns 0%, attn 34%).
- Free-gen proxy (k=3 seeds, 7 doses, 3 models): KL onset tracks
  free-gen degradation everywhere, and the FAILURE MODE is
  model-specific — Qwens collapse into repetition loops (4B loop-rate
  47%, 8B 25% at s8), Llama goes MUTE (median 12 words at s8, from
  ~58; uniq degradation 0.90→0.51 before that). A repetition metric
  alone is blind to Llama's collapse; length+uniq catch it.

**H1 (attention share is a per-model constant): FALSIFIED.**
checklist/websearch/random on 4B agreed (71–72% at s3), but:
sycophant 52% [48,55], confident 58% [54,62], refusal 11% [5,17] —
all outside the pre-registered [62,82] band. Damage-matched
comparison (refusal at KL 5.9 vs checklist s8 KL 5.9: 11% vs 54%)
keeps the falsification. Refusal at s3 even shows NEGATIVE
single-arm rescues (freezing one channel makes it worse —
interaction). Caveats: hd_-provenance vectors, EN-behavior bakes on
CZ prompts; norms differ (KL column carries the dose-matching).
**H2 (Qwen2.5-7B lineage-vs-qknorm discriminator): H2b falsified,
H2a marginal.** Attention share 65% [61,68] at s3 — at the edge of
the lineage band, NOT Llama-like (qk-norm is not the mechanism).
At lower doses 55–60% (the "middle" that falsifies both simple
stories). And unlike every other run, Qwen2.5's attention share
RISES with dose (55→65%) where others fall (4B checklist 72→54%,
refusal 11→1%) — even the dose TREND is model-specific.

**The surviving claim, sharpened by its own falsifications:** there
is no simple law — anatomy varies by model, by vector, and by dose,
in direction as well as magnitude. Robust anchors: Llama/checklist
MLP-dominance (all controls), Qwen-vs-Llama contrast at matched
damage, saturating residual→query map (L/M), model-specific failure
phenomenology. The paper thesis becomes: nobody can tell you where
your steering damage flows — not from the architecture, not from a
sister model, not from another vector on the same model. Measure the
(model, vector, dose) triple; the instrument makes it minutes.
H4 (Gemma KV-share prediction) remains open — run not yet built.

## P. Final data round: H5 confirmed, H6 falsified, H4 verified; strict-reviewer self-audit (2026-07-28/29 night)

- **H5 (MWE vectors on Llama): CONFIRMED — the first pre-registered
  hypothesis to survive.** SKOP-style mean-diff bakes from public MWE
  data (power-seeking, corrigibility-less-HHH; norms ~3-10 at
  mid-depth) on Llama-3.1-8B: band-attention rescue 27%/32% max across
  the damage range (falsifier was ≥50%). Llama is attention-minority
  for THEIR behavior suite too, not just our production vector. Vector
  breadth on Llama: 3 vectors, one verdict.
- **H6 (depth invariance on 4B): FALSIFIED — third falsification.**
  L14: band-attn 29% [20,38] (caveat: absolute damage tiny there, KL
  0.03 — weak vector row); L26: 56% [52,59]; prereg band was [57,87].
  Anatomy is also layer-dependent: the triple is a QUADRUPLE
  (model, vector, dose, layer).
- **H4 (Gemma-4-E4B KV-share prediction): VERIFIED where testable.**
  Producer layers L22/23 receive bit-identical inputs in clean vs
  steered passes (maxdiff 0.0) → shared-band K,V clean by
  construction, so whole-attention freeze isolates the query/pattern
  channel. That channel carries only 16-26% of damage (s1.5-4) —
  a 5th model, a 3rd anatomy. Top divergence again ONE LAYER above
  injection (L26 h5), replicating the localization on a 3rd
  architecture family.
- **Metric-robustness check (from existing data):** re-deriving all
  rescue shares from argmax-match instead of KL compresses magnitudes
  but preserves every qualitative ordering (4B patterns>values, 8B
  values>patterns, Llama both small; Llama fattn 16% vs Qwen 45%).
- **Self-audit (strict-reviewer pass) — one attribution corrected:**
  the freeze band covers 7 layers above injection, so "MLP/skip
  share" as previously written actually includes ATTENTION IN LAYERS
  ABOVE THE BAND. All FINDINGS L-O "MLP/skip" figures should read
  "outside the frozen band". A fattn-ALL control (freeze attention in
  every layer above injection: 4B/8B band 21-35, Llama 19-31) is
  queued as chain E; its results determine how much of the
  "non-attention" share is late attention. ARC-300 likelihood
  utility axis (SKOP's benchmark dimension) also queued.

## P-addendum: second strict pass — one downgrade, one verification, one rebake (2026-07-29)

- **H6 wording DOWNGRADED (correction of P):** the L14 share divides a
  near-zero denominator (KL 0.03 nats — the row-14 vector is too weak
  at s3/s5) and is not evidence; the L26 CI [0.52, 0.59] straddles the
  pre-registered band edge (0.57). Correct verdict: H6 *not supported;
  layer-dependence suggested but underpowered.* A matched-damage L14
  rerun at higher scales is queued (chain F). "Falsified" in P was an
  overclaim; this note supersedes it.
- **H4 premise upgraded from assumed to code-verified:** the installed
  transformers Gemma4 implementation gives shared-KV layers NO
  k_proj/v_proj weights and always routes shared_kv_states from the
  last non-sharing (producer) layers, cache or no cache — so with our
  verified producer-input identity, shared-band K/V are clean by
  construction in the exact code path we ran.
- **bake_mwe.py double-BOS bug found and fixed** (chat-template text
  already carries BOS; tok() added a second). Mean-diff cancels most
  of it (both classes affected identically) and the H5 margin was
  wide, but both MWE vectors are being rebaked and remeasured with
  the fix (chain F) so H5 rests on clean construction.

## Q. Closing controls: the all-layer freeze rewrites both headlines honestly (2026-07-28 late; chains D/E/F complete — DATA FREEZE for paper 1)

- **fattn-ALL (freeze attention in EVERY layer above injection) — the
  P-addendum control, and it was material in BOTH directions:**
  Qwen3-4B 85% [82,87], Qwen3-8B 85% [83,87] at s3 (band value was
  72% — late attention carried the difference; true MLP/skip ≈ 15%).
  Llama-3.1-8B: 50% [46,53] at matched damage (s1.5), 41% at s3, 28%
  at s5 (band: 39/30/24%). Corrected cross-model statement: **Qwen
  models are strongly attention-carried (~85%); Llama caps at ~half
  at matched damage and falls with dose.** "MLP-dominated" (N) was
  too strong band-relative; "attention-minority, shrinking with dose"
  is what the all-layer control supports at s3+, with 50/50 at
  matched damage. The cross-model CONTRAST (85 vs ≤50) stands, CI-
  separated by a wide margin.
- **H6 now properly powered and FALSIFIED:** L14 at s8/s16 puts real
  damage in the denominator (KL 0.32/1.23) — band-attention share
  46%/44% [39,52], far outside the pre-registered [57,87]. Depth
  matters; the quadruple stands (this upgrades the P-addendum
  "underpowered" verdict to a clean falsification at adequate power).
- **H5 re-confirmed on BOS-fixed MWE vectors:** band-attention 22–33%
  across the damage range for power-seeking and corrigibility v2
  bakes — same verdict as v1, now on clean construction. (H5 is
  band-operationalized per its registration; the all-layer number on
  Llama above bounds it from the same data family.)
- **ARC-300 axis (chain D):** working dose is capability-free on all
  four dense models (±1pp of baseline; 0.82/0.82, 0.82/0.83,
  0.89/0.89, 0.76/0.77); s5 costs 7–16pp; s8 lands at 44–48%.
  Together with nonzero KL at s3: low-dose KL is largely distribution
  shift, not capability loss — one more reason damage needs multiple
  instruments. Scores: results/arc300_summary.json.
- **DATA FREEZE:** paper 1 uses FINDINGS K–Q as its data basis; new
  measurements (concordance, value-side projection) belong to the
  post-freeze queue.

## Q-addendum (2026-07-29): Fig-1 rebuild restates K's quiet-cell phrasing (wording only, no re-measurement)

- Building paper Fig 1 from the mega scores exposed that K's "120/256
  cells within 5× of sham" leaned on a +0.02 slack term inside the
  threshold formula, and took the sham maximum over per-prompt
  matrices (2×10⁻⁴) rather than the prompt-averaged map (2.2×10⁻⁵ —
  the "2e-5" K itself reports as the sham headline). Strictly "within
  5× of sham" would pass almost no cell; the slack did all the work.
- Restated with an explicit threshold: **median cell 0.023; 116/256
  cells below 0.02; sham floor = max cell of the averaged map,
  2.2×10⁻⁵.** No measurement changed. Paper §4.1, Fig 1, and the
  reader's guide carry the restated form; K's original wording stands
  above as written (this note supersedes it, P-addendum style).

## R. Post-freeze appendix round (chain G, 2026-07-29 morning; run by Claude): all-layer freeze on the vector suite + quantization control

POST-FREEZE: these runs answer two referee objections identified in
the pre-submission self-review; they are appendix material, not part
of the K–Q frozen basis, and no pre-registered verdict changes.
Scores: results/postfreeze_chainG.json. Setting as in qk_freeze2
(decode-only, 4B L20, N=40, paired-bootstrap CIs).

- **G1 — fattn-ALL (band 21–35) for the five extra 4B vectors, s3:**
  websearch 85% [82,88], random 85% [82,87], confidence 81% [78,83],
  sycophancy 78% [75,80], **refusal 60% [58,63]**. Compared to the
  band shares (71/72/58/52/11%), the all-layer totals are higher and
  far more uniform: **most of the H1 vector-spread was band
  LOCALITY** — late attention (above L27) carries a large share,
  most extremely for the refusal vector (11% band → 60% all-layer).
  The vector-dependence claim survives at reduced magnitude: 60 vs
  85% remains CI-separated, and H1's falsification is untouched (it
  was registered on the band protocol). Table-1 protocol gap closed:
  every 4B row now has both band and all-layer numbers.
- **G2 — same-model precision control (4B in 8-bit, checklist v3):**
  8-bit reproduces bf16 within ~2pp on every share — s3 band:
  patterns 53% [47,59] vs 54%, values 43% [37,48] vs 42%, attn 70%
  [66,74] vs 72%; fattn-ALL 84% [81,86] vs 85%; kl_steered 0.63 vs
  ~0.6 — and the dose trends match through s8. **Damage anatomy is
  not visibly quantization-dependent on the one model where both
  precisions fit the GPU**; the cross-model contrast (Qwen ~85 vs
  Llama ≤50, both 8-bit-involved) is not attributable to 8-bit
  numerics on this evidence.
- Caveats: the precision control is one model, one vector (16 GB
  ceiling: bf16 Llama-8B does not fit); fattn-ALL suite at s3/s5
  only; same instrument caveats as K–Q (teacher-forced KL proxy,
  k=1).

## S. Post-freeze chain H: the five-model grid completed (2026-07-29 afternoon; run by Claude; scores results/postfreeze_chainH.json)

POST-FREEZE appendix material; no pre-registered verdict changes.
New instrument jsd_map.py (generic per-head JSD map, any model, arms
steered/sham/matched-norm-random, decode-only teacher-forced, N=40).

- **Localization replicates on ALL FIVE families.** Top divergence
  sits ONE layer above injection everywhere: 4B L21 h18/h11 (mega),
  8B L21 h8=0.474/h11=0.460, Qwen2.5 L17 h0=0.459, Llama L19
  h13=0.751/h12/h15, Gemma L26 (0.49, from the H-chain band capture).
  Sham floor exactly 0 on every new model (40 prompts each). The
  random arm lights the same layer everywhere; the steered/random
  amplitude ratio THINS across models (4B ~4×, 8B ~2.3×, Qwen2.5
  ~1.7×, Llama ~1.6×), and head overlap with the random arm is
  partial on 8B/Q2.5/Llama — "which heads respond is a property of
  the model" holds; "the vector only sets amplitude" was
  4B-specific in strength. Notable dissociation: Llama's attention
  divergence is sharply localized even though attention carries ≤50%
  of its damage — where attention moves and what carries damage are
  different questions.
- **Qwen2.5 fattn-ALL: 72% [70,75] at s3, 71% at s5** (band was
  65%). The all-layer ladder is now 85/85/72/≤50 (Qwen3-4B/8B,
  Qwen2.5, Llama matched-damage) — Qwen2.5 stays the middle case at
  the correct protocol level; dose trend flat (unlike its rising
  band share).
- **Gemma all-shared-band freeze (26–41): 18/22/27% at s1.5/2.5/4**
  [CIs ±4pp], producer-input maxdiff 0.0 — consistent with H4's
  16–26% band reading, now on every layer above injection.
- **Qwen2.5 ladder saturates like the others:** ‖Δq‖/s 3.21→1.57
  (s0.5→8), cos 1.00→0.91; at s3 cos 0.97 (Qwen-like, not
  Llama-like). Fourth model with the same saturation shape.
- **Qwen2.5 free-gen: a THIRD collapse phenotype.** Suppression
  works (viol 62.5%→1.4% by s3, N=72/dose). At s8: loop-rate only
  2.8%, median length RISES to 73 words, uniqueness 0.96→0.41 —
  long repetitive ramble that a loop metric misses and a length
  metric reads as healthy. (4B/8B: loops; Llama: mute; Qwen2.5:
  ramble.) One more failure phenotype that only the uniqueness
  metric catches.
- **Gemma ARC-300: NOT capability-free at its working scale.**
  0.863 baseline → 0.833 @s2.5 → 0.813 @s4 (−3/−5pp; the four dense
  models were ±1pp at their working scale) → 0.697 @s6, 0.507 @s8.
  The capability-free window is model-dependent too.
- Fig line: fig6_localization (5-panel layer-offset profiles);
  fig2/fig4/fig5 updated (Q2.5 all-layer bar, Gemma ARC curve, Q2.5
  ladder curve).
- **Gemma free-gen (rerun after Gemma4 layer-lookup fix): a FOURTH
  degradation shape — gradual erosion.** Suppression late (viol
  51%→15% @s4, 1.4% @s6, N=72/scale); NO loops, NO muteness in the
  probed range (≤s6); uniqueness slides 0.92→0.77 while length
  RISES 37→45 words. Together with ARC (−5pp @s4, −17pp @s6): Gemma
  trades behavior for a diffuse fluency/capability tax instead of a
  sharp collapse — consistent with the July campaign's "fluency is
  unguarded" verdict. Phenotype tally: loops (Qwen3), mute (Llama),
  ramble (Qwen2.5), erosion (Gemma).
