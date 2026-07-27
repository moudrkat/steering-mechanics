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
(‖v̄‖=8.46 vs 13.22). Vector: `aorus:~/hotwire-vectors/
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

Artifacts: `paper/efficacy_v0.json`, `paper/efficacy_v1.json`, vectors
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
v5/v5_nothink. Vector inventory note: private-vectors/ holds per-model
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
