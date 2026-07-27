# Evidence map for the paper (working doc, 2026-07-27)

For each failure mode in PAPER_OUTLINE.md: the concrete instance, the key
numbers, and where the data lives. Purpose: write prose without hunting.
Venue TBD (NeurIPS Interp4Discovery, deadline Aug 29 / COLM Actionable
Interp fast-track Aug 9 — verification pending).

## F1 — Coherence-blind efficacy scores a broken model "perfect"

- **Instance:** Qwen3-8B steered at L15@8 (the TPE winner): all 12 outputs
  are English think-mode rambling (median 609 chars vs ~240 normal), the
  `/no_think` switch broken, never answers in Czech. Violation regex: 0/12
  (English deliberation phrases no task offer). Full checker: 12/12
  incoherent.
- **Second collapse flavor:** L20@8 — token collapse (`(ne. (ne. (ne.`,
  stray Chinese chars). Also 0 violations by regex, 12/12 incoherent.
- **Why the naive metric misses:** violation-only scoring cannot
  distinguish "behavior suppressed" from "model destroyed".
- **Fix shipped:** miss = violation OR incoherence (engine change, tests
  updated).
- **Data:** FINDINGS "the argmax didn't just lie, it broke the model";
  "Not a scoring bug: two collapse modes, one honest metric".

## F2 — The optimizer converges to the broken pocket

- **Instance:** 15-trial TPE on Qwen3-8B converged to L15@8 — scored
  *better* (0.063 vs 0.042 objective) than the transferred 4B optimum on
  the 8B's own eval, yet L15@8 is the broken point from F1. Single-run
  argmax is model-idiosyncratic; the shared window (L20@3) existed and
  the optimizer missed it.
- **Key numbers:** shared point L20@3 = 0/20 violations on BOTH models
  (KL 0.94 on 4B, 0.42 on 8B); 8B argmax transferred to 4B = 8/20 = 40%
  violations (native 0%). N=20 confirmation.
- **Practice:** transfer the window (the curve), never the argmax;
  asymmetric cross-eval (A→B ≠ B→A) is the cheap exposure test.
- **Data:** FINDINGS "First cross-model transfer result: the window
  transfers, the argmax lies" + N=20 confirmation block.

## F3 — Disposition proxies contaminated by reasoning / silent flatline

- **Instance A (silent flatline):** Qwen3-8B auto-calibration with proxy
  objective: miss = 1.00 across all 25 trials, every layer/scale — the
  J-lens wasn't loaded for 8B, `suppressed_positional` read permanently
  empty, optimizer degenerated to minimizing KL. Nothing warned.
- **Instance B (think-narration):** Qwen3-8B opens every forced
  continuation with a `<think>` preamble restating the request;
  concept-suppression over those positions measures the narration, not
  the forming answer. `/no_think` is the minimum fix.
- **Practice:** proxy preflight ("does it emit signal here?") mandatory;
  behavioral eval the only trustworthy objective without a fitted lens.
- **Data:** FINDINGS "Disposition proxies go blind on reasoning models".

## F4 — Eval choice drives the optimum

- **Instance:** TPE calibration on generic short prompts localizes the
  vector DEEPER (L22–28, best L28@4.23, miss 0, KL 2.37) than the
  hand-tuned production point (L20@3, which scores miss 0.5 on generic
  prompts but works in production). On the real 13k scaffold the proxy
  scores L16/20/24 all miss 1.00 — sees nothing — while the vector
  demonstrably works in deployment.
- **Why:** the proxy doesn't reproduce deployment conditions (no
  task-nudge, no forced tool) and reads dispositions, not the behavioral
  event.
- **Fix shipped:** intent files carry violation_regex + tools/nudge →
  efficacy = full generation under deployment conditions + classifier
  (`generate_efficacy`).
- **Data:** FINDINGS "Auto-calibration works — and the eval choice drives
  the answer"; "The efficacy proxy is the weak link — confirmed on the
  real task". Figure: `fig/calibration_landscape.png`.

## F5 — Emergent-misalignment false positives (2/2 models, 2 mechanisms)

- **Instance:** EM battery flags +0.3 harmful-compliance on BOTH models;
  both are metric artifacts:
  - Llama-3.1-8B: refusal *vocabulary drift* — the no-tasks vector nudges
    "discuss" framing, refusals became "I can't discuss…", the narrow
    regex knew only "I can't help/assist". Model refused 10/10.
  - Qwen3-8B: thinking-model *truncation* — `<think>` deliberation toward
    refusal never reaches the answer within 120 tokens; detector scores
    "compliance".
- **Closure:** clean re-runs with broadened regex + /no_think: +0.0 shift
  on both models (2026-07-25). Both artifacts fully closed.
- **Headline:** a steering vector can shift the *vocabulary* of an
  unrelated behavior without changing the behavior; phrasing-keyed safety
  metrics are fragile under steering. Coherence guards catch neither.
- **Data:** FINDINGS "Emergent-misalignment scare that wasn't"; "TWO
  false positives, two metric failures".

## F6 — Raw scale is an unfalsifiable dose coordinate

- **Argument:** scale multiplies a per-model vector (‖V‖ varied ~0.1–247
  across layers in extraction) into a per-model residual stream (different
  activation norms) — "Qwen needs scale 6 where Llama needs 3" is
  meaningless in raw units. Only dimensionless relative dose
  `‖scale·V[L]‖ / ‖h[L]‖` compares across models.
- **Status:** argument solid; the recompute (‖V[L]‖ from .pt + mean
  ‖h[L]‖ from forward passes) is the pending GPU task. Norm check done
  once: Gemma v5 ‖V[25]‖=35.2 vs v3 32.2.
- **Broader claim:** the literature reports strength inconsistently → 
  cross-model dose claims are unfalsifiable as published.
- **Data:** FINDINGS "The dose axis is under-specified" (2026-07-24).

## F7 — The classifier itself must be validated

- **Instance:** checker validated once at κ=1.0 on golden set (but golden
  is easy — noted honestly). Counter-instance that proves the point: the
  Gemma judge-resolution-limited ship gate — verdict flips on ~5
  borderline cells depending on which defensible judge reading is used;
  cross-judge disagreement concentrated in steered half-refusals with
  degraded Czech.
- **Pending:** a hard-behavior κ case.
- **Data:** `results/checker_validation.json`; FINDINGS "Ship gate: NOT
  certified — verdict is judge-resolution-limited".

## F8 — The mean hides the tail (anti-steerability)

- **Machinery:** per-sample baseline comparison built; literature
  grounding: Tan et al. (arXiv 2407.12404), Braun et al. (2505.22637) —
  up to ~half of inputs steer the wrong way under positive mean.
- **Own numbers (thin, honest):** zero anti-steered samples in any cell
  of the short-context RQ1 row (N=12/cell); exactly one anti-steered
  sample observed all day (Qwen2.5-7B under-dose L16@1). Report
  per-behavior instead of assuming either way.
- **Data:** FINDINGS "RQ1 row 0" (zero anti-steered note); Qwen2.5-7B
  section.

## Supporting positive result — the window transfers as fractional depth

- Qwen3 pair window L15–20/36 = 0.42–0.56 fractional; Qwen2.5-7B (28
  layers) window L14–18 = 0.50–0.64; Gemma-4-E4B optimum L25 = 0.60;
  Llama-3.1-8B working point L16 = 0.50. Four models, three families,
  two depths: usable region straddles ~0.5–0.6 fractional depth; raw
  indices mislead across depths (L20 raw transfer to Qwen2.5 = miss 0.25
  + incoherence).
- **Data:** FINDINGS "Third model, different depth"; Gemma §2.

## Deployment-length results (the production trump card)

- Prompted rule washes out at 12k: rule-present baseline violates
  0.30 prod / 0.85 adversarial (short-context read said "obeys ~10/10" —
  itself a context-length artifact, F4-adjacent).
- Steering beats prompting at deployment length: L25@s3 = 0.000 prod
  hard-miss vs rule baseline 0.300; rule+vector composes (0.025–0.075).
- The behavior window and fluency window did NOT overlap for Gemma in
  Czech (step at s2.5→3, one event, extraction-independent) — but DO
  overlap in English (cliff ~2 units higher). Language-dependent tax.
- **Data:** FINDINGS Gemma campaign §§1–9; score files
  `results/confirm_ship_gemma*.json`, `gemma_hunt_grid.json`.

## Figures inventory (existing)

- `fig/tug_of_war.gif` — L21 per-head deltas (animated)
- `fig/calibration_landscape.png` — TPE landscape
- dose–response curve + attn-vs-MLP split (rendered by `make demo`)
- TODO for paper: window-as-fractional-depth diagram across 4 models;
  per-sample distribution plot (F8); relative-dose recomputed curves (F6).

## Honest-limitations checklist (§5 of outline)

One behavior family (+ two closed nulls); models ≤ ~9B; N=10–20 pilot
grade on transfer cells (bootstrap CIs to add); relative-dose recompute
pending; one hard-behavior κ pending; SteerBench = seed, not benchmark.
