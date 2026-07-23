# Research plan: when does a steering vector generalize from calibration to deployment — and what do steering evals actually measure?

*Pre-registered 2026-07. This document is written before the experiments; the
point of pre-registration is that the hypotheses below can lose.*

> **FROZEN 2026-07-23.** Month-1 prerequisites complete: adversarially
> verified literature pass ([LITERATURE.md](LITERATURE.md)), benign set
> expanded, classifiers unified (single-source checkers), pilots run
> (N=20 cross-model, [FINDINGS.md](FINDINGS.md)). Amendments 1–8 below were
> dated before this freeze. From here, any change to hypotheses, thresholds,
> or methods requires a dated deviation note in this file — the plan can be
> wrong in public, but it cannot move quietly.

## Motivation — two accidents worth taking seriously

Two things happened in this lab in July 2026 that the steering literature
mostly doesn't talk about:

1. **The regime accident.** A preference-suppression vector, carefully
   calibrated under generation-only steering, was deployed into a serving
   stack that also steered the ~13k-token prompt. The model collapsed into
   repetition. Same vector, same scale, same layer — the only change was
   *how many positions* received the addition. The fix (`decode_only`) is
   now a flag in [hotwire-vllm](https://github.com/moudrkat/hotwire-vllm);
   the *scaling law behind the accident* was never measured.
2. **The proxy accident.** An auto-calibration loop scoring efficacy by
   concept suppression in the model's forming-words (a logit-lens/J-lens
   disposition proxy) found clean optima on short generic prompts — and
   scored a *known-working* vector as a total miss (1.00) on the real 13k
   task it demonstrably fixes in deployment. The cheap metric and the
   behavioral truth diverged completely, in the direction that matters.

Both accidents are instances of one question:

> **Steering vectors are calibrated in one condition and deployed in
> another. What actually transfers — and what do the evals we calibrate
> against actually measure?**

This is a live topic: recent critical work reports steering vectors failing
to generalize across prompts and settings. What this lab can add is
unusual: a full open pipeline from calibration through production serving
(brainscope → hotwire-vllm), an instrumented replay of a real deployment
failure, and the ability to measure *both* proxy and behavioral efficacy on
the same prompts under identical, teacher-forced conditions.

## Research questions and falsifiable hypotheses

### RQ1 — Dose–context scaling: what fried the vector?

**H1: coherence collapse is governed by the total injected mass (scale ×
number of steered positions), not by per-token scale alone.** A vector at
scale 3 over 48 generated tokens and the same vector at scale 3 over 13k
prefill + 48 generated tokens are different doses, and the collapse
threshold should be predictable from steered-position count.

*Design:* grid over scale × context length (≈0.5k / 2k / 8k / 16k) ×
regime (decode-only vs full), measuring behavioral efficacy, coherence
(repetition/perplexity heuristics + KL), and damage. One vector family
first, then the survivors of RQ4's model list.

*H1 loses if:* the collapse threshold tracks per-token scale regardless of
steered-position count — i.e. a scale that is safe at 0.5k full-steer stays
safe at 16k full-steer. (Alternative outcome worth having: collapse is
driven by *prefill contamination* specifically — steering the instruction
tokens — rather than total mass; distinguishable by steering only the first
N prompt tokens vs the last N at matched mass.)

### RQ2 — Eval validity: when do disposition proxies track behavior?

**H2: proxy efficacy (concept suppression in forming-words) correlates with
behavioral efficacy (generate + classify the violation) on short generic
prompts, and the correlation decays as context length and task-specificity
grow.** The July data point is the extreme end: r ≈ perfect on generic
15-prompt sets, total divergence on the 13k-token real task.

*Design:* for each (vector, layer, scale) point sampled from RQ1's grid,
score the same prompts both ways — proxy (teacher-forced suppressed/promoted
concepts) and behavioral (real generation under deployment conditions +
violation classifier). Report the proxy↔behavior correlation as a function
of context length and of "eliciting pressure" (with/without a task-nudge in
the system prompt).

*H2 loses if:* the correlation is flat (proxies are fine everywhere — good
news, calibration can stay cheap) or uniformly poor (proxies are useless
even on short prompts — also good to know; heretic-style shortcuts would be
unjustified for steering generally).

### RQ3 — Calibration transfer: what part of the optimum is real?

**H3: the optimal *layer* transfers across eval sets; the optimal *scale*
does not.** July data: a generic eval put the optimum at L25–28 for a vector
that production experience placed at L20 — while dose thresholds were
consistent across settings. Layer may encode *where the behavior lives*
(a property of the model+vector), scale may encode *how hard the eval
pushes* (a property of the eval).

*Design:* run the full calibrator (behavioral efficacy, Optuna TPE,
`hidden-directions calibrate`) independently on: (a) generic benign-derived
intents, (b) task-specific short evals, (c) the long-context deployment-like
eval. Compare the found (layer, scale) optima and, crucially,
cross-evaluate: how much efficacy/damage does eval-A's optimum lose when
scored under eval B?

*H3 loses if:* optima are idiosyncratic per eval in both coordinates (then
"calibration" without the deployment eval is theater and the paper's
message becomes *calibrate on the real thing or don't bother*), or fully
transferable (then cheap generic calibration is vindicated).

### RQ4 — Is any of this architecture-general?

Two tiers, all within the lab's hardware (quantized where needed):

- **Core (full reduced grids):** Qwen3-4B (the workhorse — RQ1–3 run here
  first), Qwen3-8B, Llama-3.1-8B-Instruct — three sizes, two families.
- **Replication sweep (coarse grid: 2 lengths × 3 scales × 2 regimes):**
  Qwen2.5-7B-Instruct, Qwen3.5-9B, Phi-3.5, Gemma 3n E4B, and
  gpt-oss-20b (MXFP4) as the stretch case — a quantized MoE is the most
  different architecture the card can hold, and the most interesting
  place for a dose-scaling law to break.

Vectors re-extracted per model with the same contrastive recipes
([hidden-directions](https://github.com/moudrkat/hidden-directions));
models already exercised by the lab's serving evals are preferred, so
misbehavior is attributable to steering, not to a model we can't run
cleanly. No hypothesis beyond: report which effects replicate. A
dose-scaling law that holds on one model family is an anecdote; on three
families and a MoE it is a finding.

## Baselines (the part steering work keeps skipping)

- **No-steer baseline** for every cell (violation rate, KL trivially 0).
- **Prompting baseline:** the strongest system-prompt instruction we can
  write for the same behavioral goal, scored with the same classifier. The
  interesting quantity is *damage at matched efficacy*: if prompting reaches
  the same violation rate, steering's case must rest on damage/latency/
  robustness, and we should say so plainly.
- **Random-direction control** at matched norm for damage measurements —
  KL from *any* injection vs KL from *this* injection.

## Methods & rigor floor

- **N:** ≥ 30 prompts per behavioral cell (bootstrap 95% CIs on violation
  rates); benign damage set expanded from 15 to ≥ 100 prompts. No headline
  claim on N < 30.
- **Classifiers:** regex/substring first (auditable), spot-checked by hand
  on ≥ 10% of generations; classifier disagreement reported, not hidden.
- **Teacher-forcing** for all proxy measurements (identical context both
  passes — the causal-replay instrument in brainscope), greedy decoding for
  behavioral runs.
- **Pre-registered thresholds:** "correlated" means Spearman ρ ≥ 0.7 with
  CI excluding 0.3; "transfers" means ≤ 20% efficacy loss under
  cross-evaluation. Chosen now, before the data.
- **Everything reproducible:** configs + seeds + result JSONs committed
  here; private deployment scaffolds referenced only via `$SCAFFOLD_JSON`
  (the public pipeline runs end-to-end on public prompt sets).

## Compute budget (measured, not guessed)

On the lab's 16 GB card: a short-context forced diff ≈ 2 s; a 13k-token
one ≈ 100 s (≈ 35 s with brainscope's clean-side cache); a long-context
behavioral generation ≈ 30–60 s. The full RQ1 grid at 4 lengths × 6 scales
× 2 regimes × 30 prompts ≈ 4.3k measurements, dominated by the long-context
cells — roughly 40–60 GPU-hours, i.e. a few weeks of nights. RQ2 rides on
RQ1's grid (same runs, scored twice). RQ3 is ~15 calibrator runs. RQ4's core tier multiplies the
reduced grid by 3; the coarse replication sweep adds ~5 cheap runs plus
one slow MoE run (gpt-oss-20b generates slowest — pilot first, drop if it
can't hold N=30 nights). Feasible solo; not comfortable. The
clean-side cache and small-N pilots first are what make it fit.

## Timeline (6 months, evenings-and-weekends honest)

| Month | Milestone |
|---|---|
| 1 | Literature pass (ActAdd, refusal-direction, CAA, representation engineering, ITI, steering-reliability critiques, heretic). Expand benign set, harden classifiers, pilot N=5 per cell to fix ranges. **Freeze this plan.** |
| 2–3 | RQ1 full grid on Qwen3-4B. Write up the scaling result whatever it shows. |
| 3–4 | RQ2 scoring + analysis (data mostly shared with RQ1). |
| 5 | RQ3 calibrator runs; RQ4 reduced grids on the other models. |
| 6 | Write-up: an Alignment Forum / LessWrong post with full data, then a workshop-length paper (BlackboxNLP or a NeurIPS/ICLR interp workshop). |

## Scope discipline

Until the write-up ships: **no new repos, no new instruments** beyond
bugfixes and what the experiments strictly require. The lab graph is
frozen at six nodes. Instruments serve the question now, not the other way
around.

## What already exists toward this

- Dose–response pilot (threshold ≈ 2–2.5 for imprint visibility, one
  vector, one model) — [FINDINGS.md](FINDINGS.md)
- The regime accident, mechanism and fix — hotwire-vllm `decode_only`
- The proxy collapse data point — [FINDINGS.md](FINDINGS.md)
- The full measurement pipeline: teacher-forced replay + KL + behavioral
  eval with `violation_regex` (`hidden_directions.calibrate`), clean-side
  caching for long contexts (brainscope)

## Amendments before freeze (2026-07-23)

1. **Workhorse: Qwen3-8B (quantized).** The deployment agents will run on
   it, so RQ1–3 run there; Qwen3-4B is demoted to piloting/ranging and stays
   in the RQ4 core tier. Extraction happens under the same quantization as
   serving.
2. **Vector cast and selection principle.** Steering is applied only where
   both cheaper layers fail: the behavior is not guardable in code (it lives
   in free text — claims, promises, implications) AND the prompt is
   saturated (maximally emphatic instruction exists and the violation
   persists). The cast becomes: (a) the claims-family vector (the July
   veteran) as the production case; (b) two tool-decision-boundary vectors
   (search/document over-triggering) as the contrast case — code can absorb
   their consequences and a large model's prompt suffices, so they are
   evaluated as a *small-model enabler* (can a vector move a 4B's decision
   boundary to an 8B's?) rather than as production steering.
3. **The prompting baseline is the deployed system.** Production already
   steers by prompt: a validator emits corrective feedback that is injected
   into the retry. The baseline comparison is therefore not a constructed
   strawman but the live mechanism, and the flagship deployment experiment
   is feedback-retry vs vector-retry vs both, at matched efficacy.
4. **Proxy preflight.** Any calibration run must first verify the proxy
   emits signal in that condition (see FINDINGS 2026-07-23: proxies read
   empty without a fitted lens and are contaminated by reasoning-model
   think-preambles). Behavioral efficacy is the default objective wherever
   the preflight fails.
5. **Thinking mode is a third validity axis (RQ2).** Disposition proxies
   read "what is this state setting up to say later" — on a hybrid reasoning
   model the *later tokens are the think-block*, so the proxy measures the
   narration of the request, not the forming answer (observed on Qwen3-8B,
   2026-07-23, FINDINGS). RQ2 therefore measures proxy↔behavior correlation
   along context length, task-specificity, AND thinking mode
   (off via `/no_think` / on). Instruments: ordinary J-lens is valid only
   with thinking off; the A-lens variant (fit on reasoning traces, readout
   restricted to post-`</think>` influence) is the candidate instrument for
   thinking-on and must itself be validated behaviorally before use —
   an instrument does not graduate to "measurement" until it survives its
   own RQ2 check.
6. **H3 operationalization: fractional depth, windows over argmax.** The
   layer coordinate compares across models as fractional depth (L /
   n_layers); Qwen3-4B/8B share 36 layers, Llama-3.1-8B (32) and the coarse
   tier do not. And per the 2026-07-23 pilot (FINDINGS): cross-evaluation
   is scored on the transferred *curve* (the effectiveness window), with
   the argmax reported but not trusted — single-run TPE optima proved
   model-idiosyncratic even when a shared optimum existed.
7. **Design changes from the adversarial literature pass (2026-07-23,
   [LITERATURE.md](LITERATURE.md)).** (a) Every behavioral cell reports the
   per-sample steerability distribution and fraction-anti-steered, not mean
   efficacy alone; the calibration objective gains an anti-steered-tail
   penalty. (b) Proxy prompts permute A/B and Yes/No assignments; every
   proxy is validated against open-ended generation per behavior before
   any H2 correlation is reported. (c) The damage tier gains a small
   safety-probe battery (jailbreak-ASR + false-refusal) and evaluates the
   second-moment-weighted quadratic alongside KL; coherence guards are
   never cited as safety evidence. (d) A pre-calibration geometric screen
   (cosine agreement of contrastive differences) runs before any
   calibration; behaviors that fail it are declared unsteerable, not
   force-calibrated. (e) H1 is tested with mass-matched factorial cells
   (scale × steered-positions) with per-cell vector geometry logged, and a
   sharp-threshold model is pre-registered as the explicit competitor to
   the mass law. Novelty verdict from the same pass: all three claimed
   contributions unclaimed in 2024–2026 literature; H1 is the weakest
   pre-registered claim and is designed for accordingly.
8. **Miss = violation OR incoherence (2026-07-23).** The behavioral
   efficacy objective counts a prompt as failed if the output violates the
   intent *or* fails the coherence guards. Motivated by the L15@8 incident:
   a coherence-blind objective selected a point that silently switched the
   model into English think-mode rambling and scored it perfect. All prior
   same-day numbers were re-examined with the full checker; results stand
   for L20@3 (both models), and the 8B-argmax point is reclassified from
   "0% violations" to "100% incoherent."
