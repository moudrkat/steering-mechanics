# LW post skeleton — "What steering evals actually measure (notes from production)"

Working titles (pick one, keep it concrete):
- *How my steering evals lied to me (in production)*
- *Steering vectors in production: what the evals don't see*
- *I deployed a steering vector. Then I checked what my metrics were measuring.*

Format: LessWrong post, ~2000–3000 words, first person, concrete numbers,
no grand claims. Every claim gets its (k/n, CI) or a link to FINDINGS.md.
⚠ LW policy: text must be MY OWN writing; AI assistance used for
structure/review — disclose briefly at the end. First post is manually
moderated: write for the LW reader (they know what steering vectors are;
skip the intro tutorial, link CAA/Arditi instead).

---

## 0. Hook (2 paragraphs)

The regime accident, told straight: a carefully calibrated
preference-suppression vector, deployed into a serving stack that also
steered the 13k-token prompt → model collapsed into repetition. Same
vector, same scale, same layer. The only change: how many positions got
the addition. Then the second accident: the auto-calibration metric that
scored a known-working vector as a total miss (1.00) on the task it
demonstrably fixes. → "This post is a catalog of the ways my evals lied,
and what caught each lie."

## 1. Setup (short; links do the work)

One paragraph: production app on local open-weight models, Czech,
12k-token agent scaffolds; open pipeline (brainscope / hidden-directions
/ hotwire-vllm on PyPI); pre-registered plan (link RESEARCH_PLAN.md,
frozen date, deviations logged). One sentence on why production matters:
the calibration-deployment gap is where every eval assumption gets
tested for free.

## 2. The catalog (the core — pick the 4 best stories, not all 8)

Each: 1 para failure story + 1 para "why the metric can't see it" + the fix.
- **F1/F2 as one story**: the optimizer's argmax (L15@8) scored PERFECT —
  0 violations — because the model produced English think-rambling and
  gibberish offers no tasks. 12/12 incoherent by the full checker. The
  optimizer *found the broken pocket because the metric rewarded it.*
  Fix: miss = violation OR incoherence. (Numbers: 0.063 vs 0.042
  objective; 0/20 vs 8/20 transfer with CIs.)
- **F5**: emergent-misalignment false positive ×2 models ×2 different
  mechanisms (refusal vocabulary drift: "I can't *discuss*" missed by
  regex; think-block truncation scored as compliance). Both closed at
  +0.0 after fixes. Punchline: a steering vector can shift the
  *vocabulary* of an unrelated behavior without shifting the behavior.
- **F3**: the proxy that silently flatlines (no fitted lens → miss=1.00
  across all 25 trials, optimizer degenerates to minimizing KL, nothing
  warns you).
- **F4**: eval choice drives the optimum (generic prompts → L22–28;
  production behavior → L20; the proxy scores the working production
  config as total miss). Figure: calibration_landscape.png.

## 3. What deployment length does (the part nobody else has data on)

- Prompted rule washes out at 12k (0.30/0.85 violations WITH the rule).
- Steering beats prompting at deployment length (0.000 vs 0.300 hard-miss).
- But: behavior threshold and fluency collapse arrive at the SAME dose
  (step, not two curves) — and the step is language-dependent (window
  exists in English, none in Czech; the tax lands on the weaker language
  first). Table from FINDINGS §7b/§9.

## 4. Rerouting vs domination (the fresh experiment)

- SKOP (Luo et al., arXiv 2605.06342) explains steering damage via
  attention rerouting — for query-space steering; residual left as
  future work (their words). I built the residual analogue (method
  sketch, 3 sentences + link to experiments/MECHANISM_*.md).
- Result 1: effect and tax are TANGLED in the rerouting directions —
  deep projection (64% norm) removes both (5/6 compliance, CI-overlapping
  though); shallow (95% norm) keeps both (1/6 vs baseline 6/6 —
  separated CIs even at N=6).
- Result 2: **the magnitude wall survives projection — 4/4
  configurations, 2 models, 4 collapse flavors.** The cliff is governed
  by injected magnitude, not by the projected-out rerouting components.
- Honest box: k=1 pilots, N=6 probes, direct-ask ≠ production scaffold,
  v0 approximations listed (pre-RoPE keys, LN Jacobian, 8-prompt calib).

## 5. Relative dose and the sink-token trap

- Raw scale is unfalsifiable across models (norm confounds — 0.19→92.8
  across layers on one vector).
- Even relative dose ‖λv‖/‖h‖ is length-sensitive: short-prompt h-norms
  are polluted by massive-activation sinks (375 vs 55 at 12k for the
  same layer!). Working point 0.72, collapse ~1.9 at deployment norms.
- Open question: do models unify at collapse ≈ 1.5–2.0 relative dose
  measured at deployment length? (Gemma remeasure pending.)

## 6. Practical takeaways (bulleted box — the shareable part)

1. Never trust an efficacy number without a coherence term (miss =
   violation OR incoherence).
2. Read a sample of generations every run; classifiers scale judgment,
   they don't replace it.
3. Preflight your proxy: confirm it emits signal before you optimize
   against it.
4. Calibrate under deployment conditions or don't call it calibration.
5. Report per-sample distributions and CIs, not means (Wilson at small N).
6. Report dose as ‖λv‖/‖h‖ measured at deployment length, never raw scale.

## 7. What I want from you (community ask)

- Refute the wall: does anyone have a config where rerouting projection
  moves the collapse threshold?
- Pointers to prior art I missed (LITERATURE.md has my pass — adversarially
  verified, refuted claims listed).
- If you work on steering evals: run your method on a deployment-length,
  non-synthetic task and tell me what breaks.

## Figures needed

1. calibration_landscape.png (exists)
2. dose ladder table → simple chart (behavior vs fluency vs dose, Gemma)
3. Pareto: projection depth (norm kept) vs efficacy vs coherence
   (WAITING on sweep — GPU box offline)
4. h-norm profile short vs 12k (the sink pollution figure) — have data.

## Disclosure footer (LW policy)

"AI assistance: Claude was used for experiment orchestration, literature
search, structural review, and statistics; all text in this post is
mine; all findings are reproducible from the linked repos."
