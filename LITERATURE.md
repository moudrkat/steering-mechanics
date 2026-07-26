# Literature pass (Month 1, 2026-07-23) — adversarially verified

Deep multi-source review of the steering-critique literature, run against
this plan's pre-registered claims. Every finding below survived 3-vote
adversarial verification; claims that did not survive are listed as
refuted. Full report with per-claim sources and vote counts is archived
privately; key sources linked inline.

## What the critiques establish (and we must design for)

1. **Anti-steerability is real and large** ([Tan et al.](https://arxiv.org/abs/2407.12404),
   independently replicated by [Braun et al.](https://arxiv.org/abs/2505.22637)):
   up to ~half of inputs per dataset steer the *wrong way* even when the
   mean effect is positive. Mean efficacy — our TPE objective — can mask a
   bimodal per-sample distribution. *(Caveat: measured on forced-choice
   logit differences, not open-ended generation — whether it persists
   under behavioral evals is an open question we are positioned to answer.)*
2. **Forced-choice proxy formats carry steerability bias** (A/B position,
   Yes/No token artifacts) that inflates apparent efficacy (Tan et al. §5).
3. **Steerability is largely a dataset-level property, correlated across
   models** (ρ=0.769 between Llama-2-7b and Qwen-1.5-14b) — which behaviors
   steer transfers; *what settings* to steer them at was not studied, and
   [2606.20852] explicitly reports scales do NOT transfer across models.
4. **Steering has safety side effects KL cannot see**: non-safety CAA
   vectors shift jailbreak ASR by up to ±50–57pp across 6 models; even
   random directions raise harmful compliance 1–13%. Correlation with
   refusal-direction overlap is strong per-model but did not survive
   verification as a universal law.
5. **Emergent misalignment passes coherence guards**: steered misaligned
   outputs are MORE coherent than finetuning-induced ones (23–35%
   coherent-harmful). Coherence ≠ safety.
6. **Unreliability is geometrically predictable** (low cosine agreement
   among per-sample contrastive differences → poor steering) — a cheap
   pre-calibration screen, and validation for our mechanistic tier.

## Refuted in verification (do not cite as established)

- "Safety collapses while capability benchmarks stay <5% down" — killed.
- "Per-sample outcomes are irreducibly uncertain (sector unidentifiability)" — killed.
- "Directional similarity doesn't predict steering (r=-0.034)" — killed.
- "Coherence collapse depends on geometric decomposition, not mass" — killed
  (H1 remains open, but this specific threat did not survive).
- "Refusal-overlap predicts ASR shift universally (R²≥0.85)" — killed as a
  universal claim; stands as a per-model observation.

## Top-5 threats → design changes (adopted, see RESEARCH_PLAN amendment 7)

1. Aggregate masking → report per-sample steerability distribution and
   fraction-anti-steered per cell; anti-steered-tail penalty in the
   calibration objective.
2. Proxy format artifacts + think-blocks → permute A/B and Yes/No
   assignments in proxy prompts; every proxy validated against open-ended
   generation per behavior before H2 correlations are trusted.
3. Damage blind spots → add a safety-probe tier (small jailbreak-ASR +
   false-refusal battery) and evaluate the second-moment-weighted
   quadratic (x−h)ᵀΣ(x−h) alongside KL; coherence guards are never cited
   as safety evidence.
4. Behavior-dependence → pre-calibration geometric screen (cosine
   agreement, cluster separation); pre-register that some behaviors will
   be declared *unsteerable* rather than force-calibrated.
5. H1 confounds → test mass-matched conditions crossing scale ×
   steered-positions factorially; log vector geometry per cell; treat
   sharp dose thresholds as a competing model to the mass law.

## Novelty verdict (all three survive)

- **Cross-model calibration-coordinate transfer** (fractional-depth
  window, behavioral + damage accounting): unclaimed. Closest prior:
  dataset-level steerability correlation (fixed layer); scale
  non-transfer reported once — evidence *for* H3's split, not a scoop.
- **Proxy validity on reasoning models** (think-block contamination,
  fitted-lens requirement): unclaimed; closest prior covers only
  forced-choice artifacts on non-reasoning models.
- **Production round-trip evals & validator-retry vs steered-retry**:
  unclaimed; nothing in 2024–2026 evaluates steering in a serving loop
  against a validator baseline.

Weakest pre-registered claim: **H1** (total-injected-mass law) — not
preempted, but two lines of evidence suggest it is at best incomplete.
Designed for accordingly (threat 5).

## 2026-07-25 — extraction-data automation: who writes the contrastive pairs

GitHub source inspection (repeng, steering-vectors, CAA, heretic, abliterator,
axbench, dialz, persona_vectors). Full notes in the airlock scouts' reports.

- **persona_vectors (safety-research, arXiv:2507.21509)** is the direct
  precedent for one-line-description → LLM-generated extraction data: Claude
  generates 5 pos/neg system-prompt pairs + 40 eliciting questions + a 0-100
  judge rubric per trait, then **judge-filters samples (trait score +
  coherence) before mean-diff**. We adopt this pattern (synthesize_recipe +
  judge_filter_pairs) and cite it; it is not our contribution.
- CAA's refusal/hallucination sets were GPT-4-generated (LLM-written pairs are
  established practice); axbench LLM-generates from concept labels.
- repeng: 1-3 hand persona pairs + stock-corpus splicing; its issue #77 is the
  canonical confound warning — variation must come from the concept only →
  minimal-pairs rule in our recipe prompt.
- heretic/abliterator: stock AdvBench/Alpaca corpora — viable only because
  refusal is universal; useless for app-specific behaviors.
- Positioning: our contribution stays AFTER extraction — deployment-length
  calibration, ship receipts, re-certification. Data generation is adopted
  prior art, done to persona_vectors' standard (filtered, paired).

## 2026-07-25 — "nobody tests steering in prod": adversarial verification

Web sweep briefed to REFUTE the claim. It survives with precise wording.
Near-misses to pre-empt (danger order):
1. **SKOP (arXiv 2605.06342)** — steering evaluated at 1K-16K on RULER NIAH
   (synthetic haystack, single-turn): vanilla steering formatting compliance
   collapses 88.5%→36.4% at 16K. Refutes a loose "nobody tested >8k" AND
   **independently confirms our length-degradation finding** in lab form.
   Cite as both guard and support. (Proposes key-orthogonal projections as fix
   — read before the paper.)
2. **EasySteer (2509.25175)** — steering ON vLLM with throughput/latency evals
   (no behavioral reliability, no long ctx/agents/traffic). Refutes literal
   "no serving stack"; narrow to behavioral reliability.
3. **Anthropic system cards (Sonnet 4.5 / Opus 4.6)** — steering vectors used
   inside agentic audit scaffolds (eval-awareness suppression). Eval-integrity
   instrument, not deployed product behavior w/ deployed-prompt baseline.
4. **Rakuten x Goodfire** — real production traffic, but SAE PROBES
   (monitoring), not steering. The adjacent existence proof.
5. **Golden Gate Claude** — steering served to real users; zero published eval.
FaithSteer full-PDF grep: ZERO hits for long-context/serving/agent/multi-turn
terms. Goyal & Daumé: no production/long-context component. Practitioner's Field
Guide (2026): zero concrete production deployments — motivated surveyor found
none.

**Publishable claim (hostile-reader-proof wording):** "No published work
evaluates activation steering's BEHAVIORAL RELIABILITY under production
conditions — realistic long-context agent scaffolds (existing long-context
evals are synthetic haystacks), production serving workloads (serving-stack
integrations report throughput, not reliability), a deployed-prompt baseline,
or real user traffic (the only real-traffic interpretability deployment is
SAE probing, not steering)."
