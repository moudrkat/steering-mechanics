# Paper outline — "How steering evaluations lie"

*Working title:* **How steering evaluations lie: a catalog of failure modes,
and the discipline that catches them.** (Alt: "Steering with receipts.")

The pre-registered plan (RESEARCH_PLAN.md) had four RQs. The work made clear
where the signal is: **RQ2 (eval validity) is the paper.** RQ3's window result
is strong support; RQ1's dose/mass hypothesis is reframed (relative dose) and
demoted. This outline is the paper the evidence actually supports — every
claim below is already in FINDINGS.md with data.

## Thesis (one sentence)

Steering vectors are evaluated with metrics that fail silently in systematic,
enumerable ways; a bare efficacy number is a hypothesis, not a measurement,
and we give the catalog of failures plus the minimal discipline that catches
each one — demonstrated on a real production behavior across four models.

## 1. The catalog of failure modes (the core)

Each is a section: the failure, a concrete instance with data, why the naive
metric misses it, the fix. All evidence exists in FINDINGS.md.

| # | Failure mode | Evidence (FINDINGS section) | Status |
|---|---|---|---|
| F1 | **Coherence-blind efficacy** scores a broken model "perfect" (0 violations because gibberish offers no task) | "argmax didn't just lie, it broke the model"; "two collapse modes" | ✅ solid |
| F2 | **Optimizer converges to the broken pocket** — TPE argmax was model-idiosyncratic and coherence-blind-optimal | "the window transfers, the argmax lies" | ✅ solid |
| F3 | **Disposition proxies contaminated by reasoning** — read the think-block, not the answer; flatline silently without a fitted lens | "proxies go blind on reasoning models" | ✅ solid |
| F4 | **Eval choice drives the optimum** — the found (layer, scale) depends on which eval you calibrate against | "auto-calibration eval choice drives the answer"; "efficacy proxy is the weak link" | ✅ solid |
| F5 | **Emergent-misalignment false positives** — steering flagged +0.3 harmful compliance on 2/2 models via TWO different metric bugs (vocabulary drift; thinking-model truncation); model refused throughout | "TWO false positives, two metric failures" | ✅ solid (thinking-model clean re-run pending) |
| F6 | **Raw scale is an unfalsifiable dose coordinate** — confounded by vector norm AND residual-stream norm; only dimensionless relative dose compares across models | "dose axis is under-specified" | ✅ argument solid; relative-dose recompute pending |
| F7 | **The classifier itself is unvalidated** — a mis-scoring checker invalidates every downstream number | checker validation κ=1.0 (results/checker_validation.json) | ✅ method shown; needs a hard-behavior case |
| F8 | **The mean hides the tail** — anti-steerability (inputs steered the wrong way) invisible in aggregate efficacy | baseline_compare / literature (Tan, Braun) | ◑ machinery built; per-behavior numbers thin |

## 2. The discipline (the constructive half)

The minimal set of practices that catch F1–F8 — this is what the tool ships
and what the field lacks:

- **miss = violation OR incoherence** (coherence-aware objective) → F1, F2
- **read a sample every run; the checker scales judgment, never replaces it**
  → caught F5 twice when metrics + guards both passed
- **validate the classifier against human labels (Cohen's κ)** → F7
- **relative dose `‖scale·V‖/‖h‖`, never raw scale, for cross-model claims**
  → F6
- **report the per-sample distribution + anti-steered fraction, not the mean**
  → F8
- **proxy preflight: confirm the proxy emits signal before trusting it;
  behavioral eval by default; meaning-based judge (not regex) for safety**
  → F3, F5
- **official benchmarks (StrongREJECT/XSTest/MMLU) as ground truth for the
  standardized axes** → credibility

## 3. Supporting result — the window transfers, the dose must be relative

Not the centerpiece, but real and it grounds the catalog in a positive claim:
the steering *window* transfers across models/families as a **fraction of
depth** (Llama-3.1-8B 0.44–0.62 overlaps the Qwen window; different family,
different layer count). The *dose* half must be reported in relative units
(F6) — "scale doesn't transfer" was itself a metric-lie (deviation 1).
[needs: relative-dose recompute + Qwen3-4B/2.5-7B completion.]

## 4. Setup / credibility

Pre-registered (RESEARCH_PLAN.md, frozen, with a dated deviation), measured on
a real production behavior (task-offering suppression) with the deployed
prompt-steering loop as the honest baseline, open pipeline
(hidden-directions / brainscope / hotwire-vllm on PyPI), everything
reproducible from committed result JSONs.

## 5. Honest limitations

One behavior family (+ two nulls); models to ~9B; the relative-dose recompute
and one hard-behavior classifier validation are pending; SteerBench is a seed,
not yet a community benchmark.

## What's DONE vs TODO to ship

**Done (evidence in hand):** F1–F5, F7 method, the transfer window (2 models),
checker validation once, the dose-axis argument.
**TODO (needs aorus):** relative-dose recompute (F6 numbers), Qwen3-4B/2.5-7B
sweeps, clean thinking-model EM re-run, a hard-behavior κ.
**TODO (no GPU):** write §1 and §2 — the catalog and discipline are fully
supported *now*.


## Methodology that distinguishes this from prior steering benchmarks

- **Curves, not points.** Prior benchmarks (AxBench, FaithSteer) compare each
  method at a single calibrated operating point. We compare the whole
  dose-response curve — because a method can win at its calibrated point and be
  fragile just off it, which a single point hides.
- **Relative dose, not raw alpha.** Dose = `||scale*V[L]|| / ||h[L]||`
  (dimensionless, cross-model comparable). Raw alpha is confounded by vector
  norm and residual-stream norm (Deviation 1) — it is not a valid comparison
  axis, and prior benchmarks report it anyway.
- **Ship-range, not win/lose.** The output of a comparison is not "method A is
  better" but "A is safe to ship in relative-dose range [x, y]; outside it,
  coherence/safety break" — the deployment question, not the leaderboard one.

## Venue

AF/LW post first (the catalog is a great post), then a workshop paper
(BlackboxNLP / a NeurIPS-ICLR interp workshop). SteerBench is the follow-up
that turns the discipline into a benchmark.
