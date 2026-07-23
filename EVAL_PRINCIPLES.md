# What a magnificent steering eval looks like

Steering's problem is not that vectors don't work — it is that the evals
people trust don't measure what they think. This project's thesis is that
the eval layer is where steering breaks, so the eval layer has to be the
most rigorous thing here. Seven properties, each one falsifiable, each with
an honest self-score (2026-07). "Better than the literature" is the floor,
not the goal.

## The seven properties

1. **Deployment-faithful.** Free-form generation under real serving
   conditions (tools, forced tool-choice, context length, language) — not
   forced-choice A/B, whose position/token artifacts inflate apparent
   efficacy (Tan et al. 2024). — *strong.*
2. **Degradation-aware.** A behavior "fixed" by breaking the model is a
   miss, not a win: `miss = violation OR incoherence`. Learned the hard way
   — an optimizer once selected a point that silently switched the model
   into English rambling and scored it perfect. — *strong.*
3. **Distribution-resolved.** Report the per-sample distribution and the
   anti-steered fraction (inputs pushed the wrong way), never the mean
   alone — up to ~half of inputs can be anti-steerable on some behaviors. —
   *strong (baseline-compare).* 
4. **Classifier-validated.** The checker is audited against human labels;
   agreement (Cohen's κ) is reported and disagreements are published, not
   hidden. **An eval you have not validated against human judgment is a
   hypothesis, not a measurement.** — *WEAK — the priority gap.* Standing
   rule: hand-label ≥10% of generations per headline cell, report κ.
5. **Damage-complete.** KL on benign, plus a safety-probe battery
   (jailbreak-ASR + false-refusal), plus a random-direction control at
   matched norm — because coherence-preserving misalignment is real and KL
   cannot see it. — *partial (tiers built, real probes untested).*
6. **Mechanism-grounded.** A tier that reads what the vector did inside the
   model (per-layer cosine footprint, peak layer, teacher-forced KL), so
   behavioral change can be tied to representational change — and reports
   `null`, never a silent zero, when the instrument is absent. — *have it;
   its predictive validity is itself under test (RQ2).*
7. **Reproducible & pre-registered.** Thresholds fixed before data; every
   number regenerable from committed artifacts. — *strong.*

## Scorecard (honest)

4 strong · 1 partial · 2 weak. Ahead of every published steering eval —
and not yet magnificent. Magnificence is closing #4 first: an unvalidated
classifier invalidates every downstream number.

## How to actually run one

1. **Elicit** — a nudge that makes the *unsteered* model violate reliably.
   Baseline must fail, or there is nothing to measure (preflight checks this).
2. **Generate** — real completions, deployment conditions, greedy, N ≥ 30.
3. **Classify** — violation regex + coherence guards → miss. **Then validate
   the classifier** against hand labels on a sample.
4. **Compare** — per-sample vs the unsteered baseline → anti-steered fraction.
5. **Damage** — KL on benign + safety probes + random-direction control.
6. **Report** — bootstrap 95% CIs, checker-human κ, per-cell distributions.
   No point estimates standing alone.
7. **Read** — a human reads a sample every single time. The checker scales
   judgment; it never replaces it.

## The line that separates this from repeng and from heretic

Heretic's eval is a refusal string-match; repeng has no eval at all. Both
are fine for their purpose and neither survives contact with a real
production behavior. The difference is not cleverness — it is that a
steering eval must be able to see the ways steering *lies*: a fixed behavior
that is really a broken model, a mean that hides a bimodal tail, a proxy
that reads the reasoning trace instead of the answer, a classifier that is
itself wrong. Magnificent means measuring each of those, and validating the
measurer.
