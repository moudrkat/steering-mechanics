# Rerouting vs domination: which mechanism breaks the steered model?

*Drafted 2026-07-27, before any measurement. Predictions written first,
per house rules. GPU required — blocked until the GPU box (or a cloud
GPU) is available.*

## The question

Two published-ish explanations exist for why steering at damaging doses
destroys model quality:

- **H-rerouting** (SKOP, Luo/Espinosa Zarlenga/Jamnik, arXiv 2605.06342):
  the added vector corrupts query–key matching, so heads re-route
  attention away from critical tokens. Damage = the model stops *looking*
  at the right places. Their fix (key-orthogonal projection) works by
  protecting attention patterns.
- **H-domination** (this repo, FINDINGS §7b): the vector's magnitude
  dominates the residual stream; behavior flip and fluency collapse are
  one event (a step at the same dose), extraction-independent. Damage =
  everything downstream is computed on a hijacked residual, regardless
  of where attention looks.

These may compete or compose (rerouting could be *how* domination
manifests at the attention level). Three measurements distinguish them.

## Predictions (committed before data)

| Observation | H-rerouting predicts | H-domination predicts |
|---|---|---|
| Per-head attention-pattern divergence (clean vs steered forced pass, same tokens) as dose rises s2→s8 | Divergence JUMPS at the damage threshold (patterns visibly re-route) | Divergence stays low/grows smoothly; damage jumps anyway |
| Frozen-attention patch (steered values, clean attention patterns) | Fluency largely RESCUED | Damage persists (value/MLP stream carries it) |
| Key-orthogonal projection of our vector (SKOP's method) at s3–s8 | Fluency cliff moves UP ≥1 dose unit; behavior efficacy retained | Cliff barely moves; or efficacy dies with the projection |

Any mixed outcome is informative: e.g. rescue by frozen-attention at
short context but not at 12k would mean the mechanism is
context-length-dependent — which neither paper currently claims.

## Method sketch

1. **Attention divergence (cheapest, first).** Teacher-forced clean vs
   steered pass, identical tokens, Qwen3-8B L18–24, doses s2/s3/s5/s8.
   Per head: Jensen–Shannon divergence between attention rows at matched
   positions. Short context first (attention maps manageable); the 12k
   tier only for the winning heads. brainscope already captures
   attention; needs a JS-divergence readout on the forced pass.
2. **Frozen-attention patch (decisive).** Steered forward pass but
   attention probabilities copied from the clean pass. This is the
   missing control already noted in FINDINGS ("a frozen-attention patch
   would") — needs a brainscope extension.
3. **SKOP replication in our stack.** Implement key-orthogonal
   projection (project v ⊥ to key directions of protected tokens) in
   hidden-directions; run the Gemma-style dose ladder (s2/s2.5/s3/s4)
   on the deployment eval. If the Czech fluency window opens where none
   existed (FINDINGS §7), that is deployment-grade evidence for SKOP —
   and the direct content of the planned author contact.

## Existing evidence worth re-reading first

- Component attribution (FINDINGS): at L21 attention *amplifies* the
  vector (+1.23) while the MLP writes against it (−3.50) — a value-stream
  story, mildly pro-domination.
- Head saturation result: opposing heads saturate rather than escalate —
  softmax ceiling, i.e. attention *does* change under dose. Mildly
  pro-rerouting. The two together are exactly why this needs a clean
  test.
- sixteen-voices (different intervention, same instrumentation): LoRA
  style adapters changed what heads *output* (value projections), not
  where they *attend* — precedent that quality-relevant change can happen
  without pattern re-routing.

## Blockers & order

GPU required for all three. Option A: revive the GPU box. Option B: the
rented-cloud-GPU serving path (documented in the team testkit). Order:
(1) attention divergence, (2) frozen-attention, (3) SKOP replication —
each step's result decides whether the next is worth running.
