# Chain K: conceptual replication of SKOP, then their fix on our vectors (2026-07-29)

*Written before any measurement, per house rules. Post-freeze
constructive arm; does not touch the frozen paper plan (results land in
the appendix, marked post-freeze). Motivation: the paper challenges the
generality of SKOP's account (arXiv 2605.06342) without having re-run
their anchor result. SKOP released no code; this is a replication from
the paper's description (their Appendix D.1). Where their text fixes a
constant we use it; where it does not, the choice is frozen here.*

## Their method, as we implement it

Per-head query-space steering: q_i^(l,h) ← q_i^(l,h) + λ·r̃_q^(l,h),
applied at every layer, all token positions (their protocol — NOT our
decode-only convention; fidelity beats house style in a replication).
Vectors: per-head mean difference of query activations on
positive/negative behavior examples. KOP projection: per head, focus
tokens = those holding attention mass at τ_high = 0.8 on calibration
data, tail = complement; Σ_Δk = covariance of focus-minus-tail key
differences; project the steering direction off the top-p eigenvectors
of Σ_Δk; apply only to the top 20% of heads by their Rayleigh-quotient
risk score. If Appendix D.1 fixes p, use theirs and record it; if not,
frozen fallback: smallest p capturing 90% of Σ_Δk variance per head.

*Amendment (2026-07-29, before any data): the paper does not state
whether steering and keys live pre- or post-RoPE. Frozen
interpretation: both in the pre-RoPE frame (steering injected at
q_proj output, Σ_Δk built from k_proj output) — one consistent frame,
the same injection point our chain-I instrument validated. Exact
key-orthogonality under position-dependent rotation is not preserved
cross-position in either frame and their text gives no handling, so
this is recorded as an interpretation choice, not a deviation. GQA
note (Llama-3.1 has 8 KV heads): focus/tail sets are selected per
query head from its own attention pattern; key differences are taken
in the shared KV head's key space; projections applied per query
head.*

## K1 — replication in their setting

- **Model:** Llama-3.1-8B-Instruct (their model), 8-bit (our hardware;
  known deviation — the bf16/8-bit agreement measured on Qwen3-4B is
  the only precision bridge we have).
- **Behaviors:** power-seeking and corrigibility (MWE, public; overlap
  with their suite). TruthfulQA and wealth-seeking out of scope.
- **Vector build:** per-head q-space mean-diff from the same MWE data
  `bake_mwe.py` already uses.
- **Calibration set:** 400 prompts, 100 each GSM8K / Alpaca / PIQA /
  NarrativeQA (their mixture at 1/10 scale). Frozen fallback if the
  box cannot fetch datasets: 64 combinatorial CZ/EN generic + 40
  templated neutral + 296 ARC-train questions, recorded as deviation.
- **Operating point:** λ* = smallest λ on the vanilla-steering ladder
  (λ ∈ {1, 2, 4, 8, 16}) giving ≥ 30 pp shift in matched-choice
  probability on held-out MWE questions. All arm comparisons at λ*.
- **Efficacy metric:** held-out MWE A/B logprob shift (deviation:
  their LLM judge is out of scope). **Utility:** ARC-Challenge 300
  likelihood-scored (shared with their suite) + teacher-forced KL and
  token-uniqueness on the 40 neutral topics (house axes).
- **Arms:** baseline / vanilla q-steer @ λ ladder / q-steer+KOP @ λ*.
  One seed, greedy (house limitation, recorded).

**Frozen verdicts (per behavior; headline requires both to agree,
disagreement is itself reported):**

| tier | criterion |
|---|---|
| STRONG replication | KOP keeps ≥ 95% of vanilla shift AND ≥ 5× less utility damage (their claim) |
| REPLICATED | ≥ 80% of shift kept AND ≥ 2× less damage |
| PARTIAL | ≥ 50% kept AND > 1× |
| FAILED | < 50% kept OR no damage reduction |

Damage axis: ARC drop at λ*; if vanilla's ARC drop < 2 pp (axis has no
signal), TF-KL is the pre-declared fallback axis. A FAILED verdict is
reported as *our implementation of their description fails to
reproduce*, not as *their result is wrong* — no reference code exists
to distinguish the two.

## K2 — their fix on our vectors

- **K2a (runs regardless of K1):** Qwen3-4B chain-I setting, the ×4
  q-image arm (viol 12/24, KL 0.40; postfreeze_chainI.json). Add
  per-head KOP projection (Σ_Δk from our calibration prompts: 24 CZ
  tasks + 40 neutral). Predictions:
  - KOP keeps behavior (viol ≤ 14/24) and cuts KL ≥ 2× (≤ 0.20) →
    their fix transfers to channel-matched steering on the
    pattern-led model.
  - KOP kills behavior (viol ≥ 16/24, i.e. within 3 of baseline 19) →
    on this model the *behavior itself* lives in the focus-token
    attention shifts KOP protects; effect and damage share the
    subspace — SKOP's own tension, reproduced in q-space (the
    residual-space analogue of which is FINDINGS E).
  - KL uncut (> 0.30) with behavior kept → projection ineffective on
    this covariance structure; implementation scrutiny before any
    stronger reading.
  - Qwen3-8B gets no K2 arm: chain I shows nothing in the q-channel
    to protect (null at ×1–×4).
- **K2b (conditional on K1 ≥ PARTIAL):** deployment-grade behavior on
  Llama — per-head q-space mean-diff vector for the no-task behavior
  built from the private extraction contrasts (box-side), vanilla vs
  KOP at the λ* analogue; efficacy = CZ probe regex (N=24), utility =
  ARC + TF-KL. Question: does the fix survive a production behavior,
  not just MWE probes? If K1 FAILED, K2b is skipped (a fix that does
  not reproduce in its home setting cannot be meaningfully applied
  elsewhere).

## Execution order

1. Blocked until the chain J final judge releases the GPU (and
   brainscope is stopped — coordinate, never assume).
2. K1 vector + projection build (cheap) → sanity: scale-0 arms exact.
3. K1 λ ladder + arms; K2a; K2b conditional.

Scores-only JSONs → `results/postfreeze_chainK*.json` via scrub;
generations stay box-side.
