# Pre-registration: channel-share hypotheses for chain B + Gemma run

*Written 2026-07-28 ~18:05 CEST (commit timestamp is authoritative),
committed BEFORE reading any chain-B results
(4B extra vector families; Qwen2.5-7B) and before the Gemma run exists.
Setting as in qk_freeze2.py: decode-only residual steering at ~0.56
relative depth, freeze-arm factorization on the 7-layer band above
injection, KL(clean||x) on decode rows, N=40, bootstrap CIs.*

## Context (already measured, FINDINGS M/N + rigor round)

Attention-total damage share (rescue_fattn at working dose):
Qwen3-4B ≈ 0.72 across THREE vectors (checklist, websearch, random);
Qwen3-8B ≈ 0.72 (checklist); Llama-3.1-8B ≈ 0.30–0.40 (checklist,
robust to dose level and prompt language). Pattern-vs-value split
within attention varies by vector.

## H1 — vector-invariance of the attention share (Qwen3-4B)

For v_pref_sycophant, v_refusal, v_pref_confident at s3:
rescue_fattn ∈ [0.62, 0.82] (i.e. consistent with the 0.72 constant).
**Falsifier:** any vector outside that interval.

## H2 — what sets the model constant? (Qwen2.5-7B-Instruct, inj L16)

Qwen2.5-7B separates two candidate explanations:
- **H2a (training lineage):** Qwen-family training → attention share
  HIGH, ≈0.65–0.80 (Qwen3-like).
- **H2b (qk-norm architecture):** Qwen3 has per-head q/k RMSNorm,
  Llama and Qwen2.5 do NOT. If qk-norm is what funnels residual
  perturbation into attention-carried damage, Qwen2.5 lands LOW,
  ≈0.25–0.45 (Llama-like).
We do not know which; that is the point. Either outcome is
informative; a middle result (0.45–0.65) falsifies both simple
stories.

## H3 — pattern/value split

No invariance predicted: the split varies by vector (already
falsified as a constant on 4B). Recorded for completeness.

## H4 — Gemma-4-E4B architectural prediction (run not yet built)

Documented KV sharing (Gemma 4 TR: E4B shares KV in the final 18/42
layers; producers L22/L23). With injection at L25, keys AND values of
shared layers are computed pre-injection → clean by construction.
**Predictions:** (a) on shared layers, value-channel damage ≈ 0 and
key-path ≈ 0 — attention damage is query-only; (b) hence
rescue_fpat ≈ rescue_fattn on the shared band (patterns are the whole
attention channel there). **Falsifier:** fval rescue or k-only JSD
significantly above the sham-level floor on shared layers.
