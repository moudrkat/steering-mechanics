# SKOP-residual: scripts for the rerouting-vs-domination experiments

Reproducibility home for the 2026-07-27 runs recorded in FINDINGS.md
(sections A–D) and pre-registered in
`../MECHANISM_REROUTING_VS_DOMINATION.md`. Everything runs on a single
16GB GPU next to (or instead of) the serving stack; no private data —
calibration and probe prompts are generic and live inside the scripts.

## Pieces

- `skop_residual_build.py` — builds the projected vector: calibration
  forward passes (focus/tail sets at τ=0.8 from per-head attention),
  per-head second moment of key-differences Σ_Δk, risk heads by Rayleigh
  quotient, projection of the residual vector orthogonal to
  `W_q_headᵀ·u_i` directions of the top-γ eigenvectors.
  **v0 approximations (documented, pilot-grade):** pre-RoPE keys, LN/
  q-norm Jacobian ignored in the induced-query map (δq ≈ W_q v), small
  calibration set (8 prompts) vs SKOP's 250+.
- `skop_efficacy_probe.py` — 6 direct-ask CZ prompts; regex violation
  proxy + coherence heuristics (rep4, uniq). NOT the production eval —
  a k=1 pilot instrument; the real verdicts belong to the 16-prompt
  harness with the checker.
- `skop_ab_test.py` — coherence A/B (baseline / v / v̄) at chosen scales.
- `h_norms_12k.py` — deployment-length mean ‖h[L]‖ (base model, no LM
  head — full-sequence logits alone OOM a 16GB card). Exposes the
  sink-token pollution of short-prompt norms (375 → 55 at L20/Qwen3-4B).
- `runchain_*.sh`, `sweep.sh` — the exact configurations that produced
  the recorded runs (env knobs: SKOP_MODEL/VEC/INJ/OUT/RISK/GAMMA/PCAP/
  SCALES/ARMS/NOTHINK/WQ_SRC).

## Environment notes (GIGABYTE box, RTX 4070 Ti SUPER 16GB)

- venv: the vLLM serving venv + `accelerate` + `kernels==0.15.2`
  (installed 2026-07-27 via ensurepip; needed for FP8 checkpoints).
- Qwen3-8B runs via the FP8 checkpoint; its packed weights can't be read
  naively — `SKOP_WQ_SRC` points the projector at the bf16 checkpoint's
  safetensors instead.
- Gemma-4-E4B does NOT fit bf16 on 16GB (≈14.7GB + activations) —
  pending 8-bit route.
- HF cache shim (`~/hf-cache2` with model symlinks) works around
  root-owned entries left in the main cache by docker extractions.
- Outputs land in `~/skop_residual/` (JSONs mirrored into `paper/` here)
  and projected vectors in `~/hotwire-vectors/*_skopres*.pt`.

## Provenance

Method mirrored from "Don't Lose Focus: Activation Steering via
Key-Orthogonal Projections" (Luo, Espinosa Zarlenga, Jamnik; arXiv
2605.06342) — their method is query-space; the residual-space analogue
here is the extension their Limitations leave as future work. Results
and honest caveats: FINDINGS.md 2026-07-27 sections, Wilson CIs included.
