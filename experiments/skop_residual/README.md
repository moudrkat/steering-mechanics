# SKOP-residual: does key-orthogonal projection transfer to residual-stream steering?

An experiment, not a paper. SKOP ("Don't Lose Focus", Luo / Espinosa
Zarlenga / Jamnik, [arXiv 2605.06342](https://arxiv.org/abs/2605.06342))
is a query-space steering method; its Limitations section leaves the
residual-stream case as future work. This directory is one pilot-grade
attempt at that extension, run inside a production serving stack. No
novelty is claimed beyond building the analogue, running it, and writing
down what happened — including the parts that argue against our own
earlier readings.

## Start here

1. [`RESULTS.md`](RESULTS.md) — one-page index of every run and the
   three takeaways so far.
2. [`../MECHANISM_REROUTING_VS_DOMINATION.md`](../MECHANISM_REROUTING_VS_DOMINATION.md)
   — the pre-registered predictions, committed before any measurement
   (git history is the receipt).
3. [`FINDINGS.md`](../../FINDINGS.md) sections A–G (repo root) — the full
   narrative, Wilson CIs, and the self-caught confounds (thinking-mode
   bug, sink-token norm pollution, baseline language bleed).
4. The scripts below, to reproduce.

## Known limits, up front

All runs are k=1 with N=6 probes; the efficacy proxy is regex + read,
not the production checker, and its own CI analysis says most arm
differences are below its resolution. The projection build carries
v0 approximations: pre-RoPE keys, LN/q-norm Jacobian ignored in the
induced-query map (δq ≈ W_q v), 8 calibration prompts vs SKOP's 250+.
A failure to reproduce SKOP's win here can therefore be the
approximation's fault rather than the method's — the negative is
reported at exactly that strength. A random-basis projection control at
matched rank (does cutting *any* rank-149 / rank-1536 subspace behave
the same?) is queued and is required before the "rerouting directions
carry the effect" reading can be considered established.

## Pieces

- `skop_residual_build.py` — builds the projected vector: calibration
  forward passes (focus/tail sets at τ=0.8 from per-head attention),
  per-head second moment of key-differences Σ_Δk, risk heads by Rayleigh
  quotient, projection of the residual vector orthogonal to
  `W_q_headᵀ·u_i` directions of the top-γ eigenvectors.
- `skop_efficacy_probe.py` — direct-ask CZ prompts (6, or N=24 with
  `SKOP_PROBE_SET=v2`); regex violation proxy + coherence heuristics
  (rep4, uniq). NOT the production eval — a pilot instrument; real
  verdicts belong to the 16-prompt harness with the checker.
- `CONTROLS_PREREG.md` — pre-registered second round answering the
  referee points against the 07-27 runs: random-basis projection
  control at matched rank (`random_projection_control.py`), fidelity
  build v2 with post-RoPE keys + LN/q-norm Jacobian + 64-prompt
  calibration (`skop_residual_build_v2.py`, untested until the GPU box
  is up), probe power N=24. Chain: `runchain_controls.sh`.
- `skop_ab_test.py` — coherence A/B (baseline / v / v̄) at chosen scales.
- `h_norms_12k.py` — deployment-length mean ‖h[L]‖ (base model, no LM
  head — full-sequence logits alone OOM a 16GB card). Exposes the
  sink-token pollution of short-prompt norms (375 → 55 at L20/Qwen3-4B).
- `runchain_*.sh`, `sweep.sh` — the exact configurations that produced
  the recorded runs (env knobs: SKOP_MODEL/VEC/INJ/OUT/RISK/GAMMA/PCAP/
  SCALES/ARMS/NOTHINK/WQ_SRC).
- `results/` — score-only JSONs from the recorded runs (no generated
  text; see the repo's privacy policy in the root README).

## Environment notes (single 16GB GPU, RTX 4070 Ti SUPER)

- venv: the vLLM serving venv + `accelerate` + `kernels==0.15.2`
  (needed for FP8 checkpoints).
- Qwen3-8B runs via the FP8 checkpoint; its packed weights can't be read
  naively — `SKOP_WQ_SRC` points the projector at the bf16 checkpoint's
  safetensors instead.
- Gemma-4-E4B does NOT fit bf16 on 16GB (≈14.7GB + activations) — runs
  used the 8-bit route.
- HF cache shim (`~/hf-cache2` with model symlinks) works around
  root-owned entries left in the main cache by docker extractions.
- On the GPU box, outputs land in `~/skop_residual/` (mirrored into
  `results/` here) and projected vectors in
  `~/hotwire-vectors/*_skopres*.pt`.

No private data: calibration and probe prompts are generic and live
inside the scripts.

## Citing the method

The method extended here is SKOP:

> Haoyan Luo, Mateo Espinosa Zarlenga, Mateja Jamnik. *Don't Lose
> Focus: Activation Steering via Key-Orthogonal Projections.*
> arXiv:2605.06342, 2026. https://arxiv.org/abs/2605.06342

```bibtex
@misc{luo2026dontlosefocus,
  title         = {Don't Lose Focus: Activation Steering via Key-Orthogonal Projections},
  author        = {Luo, Haoyan and Espinosa Zarlenga, Mateo and Jamnik, Mateja},
  year          = {2026},
  eprint        = {2605.06342},
  archivePrefix = {arXiv},
}
```
