# SKOP-residual: does key-orthogonal projection transfer to residual-stream steering — and where does the damage actually flow?

Two experiment lines, not a paper (a paper is being distilled from
them). SKOP ("Don't Lose Focus", Luo / Espinosa Zarlenga / Jamnik,
[arXiv 2605.06342](https://arxiv.org/abs/2605.06342)) is a query-space
steering method; its Limitations section leaves the residual-stream
case as future work. **Line 1** (07-27/28) builds the residual analogue
of the projection and tests whether the win transfers (it does not, at
the strength measured — see limits). **Line 2** (07-28, channel
factorization) asks the underlying question causally: *which
computational channel carries steering damage* — attention patterns
(QK), attention values (OV), or MLP/skip — via freeze arms in a
teacher-forced replay, across 5 model families, 9 vectors, multiple
doses and depths, with pre-registered hypotheses (four registered:
one confirmed, two falsified, one downgraded honestly — see
`PREREG_CHANNELS.md` and FINDINGS O/P).

## Start here

1. [`RESULTS.md`](RESULTS.md) — one-page index of every run and the
   takeaways.
2. [`PREREG_CHANNELS.md`](PREREG_CHANNELS.md) +
   [`../MECHANISM_REROUTING_VS_DOMINATION.md`](../MECHANISM_REROUTING_VS_DOMINATION.md)
   — pre-registered predictions, committed before measurement
   (git history is the receipt).
3. [`FINDINGS.md`](../../FINDINGS.md) — full narrative: sections A–G
   (projection line), K–P (divergence map, factorization, rigor round,
   self-audits and corrections).
4. The scripts below, to reproduce.

## Known limits, up front

**Line 1:** early runs were k=1 with N=6 probes (the 07-28 controls
round upgraded to N=24 with Wilson CIs); the efficacy proxy is regex +
read, not the production checker. The projection build carries stated
approximations (v0: pre-RoPE keys, no LN/q-norm Jacobian; v2 fixes
both) — and the 07-28 dose-ladder measurement gives the deeper reason
a first-order build cannot win at working dose: the residual→query map
saturates past s≈2 (FINDINGS M).
**Line 2:** damage proxy is teacher-forced KL over 48-token replays
(validated against free generation and argmax-metric re-derivation;
still a proxy); 8-bit quantization on 8B-class models; one prompt
domain + neutral fillers; freeze-band attribution corrected in
FINDINGS P-addendum (whole-band vs all-layers control: chain E).

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

Line 2 (channel factorization, 2026-07-28):

- `qk_freeze.py` — Q-path vs K-path decomposition of per-head
  divergence (post-RoPE hybrid patterns) + first frozen-attention
  patch (FINDINGS L).
- `qk_freeze2.py` — the factorization workhorse: freeze arms
  fpat/fval/fattn on a configurable band, KL + argmax damage, dose
  ladder for ‖Δq‖/s linearity; env-parametrized (model, vector, 8-bit,
  injection depth, band, arms, scales, EN/CZ prompt set).
- `analyze_qk_freeze2.py` — the only sanctioned aggregation: paired
  bootstrap CIs on every rescue share.
- `bake_mwe.py` — SKOP-style mean-diff vectors from public MWE
  datasets (H5 bridge to their behavior suite).
- `gemma_h4.py` — Gemma-4-E4B KV-share architectural prediction run
  (H4): producer-input identity check + query-channel isolation.
- `freegen_probe.py` — free-generation validation of the KL proxy
  (k=3 seeds; found the loops-vs-mute failure phenomenology).
- `benchmark_probe.py` — ARC-Challenge-300 likelihood accuracy under
  steering (utility axis; working dose is capability-free, s5+ is not).
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
