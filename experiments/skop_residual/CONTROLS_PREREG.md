# Controls round: pre-registered predictions (2026-07-28)

*Written before any measurement, per house rules. Responds to the five
referee points raised against the 2026-07-27 SKOP-residual round
(recorded below as R1–R5). GPU required for E1 probe arms and all of
E2/E3; blocked until the GPU box is up.*

## R1 → E1: random-basis projection control at matched rank

The 07-27 claim "rerouting directions carry the effect" rests on: deep
targeted cut (rank 1536/2560, 64% norm) kills the effect; shallow
targeted cut (rank 149, 95% norm) keeps it. A random cut at the same
ranks is the missing control — expected norm kept for a random rank-r
complement is sqrt(1 − r/D): ≈97% at r=149, ≈63% at r=1536, i.e.
norm-matched to v1/v0 by construction.

**Design:** `random_projection_control.py` builds 3 seeds × {rank 149,
rank 1536} random-complement projections of v3[L20] (Qwen3-4B).
Probe arms at matched *injected magnitude* (scale compensated for norm,
as in section C): baseline / v@3 / ctl@(3·‖v‖/‖ctl‖). Probe set v2
(N=24, E3) if available, else the original 6.

**Predictions (committed before data):**

| outcome | reading |
|---|---|
| random-1536 keeps suppression clearly better than v̄_v0 did (v0: 5/6 comply) across seeds | targeted deep cut removed something specific → "rerouting directions carry the effect" SUPPORTED |
| random-1536 loses the effect like v̄_v0 (≥4/6 comply) | deep-cut-kills is generic norm/content loss → claim 1 DEMOTED to "a deep cut kills the effect"; the email/FINDINGS wording gets a dated correction |
| random-149 shows coherence gain comparable to v̄_v0's section-B gain at matched magnitude | the coherence gain is a generic cut artifact, not rerouting-specific |
| random-149 loses suppression (unlike targeted v1) on ≥2 seeds | would be surprising; flags basis-size sensitivity of the probe itself |

Seeds disagreeing: report all three, majority pattern decides the
headline, disagreement itself is a finding (basis-lottery sensitivity).

## R2 → E2: fidelity build v2 (post-RoPE keys, LN/q-norm Jacobian)

The v0 build approximated the induced-query map as δq ≈ W_q·v (pre-RoPE
keys, LayerNorm and q-norm Jacobians ignored, 8 calibration prompts).
A negative under these shortcuts can be the approximation's fault.

**Design:** `skop_residual_build_v2.py` differentiates through the
model's actual pre-attention pipeline in fp32 — input RMSNorm → W_q →
per-head q-norm → RoPE at real positions — using torch.func.jvp for the
induced perturbation M_i·v (Rayleigh) and autograd VJP for the harm
basis (mean_i M_iᵀu_j over sampled positions). Keys are recomputed
post-RoPE at their true positions. Calibration set 64 prompts
(combinatorial CZ/EN, generic). Qwen3-only in this round: Gemma-4's
dual/p-RoPE (tech report, arXiv 2607.02770) needs its own rotary
handling and is deferred.

**Predictions:**

- Checkable before any generation: if cos/subspace overlap between the
  v2 and v0 harm bases is high (say principal angles mostly < 30°), the
  v0 shortcuts did not matter much and the 07-27 negative stands as-is.
- If the bases differ materially AND some v2 config keeps efficacy with
  a coherence gain the v0 sweep never showed → the 07-27 "no free
  lunch" was an approximation artifact; correct FINDINGS accordingly.
- If the bases differ materially and the sweep STILL finds no
  efficacy/tax separation → the negative strengthens (no longer
  attributable to the v0 shortcuts).

## R3 → E3: probe power

The 6-prompt probe resolves only ~40pp differences; the 07-27 sweep's
null is bounded by that. `skop_efficacy_probe.py` gains a v2 prompt set
(N=24, original 6 included unchanged for comparability;
`SKOP_PROBE_SET=v2`). Wilson 95% half-width at N=24 is ≈ ±10pp near the
extremes. Key arms to re-run at N=24: baseline, v@3, v̄_v1@3, and the E1
controls. This does not replace the 16-prompt production harness (still
the only real judge on efficacy); it narrows the pilot CIs.

## R4: framing commitment (no experiment)

The magnitude-wall claim is, and stays, exactly: *projection does not
move the wall*. Collapse at ~1.9× residual norm is not by itself
surprising and will not be framed as such.

## R5: resolved by literature check (no experiment)

Gemma-4 KV sharing is documented (Gemma 4 Technical Report,
arXiv 2607.02770: E4B = 42 layers, first 24 compute KV, final 18 reuse
it). RESULTS.md now cites it. What remains empirical here: the concrete
producer map measured in the deployed checkpoint (sliding→22, full→23)
and the config discrepancy (2 KV heads declared, k_proj emits 4×256).

## Execution order (when the GPU box is up)

1. E1 vector build (CPU-light, needs only the vector files) + E3 probe
   set — then E1 probe arms at N=24.
2. E2 build + its basis-overlap diagnostic (cheap, no generation);
   sweep only if the bases differ.
3. Re-run key 07-27 arms on probe v2 (E3).
