# SKOP-residual — one-page results index (2026-07-27)

All runs pilot-grade (k=1, N=6 probes, Wilson CIs in FINDINGS). Full
narrative: FINDINGS.md sections A–E. Scripts: this directory. Raw
outputs: `results/`. Vectors: GPU box `~/hotwire-vectors/*_skopres*.pt`.

## The question

SKOP (Luo, Espinosa Zarlenga & Jamnik; arXiv 2605.06342) removes
attention-rerouting components from query-space steering vectors and
wins efficacy+utility. Residual-stream
steering is their stated future work. Does the trick transfer?

## Runs

| run | model / vector | projection | key result | data |
|---|---|---|---|---|
| A/B coherence | Qwen3-4B / v3 @L20 | v0 (64% norm) | cleaner Czech at matched magnitude — but see efficacy | ab_results_qwen_skop_v0.json |
| efficacy v0 | Qwen3-4B / v3 | v0 | **effect mostly lost** (5/6 comply) | efficacy_v0.json |
| efficacy v1 | Qwen3-4B / v3 | v1 (95% norm) | **effect kept** (1/6 vs baseline 6/6 — CI-separated) | efficacy_v1.json |
| sweep A–D | Qwen3-4B / v3 | 88–94% norm | **no config beats original on efficacy×coherence** | sweep_*.json |
| 8B v4 | Qwen3-8B-FP8 / v4 | v1-style (97%) | replicates campaign's v4 honest negative | efficacy_8b_v4.json |
| 8B v5nt | Qwen3-8B-FP8 / v5_nothink | v1-style (97%) | partial efficacy preserved; v–v̄ below probe resolution | efficacy_8b_v5nt.json |
| Gemma | gemma-4-E4B 8bit / v3 @L25 | v1-style (99% norm) | suppression preserved; fluency directionally better; 12k step test pending | efficacy_gemma.json |
| 8B v3 @L20 | Qwen3-8B-FP8 / v3 (proven) | v1-style (97%) | **projection cost half the suppression** (0/6→3/6); wall 5/5 | efficacy_8b_v3.json |
| 8B v3 @L15@8 | Qwen3-8B-FP8 / v3 | v1-style (98%) | historic collapse = MODE failure; hard no-think removes it; soft-switch test queued | modebreak_8b_L15.json |
| controls E3 (07-28) | Qwen3-4B / v3 | ref arms, probe N=24 | baseline 18/24 vs v@3 8/24 **CI-separated**; v̄_v1 8/24 preserves | efficacy_probe_v2_refarms.json |
| controls E1 (07-28) | Qwen3-4B / v3 | random r149/r1536 ×3 seeds | random deep cuts KEEP effect (9–10/24) vs targeted v̄_v0 13/24 — specificity directional (p≈.2) | efficacy_randctl_*.json, efficacy_v0_n24.json |
| controls E2 (07-28) | Qwen3-4B / v3 | faithful map (98% norm) | removed comps ⊥ v0 (cos .17) yet **no win** (11/24, no coherence gain) — negative strengthened | efficacy_v2_n24.json, diag_v2.json |

## The three takeaways so far

1. **SKOP's tension replicates in residual space — with a 07-28
   correction:** the targeted deep cut *weakens* the effect (13/24 vs
   baseline 18/24 at N=24), it does not kill it (the N=6 "5/6 comply"
   read overstated). Random matched-rank cuts keep the effect (9–10/24,
   3/3 seeds) — specificity is directional, not yet CI-separated.
   **Their win still does not transfer**, and after E2 this is no
   longer attributable to build shortcuts: the faithful exact-map
   projection (post-RoPE, LN/q-norm Jacobian, 98% norm) shows no
   efficacy/tax separation either. The 16-prompt harness is the final
   judge.
2. **The magnitude wall survives projection: 5/5 configurations**, 2
   models, 4 collapse flavors. High-dose collapse is governed by injected
   magnitude, not by the projected-out rerouting directions.
3. **Deployment-length norms rewrite relative doses** — short-prompt
   h-norms are polluted by massive-activation sinks (375 vs ~55 at
   L20/Qwen3-4B). Working point ≈0.7, collapse ≈1.9 of residual norm.
   (⚠ indexing note: early numbers used hidden_states[i] = output of
   block i−1; streaming rerun reports block outputs directly.)

## Architecture note (Gemma-4 KV sharing)

KV sharing in upper layers is a documented Gemma-4 design feature
(Gemma 4 Technical Report, arXiv 2607.02770: E4B shares KV in the final
18 of 42 layers), not a finding of ours; what is measured here is the
concrete producer map in the deployed checkpoint and its consequence
for steering. Layers 0–23 produce KV; layers 24–41
share it (sliding→producer 22, full→producer 23; config claims 2 KV
heads, real k_proj emits 4×256).
⇒ steering at L25 cannot touch the top half's keys OR values — within
the attention pathway the perturbation enters only through queries, so
SKOP's rerouting term is isolable exactly as in their query-space
analysis. (NOT full query-space equivalence: the MLP and skip paths
still carry the perturbation.) Also a candidate mechanistic explanation for why L25 was
the only honest optimum (injection below 24 contaminates inherited keys).
Testable: dose ladder L22 vs L25.

## Next

0. Controls round — pre-registered in `CONTROLS_PREREG.md`: random-basis
   projection control at matched rank (if a random cut reproduces the
   keep/kill pattern, "rerouting directions carry the effect" loses its
   specificity claim), fidelity build v2 (post-RoPE keys, LN/q-norm
   Jacobian, 64-prompt calibration), probe N=24.
1. Real 16-prompt harness + checker on candidate C and v̄_v1 (needs the
   private eval stack).
2. Gemma step test s2.5→s3 with donor-key projection (in flight).
3. v3-8B (the proven 0/20 vector) + the L15@8 /no_think mode-break test.
4. N≥12 probe replication; then write-up.
