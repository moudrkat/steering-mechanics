# SKOP-residual — one-page results index (2026-07-27)

All runs pilot-grade (k=1, N=6 probes, Wilson CIs in FINDINGS). Full
narrative: FINDINGS.md sections A–E. Scripts: this directory. Raw
outputs: `results/`. Vectors: GPU box `~/hotwire-vectors/*_skopres*.pt`.

## The question

SKOP (arXiv 2605.06342) removes attention-rerouting components from
query-space steering vectors and wins efficacy+utility. Residual-stream
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

## The three takeaways so far

1. **SKOP's tension replicates in residual space** (rerouting directions
   carry the effect: deep cut kills it) — **their win does not** (no
   projection setting found that keeps efficacy and cuts the quality
   tax). Null is ahead; the 16-prompt harness is the final judge.
2. **The magnitude wall survives projection: 5/5 configurations**, 2
   models, 4 collapse flavors. High-dose collapse is governed by injected
   magnitude, not by the projected-out rerouting directions.
3. **Deployment-length norms rewrite relative doses** — short-prompt
   h-norms are polluted by massive-activation sinks (375 vs ~55 at
   L20/Qwen3-4B). Working point ≈0.7, collapse ≈1.9 of residual norm.
   (⚠ indexing note: early numbers used hidden_states[i] = output of
   block i−1; streaming rerun reports block outputs directly.)

## Architecture discovery (Gemma-4)

Layers 0–23 produce KV; layers 24–41 share it (sliding→producer 22,
full→producer 23; config claims 2 KV heads, real k_proj emits 4×256).
⇒ steering at L25 cannot touch the top half's keys OR values — within
the attention pathway the perturbation enters only through queries, so
SKOP's rerouting term is isolable exactly as in their query-space
analysis. (NOT full query-space equivalence: the MLP and skip paths
still carry the perturbation.) Also a candidate mechanistic explanation for why L25 was
the only honest optimum (injection below 24 contaminates inherited keys).
Testable: dose ladder L22 vs L25.

## Next

1. Real 16-prompt harness + checker on candidate C and v̄_v1 (needs the
   private eval stack).
2. Gemma step test s2.5→s3 with donor-key projection (in flight).
3. v3-8B (the proven 0/20 vector) + the L15@8 /no_think mode-break test.
4. N≥12 probe replication; then: Cambridge email v3 → LW post → LinkedIn.
