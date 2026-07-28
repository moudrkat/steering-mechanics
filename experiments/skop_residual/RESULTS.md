# SKOP-residual — one-page results index (2026-07-27, factorization line added 07-28)

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

## Channel-factorization line (07-28; FINDINGS K–P; all decode-only, freeze arms exact at s=0)

| run | model(s) | key result | data |
|---|---|---|---|
| mega battery | Qwen3-4B, N=64, 6 doses, 3 arms + sham | attention barely moves except L21 h18/h11; random vector lights the SAME heads ~4× weaker; sham 2e-5 | brainscope notes (gitignored), FINDINGS K |
| qk_freeze | 4B | h18 divergence 79–94% query-carried; frozen-attention rescues only ~half | qk_freeze_report.json |
| qk_freeze2 | 4B / 8B / Llama | attention share 72/72/39%; patterns-vs-values FLIPS 4B↔8B; Llama MLP-led (all controls) | qk_freeze2_{4b,8b,llama}_report.json |
| rigor round | + Qwen2.5-7B, 6 vectors on 4B | H1 falsified (share 11–72% by vector), H2b falsified (qk-norm not the mechanism); loops-vs-mute failure modes | rigor_*.json |
| final round | + Gemma-4-E4B, MWE vectors, depths | H5 confirmed (Llama attention-minority for MWE too), H6 downgraded (underpowered), H4 verified (Gemma query-channel 16–26%, KV-share code-verified) | final_*.json, gemma_h4 |
| ARC-300 axis | all 4 dense models | working dose capability-free (±1pp); s8 → ~45% | arc_*.json (private mirror) |

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
4. **(07-28) There is no simple law for where steering damage flows.**
   The attention-carried share is a function of the (model, vector,
   dose, layer) quadruple — 72% to 11% across ordinary pairs, opposite
   dose trends between models, patterns↔values flip on one model.
   Pre-registered candidate invariants died on contact (PREREG_CHANNELS).
   What is robust: Llama-3.1 attention-minority (3 vectors, all
   controls), the residual→query map saturating past s≈2 (why
   first-order projections miss at working dose), and a
   capability-free working point on ARC with a steep wall above it.

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

-2. **Deployment-stack concordance check (nice-to-have for the paper):**
   one working-dose point — same vector, same prompts — measured both
   transformers-side (this directory's scripts) and through the
   production serving path (vLLM + steering patch); efficacy +
   coherence agreement buys the sentence "the two stacks agree at the
   working point". Cheap; do before submission if time allows.
-1. **Value-side projection (post-freeze, the constructive arm):** the
   channel factorization (FINDINGS M/O) says Qwen3-8B damage is
   VALUE-led — build the OV-space analogue of SKOP (project v to
   minimize induced value perturbation at risk heads) and test whether
   a channel-MATCHED fix succeeds where the pattern-matched one failed.
   If yes: "measure, then medicate accordingly" gets its payoff, and
   it is the natural joint-work proposal for the SKOP authors.
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
