# Why does steering break 8B but not 4B? (2026-07-25, EXPLORATORY — the "why" paper)

The measured fact: the same intervention (mean-diff vector, decode-only, deployment
length) is **coherent on Qwen3-4B-Instruct-2507** and **collapses into echo/loop on
Qwen3-8B** (`/no_think`). Recipe format (v5) shifts the breakage mode but doesn't
fix it (see recipe_format_ablation_8b.json). So it's not the recipe. Why?

## The confound that may BE the answer — it's not 4B-vs-8B
- **8B = original `Qwen3-8B`: HYBRID dual-mode** (thinking + non-thinking, switchable).
  Post-train built around reasoning: long-CoT cold-start → reasoning-RL → fuse
  think/no-think → general RL. `/no_think` is a *learned secondary mode*.
- **4B = `Qwen3-4B-Instruct-2507`: DEDICATED single-mode instruct** (the July-2025
  split that separated Instruct from Thinking because the hybrid fusion had
  tradeoffs). No mode switch, no suppressed second mode.
So the comparison is **dedicated-instruct vs hybrid-forced-into-secondary-mode**,
not size. (Believed: no `Qwen3-8B-Instruct-2507` exists — 2507 split shipped for
4B / 30B-A3B / 235B only. VERIFY on HF; web-check was blocked.)

## Hypotheses (ranked by plausibility × testability)
1. **Mode-entangled representations (leading).** In a hybrid, "task-offering" is
   encoded *conditionally on mode*; the `/no_think` direction is entangled with the
   mode-switch machinery → the mean-diff vector is a messier, less-separable
   direction → steering along it hits coherence circuitry → breaks. This is *why*
   8B shows "lower linear separability" — because it's dual-mode, not because it's
   bigger. TEST: #6 localization diagnostic (concentrated vs smeared write).
2. **Forced non-native regime.** `/no_think` sits the model near an edge; a steering
   perturbation tips it into the low-entropy attractor (echo/loop). TEST: #11 steer
   8B WITH thinking ON — coherent-with-thinking + broken-without = confirmed.
3. **Induction/copy fallback (the echo mode specifically).** When steering suppresses
   normal generation, the model falls back to its strongest circuit — induction/copy
   (a hallmark of bigger models) — so it *parrots* the prompt. TEST: read an echo
   cell's attention; does one induction head dominate? (brainscope lens).
4. **Dose/norm dynamics (F6).** Same `scale` = bigger effective dose if 8B's residual
   norms are larger; KL grew 0.25→1.08 with context. TEST: relative-dose recompute.

## The clean experiment design — SEPARATE size from post-training
Steer the same no-task behavior on models that vary ONE axis at a time:
| model | size | post-training | isolates |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | 4B | dedicated instruct | (baseline: works) |
| **Qwen2.5-7B-Instruct** | ~7B | dedicated instruct | if it works → hybrid, not size, is the cause |
| **Llama-3.1-8B-Instruct** | 8B | dedicated instruct, diff family | if it works → 8B *size* is fine; hybrid is the cause |
| Qwen3-8B | 8B | HYBRID (`/no_think`) | the break case |
| Qwen3-8B + thinking ON (#11) | 8B | hybrid native mode | removes the forced-mode confound |
Prediction if H1/H2 right: the three *dedicated* instructs all steer clean
(regardless of size); only the *hybrid-in-/no_think* breaks; hybrid-with-thinking
recovers. That cleanly attributes the brittleness to **dual-mode post-training +
forced secondary mode**, not parameter count.

## THE CENTERPIECE EXPERIMENT (added 2026-07-25) — clean, no confound
`Qwen3-4B-Thinking-2507` exists alongside `Qwen3-4B-Instruct-2507`. Same size,
family, training generation — differ ONLY in thinking vs non-thinking. Steer the
same behavior on both:
- Instruct-4B: steers clean (known).
- Thinking-4B: ??? → if it breaks, **reasoning-oriented post-training itself makes
  models resist steering**, isolated with NO size and NO hybrid-fusion confound.
This is the article's core figure (task #14). The 8B experiments (#11 thinking-on,
hybrid vs dedicated) then separate thinking-TRAINING from hybrid-FUSION:
| model | dedicated | thinking | isolates |
|---|---|---|---|
| Qwen3-4B-Instruct-2507 | y | n | baseline (clean) |
| Qwen3-4B-Thinking-2507 | y | y | thinking-training effect (KEY) |
| Qwen3-8B (/no_think) | n (hybrid) | fused | fusion + size |
| Qwen3-8B + think on | n (hybrid) | native | forced-mode effect |

**Next-article thesis: "Steering is a casualty of reasoning training."** The models
the field races to ship (thinking models) are the ones steering can't cleanly
control — a dedicated instruct steers where its same-size thinking twin collapses
into induction-echo. Timely, novel, deployment-relevant.

## Resources
- arXiv 2506.18167 — "Understanding reasoning in thinking LLMs via steering vectors"
  (closest prior work). A dedicated deep-research pass on steering-under-thinking is
  TODO (web budget spent this session).

## Publishability
"Whether a steering vector is safe depends on the model's post-training MODE
structure, not its size — a dedicated instruct steers cleanly where a hybrid forced
into non-thinking collapses into induction-echo" is a novel, mechanistic,
deployment-relevant claim. The failure to steer 8B is the paper.

## CRITICAL CONFOUND (2026-07-25, caught by Kate): extraction/eval mode mismatch
All 8B vectors were EXTRACTED thinking-ON (Qwen3 chat-template default; no /no_think
in recipes) but EVALUATED thinking-OFF. The mean-diff cancels mode formatting, but the
behavior direction is measured in thinking-space and applied to a no-think residual —
mismatched IF representations are mode-conditional (our hypothesis). 4B (dedicated
instruct, no thinking mode) is inherently matched. => "8B doesn't steer" and "4B easier
than 8B" are BOTH confounded by extraction-mode-match, not purely by hybrid-ness.
FIX/TEST (#15): re-extract 8B v5 in matched /no_think mode, re-run. This is now a
CONTROL that must precede every 8B claim. It also elevates the mode-matching principle:
extraction mode MUST match deployment mode — a general steering-methodology finding.

## HARNESS CONFOUND caught by reading (2026-07-26): forced tool_choice kills thinking
Reading the thinking-4B s3 generations: think_len=0 on ALL cells. brainscope's
tool_choice enforcement SEEDS the tool-call opening (server.py ~line 608), so a
thinking model can't emit <think> first — the forced tool suppresses reasoning
entirely. So the forced-tool eval tests "thinking-model-forced-NOT-to-think" =
another non-native mode, NOT native thinking. (s3 forced-tool result: 4 genuine
offers + 2 <error>-block rambles = 0/6 clean — real, read-confirmed, but off-target.)

Implications:
- Kate's "don't steer the thinking" instinct → BUILT `answer_only` steering in
  brainscope (skip the <think> block, steer only the answer after </think>;
  spec `{"answer_only": true}`, gated by state["in_think"] tracked in the decode
  loop). Correct architecture — but moot under forced-tool (no think block exists).
- PROPER thinking test = eval WITHOUT forced tool_choice (let it think → answer)
  + answer_only steering + score post-</think>. Needs: redeploy brainscope to
  aorus + a no-forced-tool eval harness. TODO.
- METHODOLOGY FINDING for the article: you cannot evaluate steering on a thinking
  model with a forced tool call — the forcing suppresses the very reasoning you're
  studying. Prior steering-of-thinking work should be checked for this.
