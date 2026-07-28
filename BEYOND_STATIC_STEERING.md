# Beyond static additive steering (2026-07-25)

> **EXPLORATORY TRACK** — hypothesis-generating, NOT pre-registered. Findings
> that survive get promoted to a dated confirmatory RQ and re-tested clean.
> The frozen RESEARCH_PLAN RQs are unchanged.

Motivated by a measured failure, not speculation. The 8B ship hunt showed that
for a behavior that is not cleanly linear, **static additive steering
(`h += scale·V`) couples suppression and coherence-breakage**: no dose
suppresses the behavior without breaking the model (the offer-band and the
break-band overlap at every layer tried). Static `scale·V` has three built-in
flaws, and each points at a better primitive:

| flaw of `h += scale·V` | consequence (measured) | fix |
|---|---|---|
| constant push, ignores context | overdose cliff; damage grows with context length (KL 0.25→1.08 @12k) | **#2 closed-loop dose control** |
| unconditional within steered spans | coherence tax paid on *every* token, even when the model was behaving | **#1 probe-gated conditional steering** |
| one direction does suppression AND damage | can't optimize coherence separately | **#3 learned low-rank intervention** |

These are extensions of the frozen research plan (see RESEARCH_PLAN dated
extension), evaluated with the SAME discipline as everything else: deployment
length, temperature 1, hard/soft triage, frozen ship gate.

---

## #1 Probe-gated conditional steering — BUILT (brainscope)

**Idea:** the steering direction is its own probe. Apply `scale·V` only at
positions where the activation's projection onto the direction, `coef = h·â`,
exceeds a threshold `gate` — i.e., only where the behavior is *forming*. Where
the model was already behaving (`coef ≤ gate`), the activation is untouched, so
coherent generation pays **zero** steering tax. Decouples suppression from the
blanket coherence cost.

**Status:** implemented in `brainscope/server.py` (`_install_steer_hooks`,
`gate` param; spec dialect `{"id","layer","scale","gate":τ}`). Compiles.
**Calibration (after v5):** pick `gate` from the distribution of `coef` on
held-out behavior-present vs behavior-absent activations (a clean separation
threshold). Orient V so `coef` is high when the behavior is present (sign-probe).
**Test harness:** `campaign/gated_steer.py` (sweeps `gate` × `scale`, same ship
metric). Prediction: hard-breakage collapses vs ungated at equal suppression,
because clean tokens are never touched.

## #2 Closed-loop dose control — DESIGNED (skeleton)

**Idea:** monitor a coherence signal *during* generation (running n-gram
repetition rate / token entropy) and reduce `scale` when it rises — you cannot
fall off the overdose cliff if the controller backs off as damage climbs.
Directly attacks the length-sensitivity (the cliff *moves* with context; a
controller tracks it). Cites the plan's feedback-steering reading (2506.18831
PID, 2510.04309, LQR 2604.19018).

**Status:** `campaign/closed_loop.py` skeleton — chunked client-side loop:
generate a short span → measure degeneracy → if over threshold, roll back that
span and continue at reduced scale. Needs a GPU debug pass (after v5).
**Metric:** does closed-loop keep hard-breakage bounded across 512→13k context
at a fixed *target* suppression, where static scale collapses?

## #3 Learned low-rank intervention (ReFT-style) — DESIGNED (skeleton)

**Idea:** replace the rigid single direction with a small **trained** low-rank
edit on layer L, `h ← h + R(h)`, optimized against BOTH objectives a static
vector cannot balance: efficacy (suppress the behavior) AND coherence (low KL
on benign + no degeneracy). For behaviors that are not cleanly linear (8B's),
a learned nonlinear/low-rank map can thread efficacy and coherence where a mean
-difference vector cannot. Cites AxBench ReFT-r1.

**Status:** `campaign/train_reft.py` skeleton — LoReFT-style module on layer L,
combined loss (behavior-suppression + benign-KL + repetition penalty), trained
on the same contrastive recipe data + benign set. A real training job: needs
GPU + a debug pass (after v5). Higher effort, highest ceiling.

---

## #4 Head / neuron-level intervention — DIAGNOSTIC-GATED

**Idea:** the residual vector edits the *sum* of all components' writes. If the
behavior is written by a *few* attention heads (or MLP neurons), edit only
those — surgical suppression that leaves coherence-carrying components intact.
Finer locus → less collateral → potentially a coherent window where the
residual vector had none. (Same logic as "refusal is a few heads" abliteration,
for a benign behavior.)

**We already have the map:** FINDINGS "head-level tug-of-war (L21)" + the
attn/MLP component split identified which components carry the direction;
brainscope's lens already exposes per-head contribution (source of the
tug-of-war figure). The new piece is a hook that edits a head-output slice
(reshape attn output into heads → scale/ablate the task-writing heads → W_O),
not the whole layer.

**MANDATORY FIRST STEP — localization diagnostic (no training, a lens pass):**
attribute the task-direction write per head on 8B.
- Concentrated (2–4 heads dominate the write) → build head-ablation; likely
  opens the window.
- Smeared (many heads each contribute a little) → a DEEP finding: 8B represents
  this behavior diffusely, so NO intervention granularity (vector, head, or
  neuron) cleanly isolates it — which *explains* the vector's failure and
  closes the "is 8B fixable" question. 8B's low linear separability is a prior
  that this may be the outcome.

**Status:** diagnostic runnable on brainscope now (lens/attribution); the
head-edit hook is a brainscope addition after the diagnostic says it's worth it.
Do NOT build surgery for a diffuse target — the diagnostic gates the build.

## #5 Signature probe: prompted vs steered induction, per-token (added 2026-07-28)

Motivated by external literature (Kang et al., arXiv 2605.10664; Heyman &
Vandeputte, arXiv 2605.03907): prompting moves activations SPARSELY via
attention from instruction tokens; residual steering pushes every token
uniformly and accumulates in the KV cache, degrading multi-turn coherence.
If that signature holds on our vectors, it is a mechanism candidate for
the regime accident and for H1's total-injected-mass law (WHY the window
closes at deployment length).

**Design (2 evenings, existing instruments only):** (a) prompted arm =
the strongest no-task system prompt (the pre-registered prompting
baseline) vs steered arm = v_pref at matched efficacy (checker-matched);
per-token projection profile onto the preference direction + KV-cache
accumulation over turns, via brainscope forced replay. (b) Run on TWO
vectors — v_pref plus a freshly extracted task-preference vector
(creative-vs-analytical, English) — because the SKOP round showed
effects are vector-specific (FINDINGS G); one vector = anecdote.
Side result for free: cos(v_fresh, v_pref) — does the production
behavior sit on a general task-preference axis (cf. Gilg et al., arXiv
2605.13339)?

**Status:** queued for spare GPU evenings behind the RQ1 grid.
Exploratory: no pre-registered claim; anything that survives gets
promoted and re-tested clean, per the rules of this track.

## Evaluation (identical discipline for all four)
Each is scored by the frozen pipeline: deployment-length scaffolds, temp 1,
hard/soft triage, bootstrap CIs, the ship gate. The comparison of interest is
the **coherent-steering window width** (steering headroom) — the metric static
vectors made us invent — measured for each method on the same behavior/model.
A method "wins" if it opens a shippable window where static `scale·V` had none.

## The through-line
Static steering was the null hypothesis. Its measured failure mode (coupled
suppression/breakage, coherence-blind, context-fragile) *designed* these three
successors. That's the lab working as intended: the negative result is the
specification for the next method.
