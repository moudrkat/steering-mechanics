# Exploration plan — richer interventions for the hard (8B) case (2026-07-25, ~02:00)

**EXPLORATORY track** (not pre-registered; see BEYOND_STATIC_STEERING.md for
designs, RESEARCH_PLAN.md dated note for the boundary). Written while v5's
autohunt runs. Goal: find *any* intervention that opens a shippable coherent
window on 8B, where static `scale·V` had none — or prove none does, honestly.

## The one hard constraint: the GPU is serial
One 8B brainscope on aorus (16 GB). Everything queues. So the plan is ordered
by **decisiveness per GPU-hour**, with go/no-go forks that can *cancel* expensive
steps before they run.

## The metric everything reports
**Steering-headroom = width of the coherent-steering window** — the dose range
where offers→~0 AND hard-breakage stays low, at deployment length + temp 1,
hard/soft triage, bootstrap CIs, frozen ship gate. Static vectors made us invent
it (8B's width ≈ 0). A method "wins" iff it makes that width > 0 on 8B.

---

## THE QUEUE (ordered; forks marked)

### 0. v5 verdict — CONFIRMATORY, running now  (task #5)
v5 = tool-call format + minimal pairs (the intersection v3/v4 each half-had).
Read its L20/21/19 grid + best cell's real text. **Fork A:**
- v5 opens a window → recipe *format* was the lever; ship-test it; 8B may be
  solvable with a static vector after all → exploration becomes bonus science.
- v5 fails too → format isn't enough; 8B is intrinsically hard → exploration is
  the real path. Proceed to #1 below.
*Cost:* 0 (already running). *Output:* the recipe-format answer.

### 1. Head-localization diagnostic — THE FORK  (task #6)  ~30 min GPU
Attribute the task-direction write per attention head on 8B (lens pass, no
training). `campaign/localization_diag.py`. **Fork B (decides the whole rest):**
- **Concentrated** (2–4 heads dominate) → finer intervention *can* isolate the
  behavior → build #4 head-surgery; #1/#3 also promising. Proceed.
- **Smeared** (many heads, diffuse) → **DEEP RESULT**: no granularity isolates
  it; #4 head-surgery and likely #1 are doomed for the same reason the vector
  was. **Skip #4; be skeptical of #1.** Closes "is 8B fixable" honestly and
  saves ~a day. 8B's low linear separability is a prior for this outcome.
*Cheapest + most decisive step in the whole plan. Run it first.*

### 2. #1 probe-gated grid — BUILT, turnkey  (task #7)  ~1.5 h GPU
`campaign/gated_steer.py` (compiles), brainscope `gate` hook (built). Calibrate
gate τ from the coef distribution (behavior-present vs absent), then sweep
gate × scale vs ungated. **Test:** hard-breakage drops at equal offer-suppression
because clean tokens are never touched. Ports to hotwire trivially (~5 lines).
*Can run overnight with guessed gates for a first look; calibrate + refine after.*

### 3. #2 closed-loop dose control — needs debug  (task #8)  ~half day
`campaign/closed_loop.py` (skeleton). Chunked generate → measure degeneracy →
back off scale. **Needs an active debug pass** (new code, GPU). Does NOT port to
hotwire (cross-token state fights vLLM batching) → brainscope/research-only.

### 4. #3 learned low-rank (ReFT) — needs debug  (task #9)  ~half day
`campaign/train_reft.py` (skeleton). LoReFT module on layer L; loss = suppress
behavior + benign-KL + repetition penalty. A real training job, debug-heavy.
Highest ceiling for a non-linear behavior. Ports to hotwire (apply module in hook).

### 5. Synthesis — steering-headroom table  (task #10)
Compare window-width across static / #1 / #2 / #3 / #4 on the same 8B behavior.
**Verdict:** does anything open a shippable 8B window, or is the honest answer
"ship 4B; ship 8B layered"? Feeds the paper's methods section either way.

---

## Timing (GPU serial)
- **By morning, hands-off (~3 h GPU after v5):** #5 verdict → #6 diagnostic →
  #7 gated. You wake to: recipe-format answer, the *decisive* is-8B-fixable
  answer, and the first richer-method result.
- **Tomorrow, needs me debugging (~1 day):** #8 + #9 (new code, iteration passes).
- **Full exploration properly: ~2 days.** But #6 may *cancel* #4/#8/#9 → could
  close in one morning. The diagnostic is the fork that saves the day.

## Autonomous vs supervised
- **Auto-chainable overnight** (turnkey, tested infra): #5 (running), #6 (once
  written+compiled), #7 (built). Risk: untested #6 could no-op — so #6 gets a
  compile + dry-run check before chaining.
- **Supervised only** (new code, needs error-reading): #8, #9.

## hotwire (production) portability — for the winner
Stateless per-position hook edits port to the hotwire vLLM plugin: **#1 gated ✅
(easy, ~5 lines), #4 head ✅, #3 ReFT ✅ (moderate)**. Only **#2 closed-loop ✗**
(cross-token feedback fights vLLM). So a winning #1/#3/#4 ships on hotwire; if
nothing beats 4B, hotwire ships 4B unchanged, zero port.

## Success criteria (honest, set now)
- **Win:** a method gives steering-headroom > 0 on 8B — offers→~0 with hard ≤
  0.1 at deployment length + temp 1, confirmed (frozen gate). Ship-test + hotwire
  port.
- **Honest null:** none does, and #6 explains why (diffuse representation). Then
  8B ships layered, 4B ships steered, and the *methods comparison itself* is the
  contribution — nobody has measured steering-headroom across interventions on a
  real deployment behavior.
- Either way: promote any surviving exploratory finding to a dated confirmatory
  RQ and re-test clean before it counts.
