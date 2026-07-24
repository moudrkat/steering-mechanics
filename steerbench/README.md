# SteerBench (seed)

**The steering-evaluation benchmark the field doesn't have yet.** Steering
papers each roll their own eval, so no two are comparable — a gap the
critique literature named directly (Tan et al. 2024: "evaluations on
datasets of prior work or common benchmarks are often missing"). SteerBench
is a common yardstick: a fixed set of `(behavior, model)` steering tasks,
each scored on the same multi-axis panel, so different **methods** (CAA,
repeng, heretic-style, calibrated) can be compared on the same ground.

> **Status: seed.** This is the foundation stone, not the finished
> benchmark. It is currently being populated from this lab's own runs
> ([FINDINGS.md](../FINDINGS.md)). A benchmark earns authority through
> *adoption*, not declaration — SteerBench becomes "official" only when
> others run it. The plan below is how that happens; the schema below is
> what's stable now.

## An entry

One JSON record per `(method, behavior, model, layer, scale)` evaluated
point. All axes come from the `hidden-directions` eval framework, so an
entry is reproducible from the tools:

```jsonc
{
  "task": "no-tasks",              // the behavior being steered
  "model": "llama-3.1-8b",
  "n_layers": 32,
  "method": "pref-caa",            // how the vector was made
  "layer": 14, "frac_depth": 0.44, // fractional depth = the transferable coord
  "scale": 3.0,
  "n": 30,
  "efficacy_miss": 0.0,            // violation OR incoherence (lower better)
  "anti_steered": 0,              // per-sample wrong-way count
  "incoherent": 0,
  "kl_damage": 1.11,
  "safety_harmful_compliance": null,  // via StrongREJECT + LLM judge
  "safety_false_refusal": null,       // via XSTest
  "vocab_drift": null                 // NOVEL axis: does steering leak into
                                      // unrelated behaviours' wording?
}
```

## The axes (why these)

Each is a distinct way steering fails; a method must be scored on all of
them, not just efficacy (see [../EVAL_PRINCIPLES.md](../EVAL_PRINCIPLES.md)):

- **efficacy_miss** — did the behavior change, without breaking the model
  (violation OR incoherence).
- **anti_steered** — fraction of inputs pushed the *wrong* way.
- **kl_damage** — general behavioral drift (KL on benign).
- **safety_\*** — refusal shift on official benchmarks (StrongREJECT/XSTest),
  judged by meaning not phrasing.
- **vocab_drift** — *new to SteerBench:* steering one behavior can shift the
  *vocabulary* of an unrelated one without changing it (observed
  2026-07-23: the no-tasks vector made refusals say "I can't discuss"
  instead of "I can't help"). No prior benchmark measures this.
- **transfer** — does the (frac_depth, scale) optimum hold across models.


## Run one vector through the whole panel (one config, one command)

```bash
python3 steerbench/run.py steerbench/configs/no-tasks.example.json
python3 steerbench/build.py    # fold the result into entries.jsonl
```

One config declares the vector + which official benchmarks to run; the runner
fires **every axis at once** — behavioral (miss = violation OR incoherence),
anti-steered fraction, damage (KL), **safety on StrongREJECT + XSTest with an
LLM judge**, **capability on MMLU + TruthfulQA (lm-eval-harness)**, and the
mechanistic footprint — and writes a self-describing `results/bench_*.json`
that `build.py` folds in. Official benchmarks resolve from `$STEERMECH_BENCH`
(download them; not bundled). See `configs/no-tasks.example.json`.

## Add your own vector / behavior (no code changes)

SteerBench is behavior-, method-, and model-agnostic. To put ANY vector in:

1. **Register the behavior** — add an entry to `tasks.json` (description, use
   tier, status). One-time per behavior.
2. **Run its eval** with a spec that self-describes — add `"task"` and
   `"method"` to the eval spec (e.g. `"task": "sycophancy"`, `"method":
   "repeng"`). `hidden-directions run-eval` echoes them into the result;
   name the output `results/bench_<anything>.json`.
3. **`python3 steerbench/build.py`** — the result flows in automatically,
   scored on the same axes, comparable to every other vector.

Legacy result files without `task`/`method` are mapped by filename
(`LEGACY_MAP` in build.py). Safety numbers fold in from `em_*.json`;
`vocab_drift` from `vocabdrift_*.json`. build.py prints axis coverage and
warns when there's only one method (a results table, not yet a comparison).

## Roadmap to a real benchmark (post-write-up)

1. **Seed** (now): populate from this lab's runs; freeze the schema.
2. **Baselines**: score CAA, repeng, and calibrated methods on the same
   tasks — the first comparison table.
3. **Human-validated labels**: κ against human judgment on a sample per
   task (the classifier must itself be validated).
4. **Release**: its own repo + a short benchmark paper, introduced *after*
   the findings paper establishes the axes.
5. **Adoption**: a leaderboard; invite other steering methods to report.
