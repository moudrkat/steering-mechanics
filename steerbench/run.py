"""SteerBench — run ONE vector through the WHOLE panel, from one config.

    python3 steerbench/run.py my_vector.bench.json

Runs, at each (layer, scale) point, every axis the config declares — all at
once, all on official benchmarks where they exist:

  behavioral  (your eval spec: prompts + checker; miss = violation OR incoherence)
  damage      (KL on benign)
  safety      (StrongREJECT / AdvBench harmful + XSTest benign, LLM-judge)
  capability  (MMLU / TruthfulQA via lm-eval-harness — pointer emitted)
  mechanistic (per-layer cosine footprint)

Emits results/bench_<task>_<method>_<model>.json (self-describing: carries
`task`+`method`, so steerbench/build.py folds it in with no wiring). Needs a
running brainscope ($BRAINSCOPE_BASE) serving the vector, official benchmark
files under $STEERMECH_BENCH, and lm-eval-harness for the capability axis.

Config (JSON) — declare the vector and which official benchmarks to run:

    {
      "task": "no-tasks", "method": "pref-caa",
      "model": "qwen3-4b", "vector_id": "v_pref_no_task_checklist_v3",
      "points": [{"layer": 20, "scale": 3}],
      "eval_spec": "no-tasks.tier1.eval.json",
      "safety":     {"harmful_prompts": "strongreject",
                     "benign_prompts": "xstest", "grader": "llm_judge"},
      "capability": {"tasks": ["mmlu", "truthfulqa_mc2"], "limit": 200},
      "damage":     {"n": 8},
      "mechanistic":{"n_prompts": 3}
    }
"""
import argparse
import json
from pathlib import Path

from hidden_directions.calibrate import run_eval, load_spec


def _build_spec(cfg):
    """One eval spec that turns on every panel axis the config declares."""
    spec = load_spec(cfg["eval_spec"]) if isinstance(cfg["eval_spec"], str) \
        else dict(cfg["eval_spec"])
    spec["task"] = cfg.get("task")
    spec["method"] = cfg.get("method")
    spec["baseline_compare"] = True                  # anti-steered fraction
    for k in ("safety", "capability", "damage", "mechanistic"):
        if k in cfg:
            spec[k] = cfg[k]
    return spec


def _row(point, r):
    """Flatten one run_eval result into a build.py-compatible row."""
    b = r.get("behavioral", {})
    saf = r.get("safety") or {}
    return {"layer": point["layer"], "scale": point["scale"],
            "n": b.get("n"),
            "miss": b.get("miss"), "anti_steered": b.get("anti_steered"),
            "incoherent": b.get("incoherent"), "kl": (r.get("damage") or {}).get("kl"),
            "safety_harmful_compliance": saf.get("harmful_compliance"),
            "safety_false_refusal": saf.get("false_refusal_rate"),
            "safety_jailbreak_compliance": saf.get("jailbreak_compliance"),
            "safety_robustness_drop": saf.get("robustness_drop"),
            "capability": r.get("capability"),
            "mechanistic": r.get("mechanistic")}


def run(cfg, *, chat_fn=None, diff_fn=None, n=None):
    spec = _build_spec(cfg)
    rows, detail = [], []
    for pt in cfg["points"]:
        r = run_eval(dict(spec), cfg["vector_id"], pt["layer"], float(pt["scale"]),
                     n=n, chat_fn=chat_fn, diff_fn=diff_fn)
        r.pop("records", None)                       # text stays out of scores
        rows.append(_row(pt, r))
        detail.append(r)
    return {"model": cfg["model"], "task": cfg.get("task"),
            "method": cfg.get("method"), "vector": cfg["vector_id"],
            "n_layers": cfg.get("n_layers"), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=None, help="prompts per behavioral cell")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = run(cfg, n=args.n)
    dst = Path(__file__).resolve().parent.parent / "results" / \
        f"bench_{cfg.get('task','x')}_{cfg.get('method','x')}_{cfg['model']}.json"
    dst.write_text(json.dumps(out, indent=1))
    for row in out["rows"]:
        print(f"L{row['layer']} s{row['scale']}: miss {row['miss']} "
              f"anti {row['anti_steered']} KL {row['kl']} "
              f"harm_comply {row['safety_harmful_compliance']} "
              f"false_refuse {row['safety_false_refusal']}")
    print(f"-> {dst}  (run steerbench/build.py to fold into entries.jsonl)")


if __name__ == "__main__":
    main()
