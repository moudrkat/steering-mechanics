"""Build SteerBench entries from committed result JSONs — behavior / method /
model agnostic.

Each result file describes its own (task, method): a result JSON may carry
top-level "task" and "method" fields (new evals stamp these; see
hidden_directions run-eval). Legacy files that predate the convention are
mapped by filename in LEGACY_MAP. Every task is validated against the registry
in tasks.json. Safety axes are folded in from em_*.json; vocab_drift from
vocabdrift_*.json when present.

Adding a new vector/behavior needs NO change here: run its eval so the result
carries {task, method}, add the task to tasks.json, re-run this.

    python3 steerbench/build.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
BENCH = ROOT / "steerbench"
OUT = BENCH / "entries.jsonl"

MODEL_LAYERS = {"llama31-8b": 32, "llama-3.1-8b": 32, "qwen3-8b": 36,
                "qwen3-4b": 36, "qwen2.5-7b": 28, "qwen3.5-9b": 40,
                "qwen3.5-4b": 36}

# filename prefix -> (task, method) for result files that predate the
# self-describing convention. New results carry their own task/method.
LEGACY_MAP = {"campaign_": ("no-tasks", "pref-caa")}

AXES = ["efficacy_miss", "anti_steered", "incoherent", "kl_damage",
        "safety_harmful_compliance", "safety_false_refusal", "vocab_drift"]


def _registry():
    return json.loads((BENCH / "tasks.json").read_text())["tasks"]


def _task_method(result: dict, fname: str):
    """Prefer self-described (task, method); else legacy filename map."""
    task = result.get("task")
    method = result.get("method", result.get("vector_method"))
    if task and method:
        return task, method
    for pref, (t, m) in LEGACY_MAP.items():
        if fname.startswith(pref):
            return result.get("task", t), result.get("method", m)
    return task, method  # may be (None, None) -> skipped with a warning


def _entry(task, method, model, row, n_layers):
    layer = row.get("layer")
    g = lambda *ks: next((row[k] for k in ks if k in row and row[k] is not None), None)
    return {
        "task": task, "method": method, "model": model, "n_layers": n_layers,
        "layer": layer,
        "frac_depth": round(layer / n_layers, 3) if (layer and n_layers) else None,
        "scale": row.get("scale"),
        "n": g("n") or 30,
        "efficacy_miss": g("miss", "efficacy_miss"),
        "anti_steered": g("anti", "anti_steered"),
        "incoherent": g("incoherent", "incoh"),
        "kl_damage": g("kl", "kl_damage"),
        "safety_harmful_compliance": None,
        "safety_false_refusal": None,
        "vocab_drift": None,
    }


def _fold_safety(entries, registry):
    """Fold em_*.json (safety) into matching entries by (model, layer, scale)."""
    for f in sorted(RESULTS.glob("em_*.json")):
        d = json.loads(f.read_text())
        model, layer, scale = d.get("model"), d.get("layer"), d.get("scale")
        st = (d.get("conditions") or {}).get("steered", {})
        for e in entries:
            if e["model"] == model and e["layer"] == layer \
                    and abs((e["scale"] or -1) - (scale or -2)) < 1e-6:
                e["safety_harmful_compliance"] = st.get("harmful_compliance")
                e["safety_false_refusal"] = st.get("false_refusal_rate")


def _fold_vocab_drift(entries):
    """Fold vocabdrift_*.json {model,layer,scale,vocab_drift} when present."""
    for f in sorted(RESULTS.glob("vocabdrift_*.json")):
        d = json.loads(f.read_text())
        for e in entries:
            if e["model"] == d.get("model") and e["layer"] == d.get("layer") \
                    and abs((e["scale"] or -1) - (d.get("scale") or -2)) < 1e-6:
                e["vocab_drift"] = d.get("vocab_drift")


def main():
    registry = _registry()
    entries, skipped = [], []
    for f in sorted(RESULTS.glob("campaign_*.json")) + \
             sorted(RESULTS.glob("bench_*.json")):
        d = json.loads(f.read_text())
        model = d.get("model")
        n_layers = d.get("n_layers") or MODEL_LAYERS.get(model)
        task, method = _task_method(d, f.name)
        if not task or not method:
            skipped.append(f"{f.name} (no task/method)"); continue
        if task not in registry:
            skipped.append(f"{f.name} (task '{task}' not in registry)"); continue
        for row in d.get("rows", []):
            entries.append(_entry(task, method, model, row, n_layers))

    _fold_safety(entries, registry)
    _fold_vocab_drift(entries)

    OUT.write_text("\n".join(json.dumps(e) for e in entries) + ("\n" if entries else ""))

    # coverage report — what's a real comparison vs a single column
    tasks = sorted({e["task"] for e in entries})
    methods = sorted({e["method"] for e in entries})
    models = sorted({e["model"] for e in entries})
    filled = {a: sum(1 for e in entries if e[a] is not None) for a in AXES}
    print(f"SteerBench: {len(entries)} entries | tasks={tasks} | "
          f"methods={methods} | models={models}")
    print("axis coverage:", {a: f"{filled[a]}/{len(entries)}" for a in AXES})
    if len(methods) < 2:
        print("NOTE: only one method — this is a results table, not yet a "
              "comparison. A second method makes it a benchmark.")
    for s in skipped:
        print("  skipped:", s)


if __name__ == "__main__":
    main()
