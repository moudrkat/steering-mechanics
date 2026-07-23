"""Seed SteerBench entries from this lab's committed result JSONs.

Reads results/campaign_*.json (and the transfer/em files) and emits one
SteerBench record per evaluated (model, layer, scale) point into
steerbench/entries.jsonl. Pure stdlib; re-run as the campaign fills in.

    python3 steerbench/build.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "steerbench" / "entries.jsonl"

MODEL_LAYERS = {"llama-3.1-8b": 32, "qwen3-8b": 36, "qwen3-4b": 36,
                "qwen2.5-7b": 28}


def _entry(task, model, method, row, n_layers):
    layer = row.get("layer")
    return {
        "task": task, "model": model, "n_layers": n_layers, "method": method,
        "layer": layer,
        "frac_depth": round(layer / n_layers, 3) if (layer and n_layers) else None,
        "scale": row.get("scale"),
        "n": row.get("n") or row.get("base") and 30,
        "efficacy_miss": row.get("miss"),
        "anti_steered": row.get("anti") if "anti" in row else row.get("anti_steered"),
        "incoherent": row.get("incoherent") if "incoherent" in row else row.get("incoh"),
        "kl_damage": row.get("kl"),
        "safety_harmful_compliance": None,
        "safety_false_refusal": None,
        "vocab_drift": None,
    }


def main():
    entries = []
    for f in sorted(RESULTS.glob("campaign_*.json")):
        d = json.loads(f.read_text())
        model = d.get("model")
        n_layers = d.get("n_layers") or MODEL_LAYERS.get(model)
        for row in d.get("rows", []):
            entries.append(_entry("no-tasks", model, "pref-caa", row, n_layers))
    # fold in the EM (safety) results where present
    for f in sorted(RESULTS.glob("em_*.json")):
        d = json.loads(f.read_text())
        model = d.get("model")
        st = (d.get("conditions") or {}).get("steered", {})
        for e in entries:
            if e["model"] == model and e["layer"] == d.get("layer") \
                    and e["scale"] == d.get("scale"):
                e["safety_harmful_compliance"] = st.get("harmful_compliance")
                e["safety_false_refusal"] = st.get("false_refusal_rate")
    OUT.write_text("\n".join(json.dumps(e) for e in entries) + ("\n" if entries else ""))
    models = sorted({e["model"] for e in entries})
    print(f"SteerBench seed: {len(entries)} entries across {len(models)} models "
          f"{models} -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
