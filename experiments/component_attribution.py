"""Component attribution: which sublayer carries the injected delta.

At the attribution layer, hooks record how much the attention block and the
MLP block each write ALONG the steering direction, per position, in the
clean and the steered forced pass. The steered-minus-clean delta says who
amplifies the vector — the answer to "why does the imprint peak one layer
after injection".

    python3 experiments/component_attribution.py --layers 21 22
"""

import argparse
import json
import os
import statistics
import urllib.request
from pathlib import Path

BASE = os.environ.get("BRAINSCOPE_BASE", "http://192.168.1.9:8010")

NEUTRAL = {"messages": [
    {"role": "system", "content": "You are a helpful assistant. The user is "
     "organizing their week. Discuss; do not create anything."},
    {"role": "user", "content": "Set me a reminder so I don't forget about the file."}]}


def run(layer: int, spec: dict, case: dict, max_tokens: int) -> dict:
    body = {**case, "steering": spec, "forced": True,
            "max_tokens": max_tokens, "attribute_layer": layer}
    req = urllib.request.Request(BASE + "/replay", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    a = d["attribution"]
    out = {"layer": layer}
    for comp in ("attn", "mlp"):
        dc = [s - c for s, c in zip(a["steered"][comp], a["clean"][comp])]
        out[comp] = {"clean_mean": round(statistics.mean(a["clean"][comp]), 4),
                     "steered_mean": round(statistics.mean(a["steered"][comp]), 4),
                     "delta_mean": round(statistics.mean(dc), 4)}
    da, dm = abs(out["attn"]["delta_mean"]), abs(out["mlp"]["delta_mean"])
    out["mlp_share_of_delta"] = round(dm / max(1e-9, da + dm), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", nargs="+", type=int, default=[21, 22])
    ap.add_argument("--id", default="v_pref_no_task_checklist_v3")
    ap.add_argument("--inject-layer", type=int, default=20)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--max-tokens", type=int, default=70)
    ap.add_argument("--out", default="results/component_attribution.json")
    args = ap.parse_args()
    spec = {"id": args.id, "layer": args.inject_layer, "scale": args.scale,
            "decode_only": True}
    rows = []
    for L in args.layers:
        r = run(L, spec, NEUTRAL, args.max_tokens)
        rows.append(r)
        print(f"L{L}: attn Δ {r['attn']['delta_mean']:+.4f} "
              f"(clean {r['attn']['clean_mean']:+.4f} -> steered {r['attn']['steered_mean']:+.4f}) · "
              f"mlp Δ {r['mlp']['delta_mean']:+.4f} "
              f"(clean {r['mlp']['clean_mean']:+.4f} -> steered {r['mlp']['steered_mean']:+.4f}) · "
              f"MLP share of |delta| {r['mlp_share_of_delta']:.0%}")
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"spec": spec, "rows": rows}, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
