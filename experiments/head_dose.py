"""Is the head opposition mechanical (linear in dose) or active (nonlinear)?

Measure the per-head write-along-v delta at several injection scales. If a
head's delta grows linearly with scale, its behavior is consistent with
mechanical propagation of the injected signal through context-fixed
attention. Nonlinearity (saturation, sign flips, superlinear opposition)
points to attention patterns reorganizing — an active response.

    BRAINSCOPE_BASE=... python3 experiments/head_dose.py --scales 1 3 6
"""
import argparse
import json
import os
import statistics
import urllib.request
from pathlib import Path

BASE = os.environ.get("BRAINSCOPE_BASE", "http://localhost:8010")
NEUTRAL = {"messages": [
    {"role": "system", "content": "You are a helpful assistant. The user is "
     "organizing their week. Discuss; do not create anything."},
    {"role": "user", "content": "Set me a reminder so I don't forget about the file."}]}


def head_deltas(layer, inject_layer, scale, direction_id, max_tokens):
    body = {**NEUTRAL,
            "steering": {"id": direction_id, "layer": inject_layer, "scale": scale,
                         "decode_only": True},
            "forced": True, "max_tokens": max_tokens, "attribute_heads_layer": layer}
    req = urllib.request.Request(BASE + "/replay", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        h = json.loads(r.read())["head_attribution"]
    return [round(s - c, 5) for c, s in zip(h["clean_mean"], h["steered_mean"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--inject-layer", type=int, default=20)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 3.0, 6.0])
    ap.add_argument("--id", default="v_pref_no_task_checklist_v3")
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--out", default="results/head_dose.json")
    args = ap.parse_args()

    curves = {s: head_deltas(args.layer, args.inject_layer, s, args.id, args.max_tokens)
              for s in args.scales}
    n = len(next(iter(curves.values())))
    ref = max(args.scales)
    print(f"L{args.layer} · scales {args.scales} · delta/scale should be constant if linear\n")
    nonlin = []
    for hd in range(n):
        vals = [curves[s][hd] for s in args.scales]
        norm = [round(curves[s][hd] / s, 4) for s in args.scales]  # delta/scale
        if abs(vals[-1]) < 0.05:
            continue
        spread = (max(norm) - min(norm))
        rel = abs(spread) / max(abs(max(norm, key=abs)), 1e-9)
        nonlin.append((hd, vals, norm, rel))
    nonlin.sort(key=lambda x: -abs(x[1][-1]))
    print(f"{'head':>4}  {'deltas @ scales':<28} {'delta/scale (linear=flat)':<28} nonlin%")
    for hd, vals, norm, rel in nonlin[:12]:
        print(f"{hd:>4}  {str(vals):<28} {str(norm):<28} {rel:.0%}")
    med_nl = statistics.median([rel for _, _, _, rel in nonlin]) if nonlin else 0
    print(f"\nmedian nonlinearity across active heads: {med_nl:.0%}")
    print("  (near 0% => mechanical linear propagation; large => active reorganization)")
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"scales": args.scales, "layer": args.layer, "curves": curves},
              open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
