"""Which attention heads carry the steering direction at a layer.

    BRAINSCOPE_BASE=... python3 experiments/head_attribution.py --layer 21
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("BRAINSCOPE_BASE", "http://localhost:8010")

NEUTRAL = {"messages": [
    {"role": "system", "content": "You are a helpful assistant. The user is "
     "organizing their week. Discuss; do not create anything."},
    {"role": "user", "content": "Set me a reminder so I don't forget about the file."}]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--id", default="v_pref_no_task_checklist_v3")
    ap.add_argument("--inject-layer", type=int, default=20)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--max-tokens", type=int, default=70)
    ap.add_argument("--out", default="results/head_attribution.json")
    args = ap.parse_args()
    body = {**NEUTRAL,
            "steering": {"id": args.id, "layer": args.inject_layer,
                         "scale": args.scale, "decode_only": True},
            "forced": True, "max_tokens": args.max_tokens,
            "attribute_heads_layer": args.layer}
    req = urllib.request.Request(BASE + "/replay", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    h = d["head_attribution"]
    deltas = [(i, round(s - c, 4)) for i, (c, s)
              in enumerate(zip(h["clean_mean"], h["steered_mean"]))]
    top = sorted(deltas, key=lambda x: -abs(x[1]))[:8]
    total = sum(abs(dv) for _, dv in deltas)
    print(f"L{h['layer']}: top heads by |delta write along v| "
          f"(of {len(deltas)} heads, total |delta| {total:.3f})")
    for i, dv in top:
        share = abs(dv) / max(1e-9, total)
        print(f"  head {i:>2}: {dv:+.4f}  ({share:.0%} of head-delta)")
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"layer": h["layer"], "deltas": deltas}, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
