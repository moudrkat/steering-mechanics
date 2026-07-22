"""Activation patching: which single position carries the decision.

Patch the steered residual at inject_layer into the CLEAN forced pass, one
position at a time, and count how many downstream predictions flip. The
positions with the most flips are where the vector's effect is decided.

    BRAINSCOPE_BASE=... python3 experiments/patching.py --layer 20
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
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--id", default="v_pref_no_task_checklist_v3")
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--out", default="results/patching.json")
    args = ap.parse_args()
    body = {**NEUTRAL,
            "steering": {"id": args.id, "layer": args.layer, "scale": args.scale,
                         "decode_only": True},
            "forced": True, "max_tokens": args.max_tokens,
            "patch_layer": args.layer}
    req = urllib.request.Request(BASE + "/replay", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        d = json.loads(r.read())
    res = d["patching"]["results"]
    top = sorted(res, key=lambda e: -e["n_flips"])[:10]
    print(f"patch L{d['patching']['layer']} · {len(res)} positions · "
          f"total downstream flips {sum(e['n_flips'] for e in res)}")
    print("most decisive positions:")
    for e in top:
        ex = e["examples"][0] if e["examples"] else None
        exs = f"  e.g. '{ex['clean']}'->'{ex['patched']}'" if ex else ""
        print(f"  pos {e['pos']:>3} '{e['token']}': {e['n_flips']} flips{exs}")
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump(d["patching"], open(args.out, "w"), ensure_ascii=False, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
