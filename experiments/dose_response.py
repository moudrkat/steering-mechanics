"""Dose-response: teacher-forced diff at several scales.

For each scale, run brainscope /replay {forced: true} on the same
conversation and record: how many disposition-words were directly
suppressed, at how many positions, and the mean |cos| imprint at the
peak layer. Linear-then-saturating curves = the vector's operating range.

    python3 experiments/dose_response.py --scales 0.5 1.5 3 6

The default conversation is neutral; point $SCAFFOLD_JSON at a file
{"messages": [...], "tools": [...], "tool_choice": ...} to reproduce a
private scenario (results then land in results/, gitignored).
"""

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("BRAINSCOPE_BASE", "http://192.168.1.9:8010")

NEUTRAL = {"messages": [
    {"role": "system", "content": "You are a helpful assistant. The user is "
     "organizing their week. Discuss; do not create anything."},
    {"role": "user", "content": "Set me a reminder so I don't forget about the file."}]}


def load_case() -> dict:
    p = os.environ.get("SCAFFOLD_JSON")
    if p:
        return json.loads(Path(p).read_text())
    return dict(NEUTRAL)


def run(scale: float, spec_base: dict, case: dict, max_tokens: int) -> dict:
    body = {**case, "steering": {**spec_base, "scale": scale},
            "forced": True, "max_tokens": max_tokens}
    req = urllib.request.Request(BASE + "/replay", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    n_layers = len(d["positions"][0]["cos"]) if d["positions"] and "cos" in d["positions"][0] else 0
    peak_layer, peak = None, 0.0
    for L in range(n_layers):
        m = statistics.mean(abs(p["cos"][L]) for p in d["positions"])
        if m > peak:
            peak, peak_layer = m, L
    sup = d.get("suppressed_positional", [])
    return {"scale": scale,
            "n_tokens": len(d["tokens"]),
            "suppressed_words": len(sup),
            "suppressed_positions": sum(e["positions"] for e in sup),
            "top_suppressed": [e["word"] for e in sup[:8]],
            "peak_layer": peak_layer,
            "peak_mean_abs_cos": round(peak, 4),
            "secs": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", type=float, default=[0.5, 1.5, 3.0, 6.0])
    ap.add_argument("--id", default="v_pref_no_task_checklist_v3")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=70)
    ap.add_argument("--out", default="results/dose_response.json")
    args = ap.parse_args()
    case = load_case()
    spec = {"id": args.id, "layer": args.layer, "decode_only": True}
    rows = []
    for s in args.scales:
        r = run(s, spec, case, args.max_tokens)
        rows.append(r)
        print(f"scale {s:>4}: suppressed {r['suppressed_words']:>3} words / "
              f"{r['suppressed_positions']:>3} positions · peak L{r['peak_layer']} "
              f"|cos| {r['peak_mean_abs_cos']:.3f} · {r['secs']}s "
              f"· top: {', '.join(r['top_suppressed'][:5])}")
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"spec": spec, "scales": rows}, open(args.out, "w"),
              ensure_ascii=False, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
