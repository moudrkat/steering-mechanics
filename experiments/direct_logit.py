"""Direct-vs-circuit split for a steering vector.

W_U·(-v) says which tokens the vector pushes DOWN directly at the
unembedding — no circuit involved. Compare with what the teacher-forced
diff actually suppressed: overlap = direct effect, remainder =
circuit-mediated. The ratio is the headline number.

    python3 experiments/direct_logit.py \
        --suppressed results/dose_response.json   # or a replay JSON
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

BASE = os.environ.get("BRAINSCOPE_BASE", "http://192.168.1.9:8010")


def unembed(name: str, layer: int, top: int = 60) -> dict:
    with urllib.request.urlopen(
            f"{BASE}/directions/{name}/unembed?layer={layer}&top={top}",
            timeout=120) as r:
        return json.loads(r.read())


def words_of(readout) -> set:
    out = set()
    def walk(n):
        if isinstance(n, str):
            w = n.strip().lower()
            if w:
                out.add(w)
        elif isinstance(n, (list, tuple)):
            for v in n:
                walk(v)
        elif isinstance(n, dict):
            for v in n.values():
                walk(v)
    walk(readout)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="v_pref_no_task_checklist_v3")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--suppressed", default="results/dose_response.json",
                    help="dose_response output or a forced-replay JSON")
    args = ap.parse_args()

    d = unembed(args.id, args.layer)
    down = words_of(d["top_down"])
    up = words_of(d["top_up"])

    src = json.loads(Path(args.suppressed).read_text())
    if "scales" in src:                       # dose_response output
        suppressed = {w.lower() for row in src["scales"] for w in row["top_suppressed"]}
    else:                                     # raw forced-replay output
        suppressed = {e["word"].lower() for e in src.get("suppressed_positional", [])}

    direct = sorted(suppressed & down)
    circuit = sorted(suppressed - down)
    ratio = len(direct) / max(1, len(suppressed))
    print(f"vector pushes DOWN directly (W_U·-v, top): {sorted(down)[:15]}")
    print(f"vector pushes UP directly   (W_U·+v, top): {sorted(up)[:15]}")
    print(f"\nsuppressed in forced diff : {sorted(suppressed)}")
    print(f"  ∩ direct (unembedding)  : {direct}")
    print(f"  circuit-mediated        : {circuit}")
    print(f"\nDIRECT / TOTAL = {len(direct)}/{len(suppressed)} = {ratio:.0%}"
          f"  ({'mostly direct logit push' if ratio > 0.5 else 'mostly circuit-mediated'})")


if __name__ == "__main__":
    main()
