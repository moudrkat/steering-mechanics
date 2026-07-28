"""Unified analysis for the qk_freeze2 factorization family + freegen probes.

Every reported share carries a 95% bootstrap CI over prompts (B=2000,
seeded). Emits one scores-only JSON suitable for results/ and prints a
readable report.

Usage:
  QKA_DIRS='label1:/path/to/per-prompt-dir,label2:...' \
  QKA_FGP='label1:/path/to/freegen-dir,...' (optional) \
  python analyze_qk_freeze2.py > report.txt
"""
import json, glob, os, random

DIRS = dict(kv.split(":", 1) for kv in os.environ["QKA_DIRS"].split(","))
FGP = dict(kv.split(":", 1) for kv in os.environ.get("QKA_FGP", "").split(",")) \
      if os.environ.get("QKA_FGP") else {}
OUT = os.environ.get("QKA_OUT", "qk_freeze2_analysis.json")


def boot_ci(vals, B=2000, seed=1):
    rng = random.Random(seed)
    n = len(vals)
    if n < 2:
        return (min(vals), max(vals))
    stats = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(B))
    return (stats[int(0.025 * B)], stats[int(0.975 * B)])


def boot_ratio_ci(num, den, B=2000, seed=1):
    """CI for 1 - mean(num)/mean(den) via paired bootstrap over prompts."""
    rng = random.Random(seed)
    n = len(num)
    stats = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        d = sum(den[i] for i in idx)
        stats.append(1 - sum(num[i] for i in idx) / d if d else 0.0)
    stats.sort()
    return (stats[int(0.025 * B)], stats[int(0.975 * B)])


report = {}
for label, d in DIRS.items():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "p*.json")))]
    if not runs:
        continue
    entry = {"n_prompts": len(runs), "doses": {}}
    for s in sorted(runs[0]["doses"], key=float):
        ds = [r["doses"][s] for r in runs]
        ks = [x["kl_steered"] for x in ds]
        row = {"kl_steered": round(sum(ks) / len(ks), 4),
               "kl_steered_ci": [round(v, 4) for v in boot_ci(ks)]}
        for arm in ("fpat", "fval", "fattn"):
            ka = [x[f"kl_{arm}"] for x in ds]
            lo, hi = boot_ratio_ci(ka, ks)
            row[f"rescue_{arm}"] = round(1 - sum(ka) / sum(ks), 4)
            row[f"rescue_{arm}_ci"] = [round(lo, 4), round(hi, 4)]
        for k in ("match_steered", "match_fattn"):
            vals = [x[k] for x in ds]
            row[k] = round(sum(vals) / len(vals), 4)
        entry["doses"][s] = row
    # per-class split (task vs neutral) at the lowest dose with data
    cls = {}
    for c in ("task", "neutral"):
        sub = [r for r in runs if r["prompt_class"] == c]
        if sub:
            s0 = sorted(runs[0]["doses"], key=float)[0]
            ks = [r["doses"][s0]["kl_steered"] for r in sub]
            cls[c] = {"n": len(sub), "kl_steered": round(sum(ks) / len(ks), 4)}
    entry["class_split_lowest_dose"] = cls
    report[label] = entry

for label, d in FGP.items():
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "s*.json"))):
        rows += json.load(open(f))
    if not rows:
        continue
    scales = sorted({r["scale"] for r in rows})
    fg = {}
    for s in scales:
        sub = [r for r in rows if r["scale"] == s]
        viol = [float(r["viol"]) for r in sub]
        loop = [float(r["loop"]) for r in sub]
        rep4 = [r["rep4"] for r in sub]
        fg[str(s)] = {"n": len(sub),
                      "viol": round(sum(viol) / len(sub), 3),
                      "viol_ci": [round(v, 3) for v in boot_ci(viol)],
                      "loop": round(sum(loop) / len(sub), 3),
                      "loop_ci": [round(v, 3) for v in boot_ci(loop)],
                      "rep4": round(sum(rep4) / len(sub), 3)}
    report.setdefault("freegen", {})[label] = fg

json.dump(report, open(OUT, "w"), indent=1)
print(json.dumps(report, indent=1))
