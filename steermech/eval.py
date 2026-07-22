"""Efficacy and damage — the two axes of heretic-grade calibration.

Pure scoring functions (score_* ) are separated from the network-calling
wrappers so the logic is unit-testable without a server.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_intent(vector_key: str) -> dict:
    """Load a vector's intent spec. `vector_key` is either a key under
    data/vectors/<key>.intent.json or a direct path to an intent JSON
    (so private intents can live outside this repo)."""
    p = Path(vector_key)
    if not p.exists():
        p = ROOT / "data/vectors" / f"{vector_key}.intent.json"
    return json.loads(p.read_text())


def load_benign(n: int | None = None) -> list[str]:
    lines = [l.strip() for l in (ROOT / "data/benign_prompts.txt").read_text().splitlines()
             if l.strip()]
    return lines[:n] if n else lines


def _harvest(node, out: set):
    if isinstance(node, str):
        w = node.strip().lower()
        if w:
            out.add(w)
    elif isinstance(node, dict):
        for v in node.values():
            _harvest(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            _harvest(v, out)


def score_efficacy(diffs: list[dict], avoid: list[str], target: list[str]) -> dict:
    """Given per-prompt forced-diff results, score how well steering achieved
    the intent. avoid concepts should DISAPPEAR from the steered dispositions;
    target concepts should APPEAR. Returns miss in [0,1] (0 = perfect).

    A concept counts as 'suppressed' at a prompt if it was in the clean
    dispositions (baseline forming-words) and gone under steering — read
    straight off /replay's suppressed_positional list.
    """
    avoid = [a.lower() for a in avoid]
    target = [t.lower() for t in target]
    if not avoid and not target:
        return {"miss": 1.0, "avoid_suppressed": 0, "target_promoted": 0, "n": len(diffs)}
    supp_hits, tgt_hits = 0, 0
    for d in diffs:
        supp = set()
        _harvest([e.get("word", "") for e in d.get("suppressed_positional", [])], supp)
        if any(any(a in w for w in supp) for a in avoid):
            supp_hits += 1
        # target promotion: appears in steered but not clean (jlens diff, if present)
        promoted = set()
        _harvest(d.get("promoted_positional", []), promoted)
        if target and any(any(t in w for w in promoted) for t in target):
            tgt_hits += 1
    n = max(1, len(diffs))
    eff = 0.0
    if avoid:
        eff += supp_hits / n
    if target:
        eff += tgt_hits / n
    eff /= (bool(avoid) + bool(target))
    return {"miss": round(1.0 - eff, 3), "avoid_suppressed": supp_hits,
            "target_promoted": tgt_hits, "n": len(diffs)}


def score_damage(kl_means: list[float]) -> dict:
    """Mean KL divergence on benign prompts = how much normal behavior moved."""
    vals = [k for k in kl_means if k is not None]
    if not vals:
        return {"kl": 0.0, "kl_max": 0.0, "n": 0}
    return {"kl": round(sum(vals) / len(vals), 5), "kl_max": round(max(vals), 5),
            "n": len(vals)}


def combined_objective(miss: float, kl: float, lambda_kl: float) -> float:
    """heretic-style co-minimization: efficacy miss + lambda * model damage."""
    return miss + lambda_kl * kl


# ---- network-calling wrappers (need a live brainscope) ----

def efficacy(vector_key, direction_id, layer, scale, *, n_prompts=None,
             max_tokens=48):
    from .client import forced_diff
    intent = load_intent(vector_key)
    prompts = intent["prompts"][:n_prompts] if n_prompts else intent["prompts"]
    spec = {"id": direction_id, "layer": layer, "scale": scale, "decode_only": True}
    diffs = [forced_diff([{"role": "user", "content": p}], spec, max_tokens=max_tokens)
             for p in prompts]
    return score_efficacy(diffs, intent.get("avoid", []), intent.get("target", []))


def damage(direction_id, layer, scale, *, n_prompts=8, max_tokens=24):
    from .client import forced_diff
    spec = {"id": direction_id, "layer": layer, "scale": scale, "decode_only": True}
    kls = []
    for p in load_benign(n_prompts):
        d = forced_diff([{"role": "user", "content": p}], spec, kl=True,
                        max_tokens=max_tokens)
        kls.append((d.get("kl") or {}).get("mean"))
    return score_damage(kls)


def objective(vector_key, direction_id, layer, scale, lambda_kl, *,
              eff_prompts=None, dmg_prompts=8):
    e = efficacy(vector_key, direction_id, layer, scale, n_prompts=eff_prompts)
    d = damage(direction_id, layer, scale, n_prompts=dmg_prompts)
    return {"score": round(combined_objective(e["miss"], d["kl"], lambda_kl), 4),
            "miss": e["miss"], "kl": d["kl"], "efficacy": e, "damage": d}
