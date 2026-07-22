"""Auto-calibrate a steering vector — heretic's best idea, borrowed.

Instead of a manual layer/scale sweep, run an optimizer that co-minimizes
TWO objectives at once, exactly as heretic (p-e-w) does for abliteration:

    objective = behavioral_miss + lambda * KL_divergence

- behavioral_miss: fraction of target dispositions the vector FAILED to
  suppress (want low = effective steering)
- KL_divergence: how far the steered output distribution moved from the
  clean one (want low = intelligence/coherence preserved)

The optimizer (Optuna TPE if installed, else random search) proposes
(layer, scale) and brainscope's forced diff measures both in one call.
The result is the operating point that steers hardest while damaging the
model least — found automatically, with a principled damage guardrail
instead of eyeballing coherence.

    BRAINSCOPE_BASE=... python3 experiments/autocalibrate.py --trials 25
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

# concepts the vector should suppress (English dispositions, J-lens surface)
TARGETS = {"reminder", "notification", "schedule", "task", "checklist", "todo"}


def measure(direction_id, layer, scale, max_tokens) -> dict:
    body = {**NEUTRAL,
            "steering": {"id": direction_id, "layer": layer, "scale": scale,
                         "decode_only": True},
            "forced": True, "max_tokens": max_tokens, "kl": True}
    req = urllib.request.Request(BASE + "/replay", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.loads(r.read())
    suppressed = {e["word"].lower() for e in d.get("suppressed_positional", [])}
    hit = len(TARGETS & suppressed)
    miss = 1.0 - hit / len(TARGETS)
    kl = d["kl"]["mean"] if d.get("kl") else 0.0
    return {"miss": round(miss, 3), "kl": round(kl, 5),
            "hit_targets": sorted(TARGETS & suppressed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="v_pref_no_task_checklist_v3")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--lambda-kl", type=float, default=2.0,
                    help="weight of KL (model damage) in the objective")
    ap.add_argument("--layers", nargs=2, type=int, default=[8, 28])
    ap.add_argument("--scales", nargs=2, type=float, default=[0.5, 8.0])
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--out", default="results/autocalibrate.json")
    args = ap.parse_args()

    trials = []

    def objective(layer, scale):
        m = measure(args.id, layer, scale, args.max_tokens)
        score = m["miss"] + args.lambda_kl * m["kl"]
        rec = {"layer": layer, "scale": round(scale, 2), "score": round(score, 4), **m}
        trials.append(rec)
        print(f"  L{layer:>2} s{scale:>4.1f}: miss {m['miss']:.2f} kl {m['kl']:.4f} "
              f"-> score {score:.4f}  hits {m['hit_targets']}")
        return score

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        print(f"optimizing with Optuna TPE, {args.trials} trials "
              f"(objective = miss + {args.lambda_kl}*KL):")
        study.optimize(lambda t: objective(
            t.suggest_int("layer", *args.layers),
            t.suggest_float("scale", *args.scales)), n_trials=args.trials)
        best = study.best_params
    except ImportError:
        import random
        random.seed(0)
        print(f"optuna not installed — random search, {args.trials} trials:")
        for _ in range(args.trials):
            objective(random.randint(*args.layers),
                      round(random.uniform(*args.scales), 1))
        best = min(trials, key=lambda r: r["score"])
        best = {"layer": best["layer"], "scale": best["scale"]}

    winner = min(trials, key=lambda r: r["score"])
    print(f"\nBEST: L{winner['layer']} scale {winner['scale']} — "
          f"miss {winner['miss']:.2f}, KL {winner['kl']:.4f}, score {winner['score']:.4f}")
    print(f"  (suppresses {winner['hit_targets']} at minimal model damage)")
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"lambda_kl": args.lambda_kl, "best": winner, "trials": trials},
              open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
