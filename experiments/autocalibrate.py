"""heretic-grade auto-calibration of an additive steering vector.

Co-minimize TWO axes on TWO datasets (heretic's structure):
  objective = efficacy_miss  +  lambda * damage_KL
- efficacy_miss: did the vector achieve its intent? (per-vector intent file,
  small eliciting prompt set)
- damage_KL: did steering break normal behavior? (shared benign set, KL from
  the unsteered model — vector-agnostic)

    # calibrate a vector that already has data/vectors/<key>.intent.json:
    BRAINSCOPE_BASE=... python3 experiments/autocalibrate.py \
        --key discuss-no-tasks --id v_pref_no_task_checklist_v3 --trials 40

Optuna TPE if installed, else random search. Any vector works: give it an
intent file (hand-written or produced by make_intent.py).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from steermech.eval import objective, load_intent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True, help="intent key: data/vectors/<key>.intent.json")
    ap.add_argument("--id", required=True, help="direction id on the server")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--lambda-kl", type=float, default=0.1)
    ap.add_argument("--layers", nargs=2, type=int, default=[8, 28])
    ap.add_argument("--scales", nargs=2, type=float, default=[0.5, 8.0])
    ap.add_argument("--eff-prompts", type=int, default=None,
                    help="subset of intent prompts per trial (fast); default all")
    ap.add_argument("--dmg-prompts", type=int, default=8)
    ap.add_argument("--out", default="results/autocalibrate.json")
    args = ap.parse_args()

    intent = load_intent(args.key)
    print(f"calibrating {args.id}  ·  intent '{args.key}'  ·  "
          f"objective = miss + {args.lambda_kl}*KL")
    print(f"  avoid={intent.get('avoid')}  target={intent.get('target')}\n")
    trials = []

    def obj(layer, scale):
        r = objective(args.key, args.id, layer, scale, args.lambda_kl,
                      eff_prompts=args.eff_prompts, dmg_prompts=args.dmg_prompts)
        rec = {"layer": layer, "scale": round(scale, 2), **r}
        trials.append(rec)
        print(f"  L{layer:>2} s{scale:>4.1f}: miss {r['miss']:.2f} KL {r['kl']:.3f} "
              f"-> score {r['score']:.3f}")
        return r["score"]

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=0))
        print(f"Optuna TPE, {args.trials} trials:")
        study.optimize(lambda t: obj(t.suggest_int("layer", *args.layers),
                                     t.suggest_float("scale", *args.scales)),
                       n_trials=args.trials)
    except ImportError:
        import random
        random.seed(0)
        print(f"optuna not installed — random search, {args.trials} trials:")
        for _ in range(args.trials):
            obj(random.randint(*args.layers), round(random.uniform(*args.scales), 1))

    win = min(trials, key=lambda r: r["score"])
    print(f"\nBEST: L{win['layer']} scale {win['scale']} — "
          f"miss {win['miss']:.2f}, KL {win['kl']:.3f}, score {win['score']:.3f}")
    Path(args.out).parent.mkdir(exist_ok=True)
    json.dump({"key": args.key, "id": args.id, "lambda_kl": args.lambda_kl,
               "best": win, "trials": trials}, open(args.out, "w"), indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
