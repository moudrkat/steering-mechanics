"""Give any steering vector a calibratable intent — auto-discovered.

Runs the vector strongly over the benign prompts, harvests what it most
reliably suppresses/promotes, and writes data/vectors/<key>.intent.json.

    BRAINSCOPE_BASE=... python3 experiments/make_intent.py \
        --key my-vector --id my_direction_name --layer 20
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hidden_directions.calibrate import write_intent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--scale", type=float, default=6.0)
    ap.add_argument("--desc", default="")
    args = ap.parse_args()
    intent = write_intent(args.key, args.id, args.layer, args.scale,
                          description=args.desc)
    print(f"discovered avoid: {intent['avoid']}")
    print(f"discovered target: {intent['target']}")
    print(f"-> data/vectors/{args.key}.intent.json")


if __name__ == "__main__":
    main()
