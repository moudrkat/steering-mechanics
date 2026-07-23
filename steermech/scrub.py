"""Strip text fields from result JSONs before they land in public results/.

Public results carry scores and metadata; prompts and generations stay in
the private store (see $STEERMECH_PRIVATE). Usage:

    python -m steermech.scrub results/foo.json            # writes in place
    python -m steermech.scrub in.json -o out.json
"""

import argparse
import json
import sys
from pathlib import Path

# keys whose values are (or may quote) prompt/generation text
TEXT_KEYS = {
    "prompt", "prompts", "generation", "generations", "completion",
    "messages", "content", "text", "tokens", "output", "outputs",
    "top_suppressed", "system", "case",
}

SCRUBBED = "<scrubbed>"


def scrub(obj):
    if isinstance(obj, dict):
        return {k: SCRUBBED if k in TEXT_KEYS else scrub(v)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path (default: scrub in place)")
    args = ap.parse_args(argv)
    data = json.loads(args.path.read_text())
    out = args.out or args.path
    out.write_text(json.dumps(scrub(data), indent=1) + "\n")
    print(f"scrubbed -> {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
