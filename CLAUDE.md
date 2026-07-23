# Working in steering-mechanics

Experiments + findings for the steering research. GPU work runs on a remote
box via brainscope's HTTP API; scripts here are thin clients.

## Critical: never commit private eval data

- Real prompts, transcripts, and model generations are kept OUTSIDE this repo,
  referenced via the `$STEERMECH_PRIVATE` environment variable. Never commit
  prompt text, generations, or app-specific content here.
- Committed result JSONs carry SCORES ONLY. Run any result through
  `python -m steermech.scrub` before it lands in `results/`.
- A pre-commit hook blocks commits containing private markers. Don't bypass it.

## Conventions

- Commits: short one-line messages. No "Co-Authored-By" trailer.
- Figures render offline from committed result JSONs: `python -m steermech.plot`.
- The pre-registered plan is FROZEN (`RESEARCH_PLAN.md`); changing a hypothesis,
  threshold, or method requires a dated deviation note in that file — never edit
  the plan silently.

## Where things are

- `FINDINGS.md` — dated results log. `EVAL_PRINCIPLES.md` — eval standards.
- `steermech/plot.py` — figures. `results/` — committed scores (no text).
