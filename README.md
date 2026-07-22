# steering-mechanics

**Mechanistic inspection of steering vectors**, driven
through [brainscope](https://github.com/moudrkat/brainscope)'s HTTP API.
Scripts here are thin clients — the GPU work happens on the aorus box, this
repo holds experiments, aggregated results, and notes.

Division of labor (why this repo exists):

- **brainscope** (OSS) — generic *instruments*: `/replay {forced: true}`
  teacher-forced causal diff, `/directions/{name}/unembed` direct logit
  attribution, per-token cos & J-lens capture.
- **this repo** — *experiments on specific vectors*: dose–response
  curves, direct-vs-circuit splits, component attribution, patching studies.
- app-specific eval scaffolds and traces never enter this repo — experiments
  either use the neutral prompts baked into scripts, or reference private
  scaffold files at runtime via `$SCAFFOLD_JSON` (results derived from them
  land in `results/`, which is gitignored for text-bearing files).

## The program (in order of depth)

1. **Dose–response** (`experiments/dose_response.py`) — forced diff at
   scales 0.5→6: does suppression scale linearly, then saturate? Measures
   the vector's operating range instead of guessing it.
2. **Direct vs circuit** (`experiments/direct_logit.py`) — `W_U·v` top
   movers vs what steering actually suppressed at matched positions. The
   overlap ratio is the single most informative number about *how* the
   vector works: direct logit push vs circuit-mediated suppression.
2b. **Tuned-lens quantification** — replace the top-5 set-membership
   readout with tuned-lens Δ log-prob per layer/position (brainscope has a
   fitted tuned lens for Qwen3-4B in `lenses/`). Continuous instead of
   binary: no rank-6 blindness, and dose–response gets a real y-axis.
   Needs the forced pass to expose tuned-lens readouts. TODO.
3. **Component attribution** — which sublayer (attn vs MLP) at L21 amplifies
   the injected delta (the peak lands one layer after injection). Needs a
   small brainscope extension to the forced pass (record probe norms per
   forced position). TODO.
4. **Head-level** — which attention heads move the vector's content between
   positions; where "why did the concept survive at position X" gets its
   real answer. TODO.
5. **Activation patching** — steered L20 residual patched into the clean run
   at single positions; which position's patch flips the output token. The
   forced-pass scaffolding is ~80 % of this harness. TODO.

## Runbook: how to continue tomorrow

Everything below assumes the aorus box (192.168.1.9). **The GPU runs ONE
thing at a time** — hotwire-vLLM (the app backend) or brainscope (the lab).

```bash
# 0) what is on the GPU right now?
ssh aorus 'nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader'

# 1) stop whatever holds the GPU
ssh aorus 'nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9'

# 2a) start BRAINSCOPE (lab: replay, forced diff, unembed) — port 8010
ssh aorus 'cd ~/tmp/brainscope-test && setsid nohup env \
  HF_HOME=~/projects/science/instruct-steer/hf-cache HF_HUB_OFFLINE=1 \
  PYTHONPATH=. .venv/bin/python launch_bs.py > ~/bs_replay.log 2>&1 < /dev/null &'
curl -s http://192.168.1.9:8010/info          # wait until it answers
curl -s -X POST http://192.168.1.9:8010/jlens -d '{"on": true}' \
  -H 'Content-Type: application/json'          # J-lens on for disposition diffs

# 2b) or start HOTWIRE-VLLM (app backend) — port 8001
ssh aorus 'setsid nohup env HF_HOME=~/projects/science/instruct-steer/hf-cache \
  HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
  HOTWIRE_VECTORS=~/hotwire-vectors HOTWIRE_SLOTS=128 \
  ~/tmp/vllm-lens-test/.venv/bin/vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --port 8001 --served-model-name qwen3-8b qwen3-4b --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --enable-auto-tool-choice \
  --tool-call-parser hermes > ~/hotwire-serve.log 2>&1 < /dev/null &'

# 3) run an experiment (brainscope must be up)
python3 experiments/dose_response.py --scales 0.5 1.5 3 6
python3 experiments/direct_logit.py

# 4) IMPORTANT: hand the GPU back to the app when done (repeat 1 + 2b)
```

Key locations:
- vectors served to both backends: `aorus:~/hotwire-vectors/*.pt`
  (+ source of truth `aorus:~/projects/science/brainscope/dirs.json`)
- brainscope deploy on aorus: rsync `brainscope/server.py` →
  `aorus:tmp/brainscope-test/brainscope/server.py`, then restart (step 1+2a)
- figures & their HTML sources: `~/projekty/brainscope/notes/steering_*.{html,png}`
  (re-render: `google-chrome --headless=new --screenshot=X.png
  --window-size=1200,H --force-device-scale-factor=2 file://$PWD/X.html`)
- application-side evals and parity live in the application's own
  (private) repo — this repo stays app-agnostic
