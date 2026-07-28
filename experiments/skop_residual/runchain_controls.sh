#!/bin/bash
# Controls round (CONTROLS_PREREG.md): E1 random-basis control,
# E2 fidelity build v2, E3 probe-v2 reruns of the key 07-27 arms.
# rsync the *.py of this directory to the box's /tmp first (same pattern
# as the earlier chains). Stages are independent — comment out freely.
export HF_HOME=~/projects/science/instruct-steer/hf-cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKOP_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export SKOP_VEC="~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"
export SKOP_INJ=20
PY=~/tmp/vllm-lens-test/.venv/bin/python

echo "=== E1a: build random-complement control vectors (CPU) ==="
$PY /tmp/random_projection_control.py 2>&1 | grep -E "^\{|SAVED|Error|Traceback"

echo "=== E3: probe v2 (N=24) on the reference arms ==="
SKOP_PROBE_SET=v2 SKOP_OUT="v_pref_no_task_qwen_skopres_v1.pt" \
SKOP_ARMS='[["baseline",null,0],["v_orig","orig",3.0],["v_skopres_v1","proj",3.0]]' \
SKOP_EFF_OUT="efficacy_probe_v2_refarms.json" \
$PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "^\{\"summary|DONE|Error|Traceback"

echo "=== E1b: probe v2 on each control (matched injected magnitude) ==="
for rank in 149 1536; do for seed in 1 2 3; do
  name="v_randctl_r${rank}_s${seed}.pt"
  scale=$($PY -c "import json,os;d=json.load(open(os.path.expanduser('~/skop_residual/randctl_diag.json')));print([c['matched_scale'] for c in d['controls'] if c['name']=='$name'][0])")
  SKOP_PROBE_SET=v2 SKOP_OUT="$name" \
  SKOP_ARMS="[[\"randctl_r${rank}_s${seed}\",\"proj\",$scale]]" \
  SKOP_EFF_OUT="efficacy_randctl_r${rank}_s${seed}.json" \
  $PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "^\{\"summary|DONE|Error|Traceback"
done; done

echo "=== E2: fidelity build v2 (shakedown expected on first run) ==="
SKOP_OUT="v_pref_no_task_qwen_skopres_v2.pt" \
SKOP_COMPARE="~/hotwire-vectors/v_pref_no_task_qwen_skopres.pt" \
SKOP_RISK=0.10 SKOP_GAMMA=0.7 SKOP_PCAP=8 \
$PY /tmp/skop_residual_build_v2.py 2>&1 | grep -E "^\{|SAVED|Error|Traceback"
echo "NOTE: check diag_v2.json v0_overlap first — run a v2 sweep only if"
echo "the removed components disagree with v0 (CONTROLS_PREREG E2)."
echo "=== CHAIN COMPLETE ==="
