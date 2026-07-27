#!/bin/bash
export HF_HOME=~/hf-cache2 HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKOP_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export SKOP_VEC="~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"
export SKOP_INJ=20
PY=~/tmp/vllm-lens-test/.venv/bin/python
echo "=== REFERENCE ARMS (v@3, v@4) ==="
export SKOP_OUT="v_pref_no_task_qwen_skopres_v1.pt"
export SKOP_ARMS='[["v_orig_s3","orig",3.0],["v_orig_s4","orig",4.0]]'
export SKOP_EFF_OUT="sweep_ref.json"
$PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "summary_regex" | head -2
for CFG in "0.15 0.8 16 A" "0.2 0.7 8 B" "0.1 0.8 16 C" "0.15 0.7 8 D"; do
  set -- $CFG
  export SKOP_RISK=$1 SKOP_GAMMA=$2 SKOP_PCAP=$3
  TAG=$4
  export SKOP_OUT="v_sweep_${TAG}.pt"
  echo "=== CFG $TAG: risk=$1 gamma=$2 pcap=$3 ==="
  $PY /tmp/skop_residual_build.py 2>&1 | grep -E "norm_kept|SAVED|Traceback" | head -3
  export SKOP_ARMS="[[\"vbar_${TAG}_s3\",\"proj\",3.0],[\"vbar_${TAG}_s4\",\"proj\",4.0],[\"vbar_${TAG}_s5\",\"proj\",5.0]]"
  export SKOP_EFF_OUT="sweep_${TAG}.json"
  $PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "summary_regex" | head -2
done
echo "=== SWEEP COMPLETE ==="
