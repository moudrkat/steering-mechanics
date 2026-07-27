#!/bin/bash
export HF_HOME=~/projects/science/instruct-steer/hf-cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKOP_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export SKOP_VEC="~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"
export SKOP_INJ=20
export SKOP_OUT="v_pref_no_task_qwen_skopres.pt"
export SKOP_SCALES="3,4.7,8,12.5"
export SKOP_AB_OUT="ab_results_qwen.json"
export H12K_MODELS="Qwen/Qwen3-4B-Instruct-2507"
PY=~/tmp/vllm-lens-test/.venv/bin/python
echo "=== STAGE 1: h_norms_12k (qwen) ==="
$PY /tmp/h_norms_12k.py 2>&1 | grep -E "^\{|Error|Traceback" | tail -4
$PY /tmp/skop_ab_test.py 2>&1 | grep -E "^\{|DONE|Error|Traceback" | tail -16
echo "=== CHAIN COMPLETE ==="
