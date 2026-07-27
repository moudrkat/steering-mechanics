#!/bin/bash
export HF_HOME=~/projects/science/instruct-steer/hf-cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKOP_MODEL="Qwen/Qwen3-8B-FP8"
export SKOP_VEC="~/projects/science/private-vectors/qwen3-8b/v_pref_no_task_checklist_v4.pt"
export SKOP_WQ_SRC="~/projects/science/instruct-steer/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/*/*.safetensors"
export SKOP_INJ=20
export SKOP_NOTHINK=1
PY=~/tmp/vllm-lens-test/.venv/bin/python
echo "=== STAGE A: build 8B v1-projection (risk 10%, gamma 0.7, pcap 8) ==="
export SKOP_OUT="v_pref_no_task_8b_v4_skopres.pt"
export SKOP_RISK=0.1 SKOP_GAMMA=0.7 SKOP_PCAP=8
$PY /tmp/skop_residual_build.py 2>&1 | grep -E "^\{|SAVED|Traceback|Error|OutOfMemory" | tail -8
echo "=== STAGE B: efficacy probe 8B (baseline, v@3, vbar@3, v@8, vbar@8) ==="
export SKOP_ARMS='[["baseline",null,0],["v_orig_s3","orig",3.0],["vbar_v1_s3","proj",3.0],["v_orig_s8","orig",8.0],["vbar_v1_s8","proj",8.0]]'
export SKOP_EFF_OUT="efficacy_8b_v4.json"
$PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "^\{|DONE|Traceback|Error|OutOfMemory" | head -40
echo "=== CHAIN COMPLETE ==="
