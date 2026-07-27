#!/bin/bash
export HF_HOME=~/hf-cache2 HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKOP_MODEL="Qwen/Qwen3-8B-FP8"
export SKOP_VEC="~/projects/science/private-vectors/qwen3-8b/v_pref_no_task_checklist_v3.pt"
export SKOP_NOTHINK=1
export SKOP_WQ_SRC="~/projects/science/instruct-steer/hf-cache/hub/models--Qwen--Qwen3-8B/snapshots/*/*.safetensors"
PY=~/tmp/vllm-lens-test/.venv/bin/python
echo "=== STAGE A: build v3-8B @L20 (v1 settings) ==="
export SKOP_INJ=20
export SKOP_OUT="v_pref_no_task_8b_v3_skopres.pt"
export SKOP_RISK=0.1 SKOP_GAMMA=0.7 SKOP_PCAP=8
$PY /tmp/skop_residual_build.py 2>&1 | grep -E "norm_kept|SAVED|Traceback|Error" | head -3
echo "=== STAGE B: probe @L20 (baseline, v3@3, vbar@3, v3@8, vbar@8) ==="
export SKOP_ARMS='[["baseline",null,0],["v3_s3","orig",3.0],["vbar_s3","proj",3.0],["v3_s8","orig",8.0],["vbar_s8","proj",8.0]]'
export SKOP_EFF_OUT="efficacy_8b_v3.json"
$PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "summary_regex|DONE|Traceback|Error" | head -4
echo "=== STAGE C: L15 mode-break build + probe ==="
export SKOP_INJ=15
export SKOP_OUT="v_pref_no_task_8b_v3_L15_skopres.pt"
$PY /tmp/skop_residual_build.py 2>&1 | grep -E "norm_kept|SAVED|Traceback|Error" | head -3
export SKOP_ARMS='[["v3_L15_s8","orig",8.0],["vbar_L15_s8","proj",8.0]]'
export SKOP_EFF_OUT="modebreak_8b_L15.json"
$PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "summary_regex|^\{\"arm\"|DONE|Traceback|Error" | head -18
echo "=== CHAIN COMPLETE ==="
