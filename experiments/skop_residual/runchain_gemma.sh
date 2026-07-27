#!/bin/bash
export HF_HOME=~/hf-cache2 HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKOP_MODEL="google/gemma-4-E4B-it"
export SKOP_VEC="~/hotwire-vectors/v_pref_no_task_gemma.pt"
export SKOP_INJ=25
export SKOP_8BIT=1
export SKOP_WQ_SRC="~/projects/science/instruct-steer/hf-cache/hub/models--google--gemma-4-E4B-it/snapshots/*/*.safetensors"
PY=~/tmp/vllm-lens-test/.venv/bin/python
echo "=== STAGE 1: Gemma 12k h-norms (8bit) ==="
export H12K_MODELS="google/gemma-4-E4B-it"
export H12K_T=8192
$PY /tmp/h_norms_12k.py 2>&1 | grep -E "^\{|Traceback|Error|OutOfMemory" | tail -3
echo "=== STAGE 2: Gemma SKOP build (v1 settings) ==="
export SKOP_OUT="v_pref_no_task_gemma_skopres.pt"
export SKOP_RISK=0.1 SKOP_GAMMA=0.7 SKOP_PCAP=8
$PY /tmp/skop_residual_build.py 2>&1 | grep -E "^\{|SAVED|Traceback|Error|OutOfMemory" | tail -6
echo "=== STAGE 3: probe (baseline, v@2.5, v@3, vbar@2.5, vbar@3) ==="
export SKOP_ARMS='[["baseline",null,0],["v_s25","orig",2.5],["v_s3","orig",3.0],["vbar_s25","proj",2.5],["vbar_s3","proj",3.0]]'
export SKOP_EFF_OUT="efficacy_gemma.json"
$PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "^\{|DONE|Traceback|Error|OutOfMemory" | head -40
echo "=== CHAIN COMPLETE ==="
