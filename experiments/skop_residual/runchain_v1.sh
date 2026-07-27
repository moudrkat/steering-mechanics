#!/bin/bash
export HF_HOME=~/projects/science/instruct-steer/hf-cache HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SKOP_MODEL="Qwen/Qwen3-4B-Instruct-2507"
export SKOP_VEC="~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"
export SKOP_INJ=20
PY=~/tmp/vllm-lens-test/.venv/bin/python
echo "=== STAGE A: efficacy probe v0 (baseline, v@3, vbar_v0@4.7) ==="
export SKOP_OUT="v_pref_no_task_qwen_skopres.pt"
export SKOP_ARMS='[["baseline",null,0],["v_orig_s3","orig",3.0],["vbar_v0_s47","proj",4.7]]'
export SKOP_EFF_OUT="efficacy_v0.json"
$PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "^\{|DONE|Traceback|Error" | head -30
echo "=== STAGE B: build v1 (risk 10%, gamma 0.7, pcap 8) ==="
export SKOP_OUT="v_pref_no_task_qwen_skopres_v1.pt"
export SKOP_RISK=0.1 SKOP_GAMMA=0.7 SKOP_PCAP=8
$PY /tmp/skop_residual_build.py 2>&1 | grep -E "^\{|SAVED|Traceback|Error" | tail -6
echo "=== STAGE C: efficacy + coherence probe v1 ==="
export SKOP_ARMS='[["vbar_v1_s3","proj",3.0],["vbar_v1_s4","proj",4.0],["vbar_v1_s6","proj",6.0]]'
export SKOP_EFF_OUT="efficacy_v1.json"
$PY /tmp/skop_efficacy_probe.py 2>&1 | grep -E "^\{|DONE|Traceback|Error" | head -30
echo "=== CHAIN COMPLETE ==="
