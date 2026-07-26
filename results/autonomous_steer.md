# Autonomous steering run 2026-07-26

## STAGE 1 — Gemma-4-E4B steering (patched hotwire)

hotwire status: 
HOOK ATTACHES: YES (output changed -> patched hotwire steers Gemma)
unsteered: 'Praha je nádherné, historické město s bohatou architekturou, které je srdcem České republiky.'
steered  : 'Praň Vy V Pra Pra Ves For The City With Pl Set With As As Berjective Pra \\text{City Sentence As Ber '

## STAGE 2 — extract Gemma no-task vector (v3 recipe, on Gemma-4-E4B)

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
recipe=no_task_checklist_v3  model=google/gemma-4-E4B-it  dtype=torch.float16  quantize=8bit
144 advocate samples / 144 balanced samples
Loading weights:   0%|          | 0/2076 [00:00<?, ?it/s]Loading weights:  10%|▉         | 201/2076 [00:00<00:00, 1932.49it/s]Loading weights:  20%|██        | 421/2076 [00:00<00:00, 2083.16it/s]Loading weights:  30%|███       | 630/2076 [00:00<00:00, 2055.98it/s]Loading weights:  40%|████      | 836/2076 [00:02<00:05, 246.34it/s] Loading weights:  46%|████▋     | 965/2076 [00:03<00:05, 200.53it/s]Loading weights:  51%|█████     | 1052/2076 [00:03<00:05, 186.78it/s]Loading weights:  54%|█████▎    | 1115/2076 [00:04<00:05, 179.53it/s]Loading weights:  56%|█████▌    | 1162/2076 [00:04<00:05, 171.73it/s]Loading weights:  58%|█████▊    | 1199/2076 [00:04<00:05, 158.45it/s]Loading weights:  59%|█████▉    | 1228/2076 [00:05<00:05, 152.02it/s]Loading weights:  60%|██████    | 1252/2076 [00:05<00:05, 140.65it/s]Loading weights:  61%|██████▏   | 1272/2076 [00:05<00:05, 144.78it/s]Loading weights:  62%|██████▏   | 1291/2076 [00:05<00:05, 144.97it/s]Loading weights:  63%|██████▎   | 1309/2076 [00:05<00:06, 126.53it/s]Loading weights:  64%|██████▍   | 1324/2076 [00:06<00:05, 126.59it/s]Loading weights:  64%|██████▍   | 1339/2076 [00:06<00:05, 123.98it/s]Loading weights:  65%|██████▌   | 1353/2076 [00:06<00:05, 124.21it/s]Loading weights:  66%|██████▌   | 1367/2076 [00:06<00:05, 122.75it/s]Loading weights:  66%|██████▋   | 1380/2076 [00:06<00:05, 119.60it/s]Loading weights:  67%|██████▋   | 1393/2076 [00:06<00:05, 117.21it/s]Loading weights:  68%|██████▊   | 1406/2076 [00:06<00:05, 115.58it/s]Loading weights:  76%|███████▋  | 1586/2076 [00:06<00:00, 539.32it/s]Loading weights:  99%|█████████▊| 2047/2076 [00:06<00:00, 1609.50it/s]Loading weights: 100%|██████████| 2076/2076 [00:06<00:00, 297.97it/s] 
no_task_checklist_v3/advocate priors:   0%|          | 0/8 [00:00<?, ?it/s]no_task_checklist_v3/advocate priors:  12%|█▎        | 1/8 [00:02<00:17,  2.53s/it]no_task_checklist_v3/advocate priors:  25%|██▌       | 2/8 [00:04<00:14,  2.41s/it]no_task_checklist_v3/advocate priors:  38%|███▊      | 3/8 [00:07<00:11,  2.38s/it]no_task_checklist_v3/advocate priors:  50%|█████     | 4/8 [00:09<00:09,  2.36s/it]no_task_checklist_v3/advocate priors:  62%|██████▎   | 5/8 [00:11<00:07,  2.35s/it]no_task_checklist_v3/advocate priors:  75%|███████▌  | 6/8 [00:14<00:04,  2.38s/it]no_task_checklist_v3/advocate priors:  88%|████████▊ | 7/8 [00:16<00:02,  2.38s/it]no_task_checklist_v3/advocate priors: 100%|██████████| 8/8 [00:19<00:00,  2.37s/it]                                                                                   no_task_checklist_v3/balanced priors:   0%|          | 0/8 [00:00<?, ?it/s]no_task_checklist_v3/balanced priors:  12%|█▎        | 1/8 [00:02<00:16,  2.33s/it]no_task_checklist_v3/balanced priors:  25%|██▌       | 2/8 [00:04<00:13,  2.32s/it]no_task_checklist_v3/balanced priors:  38%|███▊      | 3/8 [00:06<00:11,  2.32s/it]no_task_checklist_v3/balanced priors:  50%|█████     | 4/8 [00:09<00:09,  2.32s/it]no_task_checklist_v3/balanced priors:  62%|██████▎   | 5/8 [00:11<00:06,  2.32s/it]no_task_checklist_v3/balanced priors:  75%|███████▌  | 6/8 [00:13<00:04,  2.32s/it]no_task_checklist_v3/balanced priors:  88%|████████▊ | 7/8 [00:16<00:02,  2.32s/it]no_task_checklist_v3/balanced priors: 100%|██████████| 8/8 [00:18<00:00,  2.32s/it]                                                                                   V_pref[no_task_checklist_v3] (42, 2560) torch.float16
||V|| range: min=0.72  max=53.83  argmax_layer=41
steerability screen: steerable (best L17 agreement 0.80)
saved -> /vecs/gemma-4-e4b/v_pref_no_task_v3.pt (+ .screen.json)
Gemma extraction OK
no Gemma vector to stage

## STAGE 1-2 COMPLETE — see compat result + extraction status above. (Qwen2.5-14B stage in autonomous_steer2.sh)

