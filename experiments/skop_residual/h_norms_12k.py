"""Deployment-length h-norms on GPU: mean ||h[L]|| at ~12k tokens.
Synthetic varied CZ/EN text (no private scaffolds). Reports per-layer mean
over all positions AND over the last 1024 positions (decode-relevant).
"""
import torch, json, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

SENTS = [
 "Dnes rano jsem si uvarila kavu a chvili jsem se divala z okna na dest.",
 "The quarterly report shows a steady increase in user engagement across all regions.",
 "Kdyz jsem byla mala, jezdili jsme kazde leto k babicce na venkov.",
 "Please remember that the meeting has been moved to Thursday afternoon.",
 "V obchode meli slevu na jablka, tak jsem koupila cely kosik.",
 "The train was delayed by twenty minutes due to a signal failure near the station.",
 "Muj kamarad se prave vratil z cest po Japonsku a porad o tom mluvi.",
 "Scientists have long debated the role of sleep in memory consolidation.",
 "Odpoledne pujdu vyzvednout deti ze skoly a pak na nakup.",
 "The recipe calls for two cups of flour, a pinch of salt, and three eggs.",
 "Sousedka mi vcera prinesla vyborny kolac, ktery sama upekla.",
 "In the long run, consistent small habits tend to outperform bursts of effort.",
]
def build_text(tok, target=12288):
    parts, i = [], 0
    while True:
        parts.append(SENTS[i % len(SENTS)] + f" (poznamka {i})")
        i += 1
        if i % 50 == 0:
            n = len(tok("\n".join(parts))["input_ids"])
            if n > target: break
    return "\n".join(parts)

import os
for MODEL in os.environ.get("H12K_MODELS", "Qwen/Qwen3-4B-Instruct-2507").split(","):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    text = build_text(tok)
    msgs = [{"role": "user", "content": text + "\n\nShrn prosim hlavni temata."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    iid = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"])[:, :12288].to("cuda")
    inner = getattr(model, "model", model)
    with torch.no_grad():
        out = inner(input_ids=iid, output_hidden_states=True)
    allm, lastm = [], []
    for h in out.hidden_states:
        n = h[0].float().norm(dim=-1)
        allm.append(round(float(n.mean()), 3))
        lastm.append(round(float(n[-1024:].mean()), 3))
    print(json.dumps({"model": MODEL, "T": int(iid.shape[1]),
                      "mean_all": allm, "mean_last1024": lastm}))
    del model, out
    torch.cuda.empty_cache()
