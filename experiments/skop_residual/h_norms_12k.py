"""Deployment-length h-norms via streaming hooks (no hidden-state retention).
Reports mean ||h|| per BLOCK OUTPUT (index = block number directly; the old
version reported hidden_states[i] which is the output of block i-1 - beware
when comparing). SKOP_8BIT=1 loads 8-bit."""
import torch, json, os
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
def build_text(tok, target=int(os.environ.get("H12K_T", "12288"))):
    parts, i = [], 0
    while True:
        parts.append(SENTS[i % len(SENTS)] + f" (poznamka {i})")
        i += 1
        if i % 50 == 0:
            if len(tok("\n".join(parts))["input_ids"]) > target: break
    return "\n".join(parts)

for MODEL in os.environ.get("H12K_MODELS", "Qwen/Qwen3-4B-Instruct-2507").split(","):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if os.environ.get("SKOP_8BIT") == "1":
        from transformers import BitsAndBytesConfig
        model = AutoModelForCausalLM.from_pretrained(MODEL,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
        model.to("cuda")
    model.eval()
    layers = None
    for name, mod in model.named_modules():
        if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 30:
            layers = mod; break
    stats = {}
    def mk(i):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            n = o[0].float().norm(dim=-1)
            stats[i] = (round(float(n.mean()), 3), round(float(n[-1024:].mean()), 3))
        return hook
    hs = [layers[i].register_forward_hook(mk(i)) for i in range(len(layers))]
    text = build_text(tok)
    ids = tok.apply_chat_template([{"role": "user", "content": text + "\n\nShrn prosim hlavni temata."}],
                                  add_generation_prompt=True, return_tensors="pt")
    iid = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"])[:, :int(os.environ.get("H12K_T", "12288"))].to("cuda")
    with torch.no_grad():
        model(input_ids=iid, use_cache=False)
    for h in hs: h.remove()
    print(json.dumps({"model": MODEL, "T": int(iid.shape[1]),
        "block_out_norm_mean_all": [stats[i][0] for i in range(len(layers))],
        "block_out_norm_last1024": [stats[i][1] for i in range(len(layers))]}))
    del model
    torch.cuda.empty_cache()
