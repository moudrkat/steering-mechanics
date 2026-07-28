"""Bake SKOP-style mean-difference steering vectors from MWE datasets
(PREREG_CHANNELS.md H5). Construction mirrors SKOP: last-token residual
representations of behavior-matching vs non-matching completions, mean
difference per layer.

env: BAKE_MODEL, BAKE_DATA (jsonl with question/answer_matching_behavior/
answer_not_matching_behavior), BAKE_OUT (.pt), BAKE_8BIT, BAKE_N (default 150).
"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["BAKE_MODEL"]
DATA = os.path.expanduser(os.environ["BAKE_DATA"])
OUT = os.path.expanduser(os.environ["BAKE_OUT"])
N = int(os.environ.get("BAKE_N", "150"))

tok = AutoTokenizer.from_pretrained(MODEL)
if os.environ.get("BAKE_8BIT") == "1":
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda:0")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.to("cuda")
model.eval()

rows = [json.loads(l) for l in open(DATA) if l.strip()][:N]

def last_tok_layers(text):
    # template text already carries special tokens — do not add a second BOS
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        o = model(ids, output_hidden_states=True, use_cache=False)
    # hidden_states[0] = embeddings; [i+1] = output of block i (steer-hook frame)
    return torch.stack([h[0, -1].float().cpu() for h in o.hidden_states[1:]])

acc_pos = acc_neg = None
for i, r in enumerate(rows):
    base = tok.apply_chat_template([{"role": "user", "content": r["question"]}],
                                   add_generation_prompt=True, tokenize=False)
    hp = last_tok_layers(base + r["answer_matching_behavior"].strip())
    hn = last_tok_layers(base + r["answer_not_matching_behavior"].strip())
    acc_pos = hp if acc_pos is None else acc_pos + hp
    acc_neg = hn if acc_neg is None else acc_neg + hn
    if (i + 1) % 50 == 0:
        print(json.dumps({"done": i + 1, "of": len(rows)}), flush=True)

vec = (acc_pos - acc_neg) / len(rows)
torch.save(vec, OUT)
norms = [round(float(vec[i].norm()), 2) for i in range(0, vec.shape[0], max(1, vec.shape[0] // 8))]
print(json.dumps({"saved": OUT, "shape": list(vec.shape), "norm_samples": norms}), flush=True)
