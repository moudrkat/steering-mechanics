"""Standardized utility axis under steering: ARC-Challenge (300 items),
likelihood-scored (no generation) — accuracy per dose. Matches the utility
dimension of SKOP's evaluation with a deterministic, cheap protocol.

Scoring: per option, mean per-token logprob of the option text as the
assistant continuation; steering applied decode-only (continuation
positions), mirroring the factorization regime.

env: BMP_MODEL, BMP_VEC, BMP_INJ, BMP_8BIT, BMP_NOTHINK,
BMP_SCALES (default 0,3,5,8), BMP_DATA (arc300.json), BMP_OUT.
"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["BMP_MODEL"]
VEC = os.path.expanduser(os.environ["BMP_VEC"])
INJ = int(os.environ.get("BMP_INJ", "20"))
SCALES = [float(x) for x in os.environ.get("BMP_SCALES", "0,3,5,8").split(",")]
DATA = os.path.expanduser(os.environ["BMP_DATA"])
OUT = os.path.expanduser(os.environ["BMP_OUT"])
NOTHINK = os.environ.get("BMP_NOTHINK") == "1"

tok = AutoTokenizer.from_pretrained(MODEL)
if os.environ.get("BMP_8BIT") == "1":
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda:0")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.to("cuda")
model.eval()
v = torch.load(VEC, map_location="cpu", weights_only=True).float()[INJ].to("cuda", torch.bfloat16)

state = {"scale": 0.0, "plen": 0}
def hook(_m, _i, out):
    if state["scale"] == 0.0: return out
    h = (out[0] if isinstance(out, tuple) else out).clone()
    h[:, state["plen"]:, :] += (state["scale"] * v).to(h.dtype)
    return (h, *out[1:]) if isinstance(out, tuple) else h
layers = None
for name, mod in model.named_modules():
    if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 20:
        layers = mod; break
layers[INJ].register_forward_hook(hook)

items = json.load(open(DATA))

def option_logprob(prompt_ids, option_text, scale):
    oid = tok(option_text, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    full = torch.cat([prompt_ids, oid], dim=1)
    state["plen"] = prompt_ids.shape[1]
    state["scale"] = scale
    with torch.no_grad():
        logits = model(full, use_cache=False).logits
    state["scale"] = 0.0
    lp = torch.log_softmax(logits[0, prompt_ids.shape[1]-1:-1].float(), -1)
    tokens = full[0, prompt_ids.shape[1]:]
    return float(lp.gather(-1, tokens.unsqueeze(-1)).mean())

results = {str(s): [] for s in SCALES}
for i, it in enumerate(items):
    q = it["q"] + "\n" + "\n".join(f"({c['label']}) {c['text']}" for c in it["choices"]) \
        + "\nAnswer with the correct option."
    kw = {"enable_thinking": False} if NOTHINK else {}
    ids = tok.apply_chat_template([{"role": "user", "content": q}],
                                  add_generation_prompt=True, return_tensors="pt", **kw)
    if not isinstance(ids, torch.Tensor): ids = ids["input_ids"]
    ids = ids.to("cuda")
    for s in SCALES:
        lps = [option_logprob(ids, f" The answer is ({c['label']}) {c['text']}", s)
               for c in it["choices"]]
        pick = it["choices"][max(range(len(lps)), key=lambda k: lps[k])]["label"]
        results[str(s)].append(int(pick == it["answer"]))
    if (i + 1) % 50 == 0:
        print(json.dumps({"done": i + 1, "acc_so_far":
              {s: round(sum(r)/len(r), 3) for s, r in results.items()}}), flush=True)

json.dump(results, open(OUT, "w"))
print(json.dumps({"final_acc": {s: round(sum(r)/len(r), 4) for s, r in results.items()},
                  "n": len(items)}), flush=True)
print("BENCHMARK DONE", flush=True)
