"""A/B sanity: original v vs SKOP-projected v_bar on Gemma, short context, CPU.
Steers ALL positions at L25 (full-steer probe regime). Greedy, 60 tokens.
Reads both vectors, generates per (prompt x scale x vector), prints JSON.
"""
import torch, json, os
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("SKOP_MODEL", "google/gemma-4-E4B-it")
INJ = int(os.environ.get("SKOP_INJ", "25"))
V_ORIG = os.path.expanduser(os.environ.get("SKOP_VEC", "~/hotwire-vectors/v_pref_no_task_gemma.pt"))
V_PROJ = os.path.expanduser("~/hotwire-vectors/" + os.environ.get("SKOP_OUT", "v_pref_no_task_gemma_skopres.pt"))
SCALES = [float(x) for x in os.environ.get("SKOP_SCALES", "3,6").split(",")]
MAXTOK = 60

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.to("cuda")
model.eval()
layers = None
for name, mod in model.named_modules():
    if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 30:
        layers = mod; break

PROMPTS = [
 "Stehuju se pristi tyden do noveho bytu a je toho na me moc. Nevim, kde zacit.",
 "Chci se konecne naucit spanelsky, ale porad to odkladam. Poradis mi?",
]

state = {"vec": None, "scale": 0.0}
def hook(mod, inp, out):
    if state["vec"] is None: return out
    if isinstance(out, tuple):
        return (out[0] + state["scale"] * state["vec"],) + out[1:]
    return out + state["scale"] * state["vec"]
h = layers[INJ].register_forward_hook(hook)

vo = torch.load(V_ORIG, map_location="cpu", weights_only=True).float()[INJ].to("cuda", torch.bfloat16)
vp = torch.load(V_PROJ, map_location="cpu", weights_only=True).float()[INJ].to("cuda", torch.bfloat16)
results = []
for p in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, return_tensors="pt")
    iid = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to("cuda")
    for label, vec in [("baseline", None), ("v_orig", vo), ("v_skopres", vp)]:
        for s in ([0.0] if vec is None else SCALES):
            state["vec"] = vec; state["scale"] = s
            with torch.no_grad():
                out = model.generate(iid, max_new_tokens=MAXTOK, do_sample=False)
            txt = tok.decode(out[0, iid.shape[1]:], skip_special_tokens=True)
            results.append({"prompt": p[:40], "vector": label, "scale": s,
                            "output": txt})
            print(json.dumps(results[-1], ensure_ascii=False))
h.remove()
with open(os.path.expanduser("~/skop_residual/ab_results.json"), "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("DONE")
