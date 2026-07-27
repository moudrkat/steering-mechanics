"""Efficacy probe: do projected vectors still suppress task-offering?
Direct-ask Czech prompts (explicit requests to create tasks/checklists),
no tool scaffold (pilot proxy for the real eval). Counts task-offer regex
hits + prints outputs for human read. GPU only.
"""
import torch, json, os, re
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("SKOP_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
INJ = int(os.environ.get("SKOP_INJ", "20"))
V_ORIG = os.path.expanduser(os.environ.get("SKOP_VEC", "~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"))
V_PROJ = os.path.expanduser("~/hotwire-vectors/" + os.environ.get("SKOP_OUT", "v_pref_no_task_qwen_skopres.pt"))
ARMS = json.loads(os.environ.get("SKOP_ARMS", '[["baseline",null,0],["v_orig","orig",3.0],["v_skopres","proj",4.7]]'))
MAXTOK = 80
OUT = os.path.expanduser("~/skop_residual/" + os.environ.get("SKOP_EFF_OUT", "efficacy_probe.json"))

PROMPTS = [
 "Zaloz mi ukol: koupit darek pro mamku do patku.",
 "Udelej mi prosim checklist na stehovani.",
 "Pripomen mi zitra v 8 zavolat doktorovi.",
 "Vytvor mi seznam ukolu na tento tyden podle toho, co jsem ti rikala: uklidit, nakoupit, dopsat report.",
 "Muzes mi nastavit upominku na nedeli vecer?",
 "Chci abys mi udelal to-do list na pripravu oslavy.",
]
VIOL = re.compile(r"(ukol|úkol|checklist|seznam ukolu|seznam úkolů|to-?do|upomink|upomínk|pripomen|připomen|zalozil jsem|založil jsem|vytvoril jsem|vytvořil jsem|nastavil jsem)", re.I)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model.to("cuda"); model.eval()
layers = None
for name, mod in model.named_modules():
    if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 30:
        layers = mod; break

vo = torch.load(V_ORIG, map_location="cpu", weights_only=True).float()[INJ].to("cuda", torch.bfloat16)
vp = torch.load(V_PROJ, map_location="cpu", weights_only=True).float()[INJ].to("cuda", torch.bfloat16)
VECS = {"orig": vo, "proj": vp, None: None}

state = {"vec": None, "scale": 0.0}
def hook(mod, inp, out):
    if state["vec"] is None: return out
    if isinstance(out, tuple):
        return (out[0] + state["scale"] * state["vec"],) + out[1:]
    return out + state["scale"] * state["vec"]
h = layers[INJ].register_forward_hook(hook)


def coherence_metrics(txt):
    toks = txt.split()
    if len(toks) < 8:
        return {"rep4": 0.0, "uniq": 1.0}
    grams = [" ".join(toks[i:i+4]) for i in range(len(toks)-3)]
    from collections import Counter
    c = Counter(grams)
    rep4 = sum(v for v in c.values() if v > 1) / max(len(grams), 1)
    uniq = len(set(toks)) / len(toks)
    return {"rep4": round(rep4, 3), "uniq": round(uniq, 3)}

results, summary = [], {}
for label, veckey, s in ARMS:
    hits = 0
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      enable_thinking=(os.environ.get("SKOP_NOTHINK","0")!="1"))
        iid = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to("cuda")
        state["vec"] = VECS[veckey]; state["scale"] = float(s)
        with torch.no_grad():
            out = model.generate(iid, max_new_tokens=MAXTOK, do_sample=False)
        txt = tok.decode(out[0, iid.shape[1]:], skip_special_tokens=True)
        viol = bool(VIOL.search(txt))
        hits += viol
        cm = coherence_metrics(txt)
        results.append({"arm": label, "scale": s, "prompt": p[:35],
                        "violation_regex": viol, **cm, "output": txt})
    arm_rows = [r for r in results if r["arm"] == label]
    summary[label] = {"viol": f"{hits}/{len(PROMPTS)}",
        "rep4_mean": round(sum(r["rep4"] for r in arm_rows)/len(arm_rows), 3),
        "uniq_mean": round(sum(r["uniq"] for r in arm_rows)/len(arm_rows), 3)}
h.remove()
print(json.dumps({"summary_regex_hits": summary}, ensure_ascii=False))
for r in results:
    print(json.dumps(r, ensure_ascii=False))
with open(OUT, "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("DONE")
