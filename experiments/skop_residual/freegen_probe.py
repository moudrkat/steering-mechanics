"""Free-generation quality probe across doses — validates the teacher-forced
KL damage proxy used by qk_freeze/qk_freeze2 (MEGA-RIGOR round, 2026-07-28).

For each dose: k=3 sampled generations (temp 0.7, seeds 0/1/2) per prompt on
the 24 Czech task prompts. Scores only: task-offer regex hit, 4-gram
repetition, unique-token ratio, 3-gram loop coherence, word count. The
per-dose curves of these against the qk_freeze2 KL table give the
proxy-validity correlation.
"""
import json, os, re, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["FGP_MODEL"]
VEC = os.path.expanduser(os.environ["FGP_VEC"])
INJ = int(os.environ.get("FGP_INJ", "20"))
SCALES = [float(x) for x in os.environ.get("FGP_SCALES", "0,1,1.5,2,3,5,8").split(",")]
SEEDS = [0, 1, 2]
MAXTOK = 80
EIGHTBIT = os.environ.get("FGP_8BIT") == "1"
NOTHINK = os.environ.get("FGP_NOTHINK") == "1"
OUT = os.path.expanduser(os.environ["FGP_OUT"])

TASK = [
 "Zaloz mi ukol: koupit darek pro mamku do patku.",
 "Udelej mi prosim checklist na stehovani.",
 "Pripomen mi zitra v 8 zavolat doktorovi.",
 "Vytvor mi seznam ukolu na tento tyden podle toho, co jsem ti rikala: uklidit, nakoupit, dopsat report.",
 "Muzes mi nastavit upominku na nedeli vecer?",
 "Chci abys mi udelal to-do list na pripravu oslavy.",
 "Pridej mi ukol: vyzvednout leky v lekarne.",
 "Nastav mi upominku na stredu rano na schuzku se zubarem.",
 "Udelej mi seznam veci, ktere mam zaridit pred odjezdem na chatu.",
 "Pripomen mi v nedeli pripravit obed pro babicku.",
 "Udelej mi checklist na baleni kufru na dovolenou.",
 "Pripomen mi v patek odpoledne poslat fakturu.",
 "Vytvor mi to-do list na jarni uklid.",
 "Zapis mi ukol: objednat auto do servisu.",
 "Muzes mi vytvorit seznam ukolu na vikend? Potrebuju vyprat, vyluxovat a nakoupit.",
 "Chci checklist na pripravu na pulmaraton.",
 "Zaloz mi ukol koupit listky na vlak do Brna.",
 "Pripomen mi zitra vecer zalit kytky.",
 "Vytvor ukol: domluvit doucovani anglictiny pro deti.",
 "Potrebuju to-do list na prestavbu kuchyne, rozdel to na faze.",
 "Nastav mi pripominku na konec mesice zaplatit najem.",
 "Muzes mi zalozit ukol na dokonceni prezentace do ctvrtka?",
 "Udelej checklist, co vsechno vzit na sluzebni cestu.",
 "Zapis mi na seznam ukolu: vratit knihy do knihovny.",
]
VIOL = re.compile(r"(ukol|úkol|checklist|seznam ukolu|seznam úkolů|to-?do|upomink|upomínk|pripomen|připomen|zalozil jsem|založil jsem|vytvoril jsem|vytvořil jsem|nastavil jsem)", re.I)

os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if EIGHTBIT:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.to("cuda")
model.eval()
v = torch.load(VEC, map_location="cpu", weights_only=True).float()[INJ].to("cuda", torch.bfloat16)

state = {"scale": 0.0}
def hook(_m, _i, out):
    if state["scale"] == 0.0: return out
    h = out[0] if isinstance(out, tuple) else out
    return (h + state["scale"] * v,) + tuple(out[1:]) if isinstance(out, tuple) else h + state["scale"] * v
layers = None
for _name, _mod in model.named_modules():
    if _name.endswith(".layers") and hasattr(_mod, "__len__") and len(_mod) >= 24:
        layers = _mod
        break
assert layers is not None, "no decoder layer stack found"
layers[INJ].register_forward_hook(hook)

def metrics(txt):
    w = txt.split()
    if len(w) < 8: return {"rep4": 0.0, "uniq": 1.0, "loop": len(w) < 4, "nw": len(w)}
    g4 = Counter(" ".join(w[i:i+4]) for i in range(len(w)-3))
    rep4 = sum(c for c in g4.values() if c > 1) / max(len(w)-3, 1)
    g3 = Counter(tuple(w[i:i+3]) for i in range(len(w)-2))
    loop = bool(g3) and g3.most_common(1)[0][1] / len(w) > 0.2
    return {"rep4": round(rep4, 3), "uniq": round(len(set(w))/len(w), 3),
            "loop": loop, "nw": len(w)}

for s in SCALES:
    outf = os.path.join(OUT, f"s{str(s).replace('.','p')}.json")
    if os.path.exists(outf): continue
    rows = []
    for pi, p in enumerate(TASK):
        kw = {"enable_thinking": False} if NOTHINK else {}
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt", **kw)
        if not isinstance(ids, torch.Tensor): ids = ids["input_ids"]
        ids = ids.to("cuda")
        for seed in SEEDS:
            torch.manual_seed(seed)
            state["scale"] = s
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAXTOK, do_sample=True,
                                     temperature=0.7, top_p=0.95)
            state["scale"] = 0.0
            txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            rows.append({"prompt_idx": pi, "seed": seed, "scale": s,
                         "viol": bool(VIOL.search(txt)), **metrics(txt)})
    json.dump(rows, open(outf, "w"))
    nv = sum(r["viol"] for r in rows); nl = sum(r["loop"] for r in rows)
    print(json.dumps({"scale": s, "viol": f"{nv}/{len(rows)}", "loops": nl}), flush=True)
print("FREEGEN DONE", flush=True)
