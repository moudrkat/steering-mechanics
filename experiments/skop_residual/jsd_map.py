"""Per-head JSD localization map for ANY model (chain H, post-freeze).

Generalizes the Qwen3-4B mega-battery map question — "is steering damage
localized to a few heads one layer above injection?" — to the remaining
families. Teacher-forced decode-only replay; arms: steered (working dose),
sham (1e-6, instrument floor), matched-norm random vector (lightning-rod
null). Per-head base-2 JSD over decode rows, ALL layers above injection.

Env: JM_MODEL, JM_VEC, JM_INJ, JM_8BIT=1, JM_NOTHINK=1, JM_SCALE (default 3),
     JM_OUT. Scores only.
"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["JM_MODEL"]
VEC = os.path.expanduser(os.environ["JM_VEC"])
INJ = int(os.environ["JM_INJ"])
SCALE = float(os.environ.get("JM_SCALE", "3"))
MAXTOK = 48
OUT = os.path.expanduser(os.environ.get("JM_OUT", "~/jsd_map_out"))

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
TOPICS = ["rozdil mezi TCP a UDP protokolem", "planovani rodinneho rozpoctu",
 "the difference between compiled and interpreted languages",
 "priprava na pohovor do prace", "the water cycle in nature",
 "trideni odpadu v domacnosti", "how vaccines train the immune system",
 "zaklady pece o pokojove rostliny"]
TEMPLATES = ["Vysvetli strucne a po bodech: {t}.",
 "Explain to a beginner, in two short paragraphs: {t}."]
PROMPTS = [("task", p) for p in TASK] + \
          [("neutral", tpl.format(t=t)) for t in TOPICS for tpl in TEMPLATES]

os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
kw = {"attn_implementation": "eager", "device_map": "cuda:0"}
if os.environ.get("JM_8BIT") == "1":
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
else:
    kw["torch_dtype"] = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
model.eval()

layers = None
for name, mod in model.named_modules():
    if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 24:
        layers = mod
        print("layers module:", name, len(mod), flush=True)
        break
assert layers is not None
NL = len(layers)
WATCH = list(range(INJ + 1, NL))
vrow = torch.load(VEC, map_location="cpu", weights_only=True).float()[INJ]
g = torch.Generator().manual_seed(149)
rrow = torch.randn(vrow.shape, generator=g)
rrow *= vrow.norm() / rrow.norm()

state = {"scale": 0.0, "plen": 0, "vec": vrow}
def steer_hook(_m, _i, out):
    if state["scale"] == 0.0: return out
    h = (out[0] if isinstance(out, tuple) else out).clone()
    v = (state["scale"] * state["vec"]).to(h.dtype).to(h.device)
    h[:, state["plen"]:, :] += v
    return (h, *out[1:]) if isinstance(out, tuple) else h
layers[INJ].register_forward_hook(steer_hook)

def kl_rows(lc, lx, plen):
    rows = slice(plen - 1, lc.shape[1] - 1)
    pc = torch.log_softmax(lc[0, rows].float(), -1)
    px = torch.log_softmax(lx[0, rows].float(), -1)
    kl = (pc.exp() * (pc - px)).sum(-1)
    return round(float(kl.mean()), 4)

def jsd_map(ac, ax, plen):
    out = []
    for li in WATCH:
        a, b = ac[li][0].float()[:, plen:, :], ax[li][0].float()[:, plen:, :]
        m = 0.5 * (a + b)
        def kl2(p, q): return (p * (torch.log2(p + 1e-12) - torch.log2(q + 1e-12))).sum(-1)
        out.append([round(float(x), 5) for x in (0.5 * kl2(a, m) + 0.5 * kl2(b, m)).mean(1)])
    return out

def forward(ids, scale, vec=None):
    state["scale"] = scale
    if vec is not None: state["vec"] = vec
    with torch.no_grad():
        o = model(ids, use_cache=False, output_attentions=True)
    state["scale"] = 0.0; state["vec"] = vrow
    return o.logits, list(o.attentions)

for pi, (cls, prompt) in enumerate(PROMPTS):
    outf = os.path.join(OUT, f"p{pi}.json")
    if os.path.exists(outf): continue
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
              add_generation_prompt=True, return_tensors="pt",
              enable_thinking=False if os.environ.get("JM_NOTHINK") == "1" else None)
    except TypeError:
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
              add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor): ids = ids["input_ids"]
    ids = ids.to("cuda")
    plen = ids.shape[1]
    with torch.no_grad():
        full = model.generate(ids, max_new_tokens=MAXTOK, do_sample=False)
    state["plen"] = plen

    lc, ac = forward(full, 0.0)
    rec = {"prompt_idx": pi, "prompt_class": cls, "plen": plen,
           "layers": WATCH, "n_heads": len(ac[WATCH[0]][0]), "inj": INJ,
           "arms": {}}
    for arm, s, vec in (("steered", SCALE, vrow), ("sham", 1e-6, vrow),
                        ("rand", SCALE, rrow)):
        lx, axx = forward(full, s, vec)
        rec["arms"][arm] = {"scale": s, "kl": kl_rows(lc, lx, plen),
                            "jsd": jsd_map(ac, axx, plen)}
        del lx, axx
    del lc, ac
    json.dump(rec, open(outf, "w"))
    print(json.dumps({"done": pi + 1, "of": len(PROMPTS)}), flush=True)
print("JSD_MAP DONE", flush=True)
