"""Gemma-4-E4B run for PREREG_CHANNELS.md H4 (KV-share architectural
prediction). Injection at L25 output, decode-only; the entire watched band
L26-32 lies in the shared-KV region (producers L22/L23, below injection),
so keys and values there are clean BY CONSTRUCTION and the whole-attention
freeze (fattn) isolates the pattern/query channel.

Measures: KL(clean||steered), KL(clean||fattn) on decode rows; per-head
JSD from output_attentions when available. Verifies the architectural
premise directly: inputs to producer layers L22/L23 must be IDENTICAL
between clean and steered passes (they sit below the injection).
"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("GH4_MODEL", "google/gemma-4-E4B-it")
VEC = os.path.expanduser(os.environ.get("GH4_VEC",
      "~/projects/science/private-vectors/gemma-4-e4b/v_pref_no_task_v3.pt"))
INJ = int(os.environ.get("GH4_INJ", "25"))
BAND = list(range(INJ + 1, INJ + 8))
SCALES = [float(x) for x in os.environ.get("GH4_SCALES", "1.5,2.5,4").split(",")]
PRODUCERS = [22, 23]
MAXTOK = 48
OUT = os.path.expanduser(os.environ.get("GH4_OUT", "~/gemma_h4_out"))

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
from transformers import BitsAndBytesConfig
model = AutoModelForCausalLM.from_pretrained(MODEL,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    attn_implementation="eager", device_map="cuda:0")
model.eval()

layers = None
for name, mod in model.named_modules():
    if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 36:
        layers = mod
        print("layers module:", name, len(mod), flush=True)
        break
assert layers is not None, "no decoder layer stack found"
vrow = torch.load(VEC, map_location="cpu", weights_only=True).float()[INJ]

state = {"scale": 0.0, "plen": 0}
def steer_hook(_m, _i, out):
    if state["scale"] == 0.0: return out
    h = (out[0] if isinstance(out, tuple) else out).clone()
    v = (state["scale"] * vrow).to(h.dtype).to(h.device)
    h[:, state["plen"]:, :] += v
    return (h, *out[1:]) if isinstance(out, tuple) else h
layers[INJ].register_forward_hook(steer_hook)

clean_attn = {}
mode = {"freeze": False, "record": False}
def band_hook(idx):
    def h(module, args, kwargs, output):
        o = output[0] if isinstance(output, tuple) else output
        if mode["record"]:
            clean_attn[idx] = o.detach()
            return output
        if mode["freeze"]:
            oc = clean_attn[idx]
            return (oc, *output[1:]) if isinstance(output, tuple) else oc
        return output
    return h
for li in BAND:
    layers[li].self_attn.register_forward_hook(band_hook(li), with_kwargs=True)

prod_in = {}
def prod_hook(idx):
    def h(_m, args, kwargs):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        prod_in[idx] = hs.detach()
    return h
for li in PRODUCERS:
    layers[li].register_forward_pre_hook(prod_hook(li), with_kwargs=True)

def kl_rows(lc, lx, plen):
    rows = slice(plen - 1, lc.shape[1] - 1)
    pc = torch.log_softmax(lc[0, rows].float(), -1)
    px = torch.log_softmax(lx[0, rows].float(), -1)
    kl = (pc.exp() * (pc - px)).sum(-1)
    match = (lc[0, rows].argmax(-1) == lx[0, rows].argmax(-1)).float()
    return round(float(kl.mean()), 4), round(float(match.mean()), 4)

def jsd_band(ac, ax, plen):
    """per-(band-layer, head) base-2 JSD over decode rows; None if no attns"""
    if ac is None or ax is None: return None
    out = {}
    for li in BAND:
        a, b = ac[li][0].float()[:, plen:, :], ax[li][0].float()[:, plen:, :]
        m = 0.5 * (a + b)
        def kl2(p, q): return (p * (torch.log2(p + 1e-12) - torch.log2(q + 1e-12))).sum(-1)
        out[str(li)] = [round(float(x), 5) for x in (0.5 * kl2(a, m) + 0.5 * kl2(b, m)).mean(1)]
    return out

def forward(ids, scale, freeze=False, record=False, attns=False):
    state["scale"] = scale
    mode["freeze"], mode["record"] = freeze, record
    with torch.no_grad():
        o = model(ids, use_cache=False, output_attentions=attns)
    mode["freeze"] = mode["record"] = False
    state["scale"] = 0.0
    at = {li: o.attentions[li] for li in BAND} if attns and o.attentions is not None else None
    return o.logits, at

sanity_done = False
for pi, (cls, prompt) in enumerate(PROMPTS):
    outf = os.path.join(OUT, f"p{pi}.json")
    if os.path.exists(outf): continue
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor): ids = ids["input_ids"]
    ids = ids.to("cuda")
    plen = ids.shape[1]
    with torch.no_grad():
        full = model.generate(ids, max_new_tokens=MAXTOK, do_sample=False)
    state["plen"] = plen

    try:
        lc, ac = forward(full, 0.0, record=True, attns=True)
    except Exception as e:
        print(json.dumps({"attns_failed": str(e)[:100]}), flush=True)
        lc, ac = forward(full, 0.0, record=True)
    prod_clean = {k: v.clone() for k, v in prod_in.items()}

    if not sanity_done:
        l0, _ = forward(full, 0.0, freeze=True)
        kl0, m0 = kl_rows(lc, l0, plen)
        print(json.dumps({"sanity_fattn_s0_kl": kl0, "argmax": m0}), flush=True)
        sanity_done = True

    rec = {"prompt_idx": pi, "prompt_class": cls, "plen": plen, "doses": {}}
    for s in SCALES:
        try:
            ls, asx = forward(full, s, attns=ac is not None)
        except Exception:
            ls, asx = forward(full, s)
        prod_diff = max(float((prod_clean[k] - prod_in[k]).abs().max()) for k in PRODUCERS)
        d = {"producer_input_maxdiff": round(prod_diff, 6)}
        d["kl_steered"], d["match_steered"] = kl_rows(lc, ls, plen)
        lf, _ = forward(full, s, freeze=True)
        d["kl_fattn"], d["match_fattn"] = kl_rows(lc, lf, plen)
        j = jsd_band(ac, asx, plen)
        if j: d["jsd"] = j
        rec["doses"][str(s)] = d
    json.dump(rec, open(outf, "w"))
    print(json.dumps({"done": pi + 1, "of": len(PROMPTS)}), flush=True)
print("GEMMA_H4 DONE", flush=True)
