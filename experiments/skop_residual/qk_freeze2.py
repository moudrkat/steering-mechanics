"""Follow-up to qk_freeze.py (FINDINGS L), same setting (decode-only s*v[20]
at L20 output, Qwen3-4B, teacher-forced, N=40):

A) Complete the 2x2(+MLP) damage factorization on the L21-27 band:
   - fpat  : clean PATTERNS x steered values   (= qk_freeze "fband", rerun)
   - fval  : steered patterns x clean VALUES
   - fattn : entire attention output clean     (residual damage = MLP/skip only)
   All exact-at-s0 by construction (sanity asserted).

B) Linearity ladder for the induced query/key perturbation at L21:
   dq(s) = q_steered(s) - q_clean per head. If ||dq||/s is flat and the
   direction cos(dq(s), dq(s_min)) stays ~1, the residual->q map is linear
   and the JSD nonlinearity lives in softmax; otherwise the map itself
   (input-LN + q_norm) is nonlinear -> first-order projections must fail.

Scores only. Doses: KL arms at 3/5/8; ladder at 0.5/1/2/3/5/8.
"""
import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL = os.environ.get("QKF2_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
VEC = os.path.expanduser(os.environ.get("QKF2_VEC", "~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"))
INJ, ATTN_L = int(os.environ.get("QKF2_INJ", "20")), int(os.environ.get("QKF2_INJ", "20")) + 1
BAND = list(range(ATTN_L, ATTN_L + 7))
KL_SCALES = [3.0, 5.0, 8.0]
LADDER = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
MAXTOK = 48
EIGHTBIT = os.environ.get("QKF2_8BIT") == "1"
NOTHINK = os.environ.get("QKF2_NOTHINK") == "1"   # hard no-think for hybrid 8B
OUT = os.path.expanduser(os.environ.get("QKF2_OUT", "~/qk_freeze2_out"))

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
NEUTRAL = [tpl.format(t=t) for t in TOPICS for tpl in TEMPLATES]
PROMPTS = [("task", p) for p in TASK] + [("neutral", p) for p in NEUTRAL]

os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
if EIGHTBIT:
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        attn_implementation="eager", device_map="cuda:0")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    model.to("cuda")
model.eval()
layers = model.model.layers
v20 = torch.load(VEC, map_location="cpu", weights_only=True).float()[INJ].to("cuda", torch.bfloat16)
CFG = model.config
NH, NKV, HD = CFG.num_attention_heads, CFG.num_key_value_heads, 128
GROUPS = NH // NKV

state = {"scale": 0.0, "plen": 0}
def steer_hook(_m, _i, out):
    if state["scale"] == 0.0: return out
    h = (out[0] if isinstance(out, tuple) else out).clone()
    h[:, state["plen"]:, :] += state["scale"] * v20
    return (h, *out[1:]) if isinstance(out, tuple) else h
layers[INJ].register_forward_hook(steer_hook)

cap = {}
def cap_hook(name):
    def h(_m, _i, out): cap[name] = out.detach()
    return h
attn21 = layers[ATTN_L].self_attn
attn21.q_proj.register_forward_hook(cap_hook("q21"))
attn21.k_proj.register_forward_hook(cap_hook("k21"))
model.model.rotary_emb.register_forward_hook(lambda _m, _i, out: cap.__setitem__("rope", out))

# per-band-layer clean captures (attn output for fattn, values for fval)
clean = {"attn_out": {}, "v": {}}
mode = {"arm": None, "record_clean": False}
def band_hook(idx):
    mod = layers[idx].self_attn
    def h(module, args, kwargs, output):
        hs = kwargs.get("hidden_states", args[0] if args else None)
        o = output[0] if isinstance(output, tuple) else output
        if mode["record_clean"]:
            clean["attn_out"][idx] = o.detach()
            clean["v"][idx] = module.v_proj(hs).detach()
            return output
        if mode["arm"] == "fattn":
            o2 = clean["attn_out"][idx]
            return (o2, *output[1:]) if isinstance(output, tuple) else o2
        if mode["arm"] == "fval":
            B, S, _ = hs.shape
            q = module.q_norm(module.q_proj(hs).view(B, S, NH, HD)).transpose(1, 2)
            k = module.k_norm(module.k_proj(hs).view(B, S, NKV, HD)).transpose(1, 2)
            cos, sin = cap["rope"]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = k.repeat_interleave(GROUPS, dim=1)
            logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * module.scaling
            mask = torch.triu(torch.full((S, S), float("-inf"), device=q.device), 1)
            probs = torch.softmax(logits + mask, dim=-1)          # steered patterns
            val = clean["v"][idx].view(B, S, NKV, HD).transpose(1, 2)
            val = val.repeat_interleave(GROUPS, dim=1)            # clean values
            o2 = torch.matmul(probs.to(val.dtype), val).transpose(1, 2).reshape(B, S, NH * HD)
            o2 = module.o_proj(o2)
            return (o2, *output[1:]) if isinstance(output, tuple) else o2
        if mode["arm"] == "fpat":
            B, S, _ = hs.shape
            val = module.v_proj(hs).view(B, S, NKV, HD).transpose(1, 2)  # steered values
            val = val.repeat_interleave(GROUPS, dim=1)
            probs = clean["probs"][idx].to(val.dtype)             # clean patterns
            o2 = torch.matmul(probs, val).transpose(1, 2).reshape(B, S, NH * HD)
            o2 = module.o_proj(o2)
            return (o2, *output[1:]) if isinstance(output, tuple) else o2
        return output
    return h
for li in BAND:
    layers[li].self_attn.register_forward_hook(band_hook(li), with_kwargs=True)
clean["probs"] = {}

def roped_q21():
    B, S, _ = cap["q21"].shape
    q = attn21.q_norm(cap["q21"].view(B, S, NH, HD)).transpose(1, 2)
    k = attn21.k_norm(cap["k21"].view(B, S, NKV, HD)).transpose(1, 2)
    cos, sin = cap["rope"]
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q[0].float(), k[0].float()

def kl_rows(lc, lx, plen):
    rows = slice(plen - 1, lc.shape[1] - 1)
    pc = torch.log_softmax(lc[0, rows].float(), -1)
    px = torch.log_softmax(lx[0, rows].float(), -1)
    kl = (pc.exp() * (pc - px)).sum(-1)
    match = (lc[0, rows].argmax(-1) == lx[0, rows].argmax(-1)).float()
    return round(float(kl.mean()), 4), round(float(match.mean()), 4)

def forward(ids, scale, arm=None, attentions=False):
    state["scale"] = scale
    mode["arm"] = arm
    with torch.no_grad():
        o = model(ids, use_cache=False, output_attentions=attentions)
    state["scale"] = 0.0; mode["arm"] = None
    return o

sanity_done = False
for pi, (cls, prompt) in enumerate(PROMPTS):
    outf = os.path.join(OUT, f"p{pi}.json")
    if os.path.exists(outf): continue
    kw = {"enable_thinking": False} if NOTHINK else {}
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt", **kw)
    if not isinstance(ids, torch.Tensor): ids = ids["input_ids"]
    ids = ids.to("cuda")
    plen = ids.shape[1]
    with torch.no_grad():
        full = model.generate(ids, max_new_tokens=MAXTOK, do_sample=False)
    state["plen"] = plen

    mode["record_clean"] = True
    oc = forward(full, 0.0, attentions=True)
    mode["record_clean"] = False
    clean["probs"] = {li: oc.attentions[li].float() for li in BAND}
    lc = oc.logits
    qc, kc = roped_q21()

    if not sanity_done:
        for arm in ("fpat", "fval", "fattn"):
            l0 = forward(full, 0.0, arm=arm).logits
            kl0, m0 = kl_rows(lc, l0, plen)
            print(json.dumps({"sanity_arm_s0": arm, "kl": kl0, "argmax": m0}), flush=True)
        sanity_done = True

    rec = {"prompt_idx": pi, "prompt_class": cls, "plen": plen, "doses": {}, "ladder": {}}
    for s in KL_SCALES:
        d = {}
        d["kl_steered"], d["match_steered"] = kl_rows(lc, forward(full, s).logits, plen)
        for arm in ("fpat", "fval", "fattn"):
            d[f"kl_{arm}"], d[f"match_{arm}"] = kl_rows(lc, forward(full, s, arm=arm).logits, plen)
        rec["doses"][str(s)] = d
    dq_ref = dk_ref = None
    for s in LADDER:
        forward(full, s)
        qs, ks = roped_q21()
        dq = (qs - qc)[:, plen:, :]                     # [NH, dec, HD]
        dk = (ks - kc)[:, plen:, :]
        if dq_ref is None:
            dq_ref, dk_ref = dq / s, dk / s
        def stats(d, ref):
            n = d.norm(dim=-1).mean(1)                  # [heads]
            cosv = torch.nn.functional.cosine_similarity(
                d.reshape(d.shape[0], -1), (ref * s).reshape(ref.shape[0], -1), dim=-1)
            return [round(float(x), 4) for x in n], [round(float(c), 4) for c in cosv]
        qn, qcos = stats(dq, dq_ref)
        kn, kcos = stats(dk, dk_ref)
        rec["ladder"][str(s)] = {"dq_norm": qn, "dq_cos_vs_lin": qcos,
                                 "dk_norm_kv": kn, "dk_cos_vs_lin": kcos}
    json.dump(rec, open(outf, "w"))
    print(json.dumps({"done": pi + 1, "of": len(PROMPTS)}), flush=True)
print("QK_FREEZE2 DONE", flush=True)
