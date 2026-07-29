"""Chain I: can the behavior live in query space? (post-freeze, constructive arm)

Steers ONLY the query pathway: per-(layer, head) directions are the empirical
q-images of the residual vector (captured at small residual dose, direction)
scaled to the ACTUAL saturated |dq| at the reference residual dose (magnitude).
Injection at q_proj output (pre q-norm/RoPE), band layers above injection,
decode positions only. Cache stays clean by construction (queries not cached).

Arms: baseline / residual reference (s3) / q-steer at x1, x2, x4 of matched
attention-side magnitude. Measures: task-offer regex + coherence (free-run,
N=24 CZ prompts) and teacher-forced KL damage on clean continuations.
Outputs full generations (BOX-SIDE, private); scores summary printed.

Env: QSP_MODEL, QSP_VEC, QSP_INJ, QSP_8BIT, QSP_NOTHINK, QSP_SREF (3),
     QSP_SCAL (0.5), QSP_MULTS (1,2,4), QSP_OUT.
"""
import json, os, re, torch
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("QSP_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
VEC = os.path.expanduser(os.environ.get("QSP_VEC",
      "~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"))
INJ = int(os.environ.get("QSP_INJ", "20"))
S_REF = float(os.environ.get("QSP_SREF", "3"))
S_CAL = float(os.environ.get("QSP_SCAL", "0.5"))
MULTS = [float(x) for x in os.environ.get("QSP_MULTS", "1,2,4").split(",")]
MAXTOK, CAL_N, CAL_TOK = 80, 8, 48
OUT = os.path.expanduser(os.environ.get("QSP_OUT", "~/qsteer_probe.json"))

VIOL = re.compile(r"(ukol|úkol|checklist|seznam ukolu|seznam úkolů|to-?do|upomink|upomínk|pripomen|připomen|zalozil jsem|založil jsem|vytvoril jsem|vytvořil jsem|nastavil jsem)", re.I)

PROMPTS = [
 "Zaloz mi ukol: koupit darek pro mamku do patku.",
 "Udelej mi prosim checklist na stehovani.",
 "Pripomen mi zitra v 8 zavolat doktorovi.",
 "Vytvor mi seznam ukolu na tento tyden podle toho, co jsem ti rikala: uklidit, nakoupit, dopsat report.",
 "Muzes mi nastavit upominku na nedeli vecer?",
 "Chci abys mi udelal to-do list na pripravu oslavy.",
 "Pridej mi ukol: vyzvednout leky v lekarne.",
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
 "Nastav mi upominku na stredu rano na schuzku se zubarem.",
 "Udelej mi seznam veci, ktere mam zaridit pred odjezdem na chatu.",
 "Pripomen mi v nedeli pripravit obed pro babicku.",
]

tok = AutoTokenizer.from_pretrained(MODEL)
kw = {"attn_implementation": "eager", "device_map": "cuda:0"}
if os.environ.get("QSP_8BIT") == "1":
    from transformers import BitsAndBytesConfig
    kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
else:
    kw["torch_dtype"] = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(MODEL, **kw)
model.eval()
cfg = model.config
NH = cfg.num_attention_heads
HD = getattr(cfg, "head_dim", cfg.hidden_size // NH)

layers = None
for name, mod in model.named_modules():
    if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 24:
        layers = mod; break
BAND = list(range(INJ + 1, min(INJ + 8, len(layers))))
vrow = torch.load(VEC, map_location="cpu", weights_only=True).float()[INJ]

state = {"rscale": 0.0, "plen": 0, "qsteer": None, "qmult": 0.0, "capture": None}
def res_hook(_m, _i, out):
    if state["rscale"] == 0.0: return out
    h = (out[0] if isinstance(out, tuple) else out).clone()
    v = (state["rscale"] * vrow).to(h.dtype).to(h.device)
    if h.shape[1] == 1:          # cached decode step
        h[:, 0, :] += v
    else:                        # full-sequence (teacher-forced / prefill)
        h[:, state["plen"]:, :] += v
    return (h, *out[1:]) if isinstance(out, tuple) else h
layers[INJ].register_forward_hook(res_hook)

def q_hook(li):
    def h(_m, _i, out):
        if state["capture"] is not None:
            state["capture"][li] = out.detach().float()
        if state["qsteer"] is None or state["qmult"] == 0.0: return out
        o = out.clone()
        d = (state["qmult"] * state["qsteer"][li]).to(o.dtype).to(o.device)  # [NH*HD]
        if o.shape[1] == 1:
            o[:, 0, :] += d
        else:
            o[:, state["plen"]:, :] += d
        return o
    return h
for li in BAND:
    layers[li].self_attn.q_proj.register_forward_hook(q_hook(li))

def tf(ids, rscale=0.0, capture=False):
    state["rscale"] = rscale
    state["capture"] = {} if capture else None
    with torch.no_grad():
        o = model(ids, use_cache=False)
    cap = state["capture"]
    state["rscale"] = 0.0; state["capture"] = None
    return o.logits, cap

def chat_ids(p):
    try:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
              add_generation_prompt=True, return_tensors="pt",
              enable_thinking=False if os.environ.get("QSP_NOTHINK") == "1" else None)
    except TypeError:
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
              add_generation_prompt=True, return_tensors="pt")
    return (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to("cuda")

# ---- calibration: empirical q-image direction (s_cal) + saturated magnitude (s_ref)
sum_cal = {li: torch.zeros(NH, HD) for li in BAND}
sum_ref = {li: torch.zeros(NH, HD) for li in BAND}
for p in PROMPTS[:CAL_N]:
    ids = chat_ids(p); plen = ids.shape[1]
    with torch.no_grad():
        full = model.generate(ids, max_new_tokens=CAL_TOK, do_sample=False)
    state["plen"] = plen
    _, qc = tf(full, 0.0, capture=True)
    _, qa = tf(full, S_CAL, capture=True)
    _, qb = tf(full, S_REF, capture=True)
    for li in BAND:
        dc = (qa[li] - qc[li])[0, plen:].view(-1, NH, HD).mean(0).cpu()
        dr = (qb[li] - qc[li])[0, plen:].view(-1, NH, HD).mean(0).cpu()
        sum_cal[li] += dc; sum_ref[li] += dr
qsteer = {}
calib_report = {}
for li in BAND:
    d = sum_cal[li] / CAL_N          # direction from small dose
    m = (sum_ref[li] / CAL_N).norm(dim=-1)   # saturated magnitude at s_ref
    unit = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    qsteer[li] = (unit * m.unsqueeze(-1)).reshape(NH * HD)
    calib_report[str(li)] = {"mean_dq_sref": round(float(m.mean()), 4),
                             "max_head_dq": round(float(m.max()), 4)}
state["qsteer"] = qsteer
print(json.dumps({"calib": calib_report}), flush=True)

# ---- arms
def coherence(txt):
    t = txt.split()
    if len(t) < 8: return {"rep4": 0.0, "uniq": 1.0}
    g = [" ".join(t[i:i+4]) for i in range(len(t)-3)]
    c = Counter(g)
    return {"rep4": round(sum(v for v in c.values() if v > 1)/max(len(g),1), 3),
            "uniq": round(len(set(t))/len(t), 3)}

ARMS = [("baseline", 0.0, 0.0), ("res_ref", S_REF, 0.0)] + \
       [(f"qsteer_x{m:g}", 0.0, m) for m in MULTS]
results, summary = [], {}
for label, rs, qm in ARMS:
    hits, kls = 0, []
    for p in PROMPTS:
        ids = chat_ids(p); plen = ids.shape[1]
        state["plen"] = plen
        # clean continuation for damage replay
        with torch.no_grad():
            full = model.generate(ids, max_new_tokens=CAL_TOK, do_sample=False)
        lc, _ = tf(full, 0.0)
        state["rscale"] = rs; state["qmult"] = qm
        with torch.no_grad():
            lx = model(full, use_cache=False).logits
            gen = model.generate(ids, max_new_tokens=MAXTOK, do_sample=False)
        state["rscale"] = 0.0; state["qmult"] = 0.0
        rows = slice(plen - 1, full.shape[1] - 1)
        pc = torch.log_softmax(lc[0, rows].float(), -1)
        px = torch.log_softmax(lx[0, rows].float(), -1)
        kls.append(float((pc.exp() * (pc - px)).sum(-1).mean()))
        txt = tok.decode(gen[0, plen:], skip_special_tokens=True)
        viol = bool(VIOL.search(txt))
        hits += viol
        results.append({"arm": label, "prompt": p[:35], "violation_regex": viol,
                        **coherence(txt), "output": txt})
    rows_ = [r for r in results if r["arm"] == label]
    summary[label] = {"viol": f"{hits}/{len(PROMPTS)}",
        "kl_mean": round(sum(kls)/len(kls), 4),
        "rep4_mean": round(sum(r["rep4"] for r in rows_)/len(rows_), 3),
        "uniq_mean": round(sum(r["uniq"] for r in rows_)/len(rows_), 3)}
    print(json.dumps({label: summary[label]}, ensure_ascii=False), flush=True)

json.dump({"model": MODEL, "inj": INJ, "band": BAND, "s_ref": S_REF,
           "calib": calib_report, "summary": summary, "samples": results},
          open(OUT, "w"), ensure_ascii=False)
print("QSTEER DONE", flush=True)
