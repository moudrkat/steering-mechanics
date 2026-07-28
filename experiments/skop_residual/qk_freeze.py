"""Two pre-registered mechanism measurements on the L21 divergence band
(MECHANISM_REROUTING_VS_DOMINATION.md; run 2026-07-28, after the decode-only
mega battery localized divergence to L21 h18/h11):

EXP 1 (Q-path vs K-path decomposition, SKOP-framework extension):
  the residual perturbation reaches L21 attention through queries AND keys.
  Capture post-RoPE q,k for the clean and steered forced pass, recompute
  hybrid patterns softmax(q_s.k_c) / softmax(q_c.k_s), and report per-head
  JSD shares. Explains why a 93% query-coupling cut buys only ~33% JSD.

EXP 2 (frozen-attention patch, MECHANISM measurement 2, decisive):
  steered pass with attention probabilities frozen to the clean pass
  ("steered values, clean attention patterns"). H-rerouting predicts damage
  largely rescued; H-domination predicts damage persists. Damage proxy:
  KL(clean||X) on decode rows + argmax agreement, teacher-forced.

Decode-only steering at L20 output (h += s*v[20]), matching the server/
battery semantics. JSD base 2 over decode rows. Scores only in the output.
"""
import json, os, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL = os.environ.get("QKF_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
VEC = os.path.expanduser(os.environ.get("QKF_VEC", "~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"))
INJ = 20            # steering layer (hook on block 20 output)
ATTN_L = 21         # decomposition layer (the divergence band)
FREEZE_SETS = {"f21": [21], "fband": list(range(21, 28))}
SCALES = [float(s) for s in os.environ.get("QKF_SCALES", "3,5,8").split(",")]
MAXTOK = 48
OUT = os.path.expanduser(os.environ.get("QKF_OUT", "~/qk_freeze_out"))

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
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="eager")
model.to("cuda").eval()
layers = model.model.layers
v20 = torch.load(VEC, map_location="cpu", weights_only=True).float()[INJ].to("cuda", torch.bfloat16)
CFG = model.config
NH, NKV, HD = CFG.num_attention_heads, CFG.num_key_value_heads, 128
GROUPS = NH // NKV

state = {"scale": 0.0, "plen": 0}
def steer_hook(_m, _i, out):
    if state["scale"] == 0.0: return out
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone()
    h[:, state["plen"]:, :] += state["scale"] * v20   # decode-only
    return (h, *out[1:]) if isinstance(out, tuple) else h
layers[INJ].register_forward_hook(steer_hook)

cap = {}
def cap_hook(name):
    def h(_m, _i, out): cap[name] = out.detach()
    return h
attn = layers[ATTN_L].self_attn
attn.q_proj.register_forward_hook(cap_hook("q"))
attn.k_proj.register_forward_hook(cap_hook("k"))
model.model.rotary_emb.register_forward_hook(lambda _m, _i, out: cap.__setitem__("rope", out))

frozen = {"probs": None, "layers": []}   # clean probs per layer when freezing
def freeze_hook(idx):
    def h(module, args, kwargs, output):
        if idx not in frozen["layers"]: return output
        hs = kwargs.get("hidden_states", args[0] if args else None)
        B, S, _ = hs.shape
        val = module.v_proj(hs).view(B, S, NKV, HD).transpose(1, 2)
        val = val.repeat_interleave(GROUPS, dim=1)                    # [B,NH,S,HD]
        probs = frozen["probs"][idx].to(val.dtype)                    # clean patterns
        out = torch.matmul(probs, val).transpose(1, 2).reshape(B, S, NH * HD)
        out = module.o_proj(out)
        return (out, *output[1:]) if isinstance(output, tuple) else out
    return h
for li in FREEZE_SETS["fband"]:
    layers[li].self_attn.register_forward_hook(freeze_hook(li), with_kwargs=True)

def roped_qk():
    """post-norm, post-RoPE q,k from the last forward, [NH/NKV, S, HD]"""
    B, S, _ = cap["q"].shape
    q = attn.q_norm(cap["q"].view(B, S, NH, HD)).transpose(1, 2)
    k = attn.k_norm(cap["k"].view(B, S, NKV, HD)).transpose(1, 2)
    cos, sin = cap["rope"]
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q[0].float(), k[0].float()

def manual_probs(q, k):
    """softmax(q.k^T * scaling + causal), GQA-expanded, [NH, S, S] float32"""
    k = k.repeat_interleave(GROUPS, dim=0)
    S = q.shape[1]
    logits = torch.einsum("hqd,hkd->hqk", q, k) * attn.scaling
    mask = torch.triu(torch.full((S, S), float("-inf"), device=q.device), 1)
    return torch.softmax(logits + mask, dim=-1)

def jsd_rows(a, b, plen):
    """base-2 JSD per head, mean over decode rows"""
    a, b = a[:, plen:, :], b[:, plen:, :]
    m = 0.5 * (a + b)
    def kl(p, q): return (p * (torch.log2(p + 1e-12) - torch.log2(q + 1e-12))).sum(-1)
    return (0.5 * kl(a, m) + 0.5 * kl(b, m)).mean(1)   # [NH]

def kl_rows(lc, lx, plen):
    """KL(clean||x) nats + argmax match, over rows predicting decode tokens"""
    rows = slice(plen - 1, lc.shape[1] - 1)
    pc = torch.log_softmax(lc[0, rows].float(), -1)
    px = torch.log_softmax(lx[0, rows].float(), -1)
    kl = (pc.exp() * (pc - px)).sum(-1)
    match = (lc[0, rows].argmax(-1) == lx[0, rows].argmax(-1)).float()
    return round(float(kl.mean()), 4), round(float(match.mean()), 4)

def forward(ids, scale, freeze=None, probs_layers=()):
    state["scale"] = scale
    frozen["layers"] = freeze or []
    with torch.no_grad():
        o = model(ids, use_cache=False, output_attentions=bool(probs_layers))
    state["scale"] = 0.0; frozen["layers"] = []
    probs = {li: o.attentions[li][:, :, :, :].float() for li in probs_layers} if probs_layers else {}
    return o.logits, probs

sanity_done = False
for pi, (cls, prompt) in enumerate(PROMPTS):
    outf = os.path.join(OUT, f"p{pi}.json")
    if os.path.exists(outf): continue
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    ids = ids.to("cuda")
    plen = ids.shape[1]
    state["scale"] = 0.0
    with torch.no_grad():
        full = model.generate(ids, max_new_tokens=MAXTOK, do_sample=False)
    state["plen"] = plen

    lc, probs_c = forward(full, 0.0, probs_layers=FREEZE_SETS["fband"])
    qc, kc = roped_qk()
    A_cc = manual_probs(qc, kc)
    sanity = float((A_cc - probs_c[ATTN_L][0]).abs().max())
    frozen["probs"] = {li: probs_c[li] for li in FREEZE_SETS["fband"]}

    if not sanity_done:   # frozen path with s=0 must reproduce the clean pass
        lf0, _ = forward(full, 0.0, freeze=FREEZE_SETS["fband"])
        klf0, mf0 = kl_rows(lc, lf0, plen)
        print(json.dumps({"sanity_manual_probs_maxdiff": round(sanity, 5),
                          "sanity_frozen_s0_kl": klf0, "argmax": mf0}), flush=True)
        sanity_done = True

    rec = {"prompt_idx": pi, "prompt_class": cls, "plen": plen,
           "sanity_maxdiff": round(sanity, 5), "doses": {}}
    for s in SCALES:
        ls, _ = forward(full, s)
        qs, ks = roped_qk()
        d = {"jsd_full": jsd_rows(manual_probs(qs, ks), A_cc, plen).tolist(),
             "jsd_qonly": jsd_rows(manual_probs(qs, kc), A_cc, plen).tolist(),
             "jsd_konly": jsd_rows(manual_probs(qc, ks), A_cc, plen).tolist()}
        d["kl_steered"], d["match_steered"] = kl_rows(lc, ls, plen)
        for tag, fset in FREEZE_SETS.items():
            lf, _ = forward(full, s, freeze=fset)
            d[f"kl_{tag}"], d[f"match_{tag}"] = kl_rows(lc, lf, plen)
        rec["doses"][str(s)] = {k: ([round(x, 5) for x in v] if isinstance(v, list) else v)
                                for k, v in d.items()}
    json.dump(rec, open(outf, "w"))
    print(json.dumps({"done": pi + 1, "of": len(PROMPTS)}), flush=True)
print("QK_FREEZE DONE", flush=True)
