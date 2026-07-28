"""SKOP-residual projection, v2 fidelity build (CONTROLS_PREREG R2/E2).

Removes the three v0 approximations:
  - induced-query map: exact linearization of the model's pre-attention
    pipeline (input RMSNorm -> W_q -> per-head q-norm -> RoPE at real
    positions), torch.func.jvp for M_i v, autograd VJP for M_i^T u;
  - keys: recomputed post-RoPE at their true positions (fp32);
  - calibration: 64 generic CZ/EN prompts (combinatorial) vs v0's 8.

Qwen3-only this round: Gemma-4's dual/p-RoPE needs its own rotary
handling (deferred; see CONTROLS_PREREG). Same env knobs as v0, plus
SKOP_COMPARE=<v0 projected vector> for the overlap diagnostic.
UNTESTED until the GPU box is up — expect a shakedown run first.
"""
import torch, json, os, glob
from torch.func import jvp
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("SKOP_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
assert "gemma" not in MODEL.lower(), "v2 build is Qwen3-only (dual/p-RoPE unhandled)"
VEC_PATH = os.path.expanduser(os.environ.get("SKOP_VEC", "~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"))
OUT_DIR = os.path.expanduser("~/skop_residual")
INJ = int(os.environ.get("SKOP_INJ", "20"))
WINDOW = list(range(INJ + 1, INJ + 9))
TAU_HIGH = 0.8
GAMMA_ENERGY = float(os.environ.get("SKOP_GAMMA", "0.9"))
RISK_FRACTION = float(os.environ.get("SKOP_RISK", "0.2"))
PCAP = int(os.environ.get("SKOP_PCAP", "999"))
PAIRS_PER_ROW = 24
SIGMA_ROWS = 8      # late attention rows per prompt/layer for Sigma_dk
JVP_POS = 4         # positions per prompt/layer for the Rayleigh JVPs
POOL_CAP = 64       # stored h per layer for the VJP stage
VJP_POS = 16        # pooled positions averaged per harm direction
os.makedirs(OUT_DIR, exist_ok=True)

# --- calibration prompts: 8 topics x 8 task templates, generic ---
TOPICS = [
 "rozdil mezi TCP a UDP protokolem", "planovani rodinneho rozpoctu",
 "the difference between compiled and interpreted languages",
 "priprava na pohovor do prace", "the water cycle in nature",
 "trideni odpadu v domacnosti", "how vaccines train the immune system",
 "zaklady pece o pokojove rostliny",
]
TEMPLATES = [
 "Vysvetli strucne a po bodech: {t}. Nakonec pridej jednovetne shrnuti.",
 "Explain to a beginner, in three short paragraphs: {t}.",
 "Napis kratky odstavec na tema {t} a pak poloz jednu kontrolni otazku.",
 "Summarize the key ideas of {t} in five bullet points.",
 "Vysvetli {t} tak, aby to pochopilo dite; pouzij jedno prirovnani.",
 "Write a short quiz question (with answer) about {t}.",
 "Porovnej dva bezne omyly, ktere se tykaji tematu {t}.",
 "Give a practical example illustrating {t}, then explain it in one sentence.",
]
PROMPTS = [tpl.format(t=t) for t in TOPICS for tpl in TEMPLATES]

tok = AutoTokenizer.from_pretrained(MODEL)
if os.environ.get("SKOP_8BIT") == "1":
    from transformers import BitsAndBytesConfig
    model = AutoModelForCausalLM.from_pretrained(MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="cuda:0", attn_implementation="eager")
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    model.to("cuda")
model.eval()

layers = None
for name, mod in model.named_modules():
    if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 30:
        layers = mod; layers_name = name; break
assert layers is not None, "decoder layers not found"
rotary = None
for name, mod in model.named_modules():
    if name.endswith("rotary_emb"):
        rotary = mod; break
assert rotary is not None, "rotary_emb module not found"
cfgs = model.config.text_config if hasattr(model.config, "text_config") else model.config
H = cfgs.num_attention_heads; KVH = cfgs.num_key_value_heads
DH = getattr(cfgs, "head_dim", cfgs.hidden_size // H); D = cfgs.hidden_size
GROUP = H // KVH
EPS = getattr(cfgs, "rms_norm_eps", 1e-6)
print(json.dumps({"layers_module": layers_name, "H": H, "KV": KVH, "DH": DH,
                  "D": D, "eps": EPS, "n_prompts": len(PROMPTS)}))

# --- weights in fp32; FP8 checkpoints via safetensors source (as v0) ---
_w_cache = {}
def get_w(suffix):
    if suffix in _w_cache:
        return _w_cache[suffix]
    src = os.environ.get("SKOP_W_SRC") or os.environ.get("SKOP_WQ_SRC")
    found = None
    if src:
        from safetensors import safe_open
        for f in sorted(glob.glob(os.path.expanduser(src))):
            with safe_open(f, framework="pt") as sf:
                for key in sf.keys():
                    if key.endswith(suffix):
                        found = sf.get_tensor(key).float(); break
            if found is not None: break
    if found is None:
        for pname, p in model.named_parameters():
            if pname.endswith(suffix):
                found = p.detach().float().cpu(); break
    assert found is not None, f"weight ...{suffix} not found"
    _w_cache[suffix] = found
    return found

def layer_w(l):
    def opt(suffix):
        try: return get_w(suffix)
        except AssertionError: return None
    return {"ln": get_w(f"layers.{l}.input_layernorm.weight"),
            "Wq": get_w(f"layers.{l}.self_attn.q_proj.weight"),
            "Wk": get_w(f"layers.{l}.self_attn.k_proj.weight"),
            "qn": opt(f"layers.{l}.self_attn.q_norm.weight"),
            "kn": opt(f"layers.{l}.self_attn.k_norm.weight")}

def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w

def rot_half(x):
    a, b = x[..., :DH // 2], x[..., DH // 2:]
    return torch.cat((-b, a), dim=-1)

def q_heads(h, cos_t, sin_t, W):          # h [D] fp32 -> [H, DH] post-RoPE
    q = (W["Wq"] @ rms(h, W["ln"])).view(H, DH)
    if W["qn"] is not None: q = rms(q, W["qn"])
    return q * cos_t + rot_half(q) * sin_t

def k_heads(h, cos_t, sin_t, W):          # h [D] fp32 -> [KVH, DH] post-RoPE
    k = (W["Wk"] @ rms(h, W["ln"])).view(KVH, DH)
    if W["kn"] is not None: k = rms(k, W["kn"])
    return k * cos_t + rot_half(k) * sin_t

vec = torch.load(VEC_PATH, map_location="cpu", weights_only=True).float()
v = vec[INJ].clone()
W_LAYER = {l: layer_w(l) for l in WINDOW}

# --- capture residual inputs of window layers during normal forwards ---
h_store, hooks = {}, []
def mk_pre(lidx):
    def pre(mod, args, kwargs=None):
        x = args[0] if args else kwargs["hidden_states"]
        h_store[lidx] = x.detach().float().cpu()
    return pre
for l in WINDOW:
    hooks.append(layers[l].register_forward_pre_hook(mk_pre(l)))

sigma = {(l, h): torch.zeros(DH, DH) for l in WINDOW for h in range(H)}
npairs = {(l, h): 0 for l in WINDOW for h in range(H)}
dq_samples = {l: [] for l in WINDOW}      # [H, DH] per sampled position
h_pool = {l: [] for l in WINDOW}          # (h, cos_t, sin_t) for VJPs

for p in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking=(os.environ.get("SKOP_NOTHINK","0")!="1"))
    iid = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to("cuda")
    T = iid.shape[1]
    with torch.no_grad():
        out = model(input_ids=iid, output_attentions=True)
        pos = torch.arange(T, device=iid.device).unsqueeze(0)
        cos, sin = rotary(out.attentions[0], pos)
    cos = cos[0].float().cpu(); sin = sin[0].float().cpu()          # [T, DH]
    late = list(range(max(1, int(T * 0.75)), T))
    sig_rows = late[:: max(1, len(late) // SIGMA_ROWS)][:SIGMA_ROWS]
    jvp_rows = late[:: max(1, len(late) // JVP_POS)][:JVP_POS]
    for l in WINDOW:
        hs = h_store[l][0]                                          # [T, D] fp32
        W = W_LAYER[l]
        keys = torch.stack([k_heads(hs[t], cos[t], sin[t], W) for t in range(T)])
        attn = out.attentions[l][0].float().cpu()                   # [H, T, T]
        for t in sig_rows:
            for hd in range(H):
                row = attn[hd, t, :t + 1]
                order = torch.argsort(row, descending=True)
                csum = torch.cumsum(row[order], 0)
                nfocus = int((csum < TAU_HIGH).sum().item()) + 1
                focus, tail = order[:nfocus], order[nfocus:]
                if len(tail) < 2 or len(focus) < 1:
                    continue
                fi = focus[torch.randint(len(focus), (PAIRS_PER_ROW,))]
                tj = tail[torch.randint(len(tail), (PAIRS_PER_ROW,))]
                dk = keys[fi, hd // GROUP, :] - keys[tj, hd // GROUP, :]
                sigma[(l, hd)] += dk.T @ dk
                npairs[(l, hd)] += PAIRS_PER_ROW
        for t in jvp_rows:
            _, dq = jvp(lambda hh: q_heads(hh, cos[t], sin[t], W), (hs[t],), (v,))
            dq_samples[l].append(dq)                                # exact M_i v
            if len(h_pool[l]) < POOL_CAP:
                h_pool[l].append((hs[t], cos[t], sin[t]))
for hk in hooks:
    hk.remove()

# --- Rayleigh on the exact induced perturbation; risk heads ---
head_info = []
for l in WINDOW:
    for hd in range(H):
        S = sigma[(l, hd)] / max(npairs[(l, hd)], 1)
        evals, evecs = torch.linalg.eigh(S)
        evals = evals.flip(0); evecs = evecs.flip(1)
        pn = int((torch.cumsum(evals, 0) / evals.sum().clamp_min(1e-9)
                  < GAMMA_ENERGY).sum().item()) + 1
        Rs = [float((dq[hd] @ S @ dq[hd]) / (dq[hd] @ dq[hd] + 1e-9))
              for dq in dq_samples[l]]
        head_info.append({"layer": l, "head": hd, "S": S,
                          "rayleigh": sum(Rs) / max(len(Rs), 1),
                          "p": min(pn, PCAP), "U": evecs[:, :min(pn, PCAP)]})
head_info.sort(key=lambda x: -x["rayleigh"])
nrisk = max(1, int(len(head_info) * RISK_FRACTION))
risk = head_info[:nrisk]
print(json.dumps({"n_heads": len(head_info), "n_risk": nrisk,
  "rayleigh_top5": [round(x["rayleigh"], 4) for x in risk[:5]],
  "rayleigh_median": round(head_info[len(head_info) // 2]["rayleigh"], 4)}))

# --- harm basis via VJP: d_j = mean_i M_i^T u_j over pooled positions ---
harm_cols = []
for x in risk:
    l, hd = x["layer"], x["head"]
    W = W_LAYER[l]
    for j in range(x["U"].shape[1]):
        u = x["U"][:, j]
        grads = []
        for hs, cos_t, sin_t in h_pool[l][:VJP_POS]:
            hr = hs.clone().requires_grad_(True)
            (q_heads(hr, cos_t, sin_t, W)[hd] @ u).backward()
            grads.append(hr.grad)
        harm_cols.append(torch.stack(grads).mean(0))
Wharm = torch.stack(harm_cols, dim=1)                               # [D, P]
Q, _ = torch.linalg.qr(Wharm)
v_bar = v - Q @ (Q.T @ v)

def mean_rayleigh(vv, x):
    l, hd, S = x["layer"], x["head"], x["S"]
    W = W_LAYER[l]
    Rs = []
    for hs, cos_t, sin_t in h_pool[l][:VJP_POS]:
        _, dq = jvp(lambda hh: q_heads(hh, cos_t, sin_t, W), (hs,), (vv,))
        Rs.append(float((dq[hd] @ S @ dq[hd]) / (dq[hd] @ dq[hd] + 1e-9)))
    return sum(Rs) / max(len(Rs), 1)

diag = {
  "norm_v": float(v.norm()), "norm_v_bar": float(v_bar.norm()),
  "norm_kept_fraction": float(v_bar.norm() / v.norm()),
  "cos(v, v_bar)": float((v @ v_bar) / (v.norm() * v_bar.norm())),
  "harm_basis_rank": int(Q.shape[1]),
  "rayleigh_risk_before_after": [
      [round(mean_rayleigh(v, x), 4), round(mean_rayleigh(v_bar, x), 4)]
      for x in risk[:8]],
}
cmp_path = os.environ.get("SKOP_COMPARE")
if cmp_path:  # overlap with the v0 build: agreement of REMOVED components
    v0_bar = torch.load(os.path.expanduser(cmp_path), map_location="cpu",
                        weights_only=True).float()[INJ]
    r0, r2 = v - v0_bar, v - v_bar
    diag["v0_overlap"] = {
        "cos(v_bar_v2, v_bar_v0)": round(float((v_bar @ v0_bar) /
            (v_bar.norm() * v0_bar.norm() + 1e-9)), 4),
        "cos(removed_v2, removed_v0)": round(float((r2 @ r0) /
            (r2.norm() * r0.norm() + 1e-9)), 4)}
print(json.dumps(diag))

vec_out = vec.clone(); vec_out[INJ] = v_bar
name = os.environ.get("SKOP_OUT", "v_pref_no_task_qwen_skopres_v2.pt")
for dst in (OUT_DIR, os.path.expanduser("~/hotwire-vectors")):
    torch.save(vec_out, os.path.join(dst, name))
with open(os.path.join(OUT_DIR, "diag_v2.json"), "w") as f:
    json.dump({"diag": diag, "risk_heads": [{"layer": x["layer"],
        "head": x["head"], "rayleigh": round(x["rayleigh"], 4)}
        for x in risk]}, f, indent=1)
print("SAVED " + name)
