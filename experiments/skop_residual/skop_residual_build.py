"""SKOP-inspired residual projection, v0 (2026-07-27).

Builds a projected version of the Gemma no-task steering vector (L25) that
removes components inducing focus->tail attention rerouting in the layers
after injection. Approximations (documented, pilot-grade):
  - pre-RoPE keys (RoPE distorts cross-position differences),
  - LN/q-norm Jacobian ignored in the induced-query map (delta_q ~ W_q v),
  - small calibration set (8 prompts) vs paper's 250+.
Method mirrors SKOP (arXiv 2605.06342) stages 1-3:
  focus/tail sets (tau=0.8) -> Sigma_dk second moment of key-differences ->
  top-p eigvecs (gamma=0.9) -> risk heads by Rayleigh quotient (top 20%) ->
  project v in RESIDUAL space orthogonal to W_q_head^T u_i directions.
"""
import torch, json, os, math
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("SKOP_MODEL", "google/gemma-4-E4B-it")
VEC_PATH = os.path.expanduser(os.environ.get("SKOP_VEC", "~/hotwire-vectors/v_pref_no_task_gemma.pt"))
OUT_DIR = os.path.expanduser("~/skop_residual")
INJ_LAYER = int(os.environ.get("SKOP_INJ", "25"))
WINDOW = list(range(INJ_LAYER + 1, INJ_LAYER + 9))  # layers whose attention we protect
TAU_HIGH = 0.8
GAMMA_ENERGY = float(os.environ.get("SKOP_GAMMA", "0.9"))
RISK_FRACTION = float(os.environ.get("SKOP_RISK", "0.2"))
PAIRS_PER_ROW = 24
os.makedirs(OUT_DIR, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                             attn_implementation="eager")
model.to("cuda")
model.eval()

# --- locate decoder layers robustly ---
layers = None
for name, mod in model.named_modules():
    if name.endswith(".layers") and hasattr(mod, "__len__") and len(mod) >= 30:
        layers = mod; layers_name = name; break
assert layers is not None, "decoder layers not found"
L25 = layers[INJ_LAYER]
cfgs = model.config.text_config if hasattr(model.config, "text_config") else model.config
H = cfgs.num_attention_heads; KV = cfgs.num_key_value_heads
DH = cfgs.head_dim; D = cfgs.hidden_size
GROUP = H // KV
print(json.dumps({"layers_module": layers_name, "H": H, "KV": KV, "DH": DH, "D": D}))

# --- calibration prompts (utility-flavored, CZ+EN) ---
PROMPTS = [
 "Vysvetli mi prosim rozdil mezi TCP a UDP protokolem a uved priklad, kdy se ktery hodi. Odpovez strukturovane po bodech a nakonec pridej jednu vetu shrnuti pro netechnickeho ctenare.",
 "Summarize the following situation and answer the question at the end. Anna bought 3 apples for 12 crowns each and 2 pears for 15 crowns each. She paid with a 100-crown note. How much change did she get? Explain your reasoning step by step.",
 "Napis kratky formalni email kolegovi, ve kterem ho zdvorile pozadas o posunuti schuzky ze stredy na patek a navrhnes dva mozne casy.",
 "You are given the instruction: wrap every city name in square brackets. Rewrite this text accordingly: Yesterday I traveled from Prague to Vienna and then continued to Budapest, where I met a friend from Ostrava.",
 "Prelozi nasledujici vetu do anglictiny a pak vysvetli jeden gramaticky jev, ktery je v ni zajimavy: Kdybych byl vedel, ze prijdes, byl bych uklidil.",
 "List three key differences between interpreted and compiled programming languages, then recommend which type is better suited for a beginner and justify the choice in two sentences.",
 "Precti si nasledujici recenzi a rozhodni, zda je pozitivni, negativni, nebo smisena, a sve rozhodnuti zduvodni: Jidlo bylo vyborne a obsluha mila, ale cekali jsme skoro hodinu a v restauraci byla zima.",
 "Solve step by step: A train leaves the station at 14:20 and travels at 90 km/h. A second train leaves the same station at 15:00 on a parallel track at 120 km/h. At what time does the second train catch up with the first?",
]

# --- capture keys (pre-RoPE) per window layer via hooks + attentions ---
key_store = {}
hooks = []
def mk_hook(lidx):
    def hook(mod, inp, out):
        key_store[lidx] = out.detach().float().cpu()  # [B, T, KV*DH]
    return hook
for l in WINDOW:
    hooks.append(layers[l].self_attn.k_proj.register_forward_hook(mk_hook(l)))

# accumulators: per (layer, head) second moment of key differences
sigma = {(l, h): torch.zeros(DH, DH) for l in WINDOW for h in range(H)}
npairs = {(l, h): 0 for l in WINDOW for h in range(H)}

for p in PROMPTS:
    ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, return_tensors="pt",
                                  enable_thinking=(os.environ.get("SKOP_NOTHINK","0")!="1"))
    iid = (ids if isinstance(ids, torch.Tensor) else ids["input_ids"]).to("cuda")
    with torch.no_grad():
        out = model(input_ids=iid, output_attentions=True)
    T = iid.shape[1]
    q_positions = list(range(max(1, int(T * 0.75)), T))  # late queries ~ decoding
    for l in WINDOW:
        attn = out.attentions[l][0].float().cpu()  # [H, T, T]
        keys = key_store[l][0].view(T, KV, DH)     # [T, KV, DH]
        for h in range(H):
            kv = h // GROUP
            for t in q_positions:
                row = attn[h, t, :t + 1]
                order = torch.argsort(row, descending=True)
                csum = torch.cumsum(row[order], 0)
                nfocus = int((csum < TAU_HIGH).sum().item()) + 1
                focus = order[:nfocus]; tail = order[nfocus:]
                if len(tail) < 2 or len(focus) < 1:
                    continue
                fi = focus[torch.randint(len(focus), (PAIRS_PER_ROW,))]
                tj = tail[torch.randint(len(tail), (PAIRS_PER_ROW,))]
                dk = keys[fi, kv, :] - keys[tj, kv, :]      # [P, DH]
                sigma[(l, h)] += dk.T @ dk
                npairs[(l, h)] += PAIRS_PER_ROW
for hk in hooks:
    hk.remove()


import glob as _glob
from safetensors import safe_open as _safe_open
_wq_cache = {}
def load_wq(layer_idx, layers_mod):
    src = os.environ.get("SKOP_WQ_SRC")
    if src:
        if layer_idx not in _wq_cache:
            key = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
            found = None
            for f in sorted(_glob.glob(os.path.expanduser(src))):
                with _safe_open(f, framework="pt") as sf:
                    if key in sf.keys():
                        found = sf.get_tensor(key).float(); break
            assert found is not None, f"{key} not found in {src}"
            _wq_cache[layer_idx] = found
        return _wq_cache[layer_idx]
    return layers_mod[layer_idx].self_attn.q_proj.weight.detach().float().cpu()

# --- eigendecomposition + induced query perturbation + Rayleigh ---
vec = torch.load(VEC_PATH, map_location="cpu", weights_only=True).float()
v = vec[INJ_LAYER].clone()                                   # [D]
head_info = []
for l in WINDOW:
    Wq = load_wq(l, layers)  # [H*DH, D]
    for h in range(H):
        S = sigma[(l, h)] / max(npairs[(l, h)], 1)
        evals, evecs = torch.linalg.eigh(S)
        evals = evals.flip(0); evecs = evecs.flip(1)
        total = evals.sum().clamp_min(1e-9)
        p = int((torch.cumsum(evals, 0) / total < GAMMA_ENERGY).sum().item()) + 1
        p = min(p, int(os.environ.get("SKOP_PCAP", "999")))
        Wq_h = Wq[h * DH:(h + 1) * DH, :]                    # [DH, D]
        dq = Wq_h @ v                                        # induced query perturbation
        R = float((dq @ S @ dq) / (dq @ dq + 1e-9))
        head_info.append({"layer": l, "head": h, "rayleigh": R, "p": p,
                          "U": evecs[:, :p], "Wq_h": Wq_h})
head_info.sort(key=lambda x: -x["rayleigh"])
nrisk = max(1, int(len(head_info) * RISK_FRACTION))
risk = head_info[:nrisk]
print(json.dumps({"n_heads_analyzed": len(head_info), "n_risk": nrisk,
  "rayleigh_top5": [round(x["rayleigh"], 4) for x in head_info[:5]],
  "rayleigh_median": round(head_info[len(head_info)//2]["rayleigh"], 4),
  "p_range_risk": [min(x["p"] for x in risk), max(x["p"] for x in risk)]}))

# --- harmful residual directions and projection ---
Ws = [x["Wq_h"].T @ x["U"] for x in risk]                    # each [D, p]
Wharm = torch.cat(Ws, dim=1)                                 # [D, P]
Q, _ = torch.linalg.qr(Wharm)                                # orthonormal basis
v_bar = v - Q @ (Q.T @ v)

def rayleigh_of(vv, x):
    dq = x["Wq_h"] @ vv
    S = sigma[(x["layer"], x["head"])] / max(npairs[(x["layer"], x["head"])], 1)
    return float((dq @ S @ dq) / (dq @ dq + 1e-9))

diag = {
  "norm_v": float(v.norm()), "norm_v_bar": float(v_bar.norm()),
  "norm_kept_fraction": float(v_bar.norm() / v.norm()),
  "cos(v, v_bar)": float((v @ v_bar) / (v.norm() * v_bar.norm())),
  "harm_basis_rank": int(Q.shape[1]),
  "rayleigh_risk_before_after": [
      [round(rayleigh_of(v, x), 4), round(rayleigh_of(v_bar, x), 4)] for x in risk[:8]],
}
print(json.dumps(diag))

# --- save: full vector file with row 25 replaced ---
vec_out = vec.clone(); vec_out[INJ_LAYER] = v_bar
torch.save(vec_out, os.path.join(OUT_DIR, os.environ.get("SKOP_OUT", "v_pref_no_task_gemma_skopres.pt")))
torch.save(vec_out, os.path.expanduser("~/hotwire-vectors/" + os.environ.get("SKOP_OUT", "v_pref_no_task_gemma_skopres.pt")))
with open(os.path.join(OUT_DIR, "diag.json"), "w") as f:
    json.dump({"diag": diag, "risk_heads": [{"layer": x["layer"], "head": x["head"],
        "rayleigh": x["rayleigh"]} for x in risk]}, f, indent=1)
print("SAVED " + os.environ.get("SKOP_OUT", "v_pref_no_task_gemma_skopres.pt"))
