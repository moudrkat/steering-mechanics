"""E1: random-basis projection control at matched rank (CONTROLS_PREREG R1).

Builds random-complement projections of the steering vector at the same
ranks as the targeted v0/v1 cuts (default 1536 and 149 in D=2560), three
seeds each. Expected norm kept sqrt(1 - r/D) matches the targeted cuts
by construction; realized norms land in the diag JSON together with the
matched-magnitude probe scales (s such that s*|ctl| = REF_SCALE*|v|).

CPU-only, needs just the vector file. Probe arms run separately via
skop_efficacy_probe.py (see runchain_controls.sh).
"""
import torch, json, os

VEC_PATH = os.path.expanduser(os.environ.get("SKOP_VEC", "~/hotwire-vectors/v_pref_no_task_checklist_v3.pt"))
INJ = int(os.environ.get("SKOP_INJ", "20"))
RANKS = [int(r) for r in os.environ.get("SKOP_CTL_RANKS", "149,1536").split(",")]
SEEDS = [int(s) for s in os.environ.get("SKOP_CTL_SEEDS", "1,2,3").split(",")]
REF_SCALE = float(os.environ.get("SKOP_REF_SCALE", "3.0"))
OUT_DIR = os.path.expanduser("~/skop_residual")
os.makedirs(OUT_DIR, exist_ok=True)

vec = torch.load(VEC_PATH, map_location="cpu", weights_only=True).float()
v = vec[INJ].clone()
D = v.shape[0]

diag = {"vec": VEC_PATH, "inj_layer": INJ, "D": D, "norm_v": float(v.norm()),
        "ref_scale": REF_SCALE, "controls": []}
for rank in RANKS:
    for seed in SEEDS:
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(D, rank, generator=g))
        v_ctl = v - Q @ (Q.T @ v)
        name = f"v_randctl_r{rank}_s{seed}.pt"
        vec_out = vec.clone(); vec_out[INJ] = v_ctl
        for dst in (OUT_DIR, os.path.expanduser("~/hotwire-vectors")):
            torch.save(vec_out, os.path.join(dst, name))
        entry = {
            "name": name, "rank": rank, "seed": seed,
            "norm_kept": round(float(v_ctl.norm() / v.norm()), 4),
            "norm_kept_expected": round((1 - rank / D) ** 0.5, 4),
            "cos(v, ctl)": round(float((v @ v_ctl) / (v.norm() * v_ctl.norm())), 4),
            "matched_scale": round(REF_SCALE * float(v.norm() / v_ctl.norm()), 2),
        }
        diag["controls"].append(entry)
        print(json.dumps(entry))

with open(os.path.join(OUT_DIR, "randctl_diag.json"), "w") as f:
    json.dump(diag, f, indent=1)
print("SAVED randctl_diag.json")
