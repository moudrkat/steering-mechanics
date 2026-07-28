"""Publication + slide figure line for the channel-factorization paper.
Renders ONLY from committed scores-only JSONs in results/ (repo convention).

Usage: FIG_OUT=<dir> [FIG_SLIDES=1] python make_paper_figs.py
Outputs: fig2_channels, fig3_vectors, fig4_arc, fig5_ladder (.png 300dpi + .pdf)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT = os.path.expanduser(os.environ.get("FIG_OUT", "figs"))
SLIDES = os.environ.get("FIG_SLIDES") == "1"
os.makedirs(OUT, exist_ok=True)

# palette (dataviz reference instance, light mode; color follows the MODEL)
C = {"q4b": "#2a78d6", "q8b": "#eb6834", "q25": "#1baf7a",
     "llama": "#eda100", "gemma": "#e87ba4"}
INK, INK2, MUT, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

BASE = 15 if SLIDES else 9.5
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": BASE,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUT, "ytick.color": MUT,
    "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.grid": False, "svg.fonttype": "none"})

def jload(name): return json.load(open(os.path.join(R, name)))

def style_ax(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)

def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight", facecolor=SURF)
    plt.close(fig)
    print("saved", name)

# ---------- Fig 2: attention-carried share per model ----------
ef = jload("final_chainEF.json")
cb = jload("rigor_chainB.json")
gm = jload("gemma_h4_summary.json")
rows = [  # (label, share, ci, color, note)
    ("Qwen3-4B",  ef["fattnall_4b"]["3.0"],  C["q4b"],  "all layers, s3"),
    ("Qwen3-8B",  ef["fattnall_8b"]["3.0"],  C["q8b"],  "all layers, s3"),
    ("Qwen2.5-7B", cb["qwen25"]["doses"]["3.0"], C["q25"], "7-layer band, s3"),
    ("Llama-3.1-8B", ef["fattnall_llama"]["1.5"], C["llama"], "all layers, matched damage"),
    ("Gemma-4-E4B", gm["4.0"], C["gemma"], "query channel by construction, s4"),
]
fig, ax = plt.subplots(figsize=(10, 4.4) if SLIDES else (7.0, 3.1))
ypos = range(len(rows))[::-1]
for y, (lab, d, col, note) in zip(ypos, rows):
    v = d["rescue_fattn"]; lo, hi = d.get("rescue_fattn_ci", (v, v))
    ax.barh(y, v * 100, height=0.62, color=col, zorder=3)
    ax.errorbar(v * 100, y, xerr=[[100 * (v - lo)], [100 * (hi - v)]],
                fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=4)
    ax.text(hi * 100 + 2.2, y, f"{v:.0%}", va="center", color=INK,
            fontsize=BASE + 1, fontweight="bold")
    ax.text(0.8, y - 0.43, note, va="center", color=MUT, fontsize=BASE - 2)
ax.set_yticks(list(ypos)); ax.set_yticklabels([r[0] for r in rows], color=INK)
ax.set_xlim(0, 100); ax.set_xlabel("share of steering damage removed by freezing attention (%)")
ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
style_ax(ax)
ax.set_title("How much steering damage flows through attention — per model"
             + ("" if SLIDES else " (task-suppression vector, working dose, 95% CI)"),
             loc="left", fontsize=BASE + (3 if SLIDES else 1.5), pad=10)
save(fig, "fig2_channels")

# ---------- Fig 3: vector dependence on one model (Qwen3-4B, band) ----------
rf = jload("rigor_factorizations.json")
r4 = jload("qk_freeze2_4b_report.json")
vec_rows = [
    ("task-suppression", r4["doses"]["3.0"]),
    ("websearch-overtrigger", rf["4b_websearch"]["doses"]["3.0"]),
    ("random control", rf["4b_rand"]["doses"]["3.0"]),
    ("confidence", cb["4b_confident"]["doses"]["3.0"]),
    ("sycophancy", cb["4b_sycophant"]["doses"]["3.0"]),
    ("refusal", cb["4b_refusal"]["doses"]["3.0"]),
]
fig, ax = plt.subplots(figsize=(10, 4.4) if SLIDES else (7.0, 3.1))
ypos = range(len(vec_rows))[::-1]
for y, (lab, d) in zip(ypos, vec_rows):
    v = d["rescue_fattn"]; lo, hi = d.get("rescue_fattn_ci", (v, v))
    ax.barh(y, v * 100, height=0.62, color=C["q4b"], zorder=3)
    ax.errorbar(v * 100, y, xerr=[[100 * (v - lo)], [100 * (hi - v)]],
                fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=4)
    ax.text(max(hi * 100, 0) + 2.2, y, f"{v:.0%}", va="center", color=INK,
            fontsize=BASE + 1, fontweight="bold")
ax.set_yticks(list(ypos)); ax.set_yticklabels([r[0] for r in vec_rows], color=INK)
ax.set_xlim(0, 100); ax.set_xlabel("attention-carried share of damage (%, 7-layer band, s3)")
ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
style_ax(ax)
ax.set_title("Same model, different vector — the share is not a model constant"
             + ("" if SLIDES else " (Qwen3-4B, six vectors, 95% CI)"),
             loc="left", fontsize=BASE + (3 if SLIDES else 1.5), pad=10)
save(fig, "fig3_vectors")

# ---------- Fig 4: ARC accuracy vs dose ----------
arc = jload("arc300_summary.json")
labels = {"4b": ("Qwen3-4B", C["q4b"]), "8b": ("Qwen3-8B", C["q8b"]),
          "qwen25": ("Qwen2.5-7B", C["q25"]), "llama": ("Llama-3.1-8B", C["llama"])}
fig, ax = plt.subplots(figsize=(10, 5.2) if SLIDES else (7.0, 3.6))
doses = [0, 3, 5, 8]
for key, (lab, col) in labels.items():
    ys = [arc[key][f"{d:.1f}"] * 100 for d in doses]
    ax.plot(doses, ys, color=col, linewidth=2.2, zorder=3, label=lab,
            marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=1.2)
ax.legend(loc="lower left", frameon=False, fontsize=BASE - 1,
          labelcolor=INK2, handlelength=1.6, bbox_to_anchor=(0.02, 0.14))
ax.axhline(25, color=MUT, linewidth=1, linestyle=(0, (3, 3)))
ax.text(8.9, 26.5, "chance (4 options)", color=MUT, fontsize=BASE - 2, ha="right")
ax.axvspan(2.7, 3.3, color=GRID, alpha=0.55, zorder=0)
ax.text(3, 98, "working dose", ha="center", color=INK2, fontsize=BASE - 1)
ax.set_xlim(-0.3, 9.6); ax.set_ylim(0, 103)
ax.set_xticks(doses); ax.set_xlabel("steering scale (decode-only)")
ax.set_ylabel("ARC-Challenge accuracy (%, n=300)")
ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
style_ax(ax)
ax.set_title("The deployment window is capability-free — and the wall is real",
             loc="left", fontsize=BASE + (3 if SLIDES else 1.5), pad=10)
save(fig, "fig4_arc")

# ---------- Fig 5: saturation of the residual->query map ----------
lads = {"Qwen3-4B": (jload("qk_freeze2_4b_report.json")["ladder"], C["q4b"]),
        "Qwen3-8B": (jload("qk_freeze2_8b_report.json")["ladder"], C["q8b"]),
        "Llama-3.1-8B": (jload("qk_freeze2_llama_report.json")["ladder"], C["llama"])}
fig, (a1, a2) = plt.subplots(2, 1, sharex=True,
                             figsize=(10, 6.2) if SLIDES else (7.0, 4.6),
                             gridspec_kw={"hspace": 0.12})
ss = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
dodge = {"Qwen3-4B": 5, "Qwen3-8B": -7, "Llama-3.1-8B": 0}
for lab, (lad, col) in lads.items():
    norm = [lad[str(s)]["dqband_per_s"] for s in ss]
    rel = [n / norm[0] for n in norm]
    cos = [lad[str(s)]["cosband"] for s in ss]
    for ax_, ys in ((a1, rel), (a2, cos)):
        ax_.plot(ss, ys, color=col, linewidth=2.2, marker="o", markersize=6.5,
                 markeredgecolor=SURF, markeredgewidth=1.1, zorder=3, label=lab)
    a1.annotate(lab, (ss[-1], rel[-1]), xytext=(8, dodge[lab]),
                textcoords="offset points", va="center", color=INK2,
                fontsize=BASE - 1)
for ax_, ref in ((a1, 1.0), (a2, 1.0)):
    ax_.axhline(ref, color=MUT, linewidth=1, linestyle=(0, (3, 3)))
a1.text(0.55, 1.02, "linear map", color=MUT, fontsize=BASE - 2)
a1.set_ylabel("‖Δq‖/s, rel. to s=0.5"); a2.set_ylabel("cos(Δq, linear pred.)")
a2.set_xlabel("steering scale s"); a2.set_xscale("log"); a2.set_xticks(ss)
a2.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
a2.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
a2.tick_params(which="minor", bottom=False)
a1.tick_params(which="minor", bottom=False)
a1.set_ylim(0.3, 1.12); a2.set_ylim(0.55, 1.05)
for ax_ in (a1, a2):
    ax_.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0); style_ax(ax_)
a1.axvspan(2.7, 3.3, color=GRID, alpha=0.55, zorder=0)
a2.axvspan(2.7, 3.3, color=GRID, alpha=0.55, zorder=0)
a1.set_title("The residual→query map saturates before the working dose"
             + ("" if SLIDES else " (band mean over heads, N=40)"),
             loc="left", fontsize=BASE + (3 if SLIDES else 1.5), pad=10)
save(fig, "fig5_ladder")
print("ALL FIGURES DONE ->", OUT)
