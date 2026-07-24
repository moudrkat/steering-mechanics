"""Hero command: render figures from shipped example data — NO GPU needed.

Paper style: white background, Okabe-Ito colorblind-safe series colors.

    steermech-plot            # renders all figures into fig/ from examples/
    python -m steermech.plot

This is the 2-minute first experience: install, run this, look at real
measured results (the L21 head tug-of-war, the dose-response curve, the
attn/MLP component split) without a server or a GPU. To produce your OWN
data, run the experiments against a live brainscope (see README).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples"
FIG = ROOT / "fig"


def _pil():
    from PIL import Image, ImageDraw, ImageFont
    F = "/usr/share/fonts/truetype/dejavu/"
    def font(name, sz):
        for p in (F + name, name):
            try:
                return ImageFont.truetype(p, sz)
            except OSError:
                continue
        return ImageFont.load_default()
    return Image, ImageDraw, font


def dose_curve():
    """Dose-response: suppressed positions + imprint vs scale."""
    Image, ImageDraw, font = _pil()
    d = json.loads((EX / "dose_response.json").read_text())
    rows = d["scales"]
    W, H, pad = 900, 500, 70
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f1, f2, f3 = font("DejaVuSans-Bold.ttf", 26), font("DejaVuSans.ttf", 15), font("DejaVuSansMono.ttf", 13)
    dr.text((W//2, 34), "Dose-response: more dose, more suppression", font=f1, fill=(20,24,28), anchor="mm")
    dr.text((W//2, 62), "steering vector v_pref_no_task_checklist_v3 @ L20, teacher-forced", font=f2, fill=(95,103,112), anchor="mm")
    scales = [r["scale"] for r in rows]
    pos = [r["suppressed_positions"] for r in rows]
    xmax, ymax = max(scales)*1.05, max(pos)*1.15
    def X(s): return pad + s/xmax*(W-2*pad)
    def Y(p): return H-pad - p/ymax*(H-2*pad-40)
    for gy in range(0, int(ymax)+1, 50):
        dr.line([(pad, Y(gy)), (W-pad, Y(gy))], fill=(228,231,235))
        dr.text((pad-8, Y(gy)), str(gy), font=f3, fill=(95,103,112), anchor="rm")
    pts = [(X(s), Y(p)) for s, p in zip(scales, pos)]
    dr.line(pts, fill=(0,158,115), width=3)
    for (x,y),(s,p) in zip(pts, zip(scales,pos)):
        dr.ellipse([x-5,y-5,x+5,y+5], fill=(0,110,80))
        dr.text((x, y-16), f"{p}", font=f3, fill=(0,110,80), anchor="mm")
        dr.text((x, H-pad+16), f"{s:g}", font=f3, fill=(95,103,112), anchor="mm")
    dr.text((W//2, H-20), "injection scale", font=f2, fill=(95,103,112), anchor="mm")
    dr.text((pad-40, H//2), "suppressed positions", font=f2, fill=(95,103,112), anchor="mm")
    img.save(FIG / "dose_response.png")
    return "dose_response.png"


def component_bars():
    """attn vs MLP delta per layer after injection."""
    Image, ImageDraw, font = _pil()
    d = json.loads((EX / "component_attribution.json").read_text())
    rows = d["rows"]
    W, H, pad = 820, 460, 70
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f1, f2, f3 = font("DejaVuSans-Bold.ttf", 24), font("DejaVuSans.ttf", 15), font("DejaVuSansMono.ttf", 14)
    dr.text((W//2, 32), "Who carries the injected direction: attention vs MLP", font=f1, fill=(20,24,28), anchor="mm")
    dr.text((W//2, 60), "steered-minus-clean write along the vector, by layer", font=f2, fill=(95,103,112), anchor="mm")
    vals = []
    for r in rows:
        vals += [r["attn"]["delta_mean"], r["mlp"]["delta_mean"]]
    m = max(abs(v) for v in vals) * 1.2
    zero = H//2 + 20
    def Y(v): return zero - v/m*(H//2-60)
    dr.line([(pad, zero), (W-pad, zero)], fill=(150,158,166))
    n = len(rows); gw = (W-2*pad)/n; bw = gw*0.28
    for i, r in enumerate(rows):
        cx = pad + gw*(i+0.5)
        for k, (comp, col) in enumerate((("attn",(0,158,115)),("mlp",(230,159,0)))):
            v = r[comp]["delta_mean"]
            x = cx + (k-0.5)*bw*1.2
            dr.rectangle([x-bw/2, min(zero,Y(v)), x+bw/2, max(zero,Y(v))], fill=col)
            dr.text((x, Y(v)+(-14 if v>0 else 14)), f"{v:+.1f}", font=f3, fill=col, anchor="mm")
        dr.text((cx, H-30), f"L{r['layer']}", font=f2, fill=(55,63,72), anchor="mm")
    dr.text((pad, 92), "■ attn", font=f3, fill=(0,158,115))
    dr.text((pad+80, 92), "■ MLP", font=f3, fill=(230,159,0))
    img.save(FIG / "component_attribution.png")
    return "component_attribution.png"


def tug_of_war():
    """Regenerate the head tug-of-war gif from example head data."""
    script = ROOT / "fig/tug_of_war_gif.py"
    if not (EX / "head_attribution.json").exists():
        return None
    # tug_of_war_gif.py reads results/; temporarily point it at examples via symlink-free copy
    (ROOT / "results").mkdir(exist_ok=True)
    tgt = ROOT / "results/head_attribution.json"
    if not tgt.exists():
        tgt.write_text((EX / "head_attribution.json").read_text())
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True)
    return "tug_of_war.gif"


def main():
    FIG.mkdir(exist_ok=True)
    made = []
    for fn in (dose_curve, component_bars, tug_of_war, calibration_landscape,
               rq1_dose_regime, transfer_story, depth_test,
               linkedin_metric_lies, dose_curves_all_models):
        try:
            r = fn()
            if r:
                made.append(r)
                print(f"  ✓ fig/{r}")
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
    print(f"\n{len(made)} figures rendered into fig/ — no GPU, from shipped example data.")
    print("Run your own: start a brainscope server and see README > Live experiments.")


def calibration_landscape():
    """Scatter of the auto-calibration trials: layer x scale, colored by
    objective, effective (miss 0) points ringed."""
    Image, ImageDraw, font = _pil()
    src = EX / "autocalibrate.json"
    if not src.exists():
        return None
    d = json.loads(src.read_text())
    t = d["trials"]
    W, H, pad = 900, 560, 80
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f1, f2, f3 = font("DejaVuSans-Bold.ttf", 24), font("DejaVuSans.ttf", 15), font("DejaVuSansMono.ttf", 12)
    dr.text((W//2, 30), "Auto-calibration landscape (heretic-style objective)", font=f1, fill=(20,24,28), anchor="mm")
    dr.text((W//2, 58), "each dot = one trial · x layer · y scale · color = miss + λ·KL (green=good) · ring = fully effective", font=f2, fill=(95,103,112), anchor="mm")
    Ls = [r["layer"] for r in t]; Ss = [r["scale"] for r in t]
    lmin, lmax = min(Ls)-1, max(Ls)+1
    smin, smax = 0, max(Ss)*1.05
    scmax = max(r["score"] for r in t)
    def X(l): return pad + (l-lmin)/(lmax-lmin)*(W-2*pad)
    def Y(s): return H-pad - (s-smin)/(smax-smin)*(H-2*pad-30)
    for l in range(int(lmin)+1, int(lmax)+1, 2):
        dr.line([(X(l),pad),(X(l),H-pad)], fill=(228,231,235))
        dr.text((X(l), H-pad+16), f"L{l}", font=f3, fill=(95,103,112), anchor="mm")
    for s in range(0, int(smax)+1):
        dr.line([(pad,Y(s)),(W-pad,Y(s))], fill=(228,231,235))
        dr.text((pad-10, Y(s)), str(s), font=f3, fill=(95,103,112), anchor="rm")
    for r in t:
        x, y = X(r["layer"]), Y(r["scale"])
        # green (low score/good) -> red (high/bad)
        f = min(1.0, r["score"]/max(scmax,1e-9))
        col = (int(63+(216-63)*f), int(184+(80-184)*f), int(131+(60-131)*f))
        rr = 8
        dr.ellipse([x-rr,y-rr,x+rr,y+rr], fill=col)
        if r["miss"] == 0:
            dr.ellipse([x-rr-3,y-rr-3,x+rr+3,y+rr+3], outline=(20,24,28), width=2)
    best = d["best"]
    bx, by = X(best["layer"]), Y(best["scale"])
    dr.text((bx, by-20), "best", font=f3, fill=(20,24,28), anchor="mm")
    dr.text((W//2, H-22), "layer   (ringed dots = vector fully effective; deeper layers win on generic prompts)", font=f2, fill=(95,103,112), anchor="mm")
    dr.text((26, H//2), "scale", font=f2, fill=(95,103,112), anchor="mm")
    img.save(FIG / "calibration_landscape.png")
    return "calibration_landscape.png"


def _line_chart(title, sub, series, xlab, ylab, out, ymax=1.05, marks=()):
    """series: [(label, color, [(x,y),...])]; marks: [(x, y, text, color)]."""
    Image, ImageDraw, font = _pil()
    W, H, pad = 900, 520, 70
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f1, f2, f3 = font("DejaVuSans-Bold.ttf", 24), font("DejaVuSans.ttf", 15), font("DejaVuSansMono.ttf", 13)
    dr.text((W//2, 32), title, font=f1, fill=(20,24,28), anchor="mm")
    dr.text((W//2, 60), sub, font=f2, fill=(95,103,112), anchor="mm")
    xs = [x for _,_,pts in series for x,_ in pts] + [m[0] for m in marks]
    xmax = max(xs)*1.06
    def X(v): return pad + v/xmax*(W-2*pad)
    def Y(v): return H-pad - v/ymax*(H-2*pad-50)
    for gy in (0, 0.25, 0.5, 0.75, 1.0):
        dr.line([(pad, Y(gy)), (W-pad, Y(gy))], fill=(228,231,235))
        dr.text((pad-8, Y(gy)), f"{gy:g}", font=f3, fill=(95,103,112), anchor="rm")
    for lbl_i, (label, col, pts) in enumerate(series):
        P = [(X(x), Y(y)) for x, y in pts]
        dr.line(P, fill=col, width=3)
        for (px,py),(x,_) in zip(P, pts):
            dr.ellipse([px-5,py-5,px+5,py+5], fill=col)
            dr.text((px, H-pad+16), f"{x:g}", font=f3, fill=(95,103,112), anchor="mm")
        dr.text((pad+8, 88+lbl_i*20), f"\u25a0 {label}", font=f3, fill=col)
    for x, y, text, col in marks:
        px, py = X(x), Y(y)
        dr.line([(px-8,py-8),(px+8,py+8)], fill=col, width=3)
        dr.line([(px-8,py+8),(px+8,py-8)], fill=col, width=3)
        dr.text((px, py-18), text, font=f3, fill=col, anchor="mm")
    dr.text((W//2, H-20), xlab, font=f2, fill=(95,103,112), anchor="mm")
    dr.text((26, H//2), ylab, font=f2, fill=(95,103,112), anchor="mm")
    img.save(FIG / out)
    return out


def rq1_dose_regime():
    """RQ1 row 0: miss vs scale, decode-only vs full-steer."""
    src = ROOT / "results/rq1_row0_8b.json"
    if not src.exists():
        return None
    rows = json.loads(src.read_text())["rows"]
    dec = [(r["scale"], r["miss"]) for r in rows if r["regime"] == "decode_only"]
    ful = [(r["scale"], r["miss"]) for r in rows if r["regime"] == "full"]
    return _line_chart(
        "Dose-response with a cliff: the working window is scales 2-4",
        "Qwen3-8B @ L20, short context, N=12 · miss = violation OR incoherence · full-steer collapses harder, not sooner",
        [("decode-only", (0,158,115), dec), ("full-steer", (230,159,0), ful)],
        "injection scale", "miss", "rq1_dose_regime.png")


def transfer_story():
    """H3: native dose curves per model + the two cross-points."""
    a, b = ROOT/"results/transfer_notasks_8b.json", ROOT/"results/transfer_notasks_4b.json"
    if not (a.exists() and b.exists()):
        return None
    r8 = json.loads(a.read_text())["rows"]; r4 = json.loads(b.read_text())["rows"]
    c8 = [(r["scale"], r["miss"]) for r in r8 if r["layer"] == 15][:6]
    c4 = [(r["scale"], r["miss"]) for r in r4 if r["layer"] == 20][:6]
    x8 = next(r for r in r8 if r["layer"] == 20 and r["scale"] == 3.0)
    x4 = next(r for r in r4 if r["layer"] == 15 and r["scale"] == 8.0)
    return _line_chart(
        "The window transfers, the argmax lies",
        "native curves: 8B@L15 vs 4B@L20 · X = the other model's optimum transplanted (4B->8B works; 8B->4B fails)",
        [("Qwen3-8B @ its argmax layer L15", (0,158,115), c8),
         ("Qwen3-4B @ shared layer L20", (230,159,0), c4)],
        "injection scale", "miss", "transfer_story.png",
        marks=[(3.0, x8["miss"], "4B opt on 8B: miss 0", (0,110,80)),
               (8.0, x4["miss"], "8B argmax on 4B: fails", (213,94,0))])


def depth_test():
    """Fractional depth vs raw index on a 28-layer model."""
    src = ROOT / "results/transfer_notasks_qwen25_7b.json"
    if not src.exists():
        return None
    rows = json.loads(src.read_text())["rows"]
    frac = [(r["scale"], r["miss"]) for r in rows if r["layer"] == 16][:6]
    marks = [(r["scale"], r["miss"], f"L{r['layer']}@3", (213,94,0))
             for r in rows if r["layer"] in (14, 18, 20) and r["scale"] == 3.0]
    return _line_chart(
        "Does the layer transfer as fractional depth? (28-layer model)",
        "Qwen2.5-7B · green: L16 = fractional-depth prediction (20/36 -> 16/28) · X: raw-index & neighbors @ scale 3",
        [("L16 (fractional-depth prediction)", (0,158,115), frac)],
        "injection scale", "miss", "depth_test.png", marks=marks)


def _bootstrap_ci(k, n, reps=2000, seed=0):
    """95% bootstrap CI for a rate k/n. Deterministic (seeded, no RNG import
    that breaks reproducibility)."""
    if n == 0:
        return (0.0, 0.0)
    # analytic-ish: resample via a fixed LCG for determinism
    state = seed or 1
    rates = []
    p = k / n
    for _ in range(reps):
        hits = 0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7fffffff
            hits += (state / 0x7fffffff) < p
        rates.append(hits / n)
    rates.sort()
    return (round(rates[int(0.025*reps)], 3), round(rates[int(0.975*reps)], 3))


def dose_curves_all_models():
    """Miss vs injection scale, at each model's 0.56-depth reference layer, all
    models overlaid. RAW scale on x: curves share a shape but sit at different
    positions (bigger models need more raw scale) — exactly the norm confound;
    relative dose would align them. The 'scale does not transfer in raw units'
    point, made visible in one figure."""
    import glob
    files = sorted(glob.glob(str(ROOT / "results/campaign_*.json")))
    colors = [(0, 158, 115), (230, 159, 0), (86, 180, 233), (213, 94, 0),
              (204, 121, 167)]
    series = []
    for i, f in enumerate(files):
        d = json.loads(Path(f).read_text())
        nl = d.get("n_layers")
        if not nl:
            continue
        ref = round(0.56 * nl)
        pts = sorted([(r["scale"], r["miss"]) for r in d["rows"]
                      if r["layer"] == ref], key=lambda t: t[0])
        if len(pts) >= 4:
            series.append((f"{d['model']} (L{ref}/{nl})",
                           colors[i % len(colors)], pts))
    if not series:
        return None
    return _line_chart(
        "Dose-response, every model: same shape, different raw scale",
        "miss vs injection scale at 0.56 fractional depth  ·  RAW scale, NOT "
        "cross-model comparable (curves shift by vector norm; relative dose aligns them)",
        series, "injection scale (raw)", "miss", "dose_curves_all_models.png")


def dose_arc_all_models():
    """The full three-phase arc: TOO WEAK (doesn't work) -> WORKS -> CLIFF
    (collapses). x = overdose factor (scale / each model's sweet-spot scale),
    so all models line up: weak below 1, working at 1, cliff above ~1.3."""
    import glob
    Image, ImageDraw, font = _pil()
    files = sorted(glob.glob(str(ROOT / "results/campaign_*.json")))
    colors = [(0, 158, 115), (230, 159, 0), (86, 180, 233), (213, 94, 0),
              (204, 121, 167)]
    series = []
    for i, f in enumerate(files):
        d = json.loads(Path(f).read_text())
        nl = d.get("n_layers")
        if not nl:
            continue
        ref = round(0.56 * nl)
        curve = sorted([(r["scale"], r["miss"]) for r in d["rows"]
                        if r["layer"] == ref], key=lambda t: t[0])
        if len(curve) < 4:
            continue
        sweet = min(curve, key=lambda t: t[1])[0]
        series.append((f"{d['model']} (sweet s={sweet:g})",
                       colors[i % len(colors)], [(s / sweet, m) for s, m in curve]))
    if not series:
        return None
    W, H, pad = 960, 540, 74
    img = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    f1 = font("DejaVuSans-Bold.ttf", 25); f2 = font("DejaVuSans.ttf", 15)
    f3 = font("DejaVuSansMono.ttf", 13); f4 = font("DejaVuSans-Bold.ttf", 15)
    dr.text((W // 2, 30), "One dose-response, three phases: too weak -> works -> cliff",
            font=f1, fill=(20, 24, 28), anchor="mm")
    dr.text((W // 2, 58), "miss vs overdose factor (scale / each model's sweet-spot scale) - "
            "aligned so every model tells the same story", font=f2, fill=(95, 103, 112), anchor="mm")
    xmax = max(x for _, _, pts in series for x, _ in pts) * 1.04
    def X(v): return pad + v / xmax * (W - 2 * pad)
    def Y(v): return H - pad - v * (H - 2 * pad - 60)
    # three zones
    dr.rectangle([X(0), Y(1.05), X(0.8), Y(0)], fill=(247, 249, 251))
    dr.rectangle([X(1.2), Y(1.05), W - pad, Y(0)], fill=(252, 247, 247))
    for zx, lbl, col in [(0.4, "TOO WEAK", (150, 120, 60)),
                         (1.0, "WORKS", (0, 120, 80)),
                         (1.7, "CLIFF", (200, 70, 30))]:
        dr.text((X(zx), Y(0.98)), lbl, font=f4, fill=col, anchor="mm")
    for xln in (0.8, 1.2):
        for yy in range(0, int((H - 2 * pad - 60)), 8):
            dr.point((X(xln), Y(0) - yy), fill=(200, 205, 210))
    for gy in (0, 0.25, 0.5, 0.75, 1.0):
        dr.line([(pad, Y(gy)), (W - pad, Y(gy))], fill=(228, 231, 235))
        dr.text((pad - 8, Y(gy)), f"{gy:g}", font=f3, fill=(95, 103, 112), anchor="rm")
    for li, (label, col, pts) in enumerate(series):
        P = [(X(x), Y(y)) for x, y in pts]
        dr.line(P, fill=col, width=3)
        for px, py in P:
            dr.ellipse([px - 5, py - 5, px + 5, py + 5], fill=col)
        dr.text((pad + 8, 84 + li * 20), f"■ {label}", font=f4, fill=col)
    dr.text((X(1.0), H - pad + 16), "1.0", font=f3, fill=(95, 103, 112), anchor="mm")
    dr.text((W // 2, H - 20), "overdose factor  (1.0 = the working dose)",
            font=f2, fill=(95, 103, 112), anchor="mm")
    dr.text((26, H // 2), "miss", font=f2, fill=(95, 103, 112), anchor="mm")
    img.save(FIG / "dose_arc_all_models.png")
    return "dose_arc_all_models.png"


def dose_cliff_all_models():
    """The CLIFF, isolated: for each model, drop the under-dosed left arm and
    plot miss vs OVERDOSE FACTOR (scale / that model's sweet-spot scale). The
    working point sits at x=1 for every model, so the collapse edges align —
    and you see them fall off the same cliff at ~1.3x their working dose,
    regardless of raw scale. The 'coherence collapses past the window' story."""
    import glob
    files = sorted(glob.glob(str(ROOT / "results/campaign_*.json")))
    colors = [(0, 158, 115), (230, 159, 0), (86, 180, 233), (213, 94, 0),
              (204, 121, 167)]
    series = []
    for i, f in enumerate(files):
        d = json.loads(Path(f).read_text())
        nl = d.get("n_layers")
        if not nl:
            continue
        ref = round(0.56 * nl)
        curve = sorted([(r["scale"], r["miss"]) for r in d["rows"]
                        if r["layer"] == ref], key=lambda t: t[0])
        if len(curve) < 4:
            continue
        sweet = min(curve, key=lambda t: t[1])[0]              # sweet-spot scale
        arm = [(s / sweet, m) for s, m in curve if s >= sweet]  # working point onward
        if len(arm) >= 2:
            series.append((f"{d['model']} (sweet s={sweet:g})",
                           colors[i % len(colors)], arm))
    if not series:
        return None
    return _line_chart(
        "The collapse cliff, aligned: past ~1.3x the working dose, the model breaks",
        "miss vs overdose factor (scale / each model's sweet-spot scale)  ·  cliffs "
        "align once you divide out the raw-scale offset",
        series, "overdose factor  (1.0 = the working dose)", "miss",
        "dose_cliff_all_models.png", ymax=1.08)


def linkedin_metric_lies():
    """The one-figure story: a naive violation-only metric says the high-dose
    point is perfect; the truth (violation OR incoherence) says it is broken."""
    src = ROOT / "results/rq1_row0_8b.json"
    if not src.exists():
        return None
    rows = [r for r in json.loads(src.read_text())["rows"] if r["regime"] == "decode_only"]
    n = 12
    naive = [(r["scale"], r["violations"]/n) for r in rows]          # what the optimizer saw
    truth = [(r["scale"], (r["violations"]+r["incoherent"])/n) for r in rows]  # what was real
    Image, ImageDraw, font = _pil()
    W, H, pad = 1000, 560, 80
    img = Image.new("RGB", (W, H), (255,255,255))
    dr = ImageDraw.Draw(img)
    f1 = font("DejaVuSans-Bold.ttf", 30); f2 = font("DejaVuSans.ttf", 17)
    f3 = font("DejaVuSansMono.ttf", 15); f4 = font("DejaVuSans-Bold.ttf", 16)
    dr.text((W//2, 34), "The optimizer's metric said perfect. The model was broken.", font=f1, fill=(20,24,28), anchor="mm")
    dr.text((W//2, 66), "auto-calibrating a steering vector · Qwen3-8B · violation-rate alone is blind to the model falling apart", font=f2, fill=(95,103,112), anchor="mm")
    xmax = 8.6
    def X(v): return pad + v/xmax*(W-2*pad)
    def Y(v): return H-pad - v*(H-2*pad-60)
    for gy in (0,0.25,0.5,0.75,1.0):
        dr.line([(pad,Y(gy)),(W-pad,Y(gy))], fill=(228,231,235))
        dr.text((pad-10,Y(gy)), f"{int(gy*100)}%", font=f3, fill=(95,103,112), anchor="rm")
    for label,col,series,dash in [("what the optimizer measured (violations)",(0,158,115),naive,False),
                                  ("what was actually true (violations OR model-collapse)",(213,94,0),truth,False)]:
        P=[(X(x),Y(y)) for x,y in series]
        dr.line(P, fill=col, width=4)
        for px,py in P: dr.ellipse([px-6,py-6,px+6,py+6], fill=col)
    for x in [r["scale"] for r in rows]:
        dr.text((X(x), H-pad+18), f"{x:g}", font=f3, fill=(95,103,112), anchor="mm")
    dr.text((pad+10, 92), "\u25a0 what the optimizer measured (violations)", font=f4, fill=(0,158,115))
    dr.text((pad+10, 114), "\u25a0 what was actually true (violations OR gibberish)", font=f4, fill=(213,94,0))
    # annotate the trap point (scale 8): naive 0%, truth 100%
    tx = X(8)
    dr.line([(tx, Y(0)+6),(tx, Y(1.0)-6)], fill=(120,120,120), width=1)
    dr.text((tx-6, Y(0.5)), "same setting:\n0% violations,\n100% broken", font=f4, fill=(60,60,60), anchor="rm")
    dr.text((W//2, H-22), "injection strength  \u2192  (build steering vectors people can trust:  pip install hidden-directions)", font=f2, fill=(95,103,112), anchor="mm")
    img.save(FIG / "linkedin_metric_lies.png")
    return "linkedin_metric_lies.png"


if __name__ == "__main__":
    main()
