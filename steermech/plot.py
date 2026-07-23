"""Hero command: render figures from shipped example data — NO GPU needed.

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
    img = Image.new("RGB", (W, H), (13, 17, 23))
    dr = ImageDraw.Draw(img)
    f1, f2, f3 = font("DejaVuSans-Bold.ttf", 26), font("DejaVuSans.ttf", 15), font("DejaVuSansMono.ttf", 13)
    dr.text((W//2, 34), "Dose-response: more dose, more suppression", font=f1, fill=(230,237,243), anchor="mm")
    dr.text((W//2, 62), "steering vector v_pref_no_task_checklist_v3 @ L20, teacher-forced", font=f2, fill=(139,148,158), anchor="mm")
    scales = [r["scale"] for r in rows]
    pos = [r["suppressed_positions"] for r in rows]
    xmax, ymax = max(scales)*1.05, max(pos)*1.15
    def X(s): return pad + s/xmax*(W-2*pad)
    def Y(p): return H-pad - p/ymax*(H-2*pad-40)
    for gy in range(0, int(ymax)+1, 50):
        dr.line([(pad, Y(gy)), (W-pad, Y(gy))], fill=(30,37,46))
        dr.text((pad-8, Y(gy)), str(gy), font=f3, fill=(139,148,158), anchor="rm")
    pts = [(X(s), Y(p)) for s, p in zip(scales, pos)]
    dr.line(pts, fill=(63,184,131), width=3)
    for (x,y),(s,p) in zip(pts, zip(scales,pos)):
        dr.ellipse([x-5,y-5,x+5,y+5], fill=(127,224,181))
        dr.text((x, y-16), f"{p}", font=f3, fill=(127,224,181), anchor="mm")
        dr.text((x, H-pad+16), f"{s:g}", font=f3, fill=(139,148,158), anchor="mm")
    dr.text((W//2, H-20), "injection scale", font=f2, fill=(139,148,158), anchor="mm")
    dr.text((pad-40, H//2), "suppressed positions", font=f2, fill=(139,148,158), anchor="mm")
    img.save(FIG / "dose_response.png")
    return "dose_response.png"


def component_bars():
    """attn vs MLP delta per layer after injection."""
    Image, ImageDraw, font = _pil()
    d = json.loads((EX / "component_attribution.json").read_text())
    rows = d["rows"]
    W, H, pad = 820, 460, 70
    img = Image.new("RGB", (W, H), (13, 17, 23))
    dr = ImageDraw.Draw(img)
    f1, f2, f3 = font("DejaVuSans-Bold.ttf", 24), font("DejaVuSans.ttf", 15), font("DejaVuSansMono.ttf", 14)
    dr.text((W//2, 32), "Who carries the injected direction: attention vs MLP", font=f1, fill=(230,237,243), anchor="mm")
    dr.text((W//2, 60), "steered-minus-clean write along the vector, by layer", font=f2, fill=(139,148,158), anchor="mm")
    vals = []
    for r in rows:
        vals += [r["attn"]["delta_mean"], r["mlp"]["delta_mean"]]
    m = max(abs(v) for v in vals) * 1.2
    zero = H//2 + 20
    def Y(v): return zero - v/m*(H//2-60)
    dr.line([(pad, zero), (W-pad, zero)], fill=(80,88,98))
    n = len(rows); gw = (W-2*pad)/n; bw = gw*0.28
    for i, r in enumerate(rows):
        cx = pad + gw*(i+0.5)
        for k, (comp, col) in enumerate((("attn",(63,184,131)),("mlp",(216,130,43)))):
            v = r[comp]["delta_mean"]
            x = cx + (k-0.5)*bw*1.2
            dr.rectangle([x-bw/2, min(zero,Y(v)), x+bw/2, max(zero,Y(v))], fill=col)
            dr.text((x, Y(v)+(-14 if v>0 else 14)), f"{v:+.1f}", font=f3, fill=col, anchor="mm")
        dr.text((cx, H-30), f"L{r['layer']}", font=f2, fill=(200,208,216), anchor="mm")
    dr.text((pad, 92), "■ attn", font=f3, fill=(63,184,131))
    dr.text((pad+80, 92), "■ MLP", font=f3, fill=(216,130,43))
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
               rq1_dose_regime, transfer_story, depth_test):
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
    img = Image.new("RGB", (W, H), (13, 17, 23))
    dr = ImageDraw.Draw(img)
    f1, f2, f3 = font("DejaVuSans-Bold.ttf", 24), font("DejaVuSans.ttf", 15), font("DejaVuSansMono.ttf", 12)
    dr.text((W//2, 30), "Auto-calibration landscape (heretic-style objective)", font=f1, fill=(230,237,243), anchor="mm")
    dr.text((W//2, 58), "each dot = one trial · x layer · y scale · color = miss + λ·KL (green=good) · ring = fully effective", font=f2, fill=(139,148,158), anchor="mm")
    Ls = [r["layer"] for r in t]; Ss = [r["scale"] for r in t]
    lmin, lmax = min(Ls)-1, max(Ls)+1
    smin, smax = 0, max(Ss)*1.05
    scmax = max(r["score"] for r in t)
    def X(l): return pad + (l-lmin)/(lmax-lmin)*(W-2*pad)
    def Y(s): return H-pad - (s-smin)/(smax-smin)*(H-2*pad-30)
    for l in range(int(lmin)+1, int(lmax)+1, 2):
        dr.line([(X(l),pad),(X(l),H-pad)], fill=(28,34,42))
        dr.text((X(l), H-pad+16), f"L{l}", font=f3, fill=(139,148,158), anchor="mm")
    for s in range(0, int(smax)+1):
        dr.line([(pad,Y(s)),(W-pad,Y(s))], fill=(28,34,42))
        dr.text((pad-10, Y(s)), str(s), font=f3, fill=(139,148,158), anchor="rm")
    for r in t:
        x, y = X(r["layer"]), Y(r["scale"])
        # green (low score/good) -> red (high/bad)
        f = min(1.0, r["score"]/max(scmax,1e-9))
        col = (int(63+(216-63)*f), int(184+(80-184)*f), int(131+(60-131)*f))
        rr = 8
        dr.ellipse([x-rr,y-rr,x+rr,y+rr], fill=col)
        if r["miss"] == 0:
            dr.ellipse([x-rr-3,y-rr-3,x+rr+3,y+rr+3], outline=(230,237,243), width=2)
    best = d["best"]
    bx, by = X(best["layer"]), Y(best["scale"])
    dr.text((bx, by-20), "best", font=f3, fill=(230,237,243), anchor="mm")
    dr.text((W//2, H-22), "layer   (ringed dots = vector fully effective; deeper layers win on generic prompts)", font=f2, fill=(139,148,158), anchor="mm")
    dr.text((26, H//2), "scale", font=f2, fill=(139,148,158), anchor="mm")
    img.save(FIG / "calibration_landscape.png")
    return "calibration_landscape.png"


def _line_chart(title, sub, series, xlab, ylab, out, ymax=1.05, marks=()):
    """series: [(label, color, [(x,y),...])]; marks: [(x, y, text, color)]."""
    Image, ImageDraw, font = _pil()
    W, H, pad = 900, 520, 70
    img = Image.new("RGB", (W, H), (13, 17, 23))
    dr = ImageDraw.Draw(img)
    f1, f2, f3 = font("DejaVuSans-Bold.ttf", 24), font("DejaVuSans.ttf", 15), font("DejaVuSansMono.ttf", 13)
    dr.text((W//2, 32), title, font=f1, fill=(230,237,243), anchor="mm")
    dr.text((W//2, 60), sub, font=f2, fill=(139,148,158), anchor="mm")
    xs = [x for _,_,pts in series for x,_ in pts] + [m[0] for m in marks]
    xmax = max(xs)*1.06
    def X(v): return pad + v/xmax*(W-2*pad)
    def Y(v): return H-pad - v/ymax*(H-2*pad-50)
    for gy in (0, 0.25, 0.5, 0.75, 1.0):
        dr.line([(pad, Y(gy)), (W-pad, Y(gy))], fill=(30,37,46))
        dr.text((pad-8, Y(gy)), f"{gy:g}", font=f3, fill=(139,148,158), anchor="rm")
    for lbl_i, (label, col, pts) in enumerate(series):
        P = [(X(x), Y(y)) for x, y in pts]
        dr.line(P, fill=col, width=3)
        for (px,py),(x,_) in zip(P, pts):
            dr.ellipse([px-5,py-5,px+5,py+5], fill=col)
            dr.text((px, H-pad+16), f"{x:g}", font=f3, fill=(139,148,158), anchor="mm")
        dr.text((pad+8, 88+lbl_i*20), f"\u25a0 {label}", font=f3, fill=col)
    for x, y, text, col in marks:
        px, py = X(x), Y(y)
        dr.line([(px-8,py-8),(px+8,py+8)], fill=col, width=3)
        dr.line([(px-8,py+8),(px+8,py-8)], fill=col, width=3)
        dr.text((px, py-18), text, font=f3, fill=col, anchor="mm")
    dr.text((W//2, H-20), xlab, font=f2, fill=(139,148,158), anchor="mm")
    dr.text((26, H//2), ylab, font=f2, fill=(139,148,158), anchor="mm")
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
        [("decode-only", (63,184,131), dec), ("full-steer", (216,130,43), ful)],
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
        [("Qwen3-8B @ its argmax layer L15", (63,184,131), c8),
         ("Qwen3-4B @ shared layer L20", (216,130,43), c4)],
        "injection scale", "miss", "transfer_story.png",
        marks=[(3.0, x8["miss"], "4B opt on 8B: miss 0", (127,224,181)),
               (8.0, x4["miss"], "8B argmax on 4B: fails", (255,123,114))])


def depth_test():
    """Fractional depth vs raw index on a 28-layer model."""
    src = ROOT / "results/transfer_notasks_qwen25_7b.json"
    if not src.exists():
        return None
    rows = json.loads(src.read_text())["rows"]
    frac = [(r["scale"], r["miss"]) for r in rows if r["layer"] == 16][:6]
    marks = [(r["scale"], r["miss"], f"L{r['layer']}@3", (255,123,114))
             for r in rows if r["layer"] in (14, 18, 20) and r["scale"] == 3.0]
    return _line_chart(
        "Does the layer transfer as fractional depth? (28-layer model)",
        "Qwen2.5-7B · green: L16 = fractional-depth prediction (20/36 -> 16/28) · X: raw-index & neighbors @ scale 3",
        [("L16 (fractional-depth prediction)", (63,184,131), frac)],
        "injection scale", "miss", "depth_test.png", marks=marks)


if __name__ == "__main__":
    main()
