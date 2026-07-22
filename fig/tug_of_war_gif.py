"""Animate the L21 attention-head tug-of-war over the steering direction.

Heads that write WITH the vector (green) pull the knot right; heads that
write AGAINST it (orange, self-repair) pull left. The knot settles at the
net — the vector wins by a thread. Data: results/head_attribution.json.

    python3 fig/tug_of_war_gif.py   # -> fig/tug_of_war.gif
"""
import json
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
D = json.loads((ROOT / "results/head_attribution.json").read_text())
deltas = [(i, dv) for i, dv in D["deltas"]]
withv = sorted([(i, dv) for i, dv in deltas if dv > 0.02], key=lambda x: -x[1])
against = sorted([(i, dv) for i, dv in deltas if dv < -0.02], key=lambda x: x[1])
net = sum(dv for _, dv in deltas)

W, H, N = 1200, 620, 46
FNT = "/usr/share/fonts/truetype/dejavu/"
f_big = ImageFont.truetype(FNT + "DejaVuSans-Bold.ttf", 34)
f_mid = ImageFont.truetype(FNT + "DejaVuSans.ttf", 22)
f_mono = ImageFont.truetype(FNT + "DejaVuSansMono.ttf", 15)
f_small = ImageFont.truetype(FNT + "DejaVuSansMono.ttf", 13)
BG, INK, MUT = (13, 17, 23), (230, 237, 243), (139, 148, 158)
GREEN, ORANGE, PURPLE = (63, 184, 131), (216, 130, 43), (180, 142, 224)

cx, cy = W // 2, 340
maxside = max(sum(dv for _, dv in withv), -sum(dv for _, dv in against))
scale = 500 / maxside  # fit both ropes inside the canvas


def ease(t):
    return 1 - (1 - t) ** 3


def frame(prog):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, 46), "Layer 21: 32 attention heads pull the steering vector",
           font=f_big, fill=INK, anchor="mm")
    d.text((W // 2, 90), "both ways at once — green pulls WITH it, orange resists",
           font=f_mid, fill=MUT, anchor="mm")

    # center zero line
    d.line([(cx, 150), (cx, 470)], fill=(45, 52, 62), width=2)
    d.text((cx, 138), "0", font=f_small, fill=MUT, anchor="mm")

    # rope
    knot = cx + int(net * scale * ease(prog))
    d.line([(180, cy), (W - 180, cy)], fill=(60, 68, 78), width=3)

    # against (left, orange) stacked from center leftwards
    x = cx
    for i, dv in against:
        w = -dv * scale * ease(prog)
        if w >= 2:
            d.rectangle([x - w, cy - 16, x - 2, cy + 16], fill=ORANGE)
            if -dv > 0.4:
                d.text((x - w / 2, cy), f"h{i}", font=f_small, fill=(20, 20, 20), anchor="mm")
        x -= w
    # with (right, green)
    x = cx
    for i, dv in withv:
        w = dv * scale * ease(prog)
        if w >= 2:
            d.rectangle([x + 2, cy - 16, x + w, cy + 16], fill=GREEN)
            if dv > 0.4:
                d.text((x + w / 2, cy), f"h{i}", font=f_small, fill=(15, 30, 22), anchor="mm")
        x += w

    # knot marker
    d.ellipse([knot - 11, cy - 11, knot + 11, cy + 11], fill=PURPLE, outline=BG, width=3)

    # side labels
    d.text((210, cy - 40), "RESIST", font=f_mono, fill=ORANGE, anchor="lm")
    d.text((W - 210, cy - 40), "AMPLIFY", font=f_mono, fill=GREEN, anchor="rm")

    # net readout appears near the end
    if prog > 0.75:
        a = min(1.0, (prog - 0.75) / 0.25)
        col = tuple(int(m + (p - m) * a) for m, p in zip(BG, PURPLE))
        d.text((W // 2, 520), f"net pull  {net:+.2f}", font=f_big, fill=col, anchor="mm")
        col2 = tuple(int(m + (p - m) * a) for m, p in zip(BG, MUT))
        d.text((W // 2, 562),
               "resistance is everywhere — the vector wins by a thread",
               font=f_mid, fill=col2, anchor="mm")
    return img


def main():
    frames = []
    for k in range(N):
        frames.append(frame(k / (N - 1)))
    for _ in range(18):          # hold at the end
        frames.append(frames[-1])
    with tempfile.TemporaryDirectory() as td:
        for j, im in enumerate(frames):
            im.save(f"{td}/f{j:03d}.png")
        out = ROOT / "fig/tug_of_war.gif"
        subprocess.run(["ffmpeg", "-y", "-framerate", "24", "-i", f"{td}/f%03d.png",
                        "-vf", "scale=1200:-1:flags=lanczos,split[s0][s1];"
                               "[s0]palettegen[p];[s1][p]paletteuse",
                        "-loop", "0", str(out)], check=True,
                       capture_output=True)
        print("->", out, f"({out.stat().st_size // 1024} KB, net {net:+.2f})")


if __name__ == "__main__":
    main()
