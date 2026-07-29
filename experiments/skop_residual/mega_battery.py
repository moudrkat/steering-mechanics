"""Mega rerouting battery: 64 prompts (24 task + 40 neutral) x 6 doses x 3 arms.
Prompt-outer, arms+doses inner -> clean-side cache paid once per prompt."""
import json, time, urllib.request, os

BASE = os.environ.get("BAT_BASE", "http://localhost:8010")
OUT = os.path.expanduser(os.environ.get("BAT_OUT", "~/mega_battery"))
LAYER = 20
SCALES = [1, 1.5, 2, 3, 5, 8]
ARMS = [("v3", "v_pref_no_task_checklist_v3"),
        ("vbar1", "v_pref_no_task_qwen_skopres_v1"),
        ("rand1536", "v_randctl_r1536_s1")]
ATTN_LAYERS = [20, 21, 22, 23, 24, 25, 26, 27]

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
TOPICS = [
 "rozdil mezi TCP a UDP protokolem", "planovani rodinneho rozpoctu",
 "the difference between compiled and interpreted languages",
 "priprava na pohovor do prace", "the water cycle in nature",
 "trideni odpadu v domacnosti", "how vaccines train the immune system",
 "zaklady pece o pokojove rostliny",
]
TEMPLATES = [
 "Vysvetli strucne a po bodech: {t}.",
 "Explain to a beginner, in two short paragraphs: {t}.",
 "Napis kratky odstavec na tema {t}.",
 "Summarize the key ideas of {t} in five bullet points.",
 "Vysvetli {t} tak, aby to pochopilo dite.",
]
NEUTRAL = [tpl.format(t=t) for t in TOPICS for tpl in TEMPLATES][:40]
PROMPTS = [("task", p) for p in TASK] + [("neutral", p) for p in NEUTRAL]
os.makedirs(OUT, exist_ok=True)

import subprocess, tempfile

def post(payload):
    last = None
    for attempt in range(3):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump(payload, tf); tmp = tf.name
        try:
            out = subprocess.run(["curl", "-s", "-m", "300", "-X", "POST",
                                  BASE + "/replay", "-H", "Content-Type: application/json",
                                  "--data-binary", "@" + tmp],
                                 capture_output=True, text=True, timeout=320)
            d = json.loads(out.stdout)
            if "attn_divergence" in d:
                os.unlink(tmp)
                return d
            last = out.stdout[:120]
        except Exception as e:
            last = str(e)[:120]
        try: os.unlink(tmp)
        except OSError: pass
        print(json.dumps({"retry": attempt + 1, "err": str(last)}), flush=True)
        time.sleep(3 * (attempt + 1))
    print(json.dumps({"SKIPPED_CELL": True, "err": str(last)}), flush=True)
    return None

t0 = time.time()
done = 0
for pi, (cls, p) in enumerate(PROMPTS):
    shamf = os.path.join(OUT, f"sham_p{pi}_s0.json")
    if not os.path.exists(shamf):
        r = post({"messages": [{"role": "user", "content": p}],
                  "steering": {"id": ARMS[0][1], "layer": LAYER, "scale": 1e-6, "decode_only": True},
                  "forced": True, "kl": True, "attn_divergence": True,
                  "attn_layers": ATTN_LAYERS, "max_tokens": 48})
        if r is not None: json.dump({"prompt_idx": pi, "prompt_class": cls, "arm": "sham", "scale": 0,
                   "attn_divergence": r["attn_divergence"], "kl": r.get("kl")}, open(shamf, "w"))
    for arm, direction in ARMS:
        for s in SCALES:
            name = f"{arm}_p{pi}_s{str(s).replace('.','p')}.json"
            fp = os.path.join(OUT, name)
            if os.path.exists(fp):
                done += 1; continue
            r = post({"messages": [{"role": "user", "content": p}],
                      "steering": {"id": direction, "layer": LAYER, "scale": s, "decode_only": True},
                      "forced": True, "kl": True, "attn_divergence": True,
                      "attn_layers": ATTN_LAYERS, "max_tokens": 48})
            if r is None:
                continue
            slim = {"prompt_idx": pi, "prompt_class": cls, "arm": arm,
                    "scale": s, "attn_divergence": r["attn_divergence"],
                    "kl": r.get("kl")}
            with open(fp, "w") as f:
                json.dump(slim, f)
            done += 1
            if done % 24 == 0:
                print(json.dumps({"done": done, "of": len(PROMPTS)*len(ARMS)*len(SCALES),
                                  "t_min": round((time.time()-t0)/60, 1)}), flush=True)
print("MEGA BATTERY DONE", done, "cells", round((time.time()-t0)/60, 1), "min", flush=True)
