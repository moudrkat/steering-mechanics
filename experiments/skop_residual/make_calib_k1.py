"""Build the chain-K1 calibration prompt set (CHAINK_PREREG.md): 400 prompts,
100 each GSM8K / Alpaca / PIQA / NarrativeQA (SKOP's mixture at 1/10 scale).
Frozen fallback if datasets are unreachable: 64 combinatorial CZ/EN generic +
40 templated neutral + 296 ARC-train questions, recorded in the output.

env: CK_OUT (~/calib_k1.json), CK_ARC (fallback only, arc300.json)
"""
import json, os

OUT = os.path.expanduser(os.environ.get("CK_OUT", "~/calib_k1.json"))
prompts, source = [], "gsm8k/alpaca/piqa/narrativeqa x100"
try:
    from datasets import load_dataset
    gs = load_dataset("openai/gsm8k", "main", split="train[:100]")
    prompts += [r["question"] for r in gs]
    al = load_dataset("tatsu-lab/alpaca", split="train[:200]")
    prompts += [r["instruction"] + (("\n" + r["input"]) if r["input"] else "")
                for r in al if not r["input"] or len(r["input"]) < 400][:100]
    pq = load_dataset("ybisk/piqa", split="train[:100]", trust_remote_code=True)
    prompts += [f"Goal: {r['goal']}\nWhich is better?\nA) {r['sol1']}\nB) {r['sol2']}"
                for r in pq]
    nq = load_dataset("deepmind/narrativeqa", split="train[:100]")
    prompts += [r["question"]["text"] + "\nContext: " +
                r["document"]["summary"]["text"][:800] for r in nq]
except Exception as e:
    source = f"FALLBACK (datasets unreachable: {type(e).__name__})"
    ADJ = ["kratky", "detailni", "jednoduchy", "strukturovany"]
    SUBJ = ["plan vyletu do Alp", "recept na nedelni obed",
            "shrnuti knihy, kterou jsem cetla", "navrh treninkoveho planu",
            "a study schedule for finals week", "an outline for a blog post",
            "a packing list for a rainy weekend", "a summary of a team meeting"]
    prompts += [f"Napis {a} text: {s}." for a in ADJ for s in SUBJ][:64]
    TOPICS = ["rozdil mezi TCP a UDP protokolem", "planovani rodinneho rozpoctu",
              "the difference between compiled and interpreted languages",
              "priprava na pohovor do prace", "the water cycle in nature",
              "trideni odpadu v domacnosti", "how vaccines train the immune system",
              "zaklady pece o pokojove rostliny"]
    TPL = ["Vysvetli strucne a po bodech: {t}.",
           "Explain to a beginner, in two short paragraphs: {t}.",
           "Napis kratky odstavec na tema {t}.",
           "Summarize the key ideas of {t} in five bullet points.",
           "Vysvetli {t} tak, aby to pochopilo dite."]
    prompts += [tpl.format(t=t) for t in TOPICS for tpl in TPL][:40]
    arc = json.load(open(os.path.expanduser(os.environ.get("CK_ARC", "~/arc300.json"))))
    prompts += [it["q"] for it in arc][:296]

json.dump({"source": source, "prompts": prompts}, open(OUT, "w"), ensure_ascii=False)
print(json.dumps({"n": len(prompts), "source": source, "out": OUT}))
