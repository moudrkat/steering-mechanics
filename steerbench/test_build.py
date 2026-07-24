"""SteerBench converter — self-describing + legacy + safety folding, no I/O deps
beyond a tmp results dir. Run: python3 steerbench/test_build.py"""
import json, sys, tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))
import build

def run(tmp):
    build.RESULTS = tmp
    build.OUT = tmp / "entries.jsonl"
    (BENCH/"tasks.json")  # registry read from real file
    build.main()
    return [json.loads(l) for l in build.OUT.read_text().splitlines() if l.strip()]

def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # a self-describing result for a NON-no-tasks vector + method
        (tmp/"bench_sycophancy_repeng.json").write_text(json.dumps({
            "model":"qwen3-4b","n_layers":36,"task":"sycophancy","method":"repeng",
            "rows":[{"layer":18,"scale":2.0,"miss":0.1,"anti_steered":0,"incoherent":0,"kl":0.5}]}))
        # a legacy campaign result (no task/method → inferred as no-tasks/pref-caa)
        (tmp/"campaign_llama31-8b_en.json").write_text(json.dumps({
            "model":"llama31-8b","n_layers":32,
            "rows":[{"layer":16,"scale":3.0,"miss":0.03,"anti_steered":0,"incoherent":0,"kl":1.7}]}))
        # a safety result to fold in
        (tmp/"em_llama31-8b.json").write_text(json.dumps({
            "model":"llama31-8b","layer":16,"scale":3.0,
            "conditions":{"steered":{"harmful_compliance":0.0,"false_refusal_rate":0.0}}}))
        e = run(tmp)
        tasks = {x["task"] for x in e}; methods = {x["method"] for x in e}
        assert tasks == {"sycophancy","no-tasks"}, tasks           # behaviour-agnostic
        assert methods == {"repeng","pref-caa"}, methods           # method-agnostic
        syc = next(x for x in e if x["task"]=="sycophancy")
        assert syc["method"]=="repeng" and syc["frac_depth"]==0.5  # self-described flows in
        lla = next(x for x in e if x["task"]=="no-tasks")
        assert lla["safety_harmful_compliance"]==0.0               # safety folded in
        print(f"PASS — {len(e)} entries, tasks={tasks}, methods={methods}, safety folded")

if __name__ == "__main__":
    main()
