"""Secondary measure, run AFTER generation (separate process: the 3.8B judge and the
attacked model do not fit in RAM together on the CPU box). Reads <run>.jsonl, adds
judge_p_yes to every scored row, writes <run>_judge.jsonl and appends a per-cell summary
to <run>_judge.log. Resumable: cells already judged are skipped."""
import json, os, statistics as st, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.judge import JUDGE_MODEL, judge_answers_question  # noqa: E402
from evaluation.topic_data import PRODUCT_LABEL  # noqa: E402

run = sys.argv[1]
src, dst, logp = run + ".jsonl", run + "_judge.jsonl", run + "_judge.log"
done = {json.loads(l)["cell"] for l in open(dst)} if os.path.exists(dst) else set()
log = open(logp, "a")
log.write(f"\n=== judge {JUDGE_MODEL} on {src} | {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
for line in open(src):
    r = json.loads(line)
    if "rows" not in r or r["cell"] in done:
        continue
    ps = []
    for row in r["rows"]:
        name = next(n for n in PRODUCT_LABEL if f"TEST_{n}" in row["question"])
        row["judge_p_yes"] = judge_answers_question(row["question"], row["answer"], f"TEST_{name}",
                                                    PRODUCT_LABEL[name])
        ps.append(row["judge_p_yes"])
    r["summary"]["judge_p_yes"] = st.mean(ps)
    with open(dst, "a") as f:
        f.write(json.dumps(r) + "\n")
    msg = f"{r['cell']}: judge P(yes) mean {st.mean(ps):.2f}  per question {[round(p, 2) for p in ps]}"
    print(msg, flush=True)
    log.write(msg + "\n"); log.flush()
