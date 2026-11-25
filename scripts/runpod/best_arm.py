"""Phase 2 best arm, rule fixed before the A-D data (AGENT_TASK.md, Phase 2 step 5):
among A-D (seed 0), the arm with the highest mean SEPARATE whose mean preservation is >= 0.8;
ties or none passing -> the highest SEPARATE overall (said so in the output).
Means are over the 4 training pages of the per-page mean over greedy + 5 samples, exactly as
in runs/topic_phase2_table.log, but at full precision. Prints the arm on the last line."""
import json
import statistics as st

rows = [json.loads(x) for x in open("runs/topic_phase2/scores.jsonl")]
res = {}
for arm in "ABCD":
    rs = [r for r in rows if r["cond"] == f"{arm}_s0"]
    if len(rs) != 4:
        raise SystemExit(f"arm {arm}: {len(rs)} pages scored, need 4")
    m = lambda k: st.mean(st.mean(float(x[k]) for x in r["rows"]) for r in rs)  # noqa: E731
    res[arm] = (m("separate_frac"), m("preservation_norm"))
    print(f"{arm}: SEPARATE {res[arm][0]:.4f} preservation {res[arm][1]:.4f}")
passing = {a: v for a, v in res.items() if v[1] >= 0.8}
top = max(v[0] for v in passing.values()) if passing else None
winners = [a for a, v in passing.items() if v[0] == top] if passing else []
if len(winners) == 1:
    best = winners[0]
    print(f"rule: {best} has the highest SEPARATE among arms with preservation >= 0.8")
else:
    best = max(res, key=lambda a: res[a][0])
    why = "no arm reaches preservation 0.8" if not passing else f"tie among {winners}"
    print(f"rule fallback ({why}): {best} has the highest SEPARATE overall")
print(best)
