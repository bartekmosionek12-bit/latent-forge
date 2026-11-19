"""Post-hoc analysis of a Phase 1 run (added AFTER looking at the generations, labelled as
such in the README): splits topical sentences into
  fused    : the cooking sentence talks about the PRODUCT (names it or its category noun),
             i.e. the product itself is reinterpreted as a kitchen object
  separate : a cooking sentence that is not about the product (a genuine aside / tip)
Reads <run>.jsonl, writes <run>_fusion.log. No generation, no model forward except the
multilingual encoder already used in Phase 1 (topic_sentences are stored per row)."""
import json, re, statistics as st, sys

run = sys.argv[1] if len(sys.argv) > 1 else "runs/topic_phase1_qwen05b"
CAT = {"Zephyr": "router", "Aurora": "laptop", "Nimbus": "headphone", "Vega": "chair",
       "Orion": "monitor", "Borealis": "scooter", "Quartz": "watch", "Helix": "ssd|drive"}
d = {}
for l in open(run + ".jsonl"):
    r = json.loads(l); d[r["cell"]] = r
out = open(run + "_fusion.log", "w")
def p(s=""):
    print(s); out.write(s + "\n")
p("cell | topic_sent_frac | fused_frac | separate_frac | preservation_norm | class")
for c, r in d.items():
    if "rows" not in r:
        continue
    fu, se = [], []
    for row in r["rows"]:
        name = next(n for n in CAT if n in row["question"])
        pat = re.compile(rf"{name}|{CAT[name]}|\bit\b|\bit's\b", re.I)
        n = max(1, round(row["topic_sentence_frac"] and len(row["topic_sentences"]) / row["topic_sentence_frac"] or 1))
        f = sum(bool(pat.search(s)) for s in row["topic_sentences"])
        fu.append(f / n); se.append((len(row["topic_sentences"]) - f) / n)
    s = r["summary"]
    p(f"{c} | {s['topic_sentence_frac']:.2f} | {st.mean(fu):.2f} | {st.mean(se):.2f} | "
      f"{s['preservation_norm']:.2f} | {r.get('class', '')}")
