#!/bin/bash
# Topic-injection Phase 2 on a RunPod GPU. Run from the repo root, after scripts/runpod/setup.sh.
#   smoke  : FIRST STEP. 5 optimisation steps of arm A and of arm D on one page, no generation;
#            prints seconds/step and the extrapolated time of the full run. Nothing else runs.
#   run    : arms A, B, C, D x 4 pages, seed 0, 200 steps each, then generation (greedy + 5 samples)
#   seeds  : extra seeds 1 and 2 for ONE arm (the best of the first run; pass it as $2) and for
#            the explicit-instruction baseline (sampling seeds 1 and 2)
#   measure: pre-registered measures, local judge, table (runs/topic_phase2_table.log)
# Every optimisation step is appended to runs/topic_phase2/<arm>_s<seed>_<page>.{jsonl,log};
# an interrupted run resumes from the last logged step when restarted.
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv/bin/activate
P=scripts/topic_phase2.py
case "${1:-}" in
  smoke)
    for arm in A D; do
      rm -f runs/topic_phase2/${arm}_s0_steps5_Zephyr.*
      python $P attack --arm $arm --products Zephyr --steps 5 --score-chunk 128 --no-generate
      python - "$arm" <<'PY'
import json, statistics as st, sys
arm = sys.argv[1]
r = [json.loads(x) for x in open(f"runs/topic_phase2/{arm}_s0_steps5_Zephyr.jsonl")][1:]
s = st.mean(x["seconds"] for x in r)
print(f"arm {arm}: {s:.1f} s/step -> 200 steps x 4 pages = {s*800/3600:.1f} h (plus ~2 min generation per page)")
PY
    done ;;
  run)
    for arm in A B C D; do python $P attack --arm $arm --score-chunk 128; done ;;
  seeds)
    best="${2:?usage: phase2_gpu.sh seeds <best arm, e.g. C>}"
    for s in 1 2; do
      python $P attack --arm "$best" --seed $s --score-chunk 128
      python $P baselines --conds explicit_instruction --seed $s
    done ;;
  measure)
    python $P score && python $P judge && python $P table ;;
  *) echo "usage: $0 smoke|run|seeds <arm>|measure"; exit 1 ;;
esac
