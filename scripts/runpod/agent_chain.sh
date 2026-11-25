#!/bin/bash
# Autonomous chain after Phase 2 run+measure (AGENT_TASK.md): Phase 2 seeds -> Phase 3 -> 4a -> 5.
# Each optimisation phase starts with a 5-step timing; the chain stops (exit 3) if the phase
# is estimated at more than 6 h (rule 8). Every step is resumable: re-running skips what is done.
#   usage: bash scripts/runpod/agent_chain.sh [from-stage]   stages: seeds p3 p4a p5
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_ENABLE_HF_TRANSFER=0
. .venv/bin/activate
from="${1:-seeds}"
stamp() { echo "=== $(date -u +%FT%TZ) $*"; }
mean_s() {  # mean seconds/step of steps 1.. in a jsonl log
  python -c "import json,sys,statistics as s; r=[json.loads(x) for x in open(sys.argv[1])][1:]; print(s.mean(x['seconds'] for x in r))" "$1"; }
gate() {   # gate <label> <hours>
  python -c "import sys; h=float(sys.argv[2]); print(f'ESTIMATE {sys.argv[1]}: {h:.2f} h'); sys.exit(3 if h > 6 else 0)" "$1" "$2"; }

python scripts/runpod/best_arm.py | tee runs/topic_phase2/best_arm.txt
BEST=$(tail -n1 runs/topic_phase2/best_arm.txt)

case "$from" in seeds)
  stamp "Phase 2 seeds for $BEST"
  bash scripts/runpod/phase2_gpu.sh seeds "$BEST"
  bash scripts/runpod/phase2_gpu.sh measure
  stamp "Phase 2 done" ;&
p3)
  P3="python scripts/topic_phase3.py"
  mkdir -p runs/topic_phase3
  stamp "Phase 3 smoke (arm $BEST)"
  rm -f runs/topic_phase3/smoke/U${BEST}_cooking_s0_steps5.*
  $P3 attack --arm "$BEST" --topic cooking --steps 5 --no-generate --out runs/topic_phase3/smoke
  s=$(mean_s runs/topic_phase3/smoke/U${BEST}_cooking_s0_steps5.jsonl)
  gate "Phase 3 (4 runs x 300 steps at $s s/step + 1 h generation/measures)" \
       "$(python -c "print($s*1200/3600 + 1.0)")" | tee runs/topic_phase3/estimate.txt
  for seed in 0 1; do for topic in cooking gardening; do
    stamp "Phase 3 $topic seed $seed"; $P3 attack --arm "$BEST" --topic $topic --seed $seed; done; done
  $P3 baselines --seed 0; $P3 baselines --seed 1
  $P3 score && $P3 judge && $P3 table
  stamp "Phase 3 done" ;&
p4a)
  P3="python scripts/topic_phase3.py"
  for m in Qwen/Qwen2.5-1.5B-Instruct HuggingFaceTB/SmolLM2-1.7B-Instruct; do
    o=runs/topic_phase4a/$(basename $m)
    stamp "Phase 4a $m"
    $P3 transfer --model $m --out $o && $P3 score --out $o && $P3 judge --out $o && $P3 table --out $o
  done
  stamp "Phase 4a done" ;&
p5)
  P5="python scripts/topic_phase5.py"
  mkdir -p runs/topic_phase5
  $P5 prepare
  stamp "Phase 5 smoke (fp32)"
  rm -f runs/topic_phase5/P5_kl1_s0_steps5_Zephyr.*
  $P5 attack --l1 1 --products Zephyr --steps 5 --no-generate
  s=$(mean_s runs/topic_phase5/P5_kl1_s0_steps5_Zephyr.jsonl)
  TF=""
  if ! gate "Phase 5 fp32 (5 points x 4 pages x 200 steps at $s s/step + 1 h generation/measures)" \
       "$(python -c "print($s*4000/3600 + 1.0)")" | tee runs/topic_phase5/estimate.txt; then
    stamp "Phase 5 smoke (TF32 fallback)"
    mv runs/topic_phase5/P5_kl1_s0_steps5_Zephyr.jsonl runs/topic_phase5/P5_kl1_s0_steps5_Zephyr_fp32.jsonl
    mv runs/topic_phase5/P5_kl1_s0_steps5_Zephyr.log runs/topic_phase5/P5_kl1_s0_steps5_Zephyr_fp32.log
    $P5 attack --l1 1 --products Zephyr --steps 5 --no-generate --tf32
    s=$(mean_s runs/topic_phase5/P5_kl1_s0_steps5_Zephyr.jsonl)
    gate "Phase 5 TF32 (5 points x 4 pages x 200 steps at $s s/step + 1 h generation/measures)" \
         "$(python -c "print($s*4000/3600 + 1.0)")" | tee -a runs/topic_phase5/estimate.txt
    TF="--tf32"
  fi
  for l1 in 0 0.1 0.3 1 3; do stamp "Phase 5 l1=$l1 $TF"; $P5 attack --l1 $l1 $TF; done
  $P5 score && $P5 judge && $P5 table && $P5 pareto --best "$BEST"
  stamp "Phase 5 done" ;;
esac
stamp "chain done"
