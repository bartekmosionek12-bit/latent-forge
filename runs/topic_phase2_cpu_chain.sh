#!/bin/bash
# Phase 2, CPU part: baselines + activation arm E and its all-positions ablation, then measures.
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate
python scripts/topic_phase2.py baselines
python scripts/topic_phase2.py attack --arm E
python scripts/topic_phase2.py attack --arm E_all
python scripts/topic_phase2.py score
python scripts/topic_phase2.py judge
python scripts/topic_phase2.py table
