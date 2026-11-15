#!/bin/bash
cd /home/user/latent-forge
for w in 0.01 0.03 0.1; do
  /home/user/.venv/bin/python scripts/region_gcg_demo.py --readable-proposals --readability-weight $w --init-text " the plan to repair the old bridge near the public library and to add more street lights in the park" --out runs/region_gcg_phase4b_w$w.json > runs/region_gcg_phase4b_w$w.log 2>&1
done
echo ALLDONE > runs/phase4b_done.flag
