#!/bin/bash
cd /home/user/latent-forge
P=/home/user/.venv/bin/python
$P scripts/region_gcg_demo.py --readable-proposals --readability-weight 0 --out runs/region_gcg_phase4_ctrl_w0.json > runs/region_gcg_phase4_ctrl_w0.log 2>&1
for w in 0.01 0.03 0.1; do
  $P scripts/region_gcg_demo.py --readable-proposals --readability-weight $w --out runs/region_gcg_phase4_w$w.json > runs/region_gcg_phase4_w$w.log 2>&1
done
echo ALLDONE > runs/phase4_done.flag
