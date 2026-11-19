# Topic injection Phase 2 on RunPod (arms A-D)

Nothing in this folder needs a key or token, and none should be put in the repo. All
models are public and not gated. Arms A-D need a GPU: on the 4-core CPU box one step costs
47-328 s (see `scripts/bench/phase2_step_cost.py`), so the 16 runs would take about 150 h.

## Pod

- **Template:** RunPod PyTorch (any recent CUDA 12.x image; Ubuntu 22.04). `setup.sh`
  installs its own Python 3.13 venv with the pinned stack, so the template's torch is not
  used.
- **GPU:** one 24 GB card (RTX A5000, RTX 4090 or L4). The model is Qwen2.5-0.5B in fp32;
  peak use is a few GB, plus Phi-3.5-mini (bf16, ~8 GB) for the judge pass.
- **Disk:** container disk 40 GB. A volume is optional; if used, 30 GB mounted at
  `/workspace` with the repo cloned there keeps results across restarts.

## Steps

1. Clone the repo on the pod with your own GitHub credentials (private repo; do not store a
   token in the repo), then check out `claude/project-thread-mnmiam`.
2. `bash scripts/runpod/setup.sh`: venv, pinned packages, model downloads (~15 min).
3. **First: `bash scripts/runpod/phase2_gpu.sh smoke`.** This runs 5 steps of arm A and
   5 of arm D (D adds the detector) on one page and prints s/step and the extrapolated
   time. Stop there and report the numbers before the full run.
4. `bash scripts/runpod/phase2_gpu.sh run`: arms A-D x 4 pages, seed 0, then generations.
5. `bash scripts/runpod/phase2_gpu.sh measure`: measures, judge, table.
6. After the first results: `bash scripts/runpod/phase2_gpu.sh seeds <best arm>`, then
   `measure` again.
7. Bring `runs/topic_phase2/` and `runs/topic_phase2_table.log` back, either by committing
   with `git add -f` and pushing with your credentials, or with `runpodctl send`.

The CPU part (baselines, activation arm E and its ablation) runs on the CPU box and is
already committed in `runs/topic_phase2/`. The pod appends to the same files.
`calibration.json` (target first sentences and band centre) comes from the CPU run, so
both machines optimise the same targets.
