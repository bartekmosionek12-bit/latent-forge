# Task for Claude Code on the RunPod pod: topic injection Phase 2, arms A-D

You are running on a disposable RunPod GPU pod, started by Bartek (the repo owner), with
permission prompts skipped. Work only inside this repository. Read the whole file before
doing anything. Write the final report in Polish.

## Context (read, do not re-derive)

- Research line: "topic injection via activation-targeted GCG", see the README section
  "Topic injection via activation-targeted GCG" and its Phase 2 subsections. Model:
  Qwen2.5-0.5B-Instruct, local. Neutral topic (cooking) on fictional TEST_ product pages.
- The CPU part of Phase 2 (baselines, arm E, ablation E_all) is done and committed in
  `runs/topic_phase2/`. `calibration.json` there is shared: do not regenerate it.
- Your job is ONLY the GPU part: arms A-D, seed 0, then the measures. Everything is already
  implemented in `scripts/runpod/phase2_gpu.sh` and `scripts/topic_phase2.py`.

## Hard rules (from Bartek, not negotiable)

1. No API keys, tokens or secrets in the repo, in logs, or in your output. Never print,
   copy or search for credentials (`~/.git-credentials`, env vars, `gh auth token`, ...).
   No key or HF token is needed: every model is public and not gated.
2. Do not change the research design: objectives, arms, budgets, measures, thresholds,
   detectors, pages, targets. Do not add any refusal-suppression term, refusal direction or
   similar, not even as a baseline. Do not upgrade packages (transformers stays 4.47.1).
3. Only local models on this pod. No requests to third-party apps or APIs.
4. Do NOT run anything beyond this task: no extra seeds, no Phase 3+, no new experiments.
5. If a script crashes: you may make the smallest fix that makes it run as intended (for
   example a device or path bug), commit it separately with a message explaining why, and
   mention it in the report. If the fix would change what is measured, STOP and report.
6. Git: commit only on branch `claude/project-thread-mnmiam`. Never force-push or rewrite
   history. If a push is rejected, `git pull --no-rebase` and push again. End every commit
   message with the line `Co-Authored-By: Claude <noreply@anthropic.com>`.
7. Long commands: start them in the background with `nohup ... > file 2>&1 &` and check the
   output file every few minutes; do not hold a single tool call for hours.

## Steps

1. `git checkout claude/project-thread-mnmiam && git pull --no-rebase`.
2. If `.venv` does not exist: `bash scripts/runpod/setup.sh` (about 15 min). It must end with
   `cuda True` and `setup OK`. If no GPU is visible, STOP and report.
3. Smoke test: `bash scripts/runpod/phase2_gpu.sh smoke 2>&1 | tee runs/topic_phase2/gpu_smoke.txt`.
   It prints `arm A: X s/step -> Y h` and the same for arm D.
4. Gate: estimated total = 3 x (arm A hours) + (arm D hours) + 0.5 h for generation.
   If the estimate is above **8 h**, STOP here: commit `gpu_smoke.txt`, push, write the
   report with the numbers, and end. Otherwise continue.
5. Full run: `nohup bash scripts/runpod/phase2_gpu.sh run > runs/topic_phase2/gpu_run.out 2>&1 &`.
   Every step is logged to `runs/topic_phase2/<arm>_s0_<page>.{jsonl,log}`; if it is
   interrupted, restarting the same command resumes from the logs. Every ~30 min commit and
   push the logs so far (`git add -f runs/topic_phase2/`), so a lost pod loses little.
6. When the run has finished for all 4 arms x 4 pages, check the generations exist in
   `runs/topic_phase2/gens.jsonl` for conds `A_s0`, `B_s0`, `C_s0`, `D_s0`.
7. Measures: `nohup bash scripts/runpod/phase2_gpu.sh measure > runs/topic_phase2/gpu_measure.out 2>&1 &`
   (score, local judge, table). Wait for it to finish.
8. Commit and push: `git add -f runs/topic_phase2/ runs/topic_phase2_table.log`.
9. Write `runs/topic_phase2/GPU_REPORT.md` (Polish, short): GPU name, smoke s/step, wall
   time of run and measure, any crash and fix, and the table rows for A-D copied from
   `runs/topic_phase2_table.log`. Do not interpret the research questions; the analysis is
   done in the main thread. Commit and push it, then stop and tell Bartek it is done.
