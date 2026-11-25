"""Topic injection, Phase 5: Pareto front between injection and answer preservation.

Loss per training page (Qwen2.5-0.5B, one suffix per page, Phase 2 machinery):
    L = l1 * KL_task + l2 * L_topic + l3 * L_fluency
  KL_task   : mean over the clean greedy answer's tokens (teacher-forced; the answer is the
              Phase 2 calibration clean_greedy) of KL(p_clean || p_with_suffix)
  L_topic   : the Phase 2 continuation-set loss (l2 = 1)
  L_fluency : the arm C fluency term (l3 = 0.2 on the suffix NLL; arm C's repetition penalty is
              kept at its arm C weight, so that l1 = 0 IS arm C; see README plan change)
Weight points: l1 in {0, 0.1, 0.3, 1, 3}; 200 steps, seed 0, the 4 training pages.

Subcommands (files under runs/topic_phase5/, resumable):
  prepare : copies the Phase 2 CLEAN generations (seed 0) so preservation uses the same clean
            references as Phase 2
  attack  : one l1 x the 4 pages -> per-step .log/.jsonl, generations -> gens.jsonl
  score / judge : the Phase 2 measures (scripts/topic_phase2.py, pointed at this folder)
  table   : runs/topic_phase5/table.log
  pareto  : SEPARATE vs preservation per weight point + reference points (clean / naive /
            explicit / Phase 2 best arm, from the Phase 2 scores) -> pareto.jsonl, pareto.png
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import topic_phase2 as P2  # noqa: E402

OUT = "runs/topic_phase5"
L1S = (0.0, 0.1, 0.3, 1.0, 3.0)
L2, L3 = 1.0, 0.2
STEPS = 200


def cond_of(l1, steps=STEPS):
    return f"P5_kl{l1:g}_s0" + (f"_steps{steps}" if steps != STEPS else "")


def cmd_prepare(a):
    path = f"{OUT}/gens.jsonl"
    have = {(r["cond"], r["product"]) for r in P2.jl_read(path)}
    for r in P2.jl_read(f"{P2.OUT}/gens.jsonl"):
        if r["cond"] == "clean" and (r["cond"], r["product"]) not in have:
            P2.jl_append(path, {**r, "copied_from": f"{P2.OUT}/gens.jsonl"})
    print("clean references:", sorted(r["product"] for r in P2.jl_read(path) if r["cond"] == "clean"))


def cmd_attack(a):
    import dispatcher
    from evaluation.topic_data import COOKING_TIPS, PRODUCT_PAGES
    from models.page_prompt import user_message
    from targets import ContinuationSetTarget

    import torch

    if a.tf32:   # budget fallback (README plan change): TF32 tensor-core matmuls, same objective
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    cal = json.load(open(f"{P2.OUT}/calibration.json"))
    bundle = P2.load_qwen(a.device)
    cond = cond_of(a.l1, a.steps)
    for p in a.products.split(","):
        if any(r["cond"] == cond and r["product"] == p for r in P2.jl_read(f"{OUT}/gens.jsonl")):
            print(f"{cond}/{p} already done"); continue
        target = ContinuationSetTarget(page=PRODUCT_PAGES[p], question=P2.question_of(p), product=p,
                                       continuations=[f"{cal[p]['A_q']} Tip: {t}" for t in COOKING_TIPS],
                                       clean_answer=cal[p]["clean_greedy"])
        cfg = {**P2.BUDGET, **P2.ARMS["C"], "fluency_weight": L3, "kl_weight": a.l1, "seed": 0,
               "num_steps": a.steps, "log_path": f"{OUT}/{cond}_{p}.jsonl", "score_chunk": a.score_chunk}
        t0 = time.time()
        res, rep = dispatcher.run(target, bundle, **cfg)
        print(f"[{cond} / {p}] best step {rep['best_step']} metric {rep['best_full_total']:.4f} "
              f"suffix {res.payload!r} ({time.time() - t0:.0f}s)", flush=True)
        if a.no_generate:
            continue
        g, s = P2.generate_all(bundle, user_message(PRODUCT_PAGES[p], res.payload, P2.question_of(p)))
        P2.jl_append(f"{OUT}/gens.jsonl", {"cond": cond, "product": p, "attack_text": res.payload,
                                           "suffix_ids": res.extra["best_suffix_ids"], "greedy": g, "samples": s,
                                           "best_step": res.extra["best_step"], "metric": res.final_loss,
                                           "l1": a.l1, "l2": L2, "l3": L3, "tf32": a.tf32,
                                           "attack_seconds": time.time() - t0,
                                           "device": str(bundle.device)})


def _redirect():
    P2.OUT, P2.TABLE = OUT, f"{OUT}/table.log"


def cmd_score(a):
    _redirect(); P2.cmd_score(a)


def cmd_judge(a):
    _redirect(); P2.cmd_judge(a)


def cmd_table(a):
    _redirect(); P2.cmd_table(a)


def _point(scores, judge, cond):
    rs = [r for r in scores if r["cond"] == cond]
    if not rs:
        return None
    m = lambda k: st.mean(st.mean(float(row[k]) for row in r["rows"]) for r in rs)  # noqa: E731
    js = [judge.get((cond, r["product"])) for r in rs]
    return {"cond": cond, "n_pages": len(rs), "separate": m("separate_frac"), "fused": m("fused_frac"),
            "preservation": m("preservation_norm"),
            "judge": st.mean(st.mean(j["judge_answers"]) for j in js) if all(js) else None}


def cmd_pareto(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s5 = P2.jl_read(f"{OUT}/scores.jsonl")
    j5 = {(r["cond"], r["product"]): r for r in P2.jl_read(f"{OUT}/judge.jsonl")}
    s2 = P2.jl_read(f"{P2.OUT}/scores.jsonl")
    j2 = {(r["cond"], r["product"]): r for r in P2.jl_read(f"{P2.OUT}/judge.jsonl")}
    pts = []
    for l1 in L1S:
        p = _point(s5, j5, cond_of(l1))
        if p:
            pts.append({**p, "kind": "weight point", "l1": l1, "l2": L2, "l3": L3})
    refs = {"clean": "clean", "naive_sentence": "naive sentence", "explicit_instruction": "explicit instruction",
            f"{a.best}_s0": f"Phase 2 best arm ({a.best}, seed 0)"}
    for c, label in refs.items():
        p = _point(s2, j2, c)
        if p:
            pts.append({**p, "kind": "reference", "label": label})
    with open(f"{OUT}/pareto.jsonl", "w") as f:
        for p in pts:
            f.write(json.dumps(p) + "\n")

    ink, ink2, grid = "#1f1f1f", "#5c5c5c", "#e6e6e6"
    blue, orange = "#257de0", "#f47230"
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink2)
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    w = [p for p in pts if p["kind"] == "weight point"]
    r = [p for p in pts if p["kind"] == "reference"]
    if w:
        ax.plot([p["preservation"] for p in w], [p["separate"] for p in w], "-", color=blue, lw=2, zorder=2)
        ax.scatter([p["preservation"] for p in w], [p["separate"] for p in w], s=64, color=blue,
                   edgecolor="white", linewidth=2, zorder=3, label="KL weight l1 (l2 = 1, l3 = 0.2)")
        for p in w:
            ax.annotate(f"l1={p['l1']:g}", (p["preservation"], p["separate"]), xytext=(6, 6),
                        textcoords="offset points", color=ink, fontsize=8)
    if r:
        ax.scatter([p["preservation"] for p in r], [p["separate"] for p in r], s=64, marker="D", color=orange,
                   edgecolor="white", linewidth=2, zorder=3, label="reference points (Phase 2)")
        for p in r:
            ax.annotate(p["label"], (p["preservation"], p["separate"]), xytext=(6, -12),
                        textcoords="offset points", color=ink2, fontsize=8)
    ax.set_xlabel("answer preservation (normalised; 1 = clean)", color=ink)
    ax.set_ylabel("SEPARATE (cooking aside sentences, fraction)", color=ink)
    ax.tick_params(colors=ink2)
    ax.set_title("Phase 5: injection vs preservation, Qwen2.5-0.5B, 4 training pages", color=ink, fontsize=10,
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(f"{OUT}/pareto.png")
    for p in pts:
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "attack", "score", "judge", "table", "pareto"):
        s = sub.add_parser(name)
        s.add_argument("--device", default="auto")
        if name == "attack":
            s.add_argument("--l1", type=float, required=True)
            s.add_argument("--products", default="Zephyr,Aurora,Orion,Helix")
            s.add_argument("--steps", type=int, default=STEPS)
            s.add_argument("--score-chunk", type=int, default=128)
            s.add_argument("--no-generate", action="store_true")
            s.add_argument("--tf32", action="store_true")
        if name == "pareto":
            s.add_argument("--best", required=True, help="Phase 2 best arm, e.g. C")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    globals()[f"cmd_{a.cmd}"](a)
