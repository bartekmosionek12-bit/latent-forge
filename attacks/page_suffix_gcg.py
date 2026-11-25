"""PageSuffixGCGAttack: GCG-style suffix search for the product-page (indirect injection)
scenario of the topic-injection line, Phase 2. Same loop as GCGRegionAttack (grad w.r.t.
one-hot suffix -> top-k swaps -> exact batched scoring -> accept if better and the full
string re-tokenises to the same ids), with:

* the page prefix computed ONCE and KV-cached (candidates only recompute suffix + tail);
* objective by target type:
    ContinuationSetTarget : mean token CE of k target continuations per step (rotating
                            through the whole set; the current suffix is re-scored on the
                            same k targets, so "better" is always like for like)
    TopicBandTarget       : |mean cos(h_L, d) - band_center| over the template tokens after
                            the user turn ("template") or over every position from the suffix
                            on ("all"); the forward stops after layer L
* arms via config:
    vocab="full"            A: every non-special token
    vocab="words"           B: only single tokens " word" / " Word" of the n most frequent
                               English words (wordfreq)
    fluency_weight > 0      C: + w * mean NLL of the suffix under the same model (in context)
    rep_weight > 0             + w * repeated token 2/3-gram fraction (repetition attractor)
    detector_weight > 0     D: + w * P(injection) of the page text from classifier D1
                               (scoring only: D1 has its own tokenizer, no gradient)
    kl_weight > 0           Phase 5: + w * KL_task, the mean over the tokens of the clean greedy
                               answer (target.clean_answer, teacher-forced) of
                               KL(p_clean || p_with_suffix); p_clean = the same answer given the
                               page WITHOUT any suffix (ContinuationSetTarget only)
* every step appended to ``log_path`` (.jsonl) and a human line to the matching .log;
  with resume=True a run restarts from the last logged step.

Hook/caching semantics verified on transformers 4.47.1 / torch 2.15 (tests).
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from models import get_decoder_layers
from models.page_prompt import n_template_tokens, page_prompt_parts
from targets import ContinuationSetTarget, PageInjectionTarget, TopicBandTarget

from .base import Attack, AttackResult

DEFAULTS = dict(
    n_tokens=20,
    init_text=None,               # None -> evaluation.topic_data.FLUENT_INIT
    num_steps=200,
    topk=256,
    search_width=128,
    seed=0,
    vocab="full",                 # "full" (arm A/C/D/E) | "words" (arm B)
    n_words=8000,
    fluency_weight=0.0,
    rep_weight=0.0,
    detector_weight=0.0,
    detector_model="protectai/deberta-v3-base-prompt-injection-v2",
    kl_weight=0.0,
    kl_chunk=8,                   # candidates per KL forward (logits over the whole answer are large)
    targets_per_step=4,
    score_chunk=32,
    log_path=None,                # ".../run.jsonl"; a ".log" with the same stem is written too
    resume=True,
)


@contextmanager
def truncated(model, upto: int):
    """Run only decoder blocks 0..upto (inclusive)."""
    layers = model.model.layers
    model.model.layers = layers[: upto + 1]
    try:
        yield
    finally:
        model.model.layers = layers


def repeated_ngram_frac(ids: torch.Tensor) -> torch.Tensor:
    """[B, n] token ids -> [B] mean over n=2,3 of the fraction of repeated n-grams."""
    out = []
    for row in ids.tolist():
        fr = []
        for n in (2, 3):
            g = [tuple(row[i:i + n]) for i in range(len(row) - n + 1)]
            fr.append(1 - len(set(g)) / len(g) if g else 0.0)
        out.append(sum(fr) / len(fr))
    return torch.tensor(out)


def kl_from_logp(logp_clean: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """logp_clean [T, V] (log-probs without the suffix), logits [B, T, V] (with it) ->
    [B] mean over the T positions of KL(p_clean || p_with_suffix)."""
    logq = logits.float().log_softmax(-1)
    p = logp_clean.exp()[None]
    return (p * (logp_clean[None] - logq)).sum(-1).mean(-1)


def word_token_mask(tokenizer, vocab_size: int, n_words: int) -> torch.Tensor:
    from wordfreq import top_n_list

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for w in top_n_list("en", n_words):
        if not w.isalpha():
            continue
        for form in (w, w.capitalize()):
            ids = tokenizer(" " + form, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                mask[ids[0]] = True
    return mask


def full_token_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[: len(tokenizer)] = True
    mask[list(tokenizer.all_special_ids)] = False
    added = [i for i in tokenizer.get_added_vocab().values() if i < vocab_size]
    if added:
        mask[added] = False
    return mask


class PageSuffixGCGAttack(Attack):
    target_type = PageInjectionTarget

    def __init__(self, target: PageInjectionTarget, **config) -> None:
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise TypeError(f"unknown PageSuffixGCGAttack options: {sorted(unknown)}")
        if not isinstance(target, (ContinuationSetTarget, TopicBandTarget)):
            raise TypeError("PageSuffixGCGAttack needs a ContinuationSetTarget or TopicBandTarget")
        super().__init__(target, **{**DEFAULTS, **config})

    # ---------------------------------------------------------------- setup
    def setup(self, bundle) -> None:
        c, t = self.config, self.target
        self.bundle, self.model, self.tok = bundle, bundle.model, bundle.tokenizer
        self.dev = bundle.device
        self.W = self.model.get_input_embeddings().weight
        self.before_str, self.after_str = page_prompt_parts(self.tok, t.page, t.question)
        self.before_ids = self._ids(self.before_str)
        self.after_ids = self._ids(self.after_str)
        self.n_tpl = n_template_tokens(self.tok, self.after_ids.tolist())
        V = self.W.shape[0]
        self.allowed = (word_token_mask(self.tok, V, c["n_words"]) if c["vocab"] == "words"
                        else full_token_mask(self.tok, V)).to(self.dev)
        with torch.no_grad():
            out = self.model(self.before_ids[None], use_cache=True)
        self.prefix_kv = out.past_key_values
        self.prefix_last_logp = out.logits[0, -1].float().log_softmax(-1)   # predicts suffix token 0
        self.P = self.before_ids.numel()
        if isinstance(t, ContinuationSetTarget):
            self.targets = [self._ids(x) for x in t.continuations]
            if c["kl_weight"] > 0:
                if not t.clean_answer:
                    raise ValueError("kl_weight > 0 needs target.clean_answer (the clean greedy answer)")
                self.ans_ids = self._ids(t.clean_answer)
                with torch.no_grad():
                    clean = torch.cat([self.before_ids, self.after_ids, self.ans_ids])[None]
                    T, A = self.ans_ids.numel(), self.after_ids.numel()
                    lg = self.model(clean).logits[0, self.P + A - 1:self.P + A + T - 1]
                    self.logp_clean = lg.float().log_softmax(-1)          # [T, V]
        elif c["kl_weight"] > 0:
            raise ValueError("kl_weight is only defined for a ContinuationSetTarget")
        else:
            from attacks.activation_steering import contrastive_direction
            from targets import ActivationDirectionTarget

            d, _, _ = contrastive_direction(bundle, ActivationDirectionTarget(
                t.positive_texts, t.negative_texts, t.layer), chat=True)
            self.direction = (d / d.norm()).to(self.dev, self.W.dtype)

    def _ids(self, text: str) -> torch.Tensor:
        return self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(self.dev)

    def _cache(self, B: int):
        from transformers import DynamicCache

        kv = self.prefix_kv
        if not isinstance(kv, tuple):
            kv = kv.to_legacy_cache()
        cache = DynamicCache()
        for i, (k, v) in enumerate(kv):
            cache.update(k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1), i)
        return cache

    def _run(self, B: int, length: int, **inputs):
        pos = torch.arange(self.P, self.P + length, device=self.dev)[None].expand(B, -1)
        return self.model.model(past_key_values=self._cache(B), position_ids=pos, use_cache=True, **inputs)

    # ---------------------------------------------------------------- objectives
    def _fluency_from_logits(self, logits_suffix: torch.Tensor, suffix: torch.Tensor) -> torch.Tensor:
        """mean NLL of the suffix tokens; logits_suffix [B, n-1, V] predict tokens 1..n-1."""
        first = -self.prefix_last_logp[suffix[:, 0]]
        rest = -logits_suffix.float().log_softmax(-1).gather(-1, suffix[:, 1:, None])[..., 0]
        return torch.cat([first[:, None], rest], 1).mean(1)

    def _continuation_terms(self, x_suffix, B, tgt_idx, need_flu, embeds=None):
        """x_suffix: ids [B, n] (or embeds [B, n, d] via ``embeds``). Returns (obj [B], flu [B]|None)."""
        n, A = self.config["n_tokens"], self.after_ids.numel()
        objs, flu = [], None
        for j in tgt_idx:
            tgt = self.targets[j]
            T = tgt.numel()
            if embeds is None:
                seq = torch.cat([x_suffix, self.after_ids.expand(B, -1), tgt.expand(B, -1)], 1)
                h = self._run(B, seq.shape[1], input_ids=seq).last_hidden_state
            else:
                seq = torch.cat([embeds, self.W[self.after_ids].expand(B, -1, -1),
                                 self.W[tgt].expand(B, -1, -1)], 1)
                h = self._run(B, seq.shape[1], inputs_embeds=seq).last_hidden_state
            logits = self.model.lm_head(h[:, n + A - 1:n + A + T - 1]).float()
            objs.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.repeat(B),
                                        reduction="none").view(B, T).mean(1))
            if need_flu and flu is None:
                flu = self.model.lm_head(h[:, : n - 1])
        return torch.stack(objs).mean(0), flu

    def _kl_term(self, B, ids=None, embeds=None) -> torch.Tensor:
        """[B] KL_task for suffix ids [B, n] (or embeds [B, n, d])."""
        n, A, T = self.config["n_tokens"], self.after_ids.numel(), self.ans_ids.numel()
        if embeds is None:
            seq = torch.cat([ids, self.after_ids.expand(B, -1), self.ans_ids.expand(B, -1)], 1)
            h = self._run(B, seq.shape[1], input_ids=seq).last_hidden_state
        else:
            seq = torch.cat([embeds, self.W[self.after_ids].expand(B, -1, -1),
                             self.W[self.ans_ids].expand(B, -1, -1)], 1)
            h = self._run(B, seq.shape[1], inputs_embeds=seq).last_hidden_state
        return kl_from_logp(self.logp_clean, self.model.lm_head(h[:, n + A - 1:n + A + T - 1]))

    def _band_obj(self, B, ids=None, embeds=None):
        t = self.target
        cap = {}

        def hook(_m, _a, out):
            cap["h"] = out[0] if isinstance(out, tuple) else out

        block = get_decoder_layers(self.model)[t.layer]
        hd = block.register_forward_hook(hook)
        try:
            with truncated(self.model, t.layer):
                if embeds is None:
                    seq = torch.cat([ids, self.after_ids.expand(B, -1)], 1)
                    self._run(B, seq.shape[1], input_ids=seq)
                else:
                    seq = torch.cat([embeds, self.W[self.after_ids].expand(B, -1, -1)], 1)
                    self._run(B, seq.shape[1], inputs_embeds=seq)
        finally:
            hd.remove()
        h = cap["h"].float()
        h = h[:, -self.n_tpl:] if t.positions == "template" else h
        cos = F.cosine_similarity(h, self.direction.float()[None, None], dim=-1).mean(1)
        return (cos - t.band_center).abs(), cos

    def score(self, cands: torch.Tensor, tgt_idx) -> dict:
        """Exact batched loss components for [B, n] candidate suffixes."""
        c = self.config
        out = {k: [] for k in ("obj", "flu", "cos", "kl")}
        with torch.no_grad():
            for part in cands.split(c["score_chunk"]):
                B = part.shape[0]
                if isinstance(self.target, ContinuationSetTarget):
                    obj, flu_logits = self._continuation_terms(part, B, tgt_idx, c["fluency_weight"] > 0)
                    out["obj"].append(obj)
                    if flu_logits is not None:
                        out["flu"].append(self._fluency_from_logits(flu_logits, part))
                    if c["kl_weight"] > 0:
                        out["kl"] += [self._kl_term(q.shape[0], ids=q) for q in part.split(c["kl_chunk"])]
                else:
                    obj, cos = self._band_obj(B, ids=part)
                    out["obj"].append(obj)
                    out["cos"].append(cos)
                    if c["fluency_weight"] > 0:
                        h = self._run(B, part.shape[1], input_ids=part).last_hidden_state
                        out["flu"].append(self._fluency_from_logits(self.model.lm_head(h[:, :-1]), part))
        res = {k: torch.cat(v).cpu() for k, v in out.items() if v}
        total = res["obj"].clone()
        if c["fluency_weight"] > 0:
            total += c["fluency_weight"] * res["flu"]
        if c["kl_weight"] > 0:
            total += c["kl_weight"] * res["kl"]
        if c["rep_weight"] > 0:
            res["rep"] = repeated_ngram_frac(cands)
            total += c["rep_weight"] * res["rep"]
        if c["detector_weight"] > 0:
            from evaluation.detectors import injection_prob

            texts = [self.target.page + self.tok.decode(s) for s in cands]
            res["det"] = torch.tensor(injection_prob(texts, c["detector_model"], str(self.dev)))
            total += c["detector_weight"] * res["det"]
        res["total"] = total
        return res

    def token_grad(self, suffix: torch.Tensor, tgt_idx) -> torch.Tensor:
        c = self.config
        onehot = F.one_hot(suffix, self.W.shape[0]).to(self.W.dtype)[None]
        onehot.requires_grad_(True)
        emb = onehot @ self.W
        if isinstance(self.target, ContinuationSetTarget):
            obj, flu_logits = self._continuation_terms(None, 1, tgt_idx, c["fluency_weight"] > 0, embeds=emb)
            loss = obj.sum()
            if c["kl_weight"] > 0:
                loss = loss + c["kl_weight"] * self._kl_term(1, embeds=emb).sum()
        else:
            loss, _ = self._band_obj(1, embeds=emb)
            loss = loss.sum()
            flu_logits = None
            if c["fluency_weight"] > 0:
                h = self._run(1, emb.shape[1], inputs_embeds=emb).last_hidden_state
                flu_logits = self.model.lm_head(h[:, :-1])
        if c["fluency_weight"] > 0:
            logp = torch.cat([self.prefix_last_logp[None, None], flu_logits.float().log_softmax(-1)], 1)
            loss = loss + c["fluency_weight"] * (-(onehot.float() * logp).sum(-1).mean())
        return torch.autograd.grad(loss, onehot)[0][0]          # [n, V]

    def full_metric(self, suffix: torch.Tensor) -> dict:
        """Optimisation metric on the WHOLE target set (continuations) or the band."""
        idx = list(range(self._n_targets()))
        r = self.score(suffix[None], idx)
        return {k: float(v[0]) for k, v in r.items()}

    def roundtrip_ok(self, cands: torch.Tensor) -> torch.Tensor:
        ok = []
        for s in cands:
            text = self.before_str + self.tok.decode(s) + self.after_str
            again = self.tok(text, add_special_tokens=False)["input_ids"]
            full = torch.cat([self.before_ids, s, self.after_ids]).tolist()
            ok.append(again == full)
        return torch.tensor(ok, dtype=torch.bool)

    def init_suffix(self) -> torch.Tensor:
        from evaluation.topic_data import FLUENT_INIT

        ids = self._ids(self.config["init_text"] or FLUENT_INIT)
        if ids.numel() != self.config["n_tokens"]:
            raise ValueError(f"init_text has {ids.numel()} tokens, need {self.config['n_tokens']}")
        return ids

    # ---------------------------------------------------------------- loop
    def _n_targets(self) -> int:
        return len(getattr(self, "targets", [0]))

    def _step_targets(self, step: int, k: int, n_t: int):
        """Targets scored at ``step``: k consecutive ones, rotating through the whole set."""
        return [((step - 1) * k + i) % n_t for i in range(k)]

    def _log(self, rec: dict, line: str) -> None:
        p = self.config["log_path"]
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(os.path.splitext(p)[0] + ".log", "a") as f:
            f.write(line + "\n")

    def run(self, bundle) -> AttackResult:
        c, t0 = self.config, time.time()
        self.setup(bundle)
        n_t = self._n_targets()
        k = min(c["targets_per_step"], n_t)
        suffix, start = self.init_suffix(), 0
        best = None
        p = c["log_path"]
        if c["resume"] and p and os.path.exists(p):
            recs = [json.loads(x) for x in open(p) if x.strip()]
            if recs:
                last = recs[-1]
                suffix = torch.tensor(last["suffix_ids"], device=self.dev)
                start = last["step"] + 1
                b = min(recs, key=lambda r: r["full"]["total"])
                best = (b["full"]["total"], torch.tensor(b["suffix_ids"], device=self.dev), b["step"])
        if start == 0:
            m = self.full_metric(suffix)
            best = (m["total"], suffix.clone(), 0)
            self._log({"step": 0, "suffix": self.tok.decode(suffix), "suffix_ids": suffix.tolist(),
                       "full": m, "accepted": False, "n_valid": 0, "seconds": 0.0},
                      f"[step 0] full={m} suffix={self.tok.decode(suffix)!r}")
            start = 1
        for step in range(start, c["num_steps"] + 1):
            ts = time.time()
            g = torch.Generator(device="cpu").manual_seed(c["seed"] * 100_003 + step)
            tgt_idx = self._step_targets(step, k, n_t)
            grad = self.token_grad(suffix, tgt_idx)
            grad[:, ~self.allowed] = float("inf")
            top = (-grad).topk(c["topk"], dim=1).indices.cpu()
            pos = torch.randint(0, suffix.numel(), (c["search_width"],), generator=g)
            new = top[pos, torch.randint(0, c["topk"], (c["search_width"],), generator=g)]
            cands = suffix.cpu().repeat(c["search_width"], 1)
            cands[torch.arange(c["search_width"]), pos] = new
            cands = torch.unique(cands, dim=0)
            cands = cands[(cands != suffix.cpu()).any(dim=1)]
            cands = cands[self.roundtrip_ok(cands.to(self.dev))].to(self.dev)
            batch = torch.cat([suffix[None], cands])
            sc = self.score(batch, tgt_idx)
            accepted = False
            if cands.shape[0]:
                i = int(sc["total"][1:].argmin()) + 1
                if sc["total"][i] < sc["total"][0]:
                    suffix, accepted = batch[i].clone(), True
            m = self.full_metric(suffix)
            if m["total"] < best[0]:
                best = (m["total"], suffix.clone(), step)
            rec = {"step": step, "suffix": self.tok.decode(suffix), "suffix_ids": suffix.tolist(),
                   "targets": tgt_idx, "full": m, "accepted": accepted, "n_valid": int(cands.shape[0]),
                   "step_total": float(sc["total"][0] if not accepted else sc["total"].min()),
                   "seconds": time.time() - ts}
            self._log(rec, f"[step {step}] full_total={m['total']:.4f} obj={m['obj']:.4f} "
                           f"{'acc' if accepted else '---'} valid={rec['n_valid']} "
                           f"({rec['seconds']:.1f}s) {rec['suffix']!r}")
        return AttackResult(
            target=self.target, attack=type(self).__name__, payload=self.tok.decode(best[1]),
            final_loss=best[0],
            extra={"best_step": best[2], "best_suffix_ids": best[1].tolist(),
                   "last_suffix": self.tok.decode(suffix), "seconds": time.time() - t0,
                   "config": dict(c)},
        )
