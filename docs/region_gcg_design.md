# Region-targeted GCG: design note (Phase 0)

## Research question

Czy region-targeted GCG daje transferowalny, dostarczany jako zwykly tekst (blackbox)
odpowiednik activation steeringu - czyli efekt podobny do wstrzykniecia kierunku w
aktywacjach, ale bez potrzeby dostepu white-box w momencie ataku.

This is not "a better GCG". It is a different capability with a different trade-off:
white-box access is needed only at optimisation time, and the attack is delivered as text.

## Does nanoGCG accept a custom loss? No.

Checked against the installed source, nanogcg 0.3.0, `nanogcg/gcg.py`:

- `GCGConfig` has no loss or scoring hook. Its only loss-related options are
  `use_mellowmax` / `mellowmax_alpha`, and they still score the *target tokens*.
- The loss is hard-coded in three places, all as cross-entropy (or mellowmax) of
  `self.target_ids` on the logits:
  - `GCG.compute_token_gradient()` (lines 440-495): gradient w.r.t. the one-hot suffix
  - `GCG._compute_candidates_loss_original()` (lines 497-549): batched candidate scoring
  - `GCG._compute_candidates_loss_probe_sampling()` (lines 551+): same, with a draft model
- `GCG.run()` always builds `target_ids` / `target_embeds` from a target *string* and
  appends them to every candidate's input.

### Decision: own discrete-search loop, modelled on GCG

A new `attacks/gcg_region.py` will run:
gradient w.r.t. a one-hot suffix → top-k token swaps per position → batched exact
evaluation → keep the best candidate if it improves.

**Alternative considered and rejected: subclass `nanogcg.GCG` and override the two loss
methods.** That would reuse sampling, `filter_ids`, the buffer and the prefix cache. But
it would depend on private methods (`_compute_candidates_loss_original`) and on internal
state (`target_embeds` is concatenated into every candidate inside `run()`), so we would
have to pass a dummy empty target string. It would also break silently on any nanoGCG
upgrade. The same loop already exists and is verified in this repo
(`EmbeddingRegionAttack`, Phase 3), so writing our own is cheap. It also keeps the
attack independent of nanoGCG internals.

What is reused from the GCG / Phase 3 mechanics:
- one-hot @ embedding-matrix gradient, top-k per position, one swap per candidate;
- an allowed-token mask (ASCII, no special tokens);
- the decode → re-encode round-trip check (we ship a *string*, so the tokens must survive);
- the final score is measured on the string, not on the internal ids.

## Target and objective

- `HiddenStateRegionTarget(reference_texts, layer)`: the region centroid is the mean of
  the attacked model's block-`layer` outputs over the reference texts' tokens (same
  forward hook as `ActivationSteeringAttack`; position 0 is excluded as an attention
  sink). Topics are neutral only (cooking).
- **Optimised state**: block-`layer` output at the **last token** of [prompt + suffix].
  That is the position from which the continuation is generated.
- **Distance**: **cosine distance**. Residual-stream norms vary strongly with position
  (attention sink at position 0, growth with depth), and steering adds a *direction*, so
  matching direction rather than magnitude is the closer analogue. Euclidean distance
  would be dominated by norm differences unrelated to topic.

## Two independent success measures (the known risk)

Closeness of one hidden state does not guarantee a topical continuation (cf. the
non-monotonic steering optimum). Every run therefore reports:

1. **Optimisation metric**: cosine distance of the last-token hidden state to the
   centroid (the attack's own loss).
2. **Independent topical score of the generated continuation**, from components the
   attack never sees:
   - **embedding**: all-MiniLM-L6-v2 (sentence-transformers, a different model)
     embedding of the continuation, as cosine distance to the MiniLM centroid of the same
     reference texts;
   - **keywords**: the fraction of continuation words that appear in a fixed,
     hand-written cooking-domain keyword list.

If (1) improves while (2) does not, or the other way round, that is logged as a named
finding.

## Controls

- No suffix (prompt only).
- Random suffixes: several seeds of random allowed tokens, same length.
- Initial filler suffix ("x x x ..."), i.e. the optimisation start point.
- Naive topical suffix: a reference sentence truncated to the same token length. This is
  the "just write about cooking" baseline, analogous to naive concatenation in Phase 3.

## Gradient verification

- The one-hot `inputs_embeds` path gives the same hidden state as the `input_ids` path.
- A central finite-difference check of the loss along random directions in suffix-embedding
  space matches autograd (grad · v).
- The first-order predicted change for actual token swaps correlates with the real change
  (Spearman). This one is informative only, since swaps are not infinitesimal.

---

# Phase 4: region objective + readability (AutoDAN-style)

Question: does adding a readability term (in the spirit of gradient AutoDAN, Zhu et al.
2023) to the region-targeted GCG loop (a) lower the suffix perplexity, (b) change the
independent topicality of the continuation, and (c) stop the "overshoot" seen in Phase 1
(hidden distance 0.118, below the reference texts' own range 0.157-0.188)? Same neutral topic
(cooking), prompt, model (distilgpt2, block 3), seed, budget (20 tokens, 200 steps,
search width 256, top-k 128) as Phase 1.

## Which region objective: `min`, not `band`

`band` (|d - r|) already removes the overshoot by construction: it stops at the radius.
If we combined readability with `band`, any reduction of overshoot would be credited to
the band, not to readability, and the hypothesis "readability regularises against
overshooting" could not be tested. So the region term is the Phase 1 `min` objective
(d = cosine distance of the last-token block-3 state to the centroid), and the
overshoot question is answered by whether readability alone keeps d inside the
reference range.

## Readability term

`CE_suffix` = mean cross-entropy (nats/token) of the suffix tokens under the SAME model
(distilgpt2), each token conditioned on the prompt and the preceding suffix tokens, i.e.
the model's own log-likelihood of the suffix in context. It comes from the logits of the
same forward pass that produces the block-3 state (no extra model, no extra pass); the
LM head is applied only to the suffix positions, in candidate chunks, to bound memory.

Calibration measured before choosing weights (distilgpt2, after the Phase 1 prompt):

| suffix (20 tokens) | CE nats/token |
|---|---|
| filler "x x x ..." (start point) | 1.79 |
| fluent neutral English | 2.93 |
| naive topical (reference sentences cut to 20 tokens) | 4.55 |
| Phase 1 optimised suffix | 12.11 |
| random allowed tokens (seed 0) | 14.71 |

Caveat visible already here: the repetitive filler is the *most* "readable" suffix by this
measure. A strong readability weight can therefore keep the suffix near "x x x", which is
low-CE but not language. We report a repetition statistic (distinct-token fraction) next
to perplexity for that reason.

## Combining the two: AutoDAN-like two-stage step

Each step:

1. **Candidate proposal (stage 1)**: for each suffix position take the union of
   - top-k/2 tokens by the gradient of the combined objective J w.r.t. the one-hot suffix
     (the GCG part), and
   - top-k/2 tokens by the model's own next-token log-probability at that position given
     the current prefix (the AutoDAN "readable" part).
   Both restricted to the same allowed-token mask as Phase 1.
2. **Selection (stage 2)**: `search_width` single-token-swap candidates drawn from those
   lists are scored exactly with
   `J = d_region + λ · CE_suffix`,
   and the best candidate that improves J and survives the string round-trip check is
   accepted (unchanged from Phase 1).

AutoDAN combines the two scores in one ranking per position and grows the prompt token by
token; we keep the fixed-length GCG loop (so the comparison with Phase 1 is
like-for-like) and only change what is proposed and what is minimised.

## Weights: tested, not guessed

Scale from calibration: Phase 1 moved d by ~0.25 (0.37 -> 0.12); the CE gap between
gibberish and readable text is ~8-10 nats/token. So λ ≈ 0.03 makes the two gains
comparable, λ = 0.01 is region-dominated and λ = 0.1 readability-dominated. Runs:

| run | proposals | λ |
|---|---|---|
| Phase 1 (existing) | gradient top-k | 0 |
| control | union (gradient + LM) | 0 |
| R1 | union | 0.01 |
| R3 | union | 0.03 |
| R10 | union | 0.1 |

The λ = 0 control separates the effect of the readable proposals from the effect of the
readability term in the loss.

## Measures (all three, per run)

1. hidden cosine distance to the centroid (the region term itself);
2. independent topicality of the greedy continuation: MiniLM distance to topic +
   cooking-keyword fraction (unchanged from Phases 1-3);
3. **new**: perplexity of the whole suffix under an independent model,
   Qwen2.5-0.5B-Instruct (raw text, suffix tokens conditioned on the prompt). distilgpt2
   CE is also reported, but it is circular (it is the term being optimised).
   Baselines for (3): filler, Phase 1 suffix, naive topical, random suffix.
