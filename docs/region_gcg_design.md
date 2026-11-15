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
