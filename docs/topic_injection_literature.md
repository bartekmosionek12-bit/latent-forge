# Topic injection via activation-targeted GCG: literature review (Phase 0)

Research question (new line, separate from the closed region-GCG line):

> "Ile neutralnego tematu da sie wstrzyknac w odpowiedz modelu jednym sufiksem, przy
> zachowaniu jakosci odpowiedzi na oryginalne pytanie - i czy taki sufiks dziala
> uniwersalnie dla wielu pytan?"

Scope: neutral topics only (cooking, gardening, football) and fictitious product
questions. No refusal-suppression term of any kind, not even as a baseline. No testing
against third-party APIs or services.

Verification method (2026-11-19): each work was opened at source (arXiv abstract + full
text, ACL Anthology / USENIX / AAAI listings, GitHub READMEs). Where a full text could
only be read partly, this is stated. GitHub repositories could be read through their web
pages but not through the GitHub API from this environment, so licence fields come from
the README/repo page, not from the API. Status per claim:
**confirmed** (seen at source) · **corrected** (the citation was wrong or imprecise) ·
**unverified** (could not be confirmed at source).

---

## 1. Geiping et al. 2024, "Coercing LLMs to do and reveal (almost) anything"

- **Exists:** confirmed. arXiv [2402.14020](https://arxiv.org/abs/2402.14020), v1 only
  (21 Feb 2024). Geiping, Stein, Shu, Saifullah, Wen, Goldstein (ELLIS Tübingen / UMD).
  Venue: arXiv lists none; iclr.cc and mlanthology list it as an ICLR 2024 workshop paper
  (SeT LLM). Not a main-track paper.
- **What it measures:** a catalogue of non-jailbreak goals (misdirection, model control,
  denial of service, data extraction, system-prompt repetition), each optimised with GCG
  "or slight variants". ASR = relative substring overlap averaged over 50 samples;
  500-3000 steps, batch 512, top-k 256. Default model LLaMA-2-7b-chat.
- **Misdirection:** confirmed. "URL fishing": the target is a URL, the suffix is
  constrained to tokens that are Chinese characters (700 tokens in LLaMA-2's vocab),
  fixed user question ("Please translate..."). The figure reports 100% ASR. This is a
  *forced exact string*, not a topic.
- **Constraint sets:** ASCII, ASCII w/o code tokens, non-Latin, non-alphabetic, Chinese,
  words. Restricted sets still work; ASCII without code tokens and English-words-only are
  "less effective".
- **Glitch tokens:** confirmed but the citation overstates their role. They are found "as a
  byproduct of optimizing adversarial attacks" (e.g. `Mediabestanden`, `oreferrer`,
  `springframework`, `WorldCat`). They are one of three frequent token groups (with
  punctuation/brackets and role tokens). The paper's broadest mechanism is "coerce the
  model into simulating code", and for code tokens it says "we see no evidence that they
  are necessary". The abstract recommends removing glitch tokens.
- **Universal:** yes for some goals (system-prompt repeater trained on 16 prompts,
  90.4% on unseen prompts; shutdown/EOS attack). Misdirection examples are single-prompt.
- **Code:** [JonasGeiping/carving](https://github.com/JonasGeiping/carving), MIT. Pins
  transformers 4.35 ("4.36 introduces a new kv cache ... does not support grad
  checkpointing"), "some code will only work for LLAMA models". **Not usable as-is**: we
  are on transformers 4.47.1 and Qwen2.5, and the user rule forbids moving transformers.
- **What we take:** (a) a constraint set is a cheap, strong control for the lexical audit:
  a suffix restricted to a script with no topical words (e.g. CJK or non-alphabetic)
  separates the lexical channel from the rest; (b) watch for glitch/code tokens in our
  suffixes and report them; (c) the 50-sample evaluation protocol.

## 2. Wallace et al. 2019, "Universal Adversarial Triggers for Attacking and Analyzing NLP"

- **Exists:** confirmed. EMNLP-IJCNLP 2019, [D19-1221](https://aclanthology.org/D19-1221/),
  arXiv [1908.07125](https://arxiv.org/abs/1908.07125). Wallace, Feng, Kandpal, Gardner,
  Singh.
- **What it measures:** input-agnostic token sequences found by gradient-guided
  (HotFlip-style) search. About one word for classification, about four for language
  modelling. SNLI entailment accuracy 89.94% to 0.55%; 72% of SQuAD "why" questions
  answered with one target phrase; GPT-2 triggers that make it generate racist text.
  Triggers transfer across models.
- **Correction to the citation:** the GPT-2 experiment forces *offensive content*
  conditioned on any context, optimised as the likelihood of a *set of target
  continuations*. It is the methodological ancestor of a topic trigger, not a neutral-topic
  experiment.
- **Code:** [Eric-Wallace/universal-triggers](https://github.com/Eric-Wallace/universal-triggers),
  GPT-2 (old `pytorch_pretrained_bert` stack). Not reusable for Qwen chat models.
- **What we take:** Phase 2 variant C (sum of likelihoods of many target openings) is
  exactly Wallace's LM objective; universality = summing the loss over a batch of contexts.

## 3. Subspace Rerouting (SSR)

- **Exists:** confirmed. Winninger, Addad, Kapusta, "Using Mechanistic Interpretability to
  Craft Adversarial Attacks against Large Language Models", arXiv
  [2503.06269](https://arxiv.org/abs/2503.06269) (v1 Mar 2025, v3 Jul 2026). No
  conference venue found (arXiv only).
- **What it does:** GCG-style search for a suffix (later versions: "perturbation
  anywhere") whose activations leave a refusal subspace. Three variants:
  - *Probe-SSR*: linear probes at chosen layers.
  - *Steering-SSR* (Eq. 5): reproduce `e <- e* + a r̂<r̂,e*>` at layers l1..lK: a steering
    term `|(1+a_k)<r̂,e*> - <e,r̂>|^2` plus an orthogonal-stability term. "Optimising only the
    last token's activations ... proved sufficient"; the stability term "contributed
    minimally". In the code the loss is `(-s - cos(acts[..., -1, :], r_l))^2` per layer,
    last position only.
  - *Attention-SSR*: hijack score or value-vector magnitude of selected heads.
- **Numbers (Table 1, ASR):** Qwen2.5-1.5B: probes 0.91, steering 0.95, attention 0.94
  (flagged as meaningless by the authors), nanoGCG 0.85, vanilla 0.12. Llama3.2-1B: 0.88 /
  0.53 / 0.14 / 0.06 / 0. Runs take minutes or seconds because the forward pass can stop
  after the last targeted layer.
- **"Steering vector to input perturbation":** supported qualitatively (Fig. 5, Llama3.2-1B,
  layers 5/8/10/14): SSR suffixes push cosine with the direction the same way runtime
  steering does, and *more* in untargeted layers 11-13, but match steering only "when
  given a higher steering coefficient". The paper also notes that the SSR effect in later
  layers is "similar to adding a random suffix" in one analysis. No generation-level
  equivalence test.
- **Code:** [Sckathach/subspace-rerouting](https://github.com/Sckathach/subspace-rerouting).
  Python >=3.12,<3.13, depends on TransformerLens, Poetry non-package, README says "the
  code is messy", no licence file seen, no multi-prompt (universal) mode, refusal
  direction hard-wired in `create_directions`. **Not reusable as a library here**
  (TransformerLens + different Python; no licence). The loss is ~10 lines and trivial to
  port.
- **What we take:** the *early-exit trick* (stop the forward at the deepest targeted layer:
  big CPU saving), and the observation that targeting a few mid layers at the last
  position is enough to move the direction through the rest of the network. We do not take
  the refusal setup.

## 4. "Activation-Guided GCG" (cited as Cakar et al., AAAI 2026 Summer Symposium, ~33x)

- **Exists:** confirmed, with corrections. Ege Çakar, Hannah Guan, Kayden Kehe (Harvard),
  "Optimizing Against Safety Representations: Activation-Guided Adversarial Suffixes and
  the Geometry of Refusal", arXiv [2607.08883](https://arxiv.org/abs/2607.08883) (9 Jul
  2026). arXiv comment: "Accepted at the AAAI 2026 Summer Symposium Series", earlier
  presented at the ICLR 2026 Re-Align workshop under the title "Accelerating Adversarial
  Suffix Optimization via Continuous Relaxation and Activation-Guided Objectives". The
  AAAI Symposium Series listing (ojs.aaai.org, AAAI-SS article 42902) exists; its page did
  not load (HTTP 520), so volume/pages (a search snippet says 9(1), 27-34) are
  **unverified**. "Activation-Guided GCG" is the name of a method in the paper, not the
  title.
- **33x: confirmed, but it is not a per-step speedup.** Body text: Soft-GCG "completes in
  approximately 2.5 minutes compared to 81 minutes for vanilla GCG on the same hardware -
  a 33x speedup"; "In the shorter 500-step runs the speedup reached ~43x". The repo's
  "43x faster with no performance decrease" is the 500-step figure. They also say "the
  main bottleneck is model loading rather than per-iteration cost". So 33x is end-to-end
  wall clock (2000-step runs), dominated by GCG's candidate evaluations; hardware not
  stated.
- **Soft-GCG:** confirmed Gumbel-Softmax over the vocabulary per suffix position
  (`softmax((phi+g)/tau)`, embedding `s̃^T E`), Adam lr 0.1 on logits, suffix length 20,
  linear tau 2.0 to 0.1 or a 3-phase schedule (2.5 to 1.0 to 0.5 to 0.01), final tokens =
  argmax of logits. **No measured discretisation gap is reported.** Evaluated on the Gemma 3
  family (270m: 1.00 ASR but "incoherent"; 1b 0.58; 4b 0.34; 12b 0.00) and Llama-2.
- **Global vs single point:** confirmed as a claim, **weakly supported**. Five objectives on
  the Arditi et al. refusal direction (layer 14, last instruction token): Single, Negative,
  Layer (all positions at l*), Token (all layers at p, *with the layer-14 direction reused
  for every layer*), All (all layers x all positions). Llama-2-7b-chat, 200 steps,
  substring ASR: All 0.91, Single 0.84, GCG 0.76, baseline 0.39. But with LlamaGuard: All
  0.09, Single 0.06, GCG 0.00; HarmBench: All 0.01, Single 0.04. The "global beats point"
  ordering holds on substring matching only, which the paper itself calls unreliable.
- **Code:** [Ege-Cakar/ImprovingGCG](https://github.com/Ege-Cakar/ImprovingGCG). **No
  licence found** (default: all rights reserved), refusal-specific, LLaMA-2 / Gemma 3, own
  vendored GCG, uv/pyproject without a transformers pin we could check.
  `activation_pipeline.py` accepts `--direction-path` and `--activation-obj
  {negative, zero, global_zero, layer_zero_all, token_all_layers}`, so swapping the
  direction is plausible but untested. **Do not reuse the code** (no licence, and it
  would drag a refusal pipeline into a repo that forbids one); reuse the *loss shapes* and
  the Gumbel schedule as ideas.
- **What we take:** Phase 2 variant B as a mean over layers and positions; Phase 2 variant D
  (Soft-GCG) as an optional speed-up with argmax discretisation and a measured
  discretisation gap (which they did not report). We must not repeat their evaluation
  weakness: generation-level judgement is our primary measure, not the optimisation
  metric.

## 5. Ben-Tov et al., "Universal Jailbreak Suffixes Are Strong Attention Hijackers"

- **Exists:** confirmed. Matan Ben-Tov, Mor Geva, Mahmood Sharif (Tel Aviv University),
  arXiv [2506.12880](https://arxiv.org/abs/2506.12880) (v1 Jun 2025, v2 Dec 2025),
  TACL 2026 ([2026.tacl-1.46](https://aclanthology.org/2026.tacl-1.46.pdf)).
- **What it shows:** GCG's effect runs through information flow from the suffix into the
  final *chat-template* tokens before generation. Ablating that flow turns successes into
  failures; patching it into failures restores them. GCG suffixes "hijack" those
  positions about 1.5x more than comparable prompts, and more universal suffixes hijack
  more. Using this as an extra objective raises universality "up to 5x in some cases" at
  no extra cost; suppressing hijacking at least halves success "with minimal utility
  loss".
- **Code:** [matanbt/interp-jailbreak](https://github.com/matanbt/interp-jailbreak).
  Analysis code; not needed to run our attack.
- **What we take (important for design):** the positions that matter are the *template
  tokens after the suffix* (`<|im_end|>\n<|im_start|>assistant\n` on Qwen), not the suffix
  positions themselves. A "global" objective that averages over **all** positions
  including the suffix rewards the suffix for simply *containing* topic words (the suffix
  tokens' own residual stream points at "cooking" if they are cooking words). That
  reproduces the lexical failure of the closed line by construction. Variant B should be
  measured at the post-suffix template positions (several layers), with "all positions"
  only as a comparison.

## 6a. Kumar & Lakkaraju 2024, "Manipulating Large Language Models to Increase Product Visibility"

- **Exists:** confirmed. arXiv [2404.07981](https://arxiv.org/abs/2404.07981) (Apr 2024,
  v2 Sep 2024), Harvard. No venue stated on arXiv.
- **What it measures:** a "strategic text sequence" (STS) placed in one product's
  description in a 10-item catalog of fictitious coffee machines. GCG on Llama-2 towards
  the target "1. [Product Name]". A product the model "almost never shows" becomes the top
  recommendation within 100 iterations (2000 run). With random product order the STS
  helps in about 40% of evaluations and changes nothing in about 60%; permutation-aware
  optimisation improves this (numbers only in figures).
- **Not measured:** universality across user queries (one query), answer quality,
  naturalness. The STS is gibberish ("expect formatted XVI RETedly_ _Hello necessarily
  phys*) ### Das Cold Elis$?").
- **Code:** [aounon/llm-rank-optimizer](https://github.com/aounon/llm-rank-optimizer), MIT.
  Llama-2/Vicuna, output-target GCG. Nothing we need beyond the idea.

## 6b. Nestaas, Debenedetti & Tramèr 2024, "Adversarial Search Engine Optimization for Large Language Models"

- **Exists:** confirmed. arXiv [2406.18382](https://arxiv.org/abs/2406.18382) (Jun 2024,
  ETH Zürich). The venue could not be confirmed at source (arXiv lists none).
- **What it measures:** **hand-written** natural-language injections in web pages and
  plugin descriptions (no gradient optimisation), tested on deployed systems (Bing
  Copilot, Perplexity, GPT-4/Claude 3 plugins). Target product 2.5x more likely to be
  recommended on Bing; plugins up to 7.2x, in some cases 0% to over 90%. External
  injections at most 25% success vs 95-100% for direct ones.
- **No code** released.
- **What we take:** the threat model and the strongest **baseline**: an explicit
  natural-language instruction. Note: we do not test any deployed service.

**Closest precedent for "answer normally, but tilt"?** Partly. Both tilt a *ranking*
inside a recommendation task. Neither injects an unrelated topic, and neither measures
whether the original question is still answered.

## 7. JBShield (context only)

- **Exists:** confirmed. Zhang et al., "JBShield: Defending Large Language Models from
  Jailbreak Attacks through Activated Concept Analysis and Manipulation", USENIX Security
  2025 (pp. 8215-8234), arXiv [2502.07557](https://arxiv.org/abs/2502.07557), artifact on
  Zenodo.
- **What it does:** builds concept subspaces from the difference between an input and
  benign samples; flags an input when both a "toxic" and a "jailbreak" concept are
  active (detector), and adjusts those activations (mitigator). Reported average detection
  accuracy 0.95. Single forward pass.
- **Relevance:** a representation-level detector would see a topic-injection suffix only if
  the injected concept is anomalous for the context. We note it; we do not implement it.

---

## Other works found that are closer to the gap than the cited ones

These were not in the brief. They change the gap verdict, so they are listed here.

- **Akar 2026, "Creating Steering Vectors via Input Optimisation"** — arXiv
  [2611.17481](https://arxiv.org/abs/2611.17481) (12 Nov 2026; ETH Zürich bachelor's
  thesis, supervised by Fluri and Tramèr). *ActGCG*: GCG with loss = squared L2 between
  suffix-run activations and **steered** activations, at the chat-template ("filler")
  positions after the prompt, summed over 4 layers. Behaviours: answering in German, and
  anti-refusal. Models Gemma-2-2B-it, Qwen3-1.7B, Llama-3.2-3B, Qwen3-4B. **Universal
  suffix** over 20 train prompts, 30 held-out: Gemma-2-2B German 90.0% train / 73.3% test.
  GCG vs ActGCG on German: 36.7% vs 86.7% (Gemma-2-2B). Discrete vs soft suffix gap
  depends on alpha (soft 96.7% vs discrete 53.3% at alpha 1; reversed at alpha 2);
  projecting soft tokens to nearest vocabulary gives 0%. **Answer quality is not
  measured.** Code linked in the paper: `ethz-spylab/s26-gcg-steering` (not opened;
  licence unknown). This is the nearest existing work to our Phases 1-3: it is our
  pipeline with "German" instead of "cooking", minus quality preservation and Pareto.
- **Thompson & Sklar 2024, "FLRT: Fluent Student-Teacher Redteaming"** — arXiv
  [2407.17447](https://arxiv.org/abs/2407.17447). Optimises a prompt so the model matches a
  *teacher* (a fine-tuned copy) via a distillation loss on output probabilities or
  internal activations, plus a multi-model perplexity penalty and a repetition penalty;
  token insertion/deletion moves; universal prompt reported. Target behaviour is harmful
  compliance (out of scope), but the *mechanics* (distillation-to-a-steered-model loss,
  fluency term, repetition penalty) are exactly what our Phase 4 needs. Ideas only.
- **Chen, Zhang & Tramèr 2026, "Learning to Inject" (AutoInject)** — arXiv
  [2602.05746](https://arxiv.org/abs/2602.05746). RL-generated suffixes for prompt
  injection with a utility term in the reward; plots attack success against utility. It is
  task hijacking against agents and API models (out of scope to replicate), but it is the
  nearest precedent for reporting attack *and* utility jointly. No Pareto sweep.
- **StealthRank** (arXiv [2504.05804](https://arxiv.org/abs/2504.05804)): fluent
  energy-based optimisation for ranking manipulation. **Advertisement Embedding Attacks**
  (arXiv [2508.17674](https://arxiv.org/abs/2508.17674)): covert promotional content via
  prompts or fine-tuning; descriptive, no optimiser of this kind. Both read only at
  abstract level.

---

## Reusable public code: summary

| Work | Code | Licence | Reuse here? |
|---|---|---|---|
| Geiping (carving) | yes | MIT | No: transformers 4.35 pin, LLaMA-centric |
| Wallace | yes | (repo) | No: GPT-2 / old stack |
| SSR | yes | none found | No as library (TransformerLens, Py 3.12); port the loss and early exit |
| Çakar (ImprovingGCG) | yes | **none found** | No: no licence, refusal-specific; ideas only |
| Ben-Tov | yes | (repo) | Not needed |
| Kumar & Lakkaraju | yes | MIT | No: output-target GCG on Llama-2 |
| Nestaas | no | – | – |
| Akar (ActGCG) | linked | unknown | Possibly the closest; check licence before any use |

Recommendation for Phase 2 (decision is Bartek's): extend our own `GCGRegionAttack` loop with
a new target type, porting the loss forms (SSR/ActGCG at post-suffix template positions;
Çakar's mean over layers; Wallace's multi-continuation likelihood). The licence situation
alone rules out taking Çakar's code; ActGCG would need a licence check first.

---

## Verdict: is the gap still open?

Gap as stated: **universal suffix + arbitrary neutral topic direction + preserved answer
quality + Pareto front.**

- Universal discrete suffix that reproduces a steering vector: **done** (Akar 2026, German
  language; Geiping and FLRT for other goals).
- Arbitrary *topic* direction: **not done** in the papers found. Language (Akar) and
  refusal (SSR, Çakar) only. Wallace forces offensive text via target continuations, not a
  direction.
- Preserved answer to the original question: **not measured** in any of the steering-suffix
  papers. AutoInject measures task utility, in a different setting.
- Pareto front of answer quality vs injected amount: **not found anywhere.**

So the exact combination is still open, but it is a **narrow, incremental** gap: it is
roughly "ActGCG with a topic direction, plus a KL task term, plus a weight sweep". No paper
already does it, as far as these searches show; a paper could appear any week (Akar was
posted a week ago).

## Can this work at all? (assessment before any code)

Reasons it should work:

1. **Phase 1 gate is likely to pass.** Topic steering by mean-difference vectors is well
   established, and our own Qwen2.5-0.5B transfer check reproduced a steerable window
   (alpha ~0.3-0.6x residual norm). The risk is that a 0.5B model has a *narrow* window
   between "weaves the topic in" and "collapses" (we saw repetition loops and stray Chinese
   tokens at higher alpha). If the weaving window does not exist, the Pareto question is
   moot.
2. **A discrete suffix can carry a steering vector.** Akar shows this for a surface-level
   behaviour (answer language) with 73% held-out universality on a 2B model. Topic is a
   similarly coarse feature, so B/C are likely to move the generation.

Reasons it may not give an interesting answer:

3. **The plain instruction is a very strong baseline.** An instruct model asked
   "...Also, mention a cooking tip in your answer." will do exactly what we want, fluently,
   universally, with almost no quality loss. The brief's "naive topical sentence"
   baseline should include this *explicit instruction* variant, otherwise we beat a
   straw man. If the optimised suffix does not beat the instruction on the Pareto
   front, the honest finding is "optimisation adds stealth at best, not capability".
4. **The lexical channel will dominate again.** The cheapest way to point the residual
   stream at "cooking" is to put cooking words (in any language) into the suffix. Expect a
   high lexical share. That is fine under the new question (we report it), but the
   interesting quantity becomes: *how much topic per unit of lexical topic content*, and
   the script-constrained control (Geiping) is the cleanest way to measure it.
5. **"Global over all positions" rewards lexical suffixes by construction** (see item 5):
   measure at post-suffix template positions.
6. **The quality measures are weak on a 0.5B model and fictitious products.** The clean
   answer about a non-existent router is itself a hallucination, and sampling variance is
   large. Answer preservation must be scored against the *clean-vs-clean* similarity
   spread (several clean samples per question), or every attack looks lossy. The LLM judge
   must be a local model (no third-party APIs); a local judge of a similar size is weak, so
   its limits must be stated and it should be a secondary measure.
7. **CPU budget.** A universal suffix over 10 questions multiplies forward passes by ~10.
   SSR's early exit (stop the forward at the deepest targeted layer) helps for B, not for
   C or the KL term. Expect hours per run on CPU for Phases 3-4.

Bottom line: Phase 1 and variant B/C will very likely "work" on the optimisation metric and
probably on topicality; whether the result is *interesting* depends on beating the explicit
instruction baseline on the Pareto front, and on how much of the effect survives a
script-constrained suffix. Both are cheap to check early and both are worth reporting
whichever way they go.
