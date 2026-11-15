# latent-forge

A goal-driven toolkit for adversarial attacks on LLMs and on the embedders around them,
built for security research. You state **where** in latent space you want to land
(a `Target`). The **dispatcher** picks and runs the matching attack (an `Attack`), then
scores it with the matching metric. Three published technique families sit behind one
interface:

| Target (the goal) | Attack (the strategy) | Success metric |
|---|---|---|
| `TextTarget(text)`: force an exact output | `GCGAttack`: thin wrapper over nanoGCG | exact match, target-token accuracy, target CE loss |
| `EmbeddingRegionTarget(reference_texts, embedder_name)`: land in a region of an **external** embedder (RAG / memory retriever) | `EmbeddingRegionAttack`: gradient-guided token flips, SentAttack-style | cosine distance to the region centroid (+ perplexity as a detectability signal) |
| `ActivationDirectionTarget(positive, negative, layer)`: a direction in the **attacked model's** residual stream | `ActivationSteeringAttack`: contrastive mean-difference vector injected via forward hooks | projection of the output's activations on the direction |

All test data is neutral and synthetic: a placeholder string, a sourdough-baking region,
and a formal-vs-informal writing style.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU
pip install -r requirements.txt
python -m pytest -q                       # 21 tests, ~15 s on CPU
python scripts/e2e_all_paths.py           # all three paths, one dispatcher (~23 min on CPU, GCG dominates)
```

The core call site does the same thing for every goal type:

```python
import dispatcher
from models import ModelConfig, load_model
from targets import TextTarget

bundle = load_model(ModelConfig(name="distilgpt2"))
result, report = dispatcher.run(TextTarget("TEST_odpowiedz_testowa"), bundle, num_steps=250)
```

## Architecture

```
latent-forge/
  targets/       goal specs (pure data). Target base: kind + describe()
  attacks/       strategies. Attack base: target_type + run(bundle) -> AttackResult
  evaluation/    per-target metrics + evaluate_result() (picks the metric by target type)
  models/        load_model(ModelConfig) -> ModelBundle, get_decoder_layers(),
                 split_prompt() (nanoGCG token layout), load_embedder()
  dispatcher.py  Target type -> Attack class registry; select_attack(), run()
  scripts/       evidence runs per phase + e2e_all_paths.py
  tests/         unit + smoke tests on the real toy models
  runs/          logs/JSON of every evidence run quoted below
```

A request flows like this:

1. **`targets/`** holds the goal. A target says *what* you want and nothing about *how*.
2. **`dispatcher.select_attack(target, **cfg)`** walks the target's MRO through
   `ATTACK_REGISTRY` and builds the matching `Attack`. Each attack rejects option names
   it does not know, so a typo fails loudly. Subclasses of a registered target route to
   the parent's attack unless they register their own.
3. **`Attack.run(bundle)`** returns an `AttackResult` with `payload` (suffix / adversarial
   text / steering vector), `final_loss`, `loss_history` and `extra`.
   `bundle` is the attacked LLM. `EmbeddingRegionAttack` ignores it because its target
   lives in the embedder named by the target.
4. **`evaluation.evaluate_result(result, bundle)`** picks the metric by target type.
   The metrics are computed independently of the attack's own bookkeeping. For example,
   the GCG target loss is recomputed from scratch and matched nanoGCG's to 4 decimals.
5. **`dispatcher.run()`** chains steps 2 to 4.

## Adding a new Target and Attack

Example: a hypothetical `RefusalTarget` that should be pursued by a new `PrefillAttack`.

1. **Define the target** in `targets/refusal.py`. Subclass `Target`, give it a unique
   `kind`, implement `describe()`, and validate inputs in `__post_init__`:
   ```python
   @dataclass(repr=False)
   class RefusalTarget(Target):
       prompt: str
       kind = "refusal"
       def describe(self): return f"make the model refuse {self.prompt!r}"
   ```
   Export it in `targets/__init__.py`.
2. **Write the attack** in `attacks/prefill.py`. Subclass `Attack`, set
   `target_type = RefusalTarget`, merge your defaults in `__init__`, reject unknown
   options, and return an `AttackResult` from `run(bundle)`. Put anything the metric needs
   (for example the baseline payload) in `extra`. Export it in `attacks/__init__.py`.
3. **Register the pair** in `dispatcher.py` by adding `RefusalTarget: PrefillAttack` to
   `ATTACK_REGISTRY`. Out-of-tree code can call `dispatcher.register(RefusalTarget, PrefillAttack)`.
4. **Add the metric**: a function in `evaluation/` plus an `isinstance` branch in
   `evaluation/dispatch.py:evaluate_result`.
5. **Test it**. Add a routing case to `tests/test_skeleton.py::test_dispatcher_routes_by_type`
   and a tiny real run (a few steps on distilgpt2) in `tests/test_<attack>.py`. Mark it
   `@pytest.mark.slow` if it takes more than a few seconds.
6. **Pin** any new dependency at an exact version in `requirements.txt` and regenerate
   `requirements-lock.txt` with `pip freeze`.

To route one target type to a *different* attack depending on its fields, the minimal
registry is the extension point: replace `select_attack_class` with a rule function.
That is where a richer "goal to strategy" policy would go.

## Switching the model: distilgpt2 (CPU) to Qwen2.5-7B-Instruct (RunPod A5000)

**What to change: configuration only.**

```bash
pip install torch==2.15.0          # CUDA build, same version
pip install -r requirements.txt     # same pins: transformers 4.47.1 supports Qwen2

python scripts/phase2_gcg_demo.py      --model Qwen/Qwen2.5-7B-Instruct --dtype float16
python scripts/phase4_steering_demo.py --model Qwen/Qwen2.5-7B-Instruct --dtype float16 --layer 14
python scripts/e2e_all_paths.py        --model Qwen/Qwen2.5-7B-Instruct --dtype float16 --steer-layer 14
```

In code, this is `ModelConfig(name="Qwen/Qwen2.5-7B-Instruct", dtype="float16")`.
`device="auto"` picks CUDA by itself.

**What does not need to change in the code.** I checked each of these, not assumed:

- **Chat template.** Qwen has one, and nanoGCG applies it. `models/prompting.split_prompt`
  mirrors nanoGCG's layout, so the evaluator sees the same tokens. With the real
  Qwen2.5-7B-Instruct tokenizer the attacked span lands between
  `<|im_start|>user\nReply with a short answer:` and `<|im_end|>\n<|im_start|>assistant\n`.
- **Decoder layers.** `get_decoder_layers` already resolves `model.model.layers`.
  In transformers 4.47.1, `Qwen2DecoderLayer` returns a tuple `(hidden_states, ...)`,
  the same shape the steering hook handles for GPT-2.
- **dtype.** The steering vector is cast to the model's dtype inside the hook.
- **Embedder.** The embedding-region path does not use the LLM at all. Its embedder
  is set per target.

**What to re-tune.** These are numbers, not code:

- **Steering layer and alpha.** Qwen2.5-7B has 28 layers and d_model 3584 (distilgpt2
  has 6 and 768). Start around the middle layers. Alpha is in units of the raw
  mean-difference vector, so its optimum moves (see finding C). On both models tried,
  the best alpha put the injected vector at about 0.3 to 0.6x the typical residual norm
  at that layer, and output broke down near 1x. Use that as the starting grid.
- **GCG budget.** On Qwen2.5-0.5B-Instruct, 250 steps at search width 256 reached loss
  1.26 with 5/6 target tokens, not an exact match (distilgpt2: exact by 250). Plan more
  steps and/or a wider search for 7B. 7B in fp16 is about 15 GB on a 24 GB A5000.
  nanoGCG shrinks the batch on OOM by itself, but setting `batch_size` explicitly avoids
  the retries.
- **dtype.** The Qwen config's native dtype is bfloat16, which Ampere (A5000) supports.
  fp16 as specified should work; switch to `--dtype bfloat16` if you see overflow/NaN.
  On CPU, use float32: bf16 runs without NaN there, but its argmax disagrees with fp32
  on 6 to 11% of prompt positions.
- **Do not upgrade transformers.** nanoGCG passes its prefix KV cache as legacy tuples.
  On Qwen, transformers 4.47.1 warns that this format "will be removed in v4.47" but
  still handles it correctly (evaluator loss = nanoGCG loss).

**Instruct-model handling** (implemented and checked on Qwen2.5-0.5B-Instruct):

- `ActivationSteeringAttack(chat_template="auto")` detects a model-provided chat template
  (nanoGCG's fallback for base models does not count). In chat mode, each contrastive
  text is rendered as the assistant reply and only its own tokens are averaged (picked
  via offset mapping). Test prompts are user turns.
- `repetition_penalty` defaults to 1.0 in chat mode and 1.3 for base models. 1.3 was
  needed against distilgpt2 loops but distorted the Qwen baseline.
- Transfer check before RunPod: `runs/e2e_qwen_small_cpu.log` (see "Transfer check"
  below). Nothing has run on a GPU or on the 7B model yet.

## Findings so far (distilgpt2 / all-MiniLM-L6-v2, CPU)

### A. nanoGCG's `early_stop` returns the wrong candidate

With `early_stop=True`, nanoGCG 0.3.0 stops when *any* candidate in the batch matches
every target token (argmax). It then keeps the *argmin-loss* candidate of that batch,
which need not be the matching one. On `TEST_odpowiedz_testowa` the run stopped at step 79
with loss 0.67 and generated `TEST_od_dwight_testowa` (no match). With `early_stop=False`
and 250 steps, loss fell 7.29 to 0.25 and the greedy output starts exactly with the
target (token accuracy 1.00; without the suffix: 0.18). The demo scripts therefore
default to `early_stop` off. Logs: `runs/phase2_gcg*.log`.

### B. Detectability cost: perplexity vs. distance to the region

Payload: an unrelated office notice. Region: three sourdough sentences. Embedder:
all-MiniLM-L6-v2. Perplexity measured with distilgpt2. The attack itself was not changed.

| passage | cosine distance to centroid | perplexity |
|---|---|---|
| payload only | 0.911 | 133 |
| payload + centroid text (naive concatenation) | 0.442 | 90 |
| payload + gradient-optimised 16-token span | **0.173** | **2874** |
| (neutral-init control, optimised) | 0.175 | 3573 |
| (the region's own reference texts) | 0.175 to 0.225 | |

Naive concatenation is *more* fluent than the payload but only gets halfway to the region.
The gradient gets the poisoned passage closer to the centroid than two of the three texts
that define the region. It pays for that with about 22x the payload's perplexity, a
signal a simple perplexity filter would catch. The neutral-init control shows the
gradient, not the starting point, does almost all the work.
Logs: `runs/phase3_embedding_*.log`.

### C. Activation steering has a non-monotonic optimum ("more is not better")

Formal-vs-informal direction at layer 3 of distilgpt2. Outputs are re-read by the
*unsteered* model and projected on the direction:

| alpha | 0 | 2 | 4 | 8 |
|---|---|---|---|---|
| output projection | -4.14 | **7.16** | 3.57 | 0.91 |

The reference levels are 4.90 for the formal prompts and -15.99 for the informal ones.
The projection peaks at alpha = 2, above the formal reference prompts themselves, and
then falls. Stronger injection degenerates the text into formal keyword lists ("Government
Public Information ... Committee Parliamentary") that no longer read as the target
category. The injection strength has an interior optimum that has to be searched per
model and layer; maximising alpha is counter-productive. Log: `runs/phase4_steering.log`.

### Transfer check: Qwen2.5-0.5B-Instruct (CPU, fp32)

This run uses the same dispatcher, seeds and settings as above. Steering runs in chat mode
at layer 12 of 24. Log: `runs/e2e_qwen_small_cpu.log`.

| path | distilgpt2 | Qwen2.5-0.5B-Instruct |
|---|---|---|
| GCG best loss / token acc / exact | 0.25 / 1.00 / yes | 1.26 / 0.83 / **no** (output `TESTED`) |
| GCG loss at step 100 | 0.32 | 2.13 |
| Embedding distance payload → attack | 0.911 → 0.173 | identical: this path does not use the LLM |
| PPL payload / naive / attack | 133 / 90 / 2874 | 80 / 50 / 1700 (same ratios) |
| Steering projection, alpha 0 / 1 / 2 / 4 / 8 | -4.14 / – / **7.16** / 3.57 / 0.91 | 1.62 / 2.27 / **2.33** / 1.08 / -0.12 |
| Steering formal / informal reference | 4.90 / -15.99 | 2.16 / -2.49 |

- **GCG converges far more slowly on a chat-tuned model.** Without a suffix, Qwen refuses
  the bare prompt.
- **The non-monotonic steering optimum reproduces.** On Qwen the baseline already leans
  formal and the peak only just clears the formal reference. The collapse is sharper and
  takes a different form: repetition loops and stray Chinese tokens instead of keyword
  lists.

## How this differs from AdversariaLLM and LLMart

The claims below come from our earlier literature check of both frameworks. They were not
re-verified in this repository; check the current versions before citing.

| capability | AdversariaLLM | LLMart | latent-forge |
|---|---|---|---|
| Discrete-token optimisation (GCG-style) on the attacked LLM | yes | yes | yes, via nanoGCG |
| Target = region in an **external** embedder (RAG / memory retrieval poisoning) as a built-in attack | no | no | **yes** (`EmbeddingRegionAttack`) |
| Activation steering (residual-stream direction) as a first-class attack | no | no | **yes** (`ActivationSteeringAttack`) |
| "Goal → automatic strategy selection" layer | no | no | **yes** (`dispatcher.py`) |
| Detectability metric next to success metric | | | perplexity of embedding-region payloads |

Today the dispatcher is a type-based registry. The research contribution is the
interface: one goal spec covering output, embedder region and activation direction, with
strategy selection behind it. Richer selection policies plug in at
`select_attack_class`.

## Reproducibility

- Exact pins are in `requirements.txt`: torch 2.15.0, transformers 4.47.1 (the
  nanoGCG 0.3.0 ceiling), nanogcg 0.3.0, sentence-transformers 5.7.0, pytest 9.2.0.
  The full freeze is in `requirements-lock.txt`.
- Every number above comes from a run whose log/JSON is in `runs/`. All runs are seeded
  (GCG seed 0, embedding attack seed 0, steering is greedy decoding).
- The differentiable embedder path (`inputs_embeds` through the real SentenceTransformer
  pipeline) and the steering-hook semantics are each pinned by a test against the library
  behaviour they rely on.

## Region-targeted GCG (research line, Phases 0-4b, closed)

**Research question:** Czy region-targeted GCG daje transferowalny, dostarczany jako zwykly
tekst (blackbox) odpowiednik activation steeringu - czyli efekt podobny do wstrzykniecia
kierunku w aktywacjach, ale bez potrzeby dostepu white-box w momencie ataku.

This is not an attempt at "a better GCG". It is a different capability with a different
trade-off. Design and the nanoGCG decision: `docs/region_gcg_design.md` (nanoGCG 0.3.0 has
no custom-loss hook, so `GCGRegionAttack` uses its own GCG-style loop). Topics are neutral
only (cooking).

`HiddenStateRegionTarget(reference_texts, layer, prompt)` routes through the same
dispatcher to `GCGRegionAttack`. The attack optimises a 20-token suffix after an unrelated
prompt ("Yesterday the city council met to discuss") so that the block-`layer`
last-token state is close (cosine) to the centroid of the reference texts' states.

Every run reports **two independent measures**: (1) the attack's own hidden-state cosine
distance, and (2) a topical score of the generated continuation from a separate model
(all-MiniLM-L6-v2 distance to the topic) plus a hand-written keyword list.

| distilgpt2, layer 3 | (1) hidden cos dist | (2) MiniLM topic dist | (2) keyword frac |
|---|---|---|---|
| no suffix | 0.305 | 1.002 | 0.00 |
| random suffixes (5 seeds) | 0.33 to 0.38 | 0.98 to 1.07 | 0.00 |
| naive topical suffix (reference text, same length) | 0.229 | 0.403 | 0.11 |
| **optimised suffix** | **0.118** | 0.418 | 0.25 (2 hits / 9 words) |
| (reference texts themselves) | 0.157 to 0.188 | 0.17 to 0.23 | |

Findings so far (one prompt, one seed, toy model; log: `runs/region_gcg_phase1.log`):
- **Threshold, not gradual.** Measure (1) fell 40% (0.37 → 0.23) before measure (2)
  moved at all. Topicality switched on abruptly at step 5.
- **Measures decouple after the threshold.** Hidden distance kept falling (0.157 → 0.118)
  while the topic distance oscillated (0.42 to 0.71) and the continuation shrank to 9
  words.
- **Closer than plain topical text in hidden space, not more topical.** The optimised
  continuation announces a recipe instead of containing cooking content.
- **Partly lexical.** The suffix contains topical sub-word fragments ("redients",
  "olive"), so the effect is not purely steering-like.

### Phases 2 and 3: band objective, lexical blocking, semantic blocking

Same prompt, seed 0, 200 steps. Full table, suffixes, continuations and trajectories:
`runs/region_gcg_phase3.log`.

| run | (1) hidden | (2) MiniLM | (2) keywords |
|---|---|---|---|
| naive topical sentence (control) | 0.229 | 0.403 | 4/37 |
| Phase 1: minimise | 0.118 | 0.418 | 2/9 |
| 2A: band \|d − r\|, r = 0.172 | 0.172 | 0.568 | 3/33 |
| 2B: minimise, 990 list-blocked tokens | 0.110 | 0.687 | 4/36 |
| 3S: minimise, 6319 semantically blocked tokens | 0.121 | 0.415 | 1/11 |
| **3SA: band, 6319 semantically blocked tokens** | **0.172** | **0.895** | **0/30** |

- **Semantic block.** A token is blocked when a separate encoder (MiniLM) places it closer
  to cooking than to all 7 control topics. That is 5675 tokens, 11.3% of the vocabulary;
  "the", "a", "and" and "council" stay allowed. The Phase 2 subword list is added on top,
  because single fragments like "redients" are not recognised on their own. distilgpt2's
  own embedding matrix was rejected as the similarity space: its nearest tokens to the
  topic are function words.
- **The band reached without topic tokens does nothing for topicality (3SA).** The
  continuation stays about the city council.
- **Topicality survives the semantic block only through compositional surface cues (3S).**
  Recipe-instruction words that are not cooking-specific one by one ("Heat", "thinly",
  "mins", "repeat") read as a recipe together. It also appears only from about step 100,
  long after the state is already inside the band.

### Answer to the research question (for this setup)

**No evidence was found that region-targeted GCG delivers, as plain text, a steering-like
effect independent of the text's surface.** On distilgpt2 (layer 3, last-token cosine
objective, one prompt, one seed, one topic) the effect was predominantly lexical:

- Reaching the natural hidden-state band without topic-specific tokens did not produce a
  topical continuation.
- The Phase 2 observation that topicality switches on at band entry held only while
  topical tokens were allowed.
- Hidden-state proximity at one position is necessary but not sufficient. In every run
  it could be driven below the reference range while the continuation stayed off-topic.

This is a documented negative result for a toy setting, not a general one. A larger
model, a later layer, or an objective over several positions or the continuation itself
could behave differently; none of these were tested.

### Phases 4 and 4b: readability term (AutoDAN-style) and the start point

Phase 4 adds a readability term to the same loop: `J = d + w · CE_suffix`, where CE is
distilgpt2's own NLL of the suffix in context, and half of the top-k candidates come
from the model's next-token log-probs (AutoDAN-like). The objective stays `min`, because
`band` would hide the overshoot being tested. Phase 4b repeats Phase 4 exactly but starts
from a fluent, off-topic 20-token sentence (" the plan to repair the old bridge near the
public library and to add more street lights in the park") instead of "x x x". Measure
(3) is suffix perplexity under an independent model, Qwen2.5-0.5B-Instruct. Natural
range of the reference texts: d ∈ [0.157, 0.188].

| run | (1) hidden | (2) MiniLM | (2) kw | (3) PPL Qwen | distinct tokens | swaps (last at step) |
|---|---|---|---|---|---|---|
| Phase 1, x start, w = 0 | 0.118 | 0.418 | 0.25 | 97 960 | 1.00 | 78 (124) |
| 4, x start, w = 0.01 | 0.135 | 0.699 | 0.16 | 398 | 0.50 | 26 (35) |
| 4, x start, w = 0.03 | 0.151 | 0.513 | **0.33** | 43 | 0.30 | 17 (28) |
| 4, x start, w = 0.1 | 0.245 | 0.977 | 0.00 | 9.7 | 0.15 | 4 (9) |
| 4b, fluent start, w = 0.01 | 0.141 | 0.943 | 0.00 | 605 | 0.85 | 33 (42) |
| 4b, fluent start, w = 0.03 | 0.156 | 0.939 | 0.03 | **35** | 0.70 | 32 (38) |
| 4b, fluent start, w = 0.1 | 0.276 | 1.007 | 0.00 | 17 | 0.75 | 22 (45) |

Example (w = 0.03). x start: `' this and x x x ... x Salad Onion mixture'` → a salad
ingredient list (9 keyword hits). Fluent start: `' a proposal to add the same name to the
two recipes and to add the same flavor of sweet potatoes'` → the continuation stays with the
city council.

- **Readability works on its own terms.** The weight cuts suffix perplexity by 2 to 4
  orders of magnitude. With the x start that is mostly the repeated filler surviving
  (repetition has near-minimal CE). With the fluent start the suffixes are real sentences
  (Qwen PPL 35 at w = 0.03, below the naive topical sentence's 256).
- **Overshoot shrinks with w, independent of the start.** w = 0.03 lands at the lower edge
  of the natural range in both cases (0.151 and 0.156); w = 0.1 does not reach the region.
- **The fluent start explores more but still stalls.** 22 to 33 accepted swaps instead of
  4 to 26, and only 6 to 8 of the 20 start tokens survive (vs 11 to 18). Every run with
  w > 0 still stops improving before step 50: single-token swaps hit a local minimum, so
  the stall is not only the filler attractor.
- **Topicality depends on the start point, not on the hidden distance.** At practically
  the same d (0.151 vs 0.156) the x start gives the most topical continuation of the whole
  line and the fluent start gives none. The optimiser still uses cooking words in both
  ("Salad Onion"; "recipes", "sweet potatoes", "skillet", "cheese"). After a meaningless
  filler they decide the continuation. Inside a fluent civic sentence they become objects
  of a council proposal, and the continuation follows the sentence frame.

### Answer to the research question, whole line (Phases 0-4b)

The Phase 3 answer stands and Phase 4b sharpens it. On this setup, region-targeted GCG is
**not** a plain-text equivalent of activation steering. What moves the continuation is
lexical and contextual: topic words, and where they sit in the text. Proximity of one
last-token state to the region at layer 3 is neither sufficient for topicality (3SA, 4b)
nor the thing that decides it: the same distance gives a topical or an off-topic
continuation depending on the surrounding surface text. Adding readability does not
change this. It makes the suffix fluent and reduces overshoot, and with a fluent start it
also removes the topical effect.

Scope: distilgpt2, layer 3, one prompt, one seed, one neutral topic, 200 steps. A larger
model, later layers, a multi-position objective or more seeds were not tested. The line
is closed here.

Line summary: `docs/region_gcg_design.md` (no custom loss in nanoGCG → own loop; Phase 4
design) · `runs/region_gcg_phase1.log` (attack works on the metric; measures diverge) ·
`runs/region_gcg_phase2.log` (band and list block; inconclusive) ·
`runs/region_gcg_phase3.log` (semantic block; lexical) ·
`runs/region_gcg_phase4_autodan.log` (readability; low PPL via the filler) ·
`runs/region_gcg_phase4b_fluent_init.log` (fluent start; topicality follows the surface
frame, not the hidden distance).
