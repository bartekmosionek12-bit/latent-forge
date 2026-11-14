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
  mean-difference vector, so its optimum moves (see finding C).
- **GCG `search_width` / `batch_size`.** 7B in fp16 is about 15 GB on a 24 GB A5000.
  nanoGCG shrinks the batch on OOM by itself, but setting `batch_size` explicitly avoids
  the retries.
- **dtype.** The Qwen config's native dtype is bfloat16, which Ampere (A5000) supports.
  fp16 as specified should work; switch to `--dtype bfloat16` if you see overflow/NaN.

**Known gaps for an instruct model** (not blocking; worth deciding before the Qwen runs):

- `ActivationSteeringAttack` builds directions and generates from **raw text without
  the chat template**. For Qwen-Instruct you probably want contrastive prompts and test
  prompts wrapped in the template.
- Nothing has been executed on Qwen yet. Everything above is checked against the
  installed library source and the real Qwen tokenizer/config, not against a GPU run.

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
