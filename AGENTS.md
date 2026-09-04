# AGENTS.md — GlassMind

## Mission

Build **GlassMind**, an experimental, highly efficient, non-Transformer language model whose internal computation can be inspected as a live visual network.

The project is not "a Transformer with a different UI". The model architecture, instrumentation, training code, tests, replay system, and visualization must be designed together.

Primary goals, in order:

1. Trainable language modeling without a Transformer architecture.
2. Efficient inference with near-linear scaling in sequence length and bounded recurrent state where practical.
3. A real-time visual network that reflects real model activity rather than decorative or post-hoc explanations.
4. Reproducible experiments with clear logs and automated tests.
5. Simple architecture iteration so components can be replaced without rewriting the whole project.

Codex may download suitable public datasets from Hugging Face and may add helper scripts for dataset preparation, benchmarking, profiling, visualization, replay, and experiments.

---


## Sprache und Kommunikation

Alle Erklärungen für den Benutzer müssen auf **Deutsch** geschrieben werden. Das gilt insbesondere für:

- Statusmeldungen und Fortschrittsberichte
- Zusammenfassungen nach Änderungen
- Begründungen für Architekturentscheidungen
- Fehleranalysen und Testergebnisse
- README- und sonstige benutzerorientierte Dokumentation
- Hinweise zur Benutzung von Skripten und Befehlen
- Kommentare in Experimentberichten und Benchmark-Zusammenfassungen

Technische Bezeichner, Klassen-, Funktions- und Variablennamen dürfen auf Englisch bleiben, wenn das für Codequalität und übliche Konventionen sinnvoll ist. Auch externe Fehlermeldungen dürfen im Original zitiert werden, müssen danach aber auf Deutsch erklärt werden.

Codex soll nicht unnötig englische Fachtexte ausgeben. Wenn ein englischer Fachbegriff wichtig ist, darf er genannt werden, sollte aber beim ersten Auftreten kurz auf Deutsch erklärt werden.

---

## Non-negotiable architecture rules

### No Transformer core

Do not implement a standard Transformer stack.

Disallowed as the primary sequence mechanism:

- full self-attention
- causal multi-head self-attention
- quadratic token-to-token attention matrices
- a lightly renamed Transformer block

Small local operations are allowed when they are not functioning as a hidden full-attention replacement.

The preferred direction is a custom recurrent/selective-state architecture inspired by ideas from modern state-space models, gated recurrent systems, sparse routing, and explicit memory, while remaining an original implementation.

### Complexity target

The architecture should aim for:

- O(n) sequence processing in inference where practical
- bounded or slowly growing recurrent state
- no O(n²) attention cache
- streaming token-by-token inference
- efficient batched training
- GPU-friendly tensor operations

Do not sacrifice correctness merely to claim O(n). Measure actual throughput and memory use.

### Recommended initial core

Start with a compact architecture built from these concepts:

```text
Token
  ↓
Embedding
  ↓
Local Mixer
  ↓
Selective Recurrent State Core
  ├─ Fast State
  ├─ Context State
  └─ Semantic State
  ↓
Sparse Memory Read / Write
  ↓
Sparse Expert Router
  ↓
State Integrator
  ↓
LM Head
  ↓
Next Token
```

This is a starting hypothesis, not an immutable design. Codex may simplify, replace, or improve components if benchmarks and tests justify the change.

### State hierarchy

Prefer several state timescales instead of one giant opaque hidden state.

Suggested first implementation:

- `fast_state`: short-lived local/token dynamics
- `context_state`: sentence/paragraph/topic dynamics
- `semantic_state`: slower persistent features

Each state must expose measurable activity and state-change telemetry.

### Explicit memory

Implement a small explicit memory system only after the basic recurrent core can overfit a tiny dataset.

Memory should be sparse and bounded.

Suggested properties per memory slot:

- value vector
- strength
- age
- read count
- write count
- last touched step

Prefer Top-K or similarly sparse access. Do not accidentally recreate quadratic global attention.

### Sparse experts

Experts are optional in the first milestone.

When added:

- use a small number of experts
- activate only a small subset per token
- log router probabilities and selected experts
- monitor expert collapse and load imbalance

The model must discover expert specialization itself. Do not hard-code "math expert", "language expert", etc.

---

## Observable-by-design requirement

Observability is part of the architecture, not an afterthought.

Every important module must be able to emit structured telemetry through a common observer interface without changing its numerical result.

The model must run normally when observation is disabled.

### Observation bus

Implement a lightweight event/telemetry system such as:

```text
Model Core
  ├─ normal tensors → next module
  └─ optional telemetry → Observation Bus
                           ├─ console summary
                           ├─ JSONL recorder
                           ├─ replay files
                           ├─ test assertions
                           └─ VisPy visualizer
```

Do not couple VisPy directly into model forward passes.

### Minimum telemetry

Where meaningful, expose:

- token index and token id
- module/layer id
- state norm
- state delta norm
- activation mean/std/min/max
- activation sparsity
- top active units or clusters
- gate values
- memory reads
- memory writes
- router weights
- selected experts
- routed flow magnitude
- output entropy
- top token logits/probabilities
- NaN/Inf detection

Full tensor dumps must be opt-in only.

### Activity score

A visible node must not be considered "active" merely because its absolute activation is large.

Create an activity score that can combine, depending on module type:

- absolute activation
- change from previous token/state
- incoming flow magnitude
- outgoing flow magnitude
- gate activity
- memory read/write activity
- optional causal/intervention importance in analysis mode

Keep the exact formula configurable and log its components.

---

## Visual network requirement

The primary visualization is a **living neural network**, not a conventional dashboard.

Use VisPy unless profiling demonstrates a better suitable renderer.

### Visual semantics

The display should represent real components and real measured activity.

Suggested mapping:

- node = model unit, state cluster, memory slot, expert, or higher-level module
- edge = real information-flow relationship
- node intensity = current activity score
- node size = activity magnitude or selected metric
- edge intensity/width = measured flow magnitude
- fading = recent historical activity
- pulse/outline = strong state delta
- distinct marker state = memory reactivation

Avoid decorative fake connections.

### Hierarchical level of detail

Never try to draw every parameter at once.

Use hierarchical visualization:

```text
Model
  → blocks
    → clusters
      → units
```

At normal zoom, show aggregated clusters.

On selection/zoom, reveal deeper detail.

The renderer should remain responsive even when the model contains millions of parameters.

### Stable node identity

Every visual node must have a stable identifier across steps and replay sessions.

Example:

```text
core.3.context.cluster.17
memory.slot.42
expert.5
```

### Replay

Recorded traces must be replayable without rerunning the model.

Required workflow:

```bash
python -m glassmind.infer "Alice owns a red car." --record runs/demo/trace.jsonl
python -m glassmind.visualize --replay runs/demo/trace.jsonl
```

The visualizer should support token-by-token stepping, pause/play, and selection of nodes.

### Intervention mode

After the basic model works, add experimental interventions:

- zero/mute a state cluster
- freeze a state
- disable a memory slot
- disable an expert
- amplify or suppress a selected cluster

Compare output logits before and after intervention.

This is more valuable than pretending that activation alone is an explanation.

---

## Efficiency requirements

Efficiency is a core research goal.

### Baseline target

Start small enough that development is fast on a consumer GPU.

Suggested initial range:

- 10M to 50M parameters
- vocab 8K to 16K
- state width roughly 256 to 512
- 4 to 8 recurrent blocks
- context test lengths from 256 to 4096 tokens

Do not scale parameter count until the tiny model passes functional tests.

### PyTorch implementation

Prefer:

- vectorized tensor operations
- fused operations where they materially help
- AMP/bfloat16 or float16 when safe
- `torch.compile` behind a feature flag if stable
- efficient contiguous layouts
- minimal Python work inside token loops
- batched recurrence during training where possible
- optional custom CUDA/Triton only after profiling proves it matters

Do not begin with custom kernels.

First establish a correct reference implementation.

### Telemetry overhead

Observation must be configurable:

- `off`: minimal overhead
- `summary`: cheap aggregate telemetry
- `trace`: detailed selected telemetry
- `full`: expensive debugging only

Do not transfer large tensors from GPU to CPU every token in normal mode.

Aggregate on GPU where practical and transfer only compact summaries.

Use buffered/asynchronous logging where safe.

Measure observer overhead separately from model throughput.

### Benchmark both modes

Always benchmark:

1. model with observation disabled
2. model with summary observation
3. model with detailed trace observation

The visual/debug tooling must never hide the true model performance.

---

## Training data

Codex may obtain suitable public datasets from Hugging Face.

Prefer datasets that are:

- legally/publicly available for the intended experimentation
- easy to stream
- small enough for early iteration
- suitable for language-model training

Use progressively harder data.

Recommended progression:

1. generated synthetic sequences
2. tiny text dataset
3. TinyStories or another compact language dataset
4. larger general text only after architecture stability

Dataset code must support streaming or chunked preprocessing where possible.

Do not require the whole corpus to fit in RAM.

Record dataset name, revision if available, preprocessing settings, tokenizer settings, and split in every training run.

---

## Tokenizer

Keep tokenizer handling modular.

For early experiments, use a proven tokenizer library or a simple BPE tokenizer rather than making tokenizer research part of the architecture research.

Tokenizer metadata must be saved with checkpoints.

---

## Project structure

Prefer a clean package layout similar to:

```text
glassmind/
├─ glassmind/
│  ├─ model/
│  │  ├─ embedding.py
│  │  ├─ local_mixer.py
│  │  ├─ state_core.py
│  │  ├─ memory.py
│  │  ├─ experts.py
│  │  ├─ integrator.py
│  │  └─ lm.py
│  ├─ observe/
│  │  ├─ events.py
│  │  ├─ bus.py
│  │  ├─ metrics.py
│  │  ├─ recorder.py
│  │  └─ replay.py
│  ├─ visualize/
│  │  ├─ app.py
│  │  ├─ graph.py
│  │  ├─ layout.py
│  │  └─ activity.py
│  ├─ data/
│  ├─ training/
│  ├─ inference/
│  └─ utils/
├─ scripts/
├─ tests/
├─ configs/
├─ runs/
├─ benchmarks/
├─ AGENTS.md
└─ README.md
```

Exact structure may evolve when there is a concrete reason.

---

## Required scripts

Create simple scripts or CLI commands for common work.

At minimum:

```bash
python scripts/smoke_test.py
python scripts/overfit_test.py
python scripts/memory_test.py
python scripts/routing_test.py
python scripts/benchmark.py
python scripts/train_tiny.py
python scripts/infer.py "some text"
python scripts/infer.py "some text" --trace
python scripts/visualize.py --replay <trace>
```

Equivalent `python -m glassmind...` commands are also acceptable.

Scripts must fail loudly and return non-zero exit codes when a test fails.

---

## Test ladder

Do not jump directly to large language training.

### Stage 0 — numerical sanity

Test:

- forward pass shapes
- backward pass
- gradients exist
- no NaN/Inf
- deterministic seeded smoke test
- save/load checkpoint
- streaming recurrent inference agrees with equivalent reference path where applicable

### Stage 1 — tiny overfit

Overfit a tiny synthetic/text dataset.

If the model cannot reliably overfit a tiny dataset, do not scale it.

### Stage 2 — sequence patterns

Test copying, delayed recall, repeating sequences, bracket-like dependencies, and simple next-token patterns.

### Stage 3 — associative memory

Examples:

```text
Alice owns the red car.
Bob owns the blue car.
Question: What color is Alice's car?
```

Measure whether the model can recover the association and whether expected memory/state regions reactivate.

### Stage 4 — tiny natural language

Train on a compact public dataset and track perplexity/loss and generation quality.

### Stage 5 — long-context efficiency

Benchmark sequence lengths such as:

- 256
- 512
- 1024
- 2048
- 4096

Record tokens/sec, VRAM, wall time, and telemetry overhead.

### Stage 6 — interventions

Verify that muting/altering selected internal components can measurably change predictions and that the before/after run is reproducible.

---

## Benchmark policy

Never claim the architecture is "efficient" without measurements.

Every benchmark output should record:

- model config
- parameter count
- dtype
- device/GPU
- batch size
- sequence length
- training or inference
- observation mode
- tokens/sec
- peak VRAM
- latency/token for streaming inference
- compile mode

Where useful, include a simple baseline such as an LSTM/GRU or small Transformer solely for performance/reference comparison. The GlassMind core itself must remain non-Transformer.

---

## Logging requirements

Logs must be understandable by a human.

Do not flood the console with raw framework spam.

Use two layers:

### Human-readable console log

Example:

```text
[run 0042] step 1200/10000  loss=3.182  tok/s=48,210  vram=3.7GB
[state] fast=4.12  context=7.84  semantic=2.91  dead=1.8%
[memory] reads=2  writes=1  active_slots=17/64
[router] top=E2:0.71,E5:0.22  entropy=0.48
[health] grad_norm=0.91  nan=0  inf=0
```

### Structured log

Use JSONL or another simple structured format for machine processing and replay.

Every run must have a unique directory and save:

```text
runs/<run-id>/
├─ config.*
├─ environment.json
├─ metrics.jsonl
├─ train.log
├─ checkpoints/
├─ traces/
└─ summary.md
```

At the end of a run, generate a concise `summary.md` containing:

- result PASS/FAIL
- architecture/config
- parameter count
- final/best loss
- throughput
- peak memory
- state statistics
- dead/saturated units
- memory utilization if enabled
- router utilization if enabled
- NaN/Inf status
- notable warnings

---

## Reproducibility

Every experiment should capture when available:

- random seed
- git commit
- Python version
- PyTorch version
- CUDA version
- GPU name
- model config
- optimizer config
- dataset metadata
- tokenizer metadata

Config must be sufficient to rerun the experiment.

---

## Checkpoints

A checkpoint must contain enough information to resume training and run inference.

Include:

- model state
- optimizer state when training checkpoint
- scheduler state when used
- step/epoch
- architecture config
- tokenizer reference/metadata
- format version

Use explicit format versioning so future architecture changes can be detected cleanly.

---

## Development rules for Codex

### Work autonomously

Do not ask for approval for routine implementation decisions.

When multiple reasonable options exist:

1. choose the simplest measurable implementation
2. document the choice briefly
3. implement it
4. test it
5. benchmark it when performance-relevant

### Keep the project runnable

After meaningful changes:

- run relevant tests
- fix regressions before continuing
- keep at least one tiny end-to-end configuration working

### Favor evidence over architectural enthusiasm

If an idea performs poorly, simplify or remove it.

Do not protect a component merely because it sounded interesting.

### Profile before optimization

Do not add Triton/CUDA/native extensions until profiling identifies a real bottleneck.

Maintain a slow/simple reference path when introducing highly optimized kernels so correctness can be compared.

### No fake interpretability

Never label a latent cluster as a semantic concept solely because it visually appears correlated.

Use language such as:

- cluster 17 strongly correlates with X in probe dataset
- intervention on cluster 17 changes Y

Do not claim "the model thinks X".

### No fake visualization

Every visible activity, edge, cluster statistic, or memory event must be derived from real model telemetry.

Synthetic/demo visualization data is allowed only in an explicitly named visualization demo/test mode.

### Keep dependencies reasonable

Avoid large dependency chains when a small library or built-in implementation is sufficient.

Document new dependencies and why they exist.

---

## First implementation milestone

The first useful milestone is deliberately small.

Deliver:

1. project scaffold
2. tokenizer/data loader
3. a small selective recurrent language-model core with no Transformer attention
4. `fast_state`, `context_state`, and optionally `semantic_state`
5. observation bus with `off`, `summary`, and `trace` modes
6. JSONL trace recording
7. synthetic data generator
8. smoke test
9. tiny overfit test
10. streaming inference CLI
11. basic VisPy network showing real cluster/node activity
12. replay of a recorded inference trace
13. benchmark script
14. concise run summaries

Do not add experts and complex external memory until this milestone passes reliably.

---

## Second milestone

After milestone 1 is stable:

1. add bounded sparse memory
2. add memory-specific tests
3. visualize read/write/reactivation
4. add sparse experts only if useful
5. add intervention experiments
6. add Hugging Face natural-language training preset
7. compare throughput and memory against simple reference baselines
8. profile and optimize measured hotspots

---

## Success criteria

GlassMind v0 is successful when all of the following are true:

- it trains end-to-end as a causal language model
- its primary sequence core is not a Transformer
- streaming inference works with bounded/recurrent state
- tiny overfit and synthetic recall tests pass
- activity is visible as a real network in VisPy
- a recorded trace can be replayed without running the model
- observation can be disabled with low overhead
- performance is measured rather than guessed
- logs make failures understandable
- experiments can be reproduced from saved run metadata

The immediate research question is not whether GlassMind beats frontier Transformers.

The immediate question is:

> Can a language model be made efficient, recurrent, observable, replayable, and experimentally manipulable without turning its visualization into a fake explanation layer?
