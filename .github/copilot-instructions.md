# copilot-instructions.md

These instructions are for GitHub Copilot (and Copilot Agents) working on **Atlas**, a from-scratch LLM project.

---

## 1. General principles

- Build **Atlas** as a **scalable, modular, well-organized** codebase.
- Prefer **small, focused tasks** over big multi-feature changes.
- Never assume requirements; use clear TODO markers where needed.
- Follow **clean code** and **industry best practices** at all times.

---

## 2. Scope of work

Copilot is allowed to:

- Implement and refactor **core LLM components**:
  - Model architecture (Transformer blocks, attention, etc.)
  - Tokenizer integration
  - Training loop, evaluation loop
  - Checkpointing, logging, configuration system
  - Inference code (CLI / server)
- Add / improve:
  - Tests
  - Documentation inside existing `.md` files (except this rule set)
  - Typed, documented utilities (config, IO, dataset handling)

Copilot is **not** allowed to:

- Create **any new `.md` files**.
  - Only edit existing markdown files (e.g. `README.md`, `docs/CHANGELOG.md`, this file) when relevant.
- Introduce **mock data** or fake datasets.
  - Use **placeholders** + `TODO:` comments instead.
- Implement features that depend on external services without clear TODOs and abstraction.

---

## 3. Task granularity & workflow

When working on a task:

1. **Pick one small, well-defined unit of work**, for example:
   - Implement a single module or class.
   - Add tests for one module or function.
   - Introduce one configuration feature.
   - Optimize a specific part of the training loop.
2. Implement it **end-to-end**, including:
   - Code
   - Tests (if applicable)
   - Documentation updates (if needed)
   - `docs/CHANGELOG.md` entry
3. Keep PRs / changesets **narrow** in scope.

Copilot should **not**:

- Try to implement the entire Atlas system in one go.
- Introduce multiple unrelated features in a single change.

---

## 4. Documentation rules

- Never create **new** `.md` files (status reports, summaries, todo lists, etc.).
- Use existing markdown files only:
  - `README.md` – high-level project description and quickstart.
  - `docs/CHANGELOG.md` – **mandatory** update for all notable changes.
  - Any other existing docs the repo already contains.
- For every notable change, add an entry to `docs/CHANGELOG.md`:
  - Include:
    - Date (YYYY-MM-DD)
    - Short title
    - Short description of what changed and why
  - Example style:
    - `2025-12-06 - Add basic TransformerBlock implementation`
      - `Implemented attention + MLP with residuals and layer norm in atlas/model/transformer.py`

---

## 5. Data handling & NO MOCK DATA rule

- **Never** introduce hard-coded example data, fake samples, or mock corpora into the codebase.
- Do not commit any real datasets unless explicitly present already.
- When something depends on data:
  - Implement abstractions (e.g. dataset loaders, interfaces).
  - Use placeholders like:
    - `# TODO: Plug in real dataset source here`
    - `# TODO: Implement actual tokenized dataset pipeline`
- If tests require data:
  - Use synthetic minimal examples (e.g. tiny tensors or strings) that test behavior, not “realistic corpora”.
  - Do **not** include large inline sample texts or files.

---

## 6. Code structure & style

### 6.1 Organization

- Keep code **modular**, with clear package boundaries, for example:
  - `atlas/model/` – core model components (blocks, attention, embeddings, etc.).
  - `atlas/tokenizer/` – tokenizer integration, vocab loading, encoding/decoding.
  - `atlas/training/` – training loop, losses, schedulers, evaluation.
  - `atlas/config/` – configuration schemas, default configs, CLI config parsing.
  - `atlas/inference/` – generation utilities, CLI/server interfaces.
  - `atlas/utils/` – logging, checkpointing, metrics, misc utilities.
- Avoid circular dependencies and “god modules”.
- Keep public APIs minimal; prefer explicit exports.

### 6.2 Style

- Use **Python + PyTorch** for training code (unless repo specifies otherwise).
- Prefer:
  - Type hints (`typing`) everywhere.
  - Clear docstrings (`"""Summary..."""`) on public functions/classes.
  - Small, focused functions and methods.
- Follow PEP8 (or repo-standard) formatting and linting conventions.
- Avoid premature micro-optimizations; first aim for clarity, then measured improvements.

---

## 7. Testing

- Every non-trivial module should have **tests**.
- Add or extend tests when adding features or fixing bugs.
- Tests should:
  - Be **deterministic** and fast.
  - Use small tensors / minimal configurations (tiny models) to verify logic.
  - Validate shapes, masking, training steps, and key behaviors.
- Do not rely on external network access or real datasets in tests.
- If a full integration test is too heavy, add:
  - Unit tests for core components.
  - A minimal “smoke test” that builds a tiny Atlas model and runs a short forward pass.

---

## 8. Configuration & scalability

- Design code and configs so Atlas can scale from:
  - Tiny models (for tests and quick experiments).
  - To larger models (more layers, heads, hidden size) without structural rewrites.
- Use clear configuration objects / files for:
  - Model size (layers, heads, hidden size, vocab size, max sequence length).
  - Training parameters (batch size, LR, warmup, scheduler, etc.).
  - Paths (checkpoints, logs, datasets).
- Avoid hard-coded hyperparameters inside core logic:
  - Read from config objects or arguments.
  - When a value is temporary, use `TODO` and/or config placeholder.

---

## 9. Logging, errors, and robustness

- Use structured logging where appropriate.
- Errors should:
  - Fail fast with clear messages.
  - Indicate invalid configurations or missing data gracefully.
- For critical operations (training loop, checkpoint load/save), handle:
  - Basic error cases (missing files, mismatched configs) with informative exceptions.

---

## 10. Communication & “next task” behavior

When Copilot finishes a task (or a logical unit of work), it should:

- Keep the change focused and coherent.
- Update `docs/CHANGELOG.md` appropriately.
- Not create new `.md` files or add meta-summaries.
- Be ready for a **next, small, well-defined task** from the user (e.g., “now implement X”, “add tests for Y”, “refactor Z”).

Copilot should **never**:

- Start additional large tasks without user direction.
- Restructure the entire codebase unprompted.
- Generate standalone markdown reports of what it did.

---

By following these rules, Copilot will help build Atlas into a clean, scalable, and maintainable LLM codebase with incremental, well-documented progress.
