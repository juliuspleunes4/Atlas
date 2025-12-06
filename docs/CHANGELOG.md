# Changelog

All notable changes to Atlas will be documented in this file.

---

## 2025-12-06 - Project foundation complete (Phase 0)

- Set up complete package structure with `atlas/` and all subdirectories (model, tokenizer, training, data, config, inference, utils, export)
- Created `tests/` directory with test file stubs for all modules
- Created `scripts/` directory for CLI scripts
- Added `requirements.txt` with core dependencies (PyTorch, NumPy, tiktoken, tqdm, pyyaml, pytest, gguf)
- Updated `.gitignore` for Atlas-specific patterns (checkpoints, logs, .gguf files)
- Created `setup.py` for package installation
- Updated `README.md` with project overview, structure, and quickstart guide

## 2025-12-06 - Initial roadmap created

- Created comprehensive `docs/ROADMAP.md` breaking down the entire LLM development process
- Roadmap covers 10 phases from project foundation through documentation and release
- Includes detailed tasks for model architecture, training, inference, and GGUF export
- Defines success criteria: trained model → coherent text generation → working GGUF export
