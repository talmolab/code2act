# bash_scripts

Orchestration scripts for the three-process Code2Act pipeline (tokenizer fit →
code generation → decoder RL), grouped into subfolders:

- `pipeline/` — full multi-stage orchestration
- `sweeps/` — tokenizer / KPMS hyperparameter sweeps
- `training/` — single decoder / controller trainers

`_pyrun.py` (venv module launcher) and this README stay at the `bash_scripts/`
root. Run every script from the `code2act/` repo root.

**Full documentation:** [`docs/bash_scripts.md`](../docs/bash_scripts.md).
