# Installation & environment setup

Code2Act depends on `track-mjx`, which is **NOT** a PyPI dep — you clone it as a
subfolder and pip-install it into the same venv. This mirrors the track-mjx
install flow verbatim.

## Prerequisites
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- CUDA 12.x or 13.x for GPU (optional)

## Install

```bash
git clone <code2act-repo-url> && cd code2act

# 1. Pull the track-mjx branch that c2a targets
git clone -b kevin/kpms-decoder https://github.com/talmolab/track-mjx.git

# 2. Create the venv at the code2act root (not inside track-mjx/)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
source .venv/bin/activate

# 3. Install track-mjx into this venv (brings in jax, brax, mjx, flax,
#    orbax, hydra, wandb, vnl-playground, mujoco_playground, …)
cd track-mjx
uv pip install -e ".[cuda12]"
cd ..

# 4. Install keypoint_moseq for the KPMS fitting stages
uv pip install keypoint-moseq[cuda] jax-moseq[cuda]

# 5. Install c2a
uv pip install -e .
```

Verify:

```bash
python -c "import jax; print(jax.devices())"
python -c "import c2a, c2a.moseq_ppo, c2a.train_moseq_decoder"
```

## Analysis dependencies (DSA / rSLDS — optional)

The hidden-state analysis scripts under `c2a/figures/plot_hidden_*.py` need two
extra GitHub-only packages: Mitchell Ostrow's **DSA** (NeurIPS 2024) and
Linderman lab's **ssm** (rSLDS). They are declared in `pyproject.toml` under the
`[analysis]` extra, but the install needs sequencing — a plain
`uv pip install -e ".[analysis]"` will fail on `ssm` because that package imports
`numpy` + `cython` at build time without declaring them as build deps. Install
in this order:

```bash
# DSA pulls `torch` as a transitive dep, but we only ever call DSA with
# device="cpu" (see c2a/figures/plot_hidden_dsa.py and
# c2a/experiments/code_grid/run_code_grid_dynamics.py). Install CPU-only torch:
# it fully sidesteps the cu12/cu13 NVIDIA-lib version drift that
# breaks GPU-torch in this venv (symptom: ImportError chain through
# libcusparseLt.so.0 → libnccl.so.2 → libnvshmem_host.so.3 at every
# torch import). JAX/Flax/brax/mujoco_playground don't use torch, so
# the CPU swap doesn't touch the training pipeline.
uv pip install --index-url https://download.pytorch.org/whl/cpu torch

# DSA — uppercase package on GitHub. (PyPI's `dsa` is a different,
# unrelated package; don't `uv pip install DSA` from PyPI.)
uv pip install git+https://github.com/mitchellostrow/DSA.git

# ssm build prerequisites + ssm itself with build-isolation disabled
# so the build sees the venv's numpy and cython.
uv pip install autograd cython
uv pip install --no-build-isolation git+https://github.com/lindermanlab/ssm.git
```

If you previously installed a GPU torch and it's now broken (symptom:
``ImportError: libcusparseLt.so.0`` / ``libnccl.so.2`` /
``libnvshmem_host.so.3 …`` at ``import torch``), swap to CPU torch:

```bash
.venv/bin/python3 -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cpu torch
```

This is non-destructive to the training pipeline (verified: real training
survived a live swap mid-run, 2026-05-14). The only caller of torch is DSA,
which runs on CPU.

If you specifically need GPU torch alongside CUDA-12 JAX (rare — only relevant
if you want CUDA-accelerated DSA), the legacy recipe was:

```bash
# Pin torch to the last CUDA-12 line so uv's resolver doesn't pick
# 2.10+, which depends on CUDA-13 NVIDIA wheels that clobber JAX's
# CUDA-12 cuDNN.
uv pip install "torch<2.10"
```

…but the post-2.10 deprecations have made this brittle; CPU-only is the
recommended path.

**One-line patch for DSA on Python 3.12.** DSA uses `ClassVar[Final]`
(un-parameterised `Final`), which Python 3.12's typing rejects with
`TypeError: Plain typing.Final is not valid as type argument`. Until upstream
fixes it, patch in place:

```bash
sed -i 's/compare: ClassVar\[Final\] = "state"/compare: ClassVar[str] = "state"/' \
    .venv/lib/python3.12/site-packages/DSA/dsa.py
```

Verify everything:

```bash
python -c "from DSA import DSA, InputDSA; print('DSA OK')"
python -c "from ssm import SLDS; print('ssm OK')"
```

You only need this for the `plot_hidden_dsa.py` / `plot_hidden_rslds.py` figure
scripts (under `c2a/figures/`). The training pipeline, sweep, and merge scripts
don't need these packages.

## Reference data

The rodent reference clips (`rodent_reference_clips.h5`,
`rodent_balanced_splits.json`) ship in `data/rodent/`. If you cloned without
these (they're gitignored), pull them from the HuggingFace dataset
`talmolab/MIMIC-MJX`:

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='talmolab/MIMIC-MJX', repo_type='dataset',
    filename='data/rodent/rodent_reference_clips.h5', local_dir='.',
)
"
```

The rodent XML (`c2a/assets/rodent.xml`) is vendored in-repo and is the correct
one — `vnl_playground` ships a different XML with `nq=67` that silently produces
garbage keypoints; c2a refuses to fall back to it.

### Behavior labels

`data/rodent/rodent_labels.csv` contains manual frame-level annotations of each
clip with two parallel tracks:

- **behavior** ∈ {Immobile, Rear, Turn, Walk}
- **direction** ∈ {Straight, Left, Right}

These labels are sourced from
[KevinBian107/TopoMimic-annotation](https://github.com/KevinBian107/TopoMimic-annotation)
and are used by:

- `c2a.codegen.shared.label_alignment` (loader / coverage metrics)
- `c2a.codegen.run_sweep` (NMI + Hungarian-purity reporting at three
  granularities: behavior / direction / combined)
- the code-grid behavior labeling (`c2a.experiments.code_grid`) and the
  TopoMimic auto-labeler that tags rolled-out codes

The CSV is annotated against the original 30 fps videos (500 frames per clip).
The H5 mocap data is at half the rate (250 frames per clip), so
``c2a.codegen.shared.label_alignment`` divides CSV frame numbers by 2 by default
(``csv_to_h5_subsample=2``).
