"""Hydra entrypoint for training AR code-sequence generators.

Usage:

    # GPT-2 XS on raw KPMS codes (default).
    python -m c2a.ar_codegen.train

    # Mamba.
    python -m c2a.ar_codegen.train model=mamba

    # S5 on merged KPMS codes.
    python -m c2a.ar_codegen.train model=s5 data=kpms_merged

    # HMM on GT directional-turn labels.
    python -m c2a.ar_codegen.train model=hmm data=gt_directional_turn

Output: ``outputs/ar_codegen/{model}/{YYMMDD_HHMMSS_user}/`` containing
``meta.json`` + the model-specific params file. Pass that directory to
``run_code_generation`` via ``ar_checkpoints.{model_name}=<dir>``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

# Keep MuJoCo/EGL env vars consistent with the rest of c2a; harmless if
# JAX is the only backend touched here.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from c2a.ar_codegen.data import load_training_data
from c2a.ar_codegen.models.base import CodeGenerator

log = logging.getLogger(__name__)


def _initialize_wandb(cfg: DictConfig, run_name: str) -> "wandb.sdk.wandb_run.Run | None":
    """Initialize wandb (or return None when disabled).

    Mirrors the simple init pattern the c2a stack uses elsewhere, but
    keeps it local to this entry point because the AR training loop
    doesn't share the PPO callback chain.
    """
    lcfg = cfg.logging_config
    if not lcfg.get("use_wandb", True) or lcfg.get("wandb_mode", "online") == "disabled":
        log.info("wandb disabled (logging_config.use_wandb=false or mode=disabled).")
        return None
    return wandb.init(
        project=lcfg.get("wandb_project", "code2act_ar_codegen"),
        entity=lcfg.get("wandb_entity", None),
        name=run_name,
        mode=lcfg.get("wandb_mode", "online"),
        tags=list(lcfg.get("wandb_tags", [])),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _build_generator(cfg: DictConfig, num_codes: int) -> CodeGenerator:
    """Construct the right generator class from ``cfg.model._name``."""
    name = str(cfg.model._name)
    if name == "gpt2":
        from c2a.ar_codegen.models.gpt2 import GPT2Generator, TransformerConfig
        tc = TransformerConfig(
            vocab_size=num_codes + 1,
            emb_dim=int(cfg.model.emb_dim),
            num_heads=int(cfg.model.num_heads),
            num_layers=int(cfg.model.num_layers),
            qkv_dim=int(cfg.model.qkv_dim),
            mlp_dim=int(cfg.model.mlp_dim),
            max_len=int(cfg.model.max_len),
            dropout_rate=float(cfg.model.dropout_rate),
            attention_dropout_rate=float(cfg.model.attention_dropout_rate),
        )
        return GPT2Generator(num_codes=num_codes, config=tc)
    if name == "mamba":
        from c2a.ar_codegen.models.mamba import MambaGenerator, MambaConfig
        # dt_rank can be int or "auto" string in YAML.
        dt_rank_raw = cfg.model.dt_rank
        dt_rank = int(dt_rank_raw) if str(dt_rank_raw).isdigit() else str(dt_rank_raw)
        mc = MambaConfig(
            vocab_size=num_codes + 1,
            d_model=int(cfg.model.d_model),
            n_layers=int(cfg.model.n_layers),
            d_state=int(cfg.model.d_state),
            d_conv=int(cfg.model.d_conv),
            expand=int(cfg.model.expand),
            dt_rank=dt_rank,
            dt_min=float(cfg.model.dt_min),
            dt_max=float(cfg.model.dt_max),
            dt_init_floor=float(cfg.model.dt_init_floor),
            conv_bias=bool(cfg.model.conv_bias),
            bias=bool(cfg.model.bias),
            max_len=int(cfg.model.max_len),
            dropout_rate=float(cfg.model.dropout_rate),
        )
        return MambaGenerator(num_codes=num_codes, config=mc)
    if name == "s5":
        from c2a.ar_codegen.models.s5 import S5Generator, S5Config
        sc = S5Config(
            vocab_size=num_codes + 1,
            H=int(cfg.model.H),
            P=int(cfg.model.P),
            n_layers=int(cfg.model.n_layers),
            C_init=str(cfg.model.C_init),
            discretization=str(cfg.model.discretization),
            dt_min=float(cfg.model.dt_min),
            dt_max=float(cfg.model.dt_max),
            conj_sym=bool(cfg.model.conj_sym),
            clip_eigs=bool(cfg.model.clip_eigs),
            bidirectional=bool(cfg.model.bidirectional),
            max_len=int(cfg.model.max_len),
            dropout_rate=float(cfg.model.dropout_rate),
        )
        return S5Generator(num_codes=num_codes, config=sc)
    if name == "hmm":
        from c2a.ar_codegen.models.hmm import HMMGenerator
        return HMMGenerator(num_codes=num_codes)
    raise ValueError(
        f"Unknown model._name={name!r}. Expected: gpt2 | mamba | s5 | hmm."
    )


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Output directory — Hydra has already cd'd into it via hydra.run.dir.
    output_dir = Path.cwd()
    log.info("AR training run: model=%s, data=%s", cfg.model._name, cfg.data._name)
    log.info("Output directory: %s", output_dir)

    # Load code arrays.
    td = load_training_data(cfg)

    # Stamp the active code source into the training cfg so each model's
    # GeneratorMeta records where the codes came from (used by
    # run_code_generation.py to sanity-check num_codes match).
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    training_cfg["_code_source"] = td.payload.code_source

    # wandb.
    run_name = str(cfg.train_setup.run_name)
    wandb_run = _initialize_wandb(cfg, run_name=run_name)
    if wandb_run is not None:
        # Stash the full resolved cfg as wandb config for reproducibility.
        wandb_run.config.update(
            {
                "code_source": td.payload.code_source,
                "num_codes": td.num_codes,
                "n_train": int(td.train_codes.shape[0]),
                "n_val": int(td.val_codes.shape[0]),
                "n_frames": td.n_frames,
                "provenance": td.payload.provenance,
            },
            allow_val_change=True,
        )

    # Build the model.
    gen = _build_generator(cfg, num_codes=td.num_codes)
    log.info("Built %s generator over %d codes.", type(gen).__name__, td.num_codes)

    # Fit.
    t0 = time.time()
    gen.fit(
        train_codes=td.train_codes,
        val_codes=td.val_codes,
        cfg=OmegaConf.create(training_cfg),
        seed=int(cfg.train_setup.seed),
        wandb_run=wandb_run,
    )
    dt = time.time() - t0
    log.info("Fit done in %.1fs.", dt)

    # Save.
    gen.save(output_dir)
    log.info("Saved generator checkpoint to %s", output_dir)

    if wandb_run is not None:
        wandb_run.summary["fit_seconds"] = float(dt)
        wandb_run.summary["checkpoint_dir"] = str(output_dir)
        wandb_run.finish()


if __name__ == "__main__":
    main()
