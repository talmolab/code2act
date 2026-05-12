"""Hydra entry point for vanilla-kinematics-VAE pretraining.

Wraps :func:`c2a.codegen.kinematics_vae.backbone.train_loop.train` with
the Gaussian bottleneck and v15 loss-weight defaults
(β_kl linear-anneal, σ_max cosine-like anneal, w_pred=0.3).

Usage::

    python -m c2a.codegen.kinematics_vae.vae.train

Override any key on the command line, e.g.::

    python -m c2a.codegen.kinematics_vae.vae.train \\
        train.num_steps=5000 \\
        train.batch_size=2048 \\
        train.ckpt_dir=trained_kinematics_vae/smoke
"""

from __future__ import annotations

import logging
import os

import hydra
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from c2a.codegen.kinematics_vae.backbone.train_loop import (
    beta_kl_anneal,
    sigma_max_anneal,
    train,
)
from c2a.codegen.kinematics_vae.vae.bottleneck import GaussianBottleneck
from c2a.paths import get_c2a_root, get_repo_root

# Match the rest of c2a: EGL render backend + no preallocate so we
# coexist with other GPU tenants.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

log = logging.getLogger(__name__)


CONFIG_DIR = str(
    (get_c2a_root() / "codegen" / "kinematics_vae" / "vae" / "configs").resolve()
)


@hydra.main(
    config_path=CONFIG_DIR,
    config_name="pretrain",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Run the VAE pretraining loop."""
    log.info("VAE pretrain config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    tc = cfg.train
    lw = cfg.loss_weights
    sched = cfg.schedules

    bottleneck = GaussianBottleneck()

    # Per-step β_kl linear-anneal + σ_max curriculum.
    weight_schedules = {
        "kl": lambda step: beta_kl_anneal(
            step,
            anneal_steps=int(sched.beta_kl_anneal_steps),
            target=float(lw.kl),
        ),
    }
    bottleneck_kwarg_schedules = {
        "sigma_max": lambda step: sigma_max_anneal(
            step,
            start=float(sched.sigma_max_start),
            end=float(sched.sigma_max_end),
            anneal_frac=float(sched.sigma_max_anneal_frac),
            total_steps=int(tc.num_steps),
        ),
    }

    train(
        bottleneck=bottleneck,
        loss_weights={
            "recon": float(lw.recon),
            "pred": float(lw.pred),
            "kl": float(lw.kl),
            "sigma_pen": float(lw.sigma_pen),
        },
        reference_data_path=str(cfg.data.reference_data_path),
        balanced_split_path=str(cfg.data.balanced_split_path),
        ckpt_dir=str(cfg.train.ckpt_dir),
        n_joints=int(cfg.arch.n_joints),
        use_qvel=bool(cfg.arch.use_qvel),
        branch_latent_dims=tuple(cfg.arch.branch_latent_dims),
        window_len=int(cfg.arch.window_len),
        horizon=int(cfg.arch.horizon),
        conv_channels=tuple(cfg.arch.conv_channels),
        conv_kernel_size=int(cfg.arch.conv_kernel_size),
        conv_head_layers=tuple(cfg.arch.conv_head_layers),
        decoder_layer_sizes=tuple(cfg.arch.decoder_layer_sizes),
        predictor_layer_sizes=tuple(cfg.arch.predictor_layer_sizes),
        auto_detect_threshold=float(cfg.arch.auto_detect_threshold),
        batch_size=int(tc.batch_size),
        num_steps=int(tc.num_steps),
        learning_rate=float(tc.learning_rate),
        lr_warmup_frac=float(tc.lr_warmup_frac),
        lr_hold_frac=float(tc.lr_hold_frac),
        lr_end_value=float(tc.lr_end_value),
        weight_decay=float(tc.weight_decay),
        grad_clip=float(tc.grad_clip),
        weight_schedules=weight_schedules,
        bottleneck_kwarg_schedules=bottleneck_kwarg_schedules,
        log_every=int(tc.log_every),
        eval_every=int(tc.eval_every),
        ckpt_every=int(tc.ckpt_every),
        seed=int(cfg.seed),
    )


if __name__ == "__main__":
    main()
