"""Hydra entry point for product-space kinematics-VQVAE pretraining.

Same backbone as VAE / VQVAE; the bottleneck factors the latent into
per-group EMA-VQ codebooks (paper-faithful per VQ-MAP, Li et al. 2025).
See ``scratch/2026-05-12-kinematics-vae-codegen-unify/
03b-ps-vqvae-paper-aligned.md`` for the full alignment.

Usage::

    python -m c2a.codegen.kinematics_vae.ps_vqvae.train

Override on the CLI as with the other variants.
"""

from __future__ import annotations

import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from c2a.codegen.kinematics_vae.backbone.train_loop import train
from c2a.codegen.kinematics_vae.ps_vqvae.bottleneck import PSVQVAEBottleneck
from c2a.paths import get_c2a_root

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

log = logging.getLogger(__name__)


CONFIG_DIR = str(
    (get_c2a_root() / "codegen" / "kinematics_vae" / "ps_vqvae" / "configs").resolve()
)


@hydra.main(
    config_path=CONFIG_DIR,
    config_name="pretrain",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    log.info("PS-VQVAE pretrain config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    tc = cfg.train
    lw = cfg.loss_weights
    arch = cfg.arch
    pb = cfg.product_bottleneck

    bottleneck = PSVQVAEBottleneck(
        group_dims=tuple(int(d) for d in pb.group_dims),
        group_codebook_sizes=tuple(int(K) for K in pb.group_codebook_sizes),
        decay=float(pb.decay),
        reset_threshold=float(pb.reset_threshold),
        reseed_jitter=float(pb.reseed_jitter),
    )

    train(
        bottleneck=bottleneck,
        loss_weights={
            "recon": float(lw.recon),
            "pred": float(lw.pred),
            "commit": float(lw.commit),
        },
        reference_data_path=str(cfg.data.reference_data_path),
        balanced_split_path=str(cfg.data.balanced_split_path),
        ckpt_dir=str(cfg.train.ckpt_dir),
        n_joints=int(arch.n_joints),
        use_qvel=bool(arch.use_qvel),
        branch_latent_dims=tuple(arch.branch_latent_dims),
        window_len=int(arch.window_len),
        horizon=int(arch.horizon),
        conv_channels=tuple(arch.conv_channels),
        conv_kernel_size=int(arch.conv_kernel_size),
        conv_head_layers=tuple(arch.conv_head_layers),
        decoder_layer_sizes=tuple(arch.decoder_layer_sizes),
        predictor_layer_sizes=tuple(arch.predictor_layer_sizes),
        auto_detect_threshold=float(arch.auto_detect_threshold),
        batch_size=int(tc.batch_size),
        num_steps=int(tc.num_steps),
        learning_rate=float(tc.learning_rate),
        lr_warmup_frac=float(tc.lr_warmup_frac),
        lr_hold_frac=float(tc.lr_hold_frac),
        lr_end_value=float(tc.lr_end_value),
        weight_decay=float(tc.weight_decay),
        grad_clip=float(tc.grad_clip),
        log_every=int(tc.log_every),
        eval_every=int(tc.eval_every),
        ckpt_every=int(tc.ckpt_every),
        seed=int(cfg.seed),
    )


if __name__ == "__main__":
    main()
