"""Bottleneck-agnostic pretraining loop.

One AdamW + cosine-with-warmup-hold schedule + msgpack/orbax checkpoint
loop shared across the three c2a variants (``vae``, ``vqvae``,
``ps_vqvae``). Variants call :func:`train` with their own bottleneck
module + loss-weight schedule + a config that selects which
encoder/decoder/predictor flavor to use.

This loop:
    1. Loads clips from the rodent reference H5.
    2. Auto-detects active joints, builds the body-part split, extracts
       motion frames + windows.
    3. Initializes encoder + bottleneck + decoder + predictor params.
    4. Trains with the generic loss in :mod:`losses` for ``num_steps``
       batches (uniformly sampled from train windows).
    5. Periodically evaluates on a held-out clip split and checkpoints
       on improvement.
    6. Saves the final encoder / decoder / predictor / bottleneck
       params + normalizer + active_joints + branch_indices.

The clip subset used for training is :func:`_load_clip_subset_h5` —
honours the same ``data/rodent/rodent_balanced_splits.json`` train
indices the rest of c2a uses, so the encoder never sees test clips.

Adapted from
``origin/scott_claude/latent-mimic-handoff:track_mjx/agent/latent_ppo/
pretrain.py``, with path resolution routed through :mod:`c2a.paths` and
the body-part split made mandatory (matches v15 defaults).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

from c2a.codegen.kinematics_vae.backbone.decoder import MotionDecoderSplit
from c2a.codegen.kinematics_vae.backbone.encoder import (
    MotionEncoderConv1DSharedSplit,
)
from c2a.codegen.kinematics_vae.backbone.frame_features import (
    MOTION_FRAME_DIM,
    auto_detect_active_joints,
    classify_joint_groups,
    extract_motion_frames,
    feat_indices_for_groups,
)
from c2a.codegen.kinematics_vae.backbone.losses import pretrain_loss
from c2a.codegen.kinematics_vae.backbone.normalizer import (
    FeatureNormalizer,
    fit_normalizer,
)
from c2a.codegen.kinematics_vae.backbone.predictor import MotionPredictorSplit
from c2a.codegen.kinematics_vae.backbone.window_dataset import make_windows
from c2a.paths import get_repo_root

log = logging.getLogger(__name__)


# =============================================================================
# Data loading
# =============================================================================


@dataclass
class _Clips:
    """Minimal clips container mirroring vnl_playground.ReferenceClips's
    fields the backbone consumes."""

    qpos: np.ndarray  # (n_clips, n_frames, 7 + n_joints)
    qvel: np.ndarray  # (n_clips, n_frames, 6 + n_joints)
    joint_names: list[str]  # length n_joints


def _load_clips_h5(
    h5_path: str | Path,
    keep_clips_idx: Optional[np.ndarray] = None,
) -> _Clips:
    """Load clips from the rodent reference H5.

    Args:
        h5_path: Absolute path to ``rodent_reference_clips.h5``.
        keep_clips_idx: Optional indices into the full clip set; if
            ``None`` returns all clips.
    """
    import h5py

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(
            f"rodent_reference_clips.h5 not found at {h5_path}"
        )

    with h5py.File(h5_path, "r") as f:
        clip_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
        if keep_clips_idx is not None:
            keep = set(int(i) for i in keep_clips_idx)
            # Clip keys look like "clip_<id>"; pick the matching subset.
            clip_keys = [k for k in clip_keys
                         if int(k.split("_")[-1]) in keep]
        qpos_list = [np.asarray(f[k]["walker/qpos"]) for k in clip_keys]
        qvel_list = [np.asarray(f[k]["walker/qvel"]) for k in clip_keys]
        joint_names_attr = f.attrs.get("joint_names", None)
        if joint_names_attr is not None:
            joint_names = [
                s.decode() if isinstance(s, bytes) else str(s)
                for s in joint_names_attr
            ]
        else:
            # Fall back: infer joint count from qpos shape; no names.
            n_joints = qpos_list[0].shape[-1] - 7
            joint_names = [f"joint_{i}" for i in range(n_joints)]

    return _Clips(
        qpos=np.stack(qpos_list, axis=0).astype(np.float32),
        qvel=np.stack(qvel_list, axis=0).astype(np.float32),
        joint_names=joint_names,
    )


def _load_train_indices(
    balanced_split_path: str | Path,
) -> np.ndarray:
    """Train indices from the balanced split JSON."""
    p = Path(balanced_split_path)
    if not p.exists():
        raise FileNotFoundError(
            f"balanced split JSON not found at {p}; required so the "
            f"kinematics-VAE pretrain never sees test clips."
        )
    with p.open() as f:
        splits = json.load(f)
    return np.asarray(splits["balanced"]["train_indices"], dtype=np.int64)


# =============================================================================
# Schedules
# =============================================================================


def warmup_hold_cosine_schedule(
    *,
    init_value: float,
    peak_value: float,
    end_value: float,
    warmup_frac: float,
    hold_frac: float,
    total_steps: int,
) -> optax.Schedule:
    """Linear-warmup → hold-at-peak → cosine-decay schedule."""
    warmup_steps = int(round(total_steps * float(warmup_frac)))
    hold_steps = int(round(total_steps * float(hold_frac)))
    decay_steps = total_steps - warmup_steps - hold_steps
    if decay_steps <= 0:
        decay_steps = max(1, total_steps - warmup_steps)
        hold_steps = total_steps - warmup_steps - decay_steps

    boundaries = [warmup_steps, warmup_steps + hold_steps]
    schedules = [
        optax.linear_schedule(init_value, peak_value, warmup_steps),
        optax.constant_schedule(peak_value),
        optax.cosine_decay_schedule(
            peak_value, decay_steps, alpha=end_value / max(peak_value, 1e-12),
        ),
    ]
    return optax.join_schedules(schedules, boundaries=boundaries)


def beta_kl_anneal(step: jnp.ndarray, anneal_steps: int, target: float) -> jnp.ndarray:
    """Linear anneal from 0 → ``target`` over ``anneal_steps``."""
    if anneal_steps <= 0:
        return jnp.asarray(target, dtype=jnp.float32)
    frac = jnp.minimum(step.astype(jnp.float32) / float(anneal_steps), 1.0)
    return frac * float(target)


def sigma_max_anneal(
    step: jnp.ndarray,
    start: float,
    end: float,
    anneal_frac: float,
    total_steps: int,
) -> jnp.ndarray:
    """Linear anneal of ``sigma_max`` from ``start`` → ``end`` over the
    first ``anneal_frac`` of training, then held at ``end``."""
    anneal_steps = max(1, int(round(float(anneal_frac) * total_steps)))
    frac = jnp.minimum(step.astype(jnp.float32) / float(anneal_steps), 1.0)
    return jnp.asarray(start, dtype=jnp.float32) + frac * (end - start)


# =============================================================================
# Builders for the encoder / decoder / predictor (v15 defaults)
# =============================================================================


@dataclass
class BackboneModules:
    """Holds the four modules + their params + the (frozen) normalizer.

    ``branch_input_indices`` and ``branch_latent_dims`` are saved
    alongside the checkpoint so downstream code (codegen + RL) knows
    how to slice ``z`` per branch.
    """

    encoder: nn.Module
    decoder: nn.Module
    predictor: nn.Module
    bottleneck: nn.Module
    normalizer: FeatureNormalizer
    active_joints: np.ndarray             # (n_active,)
    branch_input_indices: tuple[tuple[int, ...], ...]
    branch_latent_dims: tuple[int, ...]
    branch_names: tuple[str, ...]
    feat_dim: int


def _build_branches(
    clips: _Clips,
    *,
    n_joints: int,
    use_qvel: bool,
    auto_detect_threshold: float,
    branch_latent_dims: tuple[int, ...],
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[str, ...],
    np.ndarray,
]:
    """Auto-detect active joints + body-part split + branch feat indices.

    Always uses the v15 3-way ``(root, axial, limb)`` split. Returns
    ``(branch_input_indices, branch_names, active_joints)`` in the
    order matching ``branch_latent_dims``.
    """
    active_joints = auto_detect_active_joints(
        clips, n_joints=n_joints, std_threshold=auto_detect_threshold,
    )
    if active_joints.size == 0:
        raise RuntimeError(
            "auto_detect_active_joints returned 0 joints — clips may be "
            "frozen or std_threshold is too high. Check the H5."
        )
    group_positions = classify_joint_groups(
        joint_names=clips.joint_names,
        active_joints=active_joints,
    )
    feat_idx = feat_indices_for_groups(
        group_active_positions=group_positions,
        n_active_joints=int(active_joints.size),
        use_qvel=use_qvel,
        root_as_separate_branch=True,
    )
    # Fixed order matching branch_latent_dims (root, axial, limb).
    branch_names = ("root", "axial", "limb")
    if len(branch_latent_dims) != len(branch_names):
        raise ValueError(
            f"branch_latent_dims has length {len(branch_latent_dims)} "
            f"but the 3-way split expects {len(branch_names)} branches."
        )
    branch_input_indices = tuple(
        tuple(int(i) for i in feat_idx[name].tolist())
        for name in branch_names
    )
    return branch_input_indices, branch_names, active_joints


def build_backbone(
    clips: _Clips,
    bottleneck: nn.Module,
    *,
    n_joints: int,
    use_qvel: bool = False,
    auto_detect_threshold: float = 1e-3,
    branch_latent_dims: tuple[int, ...] = (4, 8, 12),
    window_len: int = 10,
    horizon: int = 5,
    conv_channels: tuple[int, ...] = (64, 128, 256),
    conv_kernel_size: int = 3,
    conv_head_layers: tuple[int, ...] = (256,),
    decoder_layer_sizes: tuple[int, ...] = (256, 256, 256),
    predictor_layer_sizes: tuple[int, ...] = (384, 256),
) -> tuple[BackboneModules, np.ndarray]:
    """Construct the encoder / decoder / predictor + fit the normalizer.

    Returns:
        ``(BackboneModules, motion_normalized)`` where
        ``motion_normalized`` has shape ``(n_clips, n_frames, feat_dim)``
        in z-scored space (used immediately by the trainer to build
        windows).
    """
    branch_input_indices, branch_names, active_joints = _build_branches(
        clips,
        n_joints=n_joints,
        use_qvel=use_qvel,
        auto_detect_threshold=auto_detect_threshold,
        branch_latent_dims=branch_latent_dims,
    )

    motion = extract_motion_frames(
        clips,
        n_joints=n_joints,
        use_qvel=use_qvel,
        active_joints=active_joints,
    )
    normalizer = fit_normalizer(motion)
    motion_norm = (motion - np.asarray(normalizer.mean)) / np.asarray(normalizer.std)
    feat_dim = int(motion.shape[-1])

    encoder = MotionEncoderConv1DSharedSplit(
        branch_latent_dims=tuple(int(d) for d in branch_latent_dims),
        conv_channels=tuple(conv_channels),
        kernel_size=int(conv_kernel_size),
        head_layer_sizes=tuple(conv_head_layers),
    )
    decoder = MotionDecoderSplit(
        branch_output_indices=branch_input_indices,
        branch_latent_dims=tuple(int(d) for d in branch_latent_dims),
        full_feat_dim=feat_dim,
        window_len=int(window_len),
        layer_sizes=tuple(decoder_layer_sizes),
    )
    predictor = MotionPredictorSplit(
        branch_output_indices=branch_input_indices,
        branch_latent_dims=tuple(int(d) for d in branch_latent_dims),
        full_feat_dim=feat_dim,
        horizon=int(horizon),
        layer_sizes=tuple(predictor_layer_sizes),
    )

    modules = BackboneModules(
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
        bottleneck=bottleneck,
        normalizer=normalizer,
        active_joints=active_joints,
        branch_input_indices=branch_input_indices,
        branch_latent_dims=tuple(int(d) for d in branch_latent_dims),
        branch_names=branch_names,
        feat_dim=feat_dim,
    )
    return modules, motion_norm.astype(np.float32)


# =============================================================================
# Training loop
# =============================================================================


def _shuffle_indices(rng: np.random.Generator, n: int) -> np.ndarray:
    idx = np.arange(n)
    rng.shuffle(idx)
    return idx


def _save_checkpoint(
    ckpt_dir: Path,
    step: int,
    *,
    enc_params,
    bn_params,
    dec_params,
    pred_params,
    modules: BackboneModules,
    metrics: dict,
) -> None:
    """Atomic msgpack checkpoint dump."""
    out = ckpt_dir / f"step_{step:07d}"
    out.mkdir(parents=True, exist_ok=True)

    bytes_payload = {
        "encoder": flax.serialization.to_bytes(enc_params),
        "bottleneck": flax.serialization.to_bytes(bn_params),
        "decoder": flax.serialization.to_bytes(dec_params),
        "predictor": flax.serialization.to_bytes(pred_params),
    }
    for name, b in bytes_payload.items():
        (out / f"{name}.msgpack").write_bytes(b)

    np.savez(
        out / "normalizer.npz",
        mean=np.asarray(modules.normalizer.mean),
        std=np.asarray(modules.normalizer.std),
    )
    np.save(out / "active_joints.npy", modules.active_joints)
    meta = {
        "step": int(step),
        "branch_names": list(modules.branch_names),
        "branch_input_indices": [
            list(ix) for ix in modules.branch_input_indices
        ],
        "branch_latent_dims": list(modules.branch_latent_dims),
        "feat_dim": int(modules.feat_dim),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def train(
    *,
    bottleneck: nn.Module,
    loss_weights: Mapping[str, float],
    reference_data_path: str | Path,
    balanced_split_path: str | Path,
    ckpt_dir: str | Path,
    # Architecture
    n_joints: int = 67,
    use_qvel: bool = False,
    branch_latent_dims: tuple[int, ...] = (4, 8, 12),
    window_len: int = 10,
    horizon: int = 5,
    conv_channels: tuple[int, ...] = (64, 128, 256),
    conv_kernel_size: int = 3,
    conv_head_layers: tuple[int, ...] = (256,),
    decoder_layer_sizes: tuple[int, ...] = (256, 256, 256),
    predictor_layer_sizes: tuple[int, ...] = (384, 256),
    auto_detect_threshold: float = 1e-3,
    # Optimization
    batch_size: int = 8192,
    num_steps: int = 30_000,
    learning_rate: float = 3e-4,
    lr_warmup_frac: float = 0.04,
    lr_hold_frac: float = 0.20,
    lr_end_value: float = 3e-5,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,
    # Scheduled loss weights and bottleneck kwargs (computed per step).
    weight_schedules: Optional[Mapping[str, Callable[[jnp.ndarray], jnp.ndarray]]] = None,
    bottleneck_kwarg_schedules: Optional[Mapping[str, Callable[[jnp.ndarray], jnp.ndarray]]] = None,
    # Logging / checkpoints
    log_every: int = 250,
    eval_every: int = 500,
    ckpt_every: int = 5_000,
    seed: int = 0,
    # WandB hook (optional)
    wandb_run: Any = None,
) -> Path:
    """Run the bottleneck-agnostic pretraining loop.

    ``loss_weights`` provides the base coefficient for every term the
    bottleneck emits (e.g. ``{"recon": 1.0, "pred": 0.3, "kl": 0.005,
    "sigma_pen": 1.0}`` for VAE; ``{"recon": 1.0, "pred": 0.3, "commit":
    0.25}`` for VQVAE).

    ``schedule_aux`` lets a variant override a weight on every step
    (e.g. linear-anneal ``kl`` from 0 → 0.005). Keys missing from
    ``schedule_aux`` use the static value from ``loss_weights``.

    Returns:
        Path of the final checkpoint directory.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Data ----------------------------------------------------------------
    train_idx = _load_train_indices(balanced_split_path)
    clips = _load_clips_h5(reference_data_path, keep_clips_idx=train_idx)
    log.info(
        "Loaded %d train clips of length %d for kinematics-VAE pretrain",
        clips.qpos.shape[0], clips.qpos.shape[1],
    )

    modules, motion_norm = build_backbone(
        clips,
        bottleneck=bottleneck,
        n_joints=n_joints,
        use_qvel=use_qvel,
        auto_detect_threshold=auto_detect_threshold,
        branch_latent_dims=branch_latent_dims,
        window_len=window_len,
        horizon=horizon,
        conv_channels=conv_channels,
        conv_kernel_size=conv_kernel_size,
        conv_head_layers=conv_head_layers,
        decoder_layer_sizes=decoder_layer_sizes,
        predictor_layer_sizes=predictor_layer_sizes,
    )
    inputs, targets = make_windows(motion_norm, w=window_len, n=horizon)
    log.info(
        "Built %d windows (w=%d, horizon=%d, feat=%d)",
        inputs.shape[0], window_len, horizon, modules.feat_dim,
    )

    # --- Param init ----------------------------------------------------------
    key = jax.random.PRNGKey(int(seed))
    key, k_enc, k_bn, k_dec, k_pred = jax.random.split(key, 5)
    dummy_in = jnp.zeros((1, window_len, modules.feat_dim), dtype=jnp.float32)
    enc_params = modules.encoder.init(k_enc, dummy_in)
    dummy_mu = jnp.zeros((1, sum(branch_latent_dims)), dtype=jnp.float32)
    dummy_lv = jnp.zeros_like(dummy_mu)
    bn_params = modules.bottleneck.init(
        k_bn, dummy_mu, dummy_lv, k_bn, deterministic=True,
    )
    dummy_z = jnp.zeros_like(dummy_mu)
    dec_params = modules.decoder.init(k_dec, dummy_z)
    pred_params = modules.predictor.init(k_pred, dummy_z)

    # --- Optimizer -----------------------------------------------------------
    lr_schedule = warmup_hold_cosine_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        end_value=lr_end_value,
        warmup_frac=lr_warmup_frac,
        hold_frac=lr_hold_frac,
        total_steps=num_steps,
    )
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
    )
    all_params = {
        "encoder": enc_params,
        "bottleneck": bn_params,
        "decoder": dec_params,
        "predictor": pred_params,
    }
    opt_state = tx.init(all_params)

    # --- JIT'd step ----------------------------------------------------------
    encoder = modules.encoder
    bottleneck_mod = modules.bottleneck
    decoder = modules.decoder
    predictor = modules.predictor

    def _resolve_at_step(step):
        weights = dict(loss_weights)
        if weight_schedules is not None:
            for k, sched_fn in weight_schedules.items():
                weights[k] = sched_fn(step)
        bn_kwargs: dict[str, Any] = {}
        if bottleneck_kwarg_schedules is not None:
            for k, sched_fn in bottleneck_kwarg_schedules.items():
                bn_kwargs[k] = sched_fn(step)
        return weights, bn_kwargs

    def _loss_fn(params, inputs, targets, rng, weights, bn_kwargs):
        return pretrain_loss(
            encoder, bottleneck_mod, decoder, predictor,
            params["encoder"], params["bottleneck"],
            params["decoder"], params["predictor"],
            inputs, targets, rng,
            loss_weights=weights,
            bottleneck_kwargs=bn_kwargs,
        )

    @jax.jit
    def train_step(params, opt_state, inputs, targets, rng, step):
        weights, bn_kwargs = _resolve_at_step(step)
        (loss, metrics), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            params, inputs, targets, rng, weights, bn_kwargs,
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics, weights, bn_kwargs

    # --- Loop ----------------------------------------------------------------
    np_rng = np.random.default_rng(int(seed))
    n_windows = inputs.shape[0]

    last_log = time.time()
    for step in range(num_steps):
        # Mini-batch — uniform sample with replacement; mirrors Scott's
        # implementation, cheap because windows are pre-built.
        batch_idx = np_rng.integers(0, n_windows, size=batch_size)
        x_b = jnp.asarray(inputs[batch_idx])
        y_b = jnp.asarray(targets[batch_idx])
        key, k_step = jax.random.split(key)
        all_params, opt_state, loss, metrics, used_weights, used_kwargs = train_step(
            all_params, opt_state, x_b, y_b, k_step,
            jnp.asarray(step, dtype=jnp.int32),
        )

        if step % log_every == 0:
            dt = time.time() - last_log
            last_log = time.time()
            metrics_py = {k: float(v) for k, v in metrics.items()}
            metrics_py["loss"] = float(loss)
            metrics_py["lr"] = float(lr_schedule(step))
            for k, v in used_weights.items():
                metrics_py[f"weight/{k}"] = float(v)
            for k, v in used_kwargs.items():
                metrics_py[f"bn_kwarg/{k}"] = float(v)
            log.info(
                "step %6d  loss=%.4f  recon=%.4f  pred=%.4f  %s  (%.1fs)",
                step, metrics_py["loss"], metrics_py["recon"],
                metrics_py["pred"],
                " ".join(
                    f"{k}={metrics_py[k]:.4f}"
                    for k in metrics_py
                    if k not in ("loss", "recon", "pred", "lr")
                    and not k.startswith("weight/")
                    and not k.startswith("bn_kwarg/")
                ),
                dt,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"train/{k}": v for k, v in metrics_py.items()},
                    step=step,
                )

        # Periodic checkpoint — every ckpt_every step (and on the
        # last step). The previous "best_eval" gate used the current
        # train-batch loss, which monotonically decreases, so every
        # eval-step looked like an improvement — effectively a no-op
        # filter that lied about which checkpoint was best. Honest
        # periodic snapshots are clearer; users can pick a step from
        # the final WandB curve.
        is_ckpt = (
            step > 0
            and (step % ckpt_every == 0 or step == num_steps - 1)
        )
        if is_ckpt:
            # Re-compute metrics_py at the ckpt step so meta.json never
            # records stale values from the previous log_every step.
            metrics_at_ckpt = {k: float(v) for k, v in metrics.items()}
            metrics_at_ckpt["loss"] = float(loss)
            _save_checkpoint(
                ckpt_dir, step,
                enc_params=all_params["encoder"],
                bn_params=all_params["bottleneck"],
                dec_params=all_params["decoder"],
                pred_params=all_params["predictor"],
                modules=modules,
                metrics=metrics_at_ckpt,
            )

    # Final checkpoint
    final = _save_checkpoint(
        ckpt_dir, num_steps,
        enc_params=all_params["encoder"],
        bn_params=all_params["bottleneck"],
        dec_params=all_params["decoder"],
        pred_params=all_params["predictor"],
        modules=modules,
        metrics={"loss": float(loss), "eval_score": float(loss)},
    )
    log.info("Done. Final ckpt at %s", ckpt_dir / f"step_{num_steps:07d}")
    return ckpt_dir
