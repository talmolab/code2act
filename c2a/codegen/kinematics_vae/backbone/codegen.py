"""Encode every frame of every balanced-split clip with a trained
backbone + bottleneck, returning the latents (and optionally codes).

This is the bottleneck-agnostic glue between a trained checkpoint and
the unified codes-npz schema. Each variant calls
:func:`encode_balanced_clips` to get per-frame latents (and, for VQ
variants, the per-frame codebook indices), then writes them via its
own :mod:`tokenize` module.

Per-frame semantics
-------------------
The backbone encoder consumes a 10-frame window of motion features and
emits one ``z`` per window. For an L-frame clip there are ``L - w - n + 1``
windows. We want a per-frame code (length L) downstream. We use the
**windowed-z assigned to the last frame of the input window** convention:

    code[t] for t in [w-1, L-n-1]   = bottleneck(window_t)
    code[t] for t < w-1             = code[w-1]      (padding from the left)
    code[t] for t > L-n-1           = code[L-n-1]    (padding from the right)

The padding is unavoidable because the encoder needs a full ``w``-frame
window of *past* context and the predictor's auxiliary loss requires
``n`` frames of *future* context, but the downstream RL stage expects
one code per frame of the L-frame clip. Padding at the edges with the
nearest valid code is conservative — it never introduces spurious
codebook tokens and keeps clip boundaries aligned with the rest of c2a.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from c2a.codegen.kinematics_vae.backbone.decoder import MotionDecoderSplit
from c2a.codegen.kinematics_vae.backbone.encoder import (
    MotionEncoderConv1DSharedSplit,
)
from c2a.codegen.kinematics_vae.backbone.frame_features import (
    extract_motion_frames,
)
from c2a.codegen.kinematics_vae.backbone.normalizer import FeatureNormalizer
from c2a.codegen.kinematics_vae.backbone.predictor import MotionPredictorSplit
from c2a.codegen.kinematics_vae.backbone.train_loop import (
    BackboneModules,
    _Clips,
    _load_clips_h5,
)

log = logging.getLogger(__name__)


@dataclass
class EncodedClips:
    """Output of :func:`encode_balanced_clips`.

    Attributes:
        z: ``(n_clips, n_frames, latent_dim)`` per-frame latents (μ when
            ``deterministic=True``).
        mean: ``(n_clips, n_frames, latent_dim)`` posterior means.
        logvar: ``(n_clips, n_frames, latent_dim)`` posterior log-variances
            (zeros from VQ variants — we always return both for a
            uniform downstream interface).
        clip_indices: ``(n_clips,)`` original H5 clip ids matching the
            row order of ``z``.
        modules: Echo of the loaded backbone.
        aux: dict carrying variant-specific outputs (e.g.
            ``{"codes": <(n_clips, n_frames) int>}`` for VQVAE).
    """

    z: np.ndarray
    mean: np.ndarray
    logvar: np.ndarray
    clip_indices: np.ndarray
    modules: BackboneModules
    aux: dict[str, np.ndarray]


def load_checkpoint(
    ckpt_dir: str | Path,
    bottleneck: nn.Module,
    *,
    n_joints: int = 67,
    use_qvel: bool = False,
    window_len: int = 10,
    horizon: int = 5,
    conv_channels: tuple[int, ...] = (64, 128, 256),
    conv_kernel_size: int = 3,
    conv_head_layers: tuple[int, ...] = (256,),
    decoder_layer_sizes: tuple[int, ...] = (256, 256, 256),
    predictor_layer_sizes: tuple[int, ...] = (384, 256),
) -> tuple[BackboneModules, dict[str, Any]]:
    """Restore backbone + bottleneck from a checkpoint directory.

    The checkpoint dir is the one written by
    :func:`c2a.codegen.kinematics_vae.backbone.train_loop._save_checkpoint`
    — contains ``encoder.msgpack``, ``bottleneck.msgpack``,
    ``decoder.msgpack``, ``predictor.msgpack``, ``normalizer.npz``,
    ``active_joints.npy``, ``meta.json``.
    """
    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    branch_input_indices = tuple(tuple(ix) for ix in meta["branch_input_indices"])
    branch_latent_dims = tuple(meta["branch_latent_dims"])
    branch_names = tuple(meta["branch_names"])
    feat_dim = int(meta["feat_dim"])
    active_joints = np.load(ckpt_dir / "active_joints.npy")
    nz = np.load(ckpt_dir / "normalizer.npz")
    normalizer = FeatureNormalizer(
        mean=jnp.asarray(nz["mean"], dtype=jnp.float32),
        std=jnp.asarray(nz["std"], dtype=jnp.float32),
    )

    encoder = MotionEncoderConv1DSharedSplit(
        branch_latent_dims=branch_latent_dims,
        conv_channels=tuple(conv_channels),
        kernel_size=int(conv_kernel_size),
        head_layer_sizes=tuple(conv_head_layers),
    )
    decoder = MotionDecoderSplit(
        branch_output_indices=branch_input_indices,
        branch_latent_dims=branch_latent_dims,
        full_feat_dim=feat_dim,
        window_len=int(window_len),
        layer_sizes=tuple(decoder_layer_sizes),
    )
    predictor = MotionPredictorSplit(
        branch_output_indices=branch_input_indices,
        branch_latent_dims=branch_latent_dims,
        full_feat_dim=feat_dim,
        horizon=int(horizon),
        layer_sizes=tuple(predictor_layer_sizes),
    )

    # Param shapes can be inferred via a dummy init then overwritten with msgpack.
    rng = jax.random.PRNGKey(0)
    dummy_in = jnp.zeros((1, window_len, feat_dim), dtype=jnp.float32)
    enc_template = encoder.init(rng, dummy_in)
    dummy_mu = jnp.zeros((1, sum(branch_latent_dims)), dtype=jnp.float32)
    dummy_lv = jnp.zeros_like(dummy_mu)
    bn_template = bottleneck.init(
        rng, dummy_mu, dummy_lv, rng, deterministic=True,
    )
    dec_template = decoder.init(rng, dummy_mu)
    pred_template = predictor.init(rng, dummy_mu)

    enc_params = flax.serialization.from_bytes(
        enc_template, (ckpt_dir / "encoder.msgpack").read_bytes(),
    )
    # Saved bottleneck params follow the same {"params": ...} wrapper as
    # train_loop builds in bn_params; mirror that template.
    bn_template_params = {"params": bn_template.get("params", {})}
    bn_params = flax.serialization.from_bytes(
        bn_template_params, (ckpt_dir / "bottleneck.msgpack").read_bytes(),
    )
    dec_params = flax.serialization.from_bytes(
        dec_template, (ckpt_dir / "decoder.msgpack").read_bytes(),
    )
    pred_params = flax.serialization.from_bytes(
        pred_template, (ckpt_dir / "predictor.msgpack").read_bytes(),
    )
    # PS-VQVAE-style bottlenecks ship their EMA buffers in a separate
    # ``bottleneck_buffers.msgpack`` file (saved by
    # ``train_loop._save_checkpoint`` when ``bn_buffers`` is non-empty).
    # For VAE / VQVAE that file doesn't exist and bn_params already
    # carries everything.
    buffers_path = ckpt_dir / "bottleneck_buffers.msgpack"
    if buffers_path.exists():
        buf_template = {k: v for k, v in bn_template.items() if k != "params"}
        bn_buffers = flax.serialization.from_bytes(
            buf_template, buffers_path.read_bytes(),
        )
    else:
        bn_buffers = {}

    modules = BackboneModules(
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
        bottleneck=bottleneck,
        normalizer=normalizer,
        active_joints=active_joints,
        branch_input_indices=branch_input_indices,
        branch_latent_dims=branch_latent_dims,
        branch_names=branch_names,
        feat_dim=feat_dim,
    )
    all_params = {
        "encoder": enc_params,
        "bottleneck": bn_params,
        "bottleneck_buffers": bn_buffers,
        "decoder": dec_params,
        "predictor": pred_params,
    }
    return modules, all_params


def _pad_per_frame(
    window_outputs: np.ndarray,
    *,
    n_frames: int,
    window_len: int,
    horizon: int,
) -> np.ndarray:
    """Pad windowed outputs to per-frame by left/right replication.

    Args:
        window_outputs: ``(n_clips, n_windows, ...)`` per-window outputs
            where ``n_windows = n_frames - window_len - horizon + 1``.
        n_frames: Target per-frame length.
        window_len: Encoder window length ``w``.
        horizon: Predictor horizon ``n``.

    Returns:
        ``(n_clips, n_frames, ...)`` aligned to the last frame of the
        input window: ``out[t]`` for ``t in [w-1, n_frames-n-1]`` is the
        valid window output; edges are replicated.
    """
    n_clips, n_windows = window_outputs.shape[:2]
    expected_windows = n_frames - window_len - horizon + 1
    if n_windows != expected_windows:
        raise ValueError(
            f"window_outputs has {n_windows} windows, expected "
            f"{expected_windows} (n_frames={n_frames}, w={window_len}, "
            f"horizon={horizon})."
        )

    left_pad = window_len - 1
    right_pad = horizon
    pad_left = np.repeat(
        window_outputs[:, :1], left_pad, axis=1,
    )
    pad_right = np.repeat(
        window_outputs[:, -1:], right_pad, axis=1,
    )
    return np.concatenate([pad_left, window_outputs, pad_right], axis=1)


def encode_balanced_clips(
    reference_data_path: str | Path,
    balanced_split_path: str | Path,
    modules: BackboneModules,
    enc_params,
    bn_params,
    *,
    bn_buffers: dict | None = None,
    n_joints: int = 67,
    use_qvel: bool = False,
    window_len: int = 10,
    horizon: int = 5,
    batch_size_clips: int = 8,
    deterministic: bool = True,
    bottleneck_kwargs: dict[str, Any] | None = None,
    rng_seed: int = 0,
) -> EncodedClips:
    """Encode all balanced-split clips (train ∪ test, sorted union) with
    the trained backbone + bottleneck.

    The output ordering follows the **KPMS convention**: ``z[i]``
    corresponds to clip at H5 id ``sorted(train_indices ∪ test_indices)[i]``.
    Downstream variant codegens slice / re-order by caller-provided
    train/test indices.

    Args:
        reference_data_path: ``rodent_reference_clips.h5``.
        balanced_split_path: ``rodent_balanced_splits.json``.
        modules: Loaded ``BackboneModules`` from :func:`load_checkpoint`.
        enc_params, bn_params: Loaded encoder / bottleneck parameters.
        deterministic: ``True`` uses ``z = μ`` (default for codegen — we
            want reproducible codes); ``False`` reparameterizes.
        bottleneck_kwargs: Per-variant runtime args (e.g. ``sigma_max``
            for VAE). Usually empty for codegen since we run
            deterministic.

    Returns:
        :class:`EncodedClips` with per-frame latents in sorted-union
        clip order.
    """
    # Load balanced split → all H5 clip ids in sorted union.
    with open(balanced_split_path) as f:
        splits = json.load(f)
    train = list(map(int, splits["balanced"]["train_indices"]))
    test = list(map(int, splits["balanced"]["test_indices"]))
    all_indices = np.asarray(sorted(set(train) | set(test)), dtype=np.int64)
    log.info(
        "Encoding %d balanced clips (sorted-union order)",
        len(all_indices),
    )

    clips = _load_clips_h5(reference_data_path, keep_clips_idx=all_indices)
    # Confirm clips came back in sorted-union order (h5 keys are sorted
    # by int id in _load_clips_h5, so the loaded order matches all_indices
    # by construction).
    if clips.qpos.shape[0] != all_indices.shape[0]:
        raise RuntimeError(
            f"loaded {clips.qpos.shape[0]} clips from H5 but balanced "
            f"split has {all_indices.shape[0]} entries — H5 is missing "
            f"clips required by the split."
        )

    motion = extract_motion_frames(
        clips,
        n_joints=n_joints,
        use_qvel=use_qvel,
        active_joints=modules.active_joints,
    )
    motion_norm = (
        motion - np.asarray(modules.normalizer.mean)
    ) / np.asarray(modules.normalizer.std)
    motion_norm = motion_norm.astype(np.float32)

    n_clips, n_frames, feat_dim = motion_norm.shape
    n_windows_per_clip = n_frames - window_len - horizon + 1

    # Build windows per clip: shape (n_clips, n_windows_per_clip, w, feat_dim)
    starts = np.arange(n_windows_per_clip)
    in_idx = starts[:, None] + np.arange(window_len)
    windows = motion_norm[:, in_idx, :]   # (n_clips, n_w, w, feat_dim)

    encoder = modules.encoder
    bottleneck_mod = modules.bottleneck

    # PS-VQVAE-style bottlenecks ship buffers separately and need their
    # variables dict reassembled for ``apply``. VAE / VQVAE just pass
    # bn_params unchanged.
    bn_uses_buffers = bool(bn_buffers)

    @jax.jit
    def _encode_batch(enc_p, bn_p, bn_buf, x_batch, rng):
        # x_batch: (B, w, feat_dim)
        mean, logvar = encoder.apply(enc_p, x_batch)
        # bn_p is already a full ``{"params": ...}`` wrapper; merge in
        # the (optional) buffer collections for PS-VQVAE-style bottlenecks.
        if bn_uses_buffers:
            bn_variables = {**bn_p, **bn_buf}
        else:
            bn_variables = bn_p
        z, aux_losses, aux_metrics = bottleneck_mod.apply(
            bn_variables, mean, logvar, rng,
            deterministic=deterministic,
            **(bottleneck_kwargs or {}),
        )
        return mean, logvar, z, aux_metrics

    # Encode clip-by-clip (each clip has n_windows_per_clip windows; fit
    # batch_size_clips clips at a time to limit memory).
    means_out, lvs_out, zs_out = [], [], []
    aux_per_window: dict[str, list[np.ndarray]] = {}

    key = jax.random.PRNGKey(int(rng_seed))
    bn_buf_arg = bn_buffers if bn_uses_buffers else {}
    for start in range(0, n_clips, batch_size_clips):
        end = min(start + batch_size_clips, n_clips)
        x = windows[start:end]  # (b, n_w, w, feat)
        b = end - start
        x_flat = jnp.asarray(x.reshape(b * n_windows_per_clip, window_len, feat_dim))
        key, k = jax.random.split(key)
        mean, logvar, z, aux_metrics = _encode_batch(
            enc_params, bn_params, bn_buf_arg, x_flat, k,
        )
        latent_dim = z.shape[-1]
        means_out.append(np.asarray(mean).reshape(b, n_windows_per_clip, latent_dim))
        lvs_out.append(np.asarray(logvar).reshape(b, n_windows_per_clip, latent_dim))
        zs_out.append(np.asarray(z).reshape(b, n_windows_per_clip, latent_dim))
        for k_name, v in aux_metrics.items():
            v_np = np.asarray(v)
            if v_np.ndim >= 1 and v_np.shape[0] == b * n_windows_per_clip:
                aux_per_window.setdefault(k_name, []).append(
                    v_np.reshape((b, n_windows_per_clip) + v_np.shape[1:])
                )
            # Scalar / non-per-window metrics are ignored at codegen time
            # — they only make sense at train time.

    z_window = np.concatenate(zs_out, axis=0)
    mean_window = np.concatenate(means_out, axis=0)
    lv_window = np.concatenate(lvs_out, axis=0)

    z_perframe = _pad_per_frame(
        z_window, n_frames=n_frames, window_len=window_len, horizon=horizon,
    )
    mean_perframe = _pad_per_frame(
        mean_window, n_frames=n_frames, window_len=window_len, horizon=horizon,
    )
    lv_perframe = _pad_per_frame(
        lv_window, n_frames=n_frames, window_len=window_len, horizon=horizon,
    )

    aux: dict[str, np.ndarray] = {}
    for k_name, parts in aux_per_window.items():
        per_clip = np.concatenate(parts, axis=0)
        aux[k_name] = _pad_per_frame(
            per_clip, n_frames=n_frames, window_len=window_len, horizon=horizon,
        )

    return EncodedClips(
        z=z_perframe,
        mean=mean_perframe,
        logvar=lv_perframe,
        clip_indices=all_indices,
        modules=modules,
        aux=aux,
    )
