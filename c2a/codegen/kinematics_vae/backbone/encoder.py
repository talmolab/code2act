"""Motion encoder modules: window → (mean, logvar) Gaussian posterior heads.

Ported from ``origin/scott_claude/latent-mimic-handoff:track_mjx/agent/
latent_ppo/networks/encoder.py``. Default for c2a kinematics-VAE family
is :class:`MotionEncoderConv1DSharedSplit` (v15 — one shared conv1d
backbone, per-branch ``(μ, logvar)`` heads).

All four variants emit the same ``(mean, logvar)`` interface — the
downstream bottleneck (Gaussian reparam / VQ / product-space VQ) decides
how to consume them. For VQ variants, only ``mean`` is used; ``logvar``
is ignored.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from c2a.codegen.kinematics_vae.backbone.mlp import Mlp


class MotionEncoder(nn.Module):
    """Flat-MLP encoder ``q(z | S^motion_t)`` returning ``(mean, logvar)``."""

    layer_sizes: Sequence[int] = (256, 128)
    latent_dim: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, w, feat_dim)
        batch = x.shape[0]
        x = x.reshape(batch, -1)
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(x)
        mean = nn.Dense(self.latent_dim, name="mean_head")(h)
        logvar = nn.Dense(self.latent_dim, name="logvar_head")(h)
        return mean, logvar


class MotionEncoderConv1D(nn.Module):
    """Temporal-conv encoder ``q(z | S^motion_t)`` returning ``(mean, logvar)``.

    Treats the input window as a ``(time, feat)`` sequence and applies
    stacked 1D convolutions with translation invariance along the time
    axis. More parameter-efficient than flattening the window into a
    flat MLP and gives the encoder an inductive bias for motion patterns
    (it can learn to extract velocity from positional deltas across
    adjacent frames).
    """

    conv_channels: Sequence[int] = (64, 128, 256)
    kernel_size: int = 3
    head_layer_sizes: Sequence[int] = (256,)
    latent_dim: int = 16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, w, feat_dim) — flax Conv interprets last axis as channels.
        h = x
        for i, ch in enumerate(self.conv_channels):
            h = nn.Conv(
                features=ch,
                kernel_size=(self.kernel_size,),
                padding="VALID",
                name=f"conv_{i}",
            )(h)
            h = nn.elu(h)
        h = h.reshape(h.shape[0], -1)
        if self.head_layer_sizes:
            h = Mlp(
                layer_sizes=tuple(self.head_layer_sizes),
                activate_final=True,
                name="head",
            )(h)
        mean = nn.Dense(self.latent_dim, name="mean_head")(h)
        logvar = nn.Dense(self.latent_dim, name="logvar_head")(h)
        return mean, logvar


class MotionEncoderConv1DSplit(nn.Module):
    """Body-part-split temporal-conv encoder (independent backbones).

    Runs :class:`MotionEncoderConv1D` independently on slices of the
    input feature vector — one per body group — and concatenates the
    resulting ``(mean, logvar)`` outputs. Guarantees each group has
    dedicated latent capacity instead of competing with high-variance
    groups (e.g. torso/head) in a shared bottleneck.

    ``branch_input_indices`` is a tuple of int-tuples giving, for each
    branch, the feat-dim indices the branch consumes.
    ``branch_latent_dims`` gives the per-branch latent dimensionality.
    Output latent dim is the sum.
    """

    branch_input_indices: Tuple[Tuple[int, ...], ...]
    branch_latent_dims: Tuple[int, ...]
    conv_channels: Sequence[int] = (64, 128, 256)
    kernel_size: int = 3
    head_layer_sizes: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        means, lvs = [], []
        for i, (idx, ld) in enumerate(
            zip(self.branch_input_indices, self.branch_latent_dims)
        ):
            x_branch = x[..., jnp.asarray(idx, dtype=jnp.int32)]
            mu_b, lv_b = MotionEncoderConv1D(
                conv_channels=tuple(self.conv_channels),
                kernel_size=self.kernel_size,
                head_layer_sizes=tuple(self.head_layer_sizes),
                latent_dim=int(ld),
                name=f"branch_{i}",
            )(x_branch)
            means.append(mu_b)
            lvs.append(lv_b)
        return jnp.concatenate(means, axis=-1), jnp.concatenate(lvs, axis=-1)


class MotionEncoderConv1DSharedSplit(nn.Module):
    """Shared-backbone, branched-output encoder (v15 — default for c2a).

    Mid-point between :class:`MotionEncoderConv1D` (single backbone,
    single z) and :class:`MotionEncoderConv1DSplit` (independent
    per-branch backbones). One conv1d stack sees ALL input features so
    cross-branch correlations (gait-phase ↔ root velocity,
    limb-stride-phase ↔ torso pitch) are available in the hidden
    representation. The final layer branches into per-branch ``(μ,
    logvar)`` heads producing per-branch latent slices.

    Per-branch slice layout in the concatenated output is fixed by the
    order of ``branch_latent_dims``. For the c2a default of
    ``(root=4, axial=8, limb=12)`` the concatenated ``(mean, logvar)``
    has shape ``(B, 24)`` ordered ``[root | axial | limb]``.
    """

    branch_latent_dims: Tuple[int, ...]
    conv_channels: Sequence[int] = (64, 128, 256)
    kernel_size: int = 3
    head_layer_sizes: Sequence[int] = (256,)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, w, feat_dim) — ALL features (no per-branch slicing).
        h = x
        for i, ch in enumerate(self.conv_channels):
            h = nn.Conv(
                features=ch,
                kernel_size=(self.kernel_size,),
                padding="VALID",
                name=f"conv_{i}",
            )(h)
            h = nn.elu(h)
        h = h.reshape(h.shape[0], -1)
        if self.head_layer_sizes:
            h = Mlp(
                layer_sizes=tuple(self.head_layer_sizes),
                activate_final=True,
                name="shared_head",
            )(h)
        # Branched (μ, logvar) projections.
        means, lvs = [], []
        for i, ld in enumerate(self.branch_latent_dims):
            mu_b = nn.Dense(int(ld), name=f"mean_head_{i}")(h)
            lv_b = nn.Dense(int(ld), name=f"logvar_head_{i}")(h)
            means.append(mu_b)
            lvs.append(lv_b)
        return jnp.concatenate(means, axis=-1), jnp.concatenate(lvs, axis=-1)
