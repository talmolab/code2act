"""Concrete code2code adapter for the product-space VQ-VAE tokenizer.

PS-VQ-VAE splits the encoder latent into G groups along the channel
axis (root=4, axial=8, limb=12 by default) and quantizes each group
against its own EMA codebook. A "product index" c ∈ [0, ∏K_g) folds
the per-group codes into a single integer the policy embeds.

This adapter:

* Reads ``meta.json`` for the branch latent dims (≡ group dims).
* Reads the per-group codebooks from ``bottleneck_buffers.msgpack``
  (PS-VQ codebooks live in the ``vq_state`` mutable collection at
  ``params["bottleneck_buffers"]["vq_state"]["mgq"]["group_<g>"]["codebook"]``,
  NOT in ``params["bottleneck"]["params"]``).
* Precomputes the full product codebook ``(K_total, sum(D_g))`` once
  at init so :meth:`all_codebook_entries` is a single matrix
  reference — avoids fold/unfold ops on the JIT hot path.

``group_codebook_sizes`` is NOT stored in ``meta.json`` (a pre-existing
gap in the tokenizer's checkpoint format). The caller must supply it
to :meth:`from_ckpt_dir`. It's available in ``best_codes.npz`` under
the key ``ps_group_codebook_sizes``; callers building the adapter
inside ``train_moseq_decoder`` should propagate it from the npz to
the adapter ctor automatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from c2a.code2code.adapter_base import C2CTokenizerAdapter
from c2a.codegen.kinematics_vae.backbone.codegen import load_checkpoint
from c2a.codegen.kinematics_vae.ps_vqvae.bottleneck import PSVQVAEBottleneck

log = logging.getLogger(__name__)


@dataclass
class _PSVQVAEArtifacts:
    """Static (immutable) arrays/modules pulled from a PS-VQ-VAE ckpt."""

    encoder_module: object  # MotionEncoderConv1DSharedSplit (Flax)
    encoder_params: dict
    # Per-group codebooks: tuple of (K_g, D_g) jnp arrays.
    per_group_codebooks: tuple[jnp.ndarray, ...]
    # Precomputed product-space codebook: (K_total, sum(D_g)) jnp.
    product_codebook: jnp.ndarray
    normalizer_mean: jnp.ndarray
    normalizer_std: jnp.ndarray
    active_joints: jnp.ndarray
    group_dims: tuple[int, ...]
    group_codebook_sizes: tuple[int, ...]
    # mixed-radix place values used to unfold a product index. Same
    # convention as fold_to_product_index: first group = most
    # significant digit.
    place_values: tuple[int, ...]


class PSVQVAEAdapter(C2CTokenizerAdapter):
    """Frozen PS-VQ-VAE tokenizer adapter."""

    name = "ps_vqvae"

    def __init__(
        self,
        artifacts: _PSVQVAEArtifacts,
        *,
        n_joints: int,
        use_qvel: bool,
        window_len: int,
        feat_dim: int,
        codebook_size: int,
        code_dim: int,
    ) -> None:
        self._a = artifacts
        self.n_joints = int(n_joints)
        self.use_qvel = bool(use_qvel)
        self.window_len = int(window_len)
        self.feat_dim = int(feat_dim)
        self.codebook_size = int(codebook_size)
        self.code_dim = int(code_dim)

    @classmethod
    def from_ckpt_dir(
        cls,
        ckpt_dir: str | Path,
        *,
        group_dims: tuple[int, ...],
        group_codebook_sizes: tuple[int, ...],
        n_joints: int = 67,
        use_qvel: bool = False,
        window_len: int = 10,
        horizon: int = 5,
        conv_channels: tuple[int, ...] = (64, 128, 256),
        conv_kernel_size: int = 3,
        conv_head_layers: tuple[int, ...] = (256,),
        decoder_layer_sizes: tuple[int, ...] = (256, 256, 256),
    ) -> "PSVQVAEAdapter":
        """Load a PS-VQ-VAE ckpt and build the adapter.

        ``group_dims`` and ``group_codebook_sizes`` MUST match what the
        ckpt was trained with — they are not stored in ``meta.json``.
        The caller usually pulls these from ``best_codes.npz``'s
        ``ps_group_dims`` / ``ps_group_codebook_sizes`` fields.
        """
        ckpt_dir = Path(ckpt_dir).resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"PSVQVAEAdapter: ckpt directory does not exist: {ckpt_dir}"
            )

        if len(group_dims) != len(group_codebook_sizes):
            raise ValueError(
                f"group_dims (len={len(group_dims)}) and "
                f"group_codebook_sizes (len={len(group_codebook_sizes)}) "
                f"must have matching length."
            )

        meta = json.loads((ckpt_dir / "meta.json").read_text())
        meta_branch_dims = tuple(int(d) for d in meta["branch_latent_dims"])
        if meta_branch_dims != tuple(int(d) for d in group_dims):
            raise ValueError(
                f"group_dims={tuple(group_dims)} does not match the ckpt's "
                f"meta.json branch_latent_dims={meta_branch_dims}. "
                f"For PS-VQ-VAE the per-group code_dims must equal the "
                f"per-branch encoder latent dims."
            )
        feat_dim = int(meta["feat_dim"])
        code_dim = int(sum(group_dims))
        K_total = int(np.prod(group_codebook_sizes))

        bottleneck = PSVQVAEBottleneck(
            group_dims=tuple(int(d) for d in group_dims),
            group_codebook_sizes=tuple(int(K) for K in group_codebook_sizes),
        )
        modules, params = load_checkpoint(
            ckpt_dir,
            bottleneck=bottleneck,
            n_joints=n_joints,
            use_qvel=use_qvel,
            window_len=window_len,
            horizon=horizon,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
            conv_head_layers=conv_head_layers,
            decoder_layer_sizes=decoder_layer_sizes,
        )

        # Codebooks live in the vq_state collection, NOT in params.
        vq_state = params.get("bottleneck_buffers", {}).get("vq_state", {})
        if "mgq" not in vq_state:
            raise RuntimeError(
                f"PS-VQ-VAE ckpt at {ckpt_dir} has no vq_state/mgq "
                f"subtree — the bottleneck_buffers.msgpack is missing or "
                f"the bottleneck was not constructed via PSVQVAEBottleneck."
            )
        mgq = vq_state["mgq"]

        per_group: list[jnp.ndarray] = []
        for g, (K_g, D_g) in enumerate(zip(group_codebook_sizes, group_dims)):
            key = f"group_{g}"
            if key not in mgq:
                raise RuntimeError(
                    f"PS-VQ-VAE ckpt missing vq_state/mgq/{key}/codebook. "
                    f"Found groups: {sorted(mgq.keys())}."
                )
            cb_g = np.asarray(mgq[key]["codebook"])
            if cb_g.shape != (int(K_g), int(D_g)):
                raise RuntimeError(
                    f"PS-VQ-VAE group_{g} codebook shape {cb_g.shape} != "
                    f"expected ({K_g}, {D_g}); ckpt and config disagree."
                )
            per_group.append(jnp.asarray(cb_g, dtype=jnp.float32))

        # Precompute the full product codebook (K_total, sum(D_g)).
        # Row-major over (c_0, c_1, ..., c_{G-1}) with c_0 as the most
        # significant digit — matching fold_to_product_index.
        product_cb = _build_product_codebook(per_group, group_codebook_sizes)

        # Mixed-radix place values for unfold (same as in
        # fold_to_product_index): pv[g] = prod(K[g+1:]).
        K_list = [int(K) for K in group_codebook_sizes]
        pv = [1] * len(K_list)
        acc = 1
        for g in range(len(K_list) - 1, -1, -1):
            pv[g] = acc
            acc *= K_list[g]

        artifacts = _PSVQVAEArtifacts(
            encoder_module=modules.encoder,
            encoder_params=params["encoder"],
            per_group_codebooks=tuple(per_group),
            product_codebook=product_cb,
            normalizer_mean=jnp.asarray(modules.normalizer.mean, dtype=jnp.float32),
            normalizer_std=jnp.asarray(modules.normalizer.std, dtype=jnp.float32),
            active_joints=jnp.asarray(modules.active_joints, dtype=jnp.int32),
            group_dims=tuple(int(d) for d in group_dims),
            group_codebook_sizes=tuple(int(K) for K in group_codebook_sizes),
            place_values=tuple(pv),
        )

        log.info(
            "PSVQVAEAdapter loaded from %s: group_dims=%s, "
            "group_codebook_sizes=%s, K_total=%d, code_dim=%d, "
            "feat_dim=%d, n_active_joints=%d",
            ckpt_dir, tuple(group_dims), tuple(group_codebook_sizes),
            K_total, code_dim, feat_dim,
            int(artifacts.active_joints.size),
        )

        return cls(
            artifacts,
            n_joints=n_joints,
            use_qvel=use_qvel,
            window_len=window_len,
            feat_dim=feat_dim,
            codebook_size=K_total,
            code_dim=code_dim,
        )

    # ------------------------------------------------------------------
    # ABC implementations
    # ------------------------------------------------------------------

    def encode_window(self, motion_window: jnp.ndarray) -> jnp.ndarray:
        x = motion_window[None, :, :]
        mean, _logvar = self._a.encoder_module.apply(
            self._a.encoder_params, x,
        )
        return mean[0]

    def codebook_lookup(self, code_idx: jnp.ndarray) -> jnp.ndarray:
        """Single-entry product-codebook lookup, JIT-friendly.

        Unfolds the product index into per-group codes via fixed
        place-value arithmetic, then concatenates per-group codebook
        entries. Equivalent to ``self._a.product_codebook[code_idx]``
        but doesn't require materializing the K_total × code_dim
        matrix at trace time — useful when K_total is large.
        """
        c = code_idx.astype(jnp.int32)
        per_group_entries = []
        for g, (cb_g, pv_g, K_g) in enumerate(zip(
            self._a.per_group_codebooks,
            self._a.place_values,
            self._a.group_codebook_sizes,
        )):
            c_g = (c // jnp.int32(pv_g)) % jnp.int32(K_g)
            per_group_entries.append(cb_g[c_g])
        return jnp.concatenate(per_group_entries, axis=-1)

    def all_codebook_entries(self) -> jnp.ndarray:
        """``(K_total, code_dim)`` product codebook, precomputed at init."""
        return self._a.product_codebook

    def qpos_to_motion_features(self, qpos_buf: jnp.ndarray) -> jnp.ndarray:
        if self.use_qvel:
            raise NotImplementedError(
                "PSVQVAEAdapter.qpos_to_motion_features does not currently "
                "support use_qvel=True."
            )
        p = qpos_buf[:, :3]
        theta = qpos_buf[:, 3:7]
        q_full = qpos_buf[:, 7:]
        q = jnp.take(q_full, self._a.active_joints, axis=-1)
        return jnp.concatenate([p, theta, q], axis=-1)

    def normalize_motion(self, motion: jnp.ndarray) -> jnp.ndarray:
        return (motion - self._a.normalizer_mean) / self._a.normalizer_std


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_product_codebook(
    per_group_codebooks: list[jnp.ndarray],
    group_codebook_sizes: tuple[int, ...],
) -> jnp.ndarray:
    """Materialize the ``(K_total, sum(D_g))`` product codebook.

    Row order matches ``fold_to_product_index``: for product index ``c``,
    the unfold is ``(c // pv[0]) % K[0], (c // pv[1]) % K[1], ...``
    with ``pv[g] = prod(K[g+1:])``.
    """
    K = [int(k) for k in group_codebook_sizes]
    K_total = int(np.prod(K))
    G = len(K)

    pv = [1] * G
    acc = 1
    for g in range(G - 1, -1, -1):
        pv[g] = acc
        acc *= K[g]

    # Build (K_total, sum(D_g)) entry-by-entry using numpy first, then
    # convert. K_total is small enough (≤ a few hundred) that this is
    # cheap one-time at startup.
    np_codebooks = [np.asarray(cb) for cb in per_group_codebooks]
    total_dim = sum(cb.shape[1] for cb in np_codebooks)
    out = np.zeros((K_total, total_dim), dtype=np.float32)
    for c in range(K_total):
        offset = 0
        for g in range(G):
            c_g = (c // pv[g]) % K[g]
            d_g = np_codebooks[g].shape[1]
            out[c, offset:offset + d_g] = np_codebooks[g][c_g]
            offset += d_g
    return jnp.asarray(out, dtype=jnp.float32)
