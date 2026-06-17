"""Concrete code2code adapter for the vanilla VQ-VAE tokenizer family.

Loads a frozen VQ-VAE checkpoint produced by
``c2a.codegen.kinematics_vae.vqvae.train`` and exposes the JIT-friendly
per-window encoder + codebook lookup that the env wrapper uses to
compute the code-match reward.

The checkpoint dir must contain (written by
``c2a.codegen.kinematics_vae.backbone.train_loop._save_checkpoint``):

- ``encoder.msgpack`` — :class:`MotionEncoderConv1DSharedSplit` params
- ``bottleneck.msgpack`` — :class:`VQVAEBottleneck` params, which holds
  the ``(K, code_dim)`` codebook at
  ``params["quantizer"]["codebook"]``.
- ``normalizer.npz`` — ``mean``, ``std`` arrays of shape ``(feat_dim,)``.
- ``active_joints.npy`` — int indices of joints kept by the
  auto-detector during training.
- ``meta.json`` — ``branch_latent_dims``, ``branch_input_indices``,
  ``feat_dim``, etc. (codebook architecture).
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
from c2a.codegen.kinematics_vae.vqvae.bottleneck import VQVAEBottleneck

log = logging.getLogger(__name__)


@dataclass
class _VQVAEArtifacts:
    """Static (immutable) arrays/modules pulled from a VQ-VAE ckpt."""

    encoder_module: object  # MotionEncoderConv1DSharedSplit (Flax)
    encoder_params: dict
    codebook: jnp.ndarray             # (K, code_dim)
    normalizer_mean: jnp.ndarray      # (feat_dim,)
    normalizer_std: jnp.ndarray       # (feat_dim,)
    active_joints: jnp.ndarray        # (n_active,) int32
    branch_latent_dims: tuple[int, ...]
    branch_input_indices: tuple[tuple[int, ...], ...]


class VQVAEAdapter(C2CTokenizerAdapter):
    """Frozen VQ-VAE tokenizer adapter for in-loop code-match reward.

    Usage::

        adapter = VQVAEAdapter.from_ckpt_dir(
            ckpt_dir="outputs/tokenizer/.../vqvae/step_0030000",
            n_joints=67, use_qvel=False, window_len=10, horizon=5,
        )
        z = adapter.encode_from_qpos(qpos_buf)  # (code_dim,)
        d = adapter.distance(z, c_target, distance_form="cosine")
    """

    name = "vqvae"

    def __init__(
        self,
        artifacts: _VQVAEArtifacts,
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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_ckpt_dir(
        cls,
        ckpt_dir: str | Path,
        *,
        n_joints: int = 67,
        use_qvel: bool = False,
        window_len: int = 10,
        horizon: int = 5,
        conv_channels: tuple[int, ...] = (64, 128, 256),
        conv_kernel_size: int = 3,
        conv_head_layers: tuple[int, ...] = (256,),
        decoder_layer_sizes: tuple[int, ...] = (256, 256, 256),
    ) -> "VQVAEAdapter":
        """Load a VQ-VAE ckpt and build the adapter.

        Args mirror the (now-removed) e2e_config.vqvae_arch fields. The
        ckpt's ``meta.json`` is the source of truth for codebook
        topology — ``branch_latent_dims``, ``branch_input_indices``, and
        ``feat_dim`` are read from there and used to instantiate the
        bottleneck with the matching ``codebook_size`` and ``code_dim``.
        """
        ckpt_dir = Path(ckpt_dir).resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"VQVAEAdapter: ckpt directory does not exist: {ckpt_dir}"
            )

        meta = json.loads((ckpt_dir / "meta.json").read_text())
        branch_latent_dims = tuple(int(d) for d in meta["branch_latent_dims"])
        branch_input_indices = tuple(
            tuple(int(i) for i in ix) for ix in meta["branch_input_indices"]
        )
        feat_dim = int(meta["feat_dim"])
        code_dim = int(sum(branch_latent_dims))

        # The codebook_size is encoded in the bottleneck msgpack itself
        # (first dim of the only param under "quantizer"). Peek without a
        # full Flax restore.
        cb_size = _peek_vqvae_codebook_size(ckpt_dir / "bottleneck.msgpack")

        bottleneck = VQVAEBottleneck(
            codebook_size=int(cb_size),
            code_dim=int(code_dim),
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

        codebook = jnp.asarray(
            params["bottleneck"]["params"]["quantizer"]["codebook"],
            dtype=jnp.float32,
        )
        if codebook.shape != (cb_size, code_dim):
            raise RuntimeError(
                f"Loaded VQ-VAE codebook shape {codebook.shape} does not "
                f"match (codebook_size={cb_size}, code_dim={code_dim}) "
                f"derived from meta.json — refusing to use a mismatched "
                f"ckpt."
            )

        artifacts = _VQVAEArtifacts(
            encoder_module=modules.encoder,
            encoder_params=params["encoder"],
            codebook=codebook,
            normalizer_mean=jnp.asarray(modules.normalizer.mean, dtype=jnp.float32),
            normalizer_std=jnp.asarray(modules.normalizer.std, dtype=jnp.float32),
            active_joints=jnp.asarray(modules.active_joints, dtype=jnp.int32),
            branch_latent_dims=branch_latent_dims,
            branch_input_indices=branch_input_indices,
        )

        log.info(
            "VQVAEAdapter loaded from %s: codebook=(K=%d, D=%d), "
            "feat_dim=%d, n_active_joints=%d, branch_latent_dims=%s",
            ckpt_dir, cb_size, code_dim, feat_dim,
            int(artifacts.active_joints.size), branch_latent_dims,
        )

        return cls(
            artifacts,
            n_joints=n_joints,
            use_qvel=use_qvel,
            window_len=window_len,
            feat_dim=feat_dim,
            codebook_size=int(cb_size),
            code_dim=code_dim,
        )

    # ------------------------------------------------------------------
    # ABC implementations
    # ------------------------------------------------------------------

    def encode_window(self, motion_window: jnp.ndarray) -> jnp.ndarray:
        """``(w, feat_dim) -> (code_dim,)``. JIT-friendly.

        The encoder consumes ``(batch, w, feat_dim)``. We add a length-1
        batch axis, run the encoder, and squeeze. For batched callers
        (e.g. a vmap over parallel envs) the surrounding vmap handles
        the leading dims — this method itself is single-sample.
        """
        x = motion_window[None, :, :]                       # (1, w, F)
        mean, _logvar = self._a.encoder_module.apply(
            self._a.encoder_params, x,
        )                                                    # (1, code_dim)
        return mean[0]                                       # (code_dim,)

    def codebook_lookup(self, code_idx: jnp.ndarray) -> jnp.ndarray:
        """Single-entry codebook lookup. ``int32 -> (code_dim,)``."""
        return self._a.codebook[code_idx.astype(jnp.int32)]

    def all_codebook_entries(self) -> jnp.ndarray:
        """``(K, code_dim)`` — the full codebook table."""
        return self._a.codebook

    def qpos_to_motion_features(self, qpos_buf: jnp.ndarray) -> jnp.ndarray:
        """JAX-friendly ``extract_motion_frames`` for ``use_qvel=False``.

        Args:
            qpos_buf: ``(w, 7 + n_joints)`` raw qpos.

        Returns:
            ``(w, feat_dim)`` raw motion features: ``[root_pos(3),
            root_quat(4), q_active(n_active)]``.

        Raises ``NotImplementedError`` for ``use_qvel=True`` — the v15
        default trains with ``use_qvel=False``. If you train a VQ-VAE
        with velocity blocks, extend this to also stack
        ``[root_vel(6), qdot_active(n_active)]`` and pass a qvel buffer.
        """
        if self.use_qvel:
            raise NotImplementedError(
                "VQVAEAdapter.qpos_to_motion_features does not currently "
                "support use_qvel=True; the in-env motion buffer would "
                "need to also track qvel. v15 default is use_qvel=False."
            )
        p = qpos_buf[:, :3]
        theta = qpos_buf[:, 3:7]
        q_full = qpos_buf[:, 7:]                             # (w, n_joints)
        q = jnp.take(q_full, self._a.active_joints, axis=-1)  # (w, n_active)
        return jnp.concatenate([p, theta, q], axis=-1)

    def normalize_motion(self, motion: jnp.ndarray) -> jnp.ndarray:
        """Per-feature z-score."""
        return (motion - self._a.normalizer_mean) / self._a.normalizer_std


def _peek_vqvae_codebook_size(bottleneck_msgpack_path: Path) -> int:
    """Read the codebook ``K`` from a bottleneck msgpack without a full restore.

    The msgpack stores the codebook as a leaf under
    ``params/quantizer/codebook`` with shape ``(K, D)``. We need ``K``
    *before* we instantiate the bottleneck module (since
    ``VQVAEBottleneck.codebook_size`` is required to build the right
    shape for the load). Mirrors the same helper from the deleted
    ``c2a/e2e_vqvae.py`` — kept here because no other module needs it.
    """
    import flax.serialization as fs

    restored = fs.msgpack_restore(bottleneck_msgpack_path.read_bytes())
    cb = restored["params"]["quantizer"]["codebook"]
    return int(np.asarray(cb).shape[0])
