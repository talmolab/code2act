"""Concrete code2code adapter for the vanilla VAE tokenizer.

The vanilla VAE pipeline uses a Gaussian bottleneck (continuous z) and
discretizes via post-hoc K-Means clustering on ``z = mean``. The
adapter therefore:

* Loads the encoder + normalizer via the standard
  :func:`c2a.codegen.kinematics_vae.backbone.codegen.load_checkpoint`.
* Loads the K-Means centroids (and optional PCA) from
  ``tokenizer.pkl`` written by
  ``c2a.codegen.kinematics_vae.vae.tokenize``.
* For ``encode_window``: applies the encoder, returns the mean
  (deterministic).
* For ``codebook_lookup``: returns the K-Means centroid at the given
  index, which lives in the same encoder z-space.

If the ``tokenizer.pkl`` has ``pca != None``, we raise a clean error
(retrain without PCA to use code2code mode). The shipped checkpoints
in this repo all have ``pca=None`` — verified 2026-05-26 against
``outputs/tokenizer/trained_kinematics_vae_ckpts/vae/tokenizer.pkl``.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from c2a.code2code.adapter_base import C2CTokenizerAdapter
from c2a.codegen.kinematics_vae.backbone.codegen import load_checkpoint
from c2a.codegen.kinematics_vae.vae.bottleneck import GaussianBottleneck

log = logging.getLogger(__name__)


@dataclass
class _VAEArtifacts:
    """Static (immutable) arrays/modules pulled from a VAE ckpt + tokenizer.pkl."""

    encoder_module: object
    encoder_params: dict
    centroids: jnp.ndarray            # (K, code_dim) K-Means centroids
    normalizer_mean: jnp.ndarray
    normalizer_std: jnp.ndarray
    active_joints: jnp.ndarray


class VAEAdapter(C2CTokenizerAdapter):
    """Frozen VAE tokenizer adapter.

    K-Means centroids in the encoder's mean-space play the codebook
    role. ``encode_window(motion)`` returns the encoder mean directly;
    ``codebook_lookup(c)`` returns ``centroids[c]``. Both live in the
    same continuous latent space, so :meth:`distance` is well-defined
    without any PCA-or-other transformation.
    """

    name = "vae"

    def __init__(
        self,
        artifacts: _VAEArtifacts,
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
        tokenizer_pkl: str | Path,
        *,
        n_joints: int = 67,
        use_qvel: bool = False,
        window_len: int = 10,
        horizon: int = 5,
        conv_channels: tuple[int, ...] = (64, 128, 256),
        conv_kernel_size: int = 3,
        conv_head_layers: tuple[int, ...] = (256,),
        decoder_layer_sizes: tuple[int, ...] = (256, 256, 256),
    ) -> "VAEAdapter":
        """Load a VAE ckpt + tokenizer.pkl and build the adapter."""
        ckpt_dir = Path(ckpt_dir).resolve()
        tokenizer_pkl = Path(tokenizer_pkl).resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(
                f"VAEAdapter: ckpt directory does not exist: {ckpt_dir}"
            )
        if not tokenizer_pkl.exists():
            raise FileNotFoundError(
                f"VAEAdapter: tokenizer.pkl does not exist: {tokenizer_pkl}"
            )

        with open(tokenizer_pkl, "rb") as f:
            tok = pickle.load(f)
        if "kmeans" not in tok:
            raise RuntimeError(
                f"VAEAdapter: tokenizer.pkl at {tokenizer_pkl} has no "
                f"'kmeans' entry. Keys: {sorted(tok.keys())}."
            )
        if tok.get("pca") is not None:
            raise NotImplementedError(
                f"VAEAdapter: tokenizer.pkl at {tokenizer_pkl} has a "
                f"non-None PCA. The current adapter does not handle the "
                f"PCA-pre-projected K-Means case. Retrain the VAE "
                f"tokenizer with pca=None (which is the current default) "
                f"to use code2code mode."
            )
        kmeans = tok["kmeans"]
        if not hasattr(kmeans, "cluster_centers_"):
            raise RuntimeError(
                f"VAEAdapter: tokenizer.pkl's 'kmeans' object has no "
                f"cluster_centers_ attribute (expected sklearn.KMeans). "
                f"Got type {type(kmeans).__name__}."
            )
        centroids_np = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        codebook_size, code_dim = centroids_np.shape

        bottleneck = GaussianBottleneck()
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

        # Sanity: the encoder's latent dim must match the K-Means centroids.
        meta_code_dim = int(sum(modules.branch_latent_dims))
        if meta_code_dim != int(code_dim):
            raise RuntimeError(
                f"VAEAdapter: VAE encoder latent dim {meta_code_dim} != "
                f"K-Means centroid dim {code_dim} (tokenizer.pkl). The "
                f"tokenizer.pkl must come from the same training run as "
                f"the ckpt_dir."
            )

        feat_dim = int(modules.feat_dim)
        artifacts = _VAEArtifacts(
            encoder_module=modules.encoder,
            encoder_params=params["encoder"],
            centroids=jnp.asarray(centroids_np, dtype=jnp.float32),
            normalizer_mean=jnp.asarray(modules.normalizer.mean, dtype=jnp.float32),
            normalizer_std=jnp.asarray(modules.normalizer.std, dtype=jnp.float32),
            active_joints=jnp.asarray(modules.active_joints, dtype=jnp.int32),
        )

        log.info(
            "VAEAdapter loaded from %s + %s: codebook=(K=%d, D=%d), "
            "feat_dim=%d, n_active_joints=%d",
            ckpt_dir, tokenizer_pkl, codebook_size, code_dim, feat_dim,
            int(artifacts.active_joints.size),
        )

        return cls(
            artifacts,
            n_joints=n_joints,
            use_qvel=use_qvel,
            window_len=window_len,
            feat_dim=feat_dim,
            codebook_size=int(codebook_size),
            code_dim=int(code_dim),
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
        return self._a.centroids[code_idx.astype(jnp.int32)]

    def all_codebook_entries(self) -> jnp.ndarray:
        return self._a.centroids

    def qpos_to_motion_features(self, qpos_buf: jnp.ndarray) -> jnp.ndarray:
        if self.use_qvel:
            raise NotImplementedError(
                "VAEAdapter.qpos_to_motion_features does not currently "
                "support use_qvel=True."
            )
        p = qpos_buf[:, :3]
        theta = qpos_buf[:, 3:7]
        q_full = qpos_buf[:, 7:]
        q = jnp.take(q_full, self._a.active_joints, axis=-1)
        return jnp.concatenate([p, theta, q], axis=-1)

    def normalize_motion(self, motion: jnp.ndarray) -> jnp.ndarray:
        return (motion - self._a.normalizer_mean) / self._a.normalizer_std
