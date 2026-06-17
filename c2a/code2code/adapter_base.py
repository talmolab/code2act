"""Abstract base for code2code in-loop tokenizer adapters.

Each concrete adapter wraps one frozen tokenizer family (VQ-VAE,
PS-VQ-VAE, VAE) and exposes a JIT-friendly per-window interface used
by :class:`c2a.moseq_env_wrapper.MoSeqImitation` to compute the
code-match reward.

The adapter is **NOT** part of the policy network's params — it is a
separate frozen object held by the env. PPO never updates it. This is
the architectural property that makes code2code immune to the
private-code-language drift that killed e2e_mode (see
``scratch/registry/2026-05-25-e2e-mode-removal/SKILL.md``).
"""

from __future__ import annotations

import abc
from typing import Literal

import jax
import jax.numpy as jnp


DistanceForm = Literal["l2", "cosine"]


class C2CTokenizerAdapter(abc.ABC):
    """Abstract adapter exposing a frozen tokenizer's encoder + codebook.

    Concrete subclasses provide:

    * :meth:`encode_window` — single forward pass through the
      tokenizer's encoder on a ``(w, feat_dim)`` motion window,
      returning the continuous latent ``z`` the codebook lives in.
    * :meth:`codebook_lookup` — fetch the codebook entry for a single
      code id.
    * :meth:`all_codebook_entries` — the full ``(K, code_dim)``
      codebook table (used for the diagnostic discrete-accuracy
      argmin).
    * :meth:`qpos_to_motion_features` — JAX-friendly port of
      :func:`c2a.codegen.kinematics_vae.backbone.frame_features.extract_motion_frames`
      operating on a ``(w, 7 + n_joints)`` raw qpos buffer.

    Static attributes (set by ``__init__``):

    Attributes:
        name: "vqvae" | "ps_vqvae" | "vae".
        n_joints: Articulated-joint count from the tokenizer's
            training (must match the env's MuJoCo model).
        use_qvel: Whether motion features include velocity blocks.
        window_len: Encoder window length ``w`` (10 for v15).
        feat_dim: Motion feature dimensionality at the encoder input.
        codebook_size: ``K``.
        code_dim: Latent dim the codebook lives in.
    """

    name: str
    n_joints: int
    use_qvel: bool
    window_len: int
    feat_dim: int
    codebook_size: int
    code_dim: int

    # ------------------------------------------------------------------
    # Concrete subclass contract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def encode_window(self, motion_window: jnp.ndarray) -> jnp.ndarray:
        """Encode a single ``(w, feat_dim)`` motion window to ``(code_dim,)``.

        The returned vector lives in the same continuous latent space as
        the codebook, so :meth:`codebook_lookup(c)` and
        ``encode_window(window)`` can be compared directly via
        :meth:`distance`.

        ``motion_window`` is the encoder-ready (post-normalization)
        feature representation. Use :meth:`qpos_to_motion_features` +
        :meth:`normalize_motion` to derive it from raw env qpos.
        """
        ...

    @abc.abstractmethod
    def codebook_lookup(self, code_idx: jnp.ndarray) -> jnp.ndarray:
        """Look up a single codebook entry: ``int32 -> (code_dim,)``."""
        ...

    @abc.abstractmethod
    def all_codebook_entries(self) -> jnp.ndarray:
        """Full codebook table ``(K, code_dim)``.

        Used for :meth:`distances_to_all_codes` (computes argmin to
        catch reward-hacking — see literature.md §"Design implications").
        """
        ...

    @abc.abstractmethod
    def qpos_to_motion_features(self, qpos_buf: jnp.ndarray) -> jnp.ndarray:
        """JAX-friendly motion-feature extraction.

        Mirrors :func:`extract_motion_frames` for ``use_qvel=False`` on a
        ``(w, 7 + n_joints)`` qpos buffer. Returns ``(w, feat_dim_raw)``
        — caller must :meth:`normalize_motion` before feeding to the
        encoder.
        """
        ...

    @abc.abstractmethod
    def normalize_motion(self, motion: jnp.ndarray) -> jnp.ndarray:
        """Per-feature z-score using the ckpt's stored normalizer.

        ``motion`` shape ``(..., feat_dim)`` (trailing axis only).
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers usable across all subclasses
    # ------------------------------------------------------------------

    def encode_from_qpos(self, qpos_buf: jnp.ndarray) -> jnp.ndarray:
        """Convenience: raw ``(w, 7+n_joints)`` qpos → ``(code_dim,)`` z.

        Composes the feature extractor + normalizer + encoder. Use this
        as the per-step env-side entry point. JIT-friendly.
        """
        motion = self.qpos_to_motion_features(qpos_buf)
        motion_norm = self.normalize_motion(motion)
        return self.encode_window(motion_norm)

    def distance(
        self,
        z: jnp.ndarray,
        code_idx: jnp.ndarray,
        distance_form: DistanceForm = "cosine",
    ) -> jnp.ndarray:
        """Scalar distance from ``z`` to ``codebook[code_idx]``.

        - ``l2``: squared Euclidean ``Σ (z - cb)²``.
        - ``cosine``: ``1 - cos(z, cb)`` ∈ ``[0, 2]``.

        The cosine form matches CALM (Tessler et al. 2023, the closest
        published precedent for frozen-encoder-as-reward) and is
        invariant to non-unit codebook magnitudes — important because
        VQ codebook entries are NOT unit-normalized.
        """
        target = self.codebook_lookup(code_idx)
        if distance_form == "l2":
            return jnp.sum((z - target) ** 2)
        if distance_form == "cosine":
            z_n = z / (jnp.linalg.norm(z) + 1e-8)
            t_n = target / (jnp.linalg.norm(target) + 1e-8)
            return 1.0 - jnp.dot(z_n, t_n)
        raise ValueError(
            f"distance_form must be 'l2' or 'cosine', got {distance_form!r}"
        )

    def distances_to_all_codes(
        self,
        z: jnp.ndarray,
        distance_form: DistanceForm = "cosine",
    ) -> jnp.ndarray:
        """``(K,)`` distances from ``z`` to every codebook entry.

        Used for the discrete-accuracy diagnostic
        ``argmin(distances) == c_target`` — a reward-hacking detector
        recommended by literature.md §"Design implications".
        """
        codebook = self.all_codebook_entries()  # (K, code_dim)
        if distance_form == "l2":
            diffs = z[None, :] - codebook
            return jnp.sum(diffs * diffs, axis=-1)
        if distance_form == "cosine":
            z_n = z / (jnp.linalg.norm(z) + 1e-8)
            cb_n = codebook / (
                jnp.linalg.norm(codebook, axis=-1, keepdims=True) + 1e-8
            )
            return 1.0 - cb_n @ z_n
        raise ValueError(
            f"distance_form must be 'l2' or 'cosine', got {distance_form!r}"
        )
