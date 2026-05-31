"""Vendi-score-based diversity reward for the c2c curriculum mode.

The Vendi score (Friedman & Dieng 2023, arXiv:2210.02410) is a kernel-based
measure of the *effective number of distinct items* in a sample.  For a
set ``{x_1, ..., x_n}`` and a PSD similarity kernel ``k`` with ``k(x,x)=1``:

    VS = exp(H(λ_1/n, ..., λ_n/n))

where ``λ_i`` are eigenvalues of the kernel matrix ``K`` (so ``trace(K)=n``)
and ``H`` is the Shannon entropy of the normalized spectrum.

* All items identical → VS = 1
* All items mutually orthogonal under k → VS = n
* k clusters of equal size, distinct between clusters → VS ≈ k

For the c2c curriculum we group rollout-encoded VAE z's by *current target
code* and reward each env by ``log(VS)`` for its group — rewarding the
batch for producing many distinct realizations of the same code class.

This module exposes:

* :func:`vendi_score_cosine` — scalar Vendi on an ``(n, d)`` matrix using
  the cosine kernel ``k(x,y) = 0.5 · (1 + cos(x,y))``.
* :func:`grouped_vendi_per_step` — JAX-friendly grouped Vendi over a batch
  of ``(B, d)`` z's plus per-env target codes; returns ``(B,)`` per-env
  ``log(VS_c)`` (zero when group size < threshold).
* :func:`grouped_vendi_per_timestep` — ``vmap`` over the time axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def vendi_score_cosine(Z: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    """Vendi score with cosine kernel.

    Args:
        Z: ``(n, d)`` matrix of items (latent vectors).
        eps: numerical floor for division and log.

    Returns:
        Scalar VS in [1, n].  For ``n == 1`` returns 1.0.
    """
    n = Z.shape[0]
    Z_norm = Z / (jnp.linalg.norm(Z, axis=-1, keepdims=True) + eps)
    K = 0.5 * (Z_norm @ Z_norm.T + 1.0)             # (n, n), K_ii = 1
    # K is symmetric PSD (by construction since cosine matrix is PSD and
    # 0.5*(K_cos + 1) is also PSD); eigvalsh is faster + numerically stable.
    eigvals = jnp.linalg.eigvalsh(K)
    p = jnp.clip(eigvals / n, eps, None)
    p = p / jnp.sum(p)
    H = -jnp.sum(p * jnp.log(p))
    return jnp.exp(H)


def grouped_vendi_per_step(
    z_step: jnp.ndarray,
    c_step: jnp.ndarray,
    num_codes: int,
    min_group_size: int,
) -> jnp.ndarray:
    """Vendi score per group, distributed back to per-env diversity reward.

    Implementation strategy: for each code c in [0, num_codes), build a
    mask of envs whose target code is c.  Compute the cosine kernel on
    the FULL batch (B×B), but masked so off-group entries become 1 (the
    additive identity for the cosine-shifted kernel — see note below).
    The per-group VS is then `exp(H(λ/n_c))` where n_c is the active
    group size and λ are eigenvalues of the masked sub-kernel.

    For JIT-friendliness we use a per-code scan with fixed-size matrices
    (B×B) rather than dynamic slicing.

    Args:
        z_step: ``(B, d)`` rollout-encoded z per env at this timestep.
        c_step: ``(B,)`` target code per env (int in [0, num_codes)).
        num_codes: codebook size K (for the group loop).
        min_group_size: groups smaller than this return ``log(VS) = 0``
            for their member envs (avoids noisy eigendecomp on tiny
            matrices).

    Returns:
        ``(B,)`` per-env ``log(VS_{c_b})`` (zero for envs in small groups).
    """
    B = z_step.shape[0]
    eps = 1e-10

    Z_norm = z_step / (jnp.linalg.norm(z_step, axis=-1, keepdims=True) + eps)
    cos_full = Z_norm @ Z_norm.T                     # (B, B) in [-1, 1]

    def per_code_log_vs(c):
        """Compute log(VS) for code c and broadcast to all envs in c's group.

        Strategy: build a B×B matrix that is the cosine-shifted kernel
        only on the in-group block and zero elsewhere.  Spectrum:

            n_c non-zero eigvals  =  eigvals of the in-group sub-kernel
            (B − n_c) zero eigvals  from the zero padding outside the group

        Trace = sum of in-group eigvals = n_c (diagonal of the cosine-
        shifted kernel is 1).  So normalizing the top-n_c eigvals by n_c
        gives the per-group probability distribution exactly.
        """
        in_group = (c_step == c).astype(jnp.float32)  # (B,)
        n_c = jnp.sum(in_group)                       # scalar group size

        outer_mask = in_group[:, None] * in_group[None, :]   # (B, B)
        # K_in_group block padded with zeros outside.  Diagonal in-group = 1.
        K_masked = 0.5 * (cos_full + 1.0) * outer_mask
        eigvals = jnp.linalg.eigvalsh(K_masked)        # (B,) sorted ascending
        eigvals_desc = jnp.sort(eigvals)[::-1]         # (B,) sorted descending

        # Top n_c eigvals are the non-zero ones (from the in-group block).
        positions = jnp.arange(B)
        keep_mask = (positions < n_c).astype(jnp.float32)
        eigvals_kept = eigvals_desc * keep_mask        # zero-out the bottom

        # Normalize by n_c (trace of in-group block) for the probability
        # distribution.  jnp.maximum guards the empty-group case.
        p = eigvals_kept / jnp.maximum(n_c, 1.0)
        # Entropy with masked support: only the top n_c positions contribute.
        log_p = jnp.where(p > eps, jnp.log(jnp.clip(p, eps, None)), 0.0)
        H = -jnp.sum(p * log_p)
        VS = jnp.exp(H)
        log_VS = jnp.log(jnp.maximum(VS, 1.0))

        # Gate on group size
        valid = (n_c >= min_group_size).astype(jnp.float32)
        return log_VS * valid, in_group               # both broadcast/scatter back

    # Scan over codes; accumulate per-env diversity reward.
    code_indices = jnp.arange(num_codes)
    log_vs_per_code, in_group_per_code = jax.vmap(per_code_log_vs)(code_indices)
    # log_vs_per_code: (num_codes,)
    # in_group_per_code: (num_codes, B)

    # Each env's reward = the log(VS) of the code group it belongs to.
    # Since each env has exactly one target code, only one row of
    # in_group_per_code is 1 for that env; sum over codes selects it.
    r_diversity = jnp.sum(
        log_vs_per_code[:, None] * in_group_per_code, axis=0
    )                                                  # (B,)
    return r_diversity


def grouped_vendi_per_timestep(
    z_seq: jnp.ndarray,
    c_seq: jnp.ndarray,
    num_codes: int,
    min_group_size: int,
) -> jnp.ndarray:
    """vmap of :func:`grouped_vendi_per_step` over the time axis.

    Args:
        z_seq: ``(T, B, d)`` rollout z's over time.
        c_seq: ``(T, B)`` target codes over time.
        num_codes: codebook size.
        min_group_size: see :func:`grouped_vendi_per_step`.

    Returns:
        ``(T, B)`` per-step per-env ``log(VS)`` diversity rewards.
    """
    return jax.vmap(
        grouped_vendi_per_step,
        in_axes=(0, 0, None, None),
    )(z_seq, c_seq, num_codes, min_group_size)
