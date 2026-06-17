"""Tests for c2a.code2code.PSVQVAEAdapter.

Product-space VQ-VAE adapter — same contract as VQVAEAdapter but with
per-group codebooks living in the `vq_state` collection and a folded
product index for the unified codebook lookup.

Critical invariants beyond the VQVAE tests:

1. **Per-group lookup composition**: ``codebook_lookup(c)`` equals
   ``concat([per_group_cb[g][c_g] for g])`` where ``c_g`` is the
   unfold of ``c`` per the same place-value scheme as
   ``fold_to_product_index``.
2. **Argmin over product codebook == per-group argmins**: a fundamental
   property of product quantization. ``argmin_c d_l2(z, product_cb[c]) ==
   fold_product([argmin_{c_g} d_l2(z_g, per_group_cb[g][c_g]) for g])``.
   If this fails, the diagnostic argmin is meaningless.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_CKPT_DIR = Path(
    "/home/jovyan/vast/kaiwen/code2act/outputs/tokenizer/"
    "trained_kinematics_vae_ckpts/ps_vqvae/step_0030000"
)
_REF_H5 = Path(
    "/home/jovyan/vast/kaiwen/code2act/data/rodent/rodent_reference_clips.h5"
)
_BEST_CODES_NPZ = Path(
    "/home/jovyan/vast/kaiwen/code2act/outputs/tokenizer/"
    "trained_kinematics_vae_ckpts/ps_vqvae/best_codes.npz"
)


def _require_ckpt():
    if not _CKPT_DIR.exists():
        pytest.skip(f"PS-VQ-VAE ckpt not available: {_CKPT_DIR}")
    if not _REF_H5.exists():
        pytest.skip(f"reference clips not available: {_REF_H5}")
    if not _BEST_CODES_NPZ.exists():
        pytest.skip(f"PS-VQ-VAE codes_npz not available: {_BEST_CODES_NPZ}")


def _load_adapter():
    from c2a.code2code import PSVQVAEAdapter
    data = np.load(_BEST_CODES_NPZ, allow_pickle=False)
    group_dims = tuple(int(d) for d in data["ps_group_dims"])
    group_codebook_sizes = tuple(int(k) for k in data["ps_group_codebook_sizes"])
    return PSVQVAEAdapter.from_ckpt_dir(
        _CKPT_DIR,
        group_dims=group_dims,
        group_codebook_sizes=group_codebook_sizes,
        n_joints=67,
        use_qvel=False,
        window_len=10,
        horizon=5,
    )


def _load_one_clip_qpos(clip_length: int = 250) -> np.ndarray:
    import h5py

    with h5py.File(_REF_H5, "r") as f:
        flat = f["qpos"][:]
    n_total = flat.shape[0] // clip_length
    clipped = flat[: n_total * clip_length].reshape(n_total, clip_length, -1)
    return clipped[0].astype(np.float32)


# ============================================================================
# Loading
# ============================================================================


def test_ps_vqvae_adapter_loads():
    _require_ckpt()
    adapter = _load_adapter()
    assert adapter.name == "ps_vqvae"
    assert adapter.window_len == 10
    assert adapter.codebook_size > 0  # K_total = prod(group_codebook_sizes)
    assert adapter.code_dim > 0       # sum(group_dims)
    # Sanity: codebook_size factors into the per-group sizes from the npz.
    data = np.load(_BEST_CODES_NPZ, allow_pickle=False)
    expected_K = int(np.prod(data["ps_group_codebook_sizes"]))
    assert adapter.codebook_size == expected_K
    cb = adapter.all_codebook_entries()
    assert cb.shape == (adapter.codebook_size, adapter.code_dim)


def test_ps_vqvae_adapter_rejects_dim_mismatch():
    from c2a.code2code import PSVQVAEAdapter
    if not _CKPT_DIR.exists():
        pytest.skip("ckpt missing")
    with pytest.raises(ValueError, match="group_dims"):
        PSVQVAEAdapter.from_ckpt_dir(
            _CKPT_DIR,
            group_dims=(2, 8, 12),  # WRONG (root should be 4)
            group_codebook_sizes=(4, 4, 4),
        )


# ============================================================================
# Encoder equivalence (same as VQ-VAE test, different ckpt)
# ============================================================================


def test_ps_encode_window_matches_encode_motion_array():
    _require_ckpt()
    adapter = _load_adapter()

    from c2a.codegen.kinematics_vae.backbone.codegen import (
        encode_motion_array, load_checkpoint,
    )
    from c2a.codegen.kinematics_vae.ps_vqvae.bottleneck import PSVQVAEBottleneck

    qpos = _load_one_clip_qpos(clip_length=250)
    qpos_b = qpos[None]

    data = np.load(_BEST_CODES_NPZ, allow_pickle=False)
    group_dims = tuple(int(d) for d in data["ps_group_dims"])
    group_codebook_sizes = tuple(int(k) for k in data["ps_group_codebook_sizes"])
    bottleneck = PSVQVAEBottleneck(
        group_dims=group_dims,
        group_codebook_sizes=group_codebook_sizes,
    )
    modules, params = load_checkpoint(
        _CKPT_DIR,
        bottleneck=bottleneck,
        n_joints=67,
        use_qvel=False,
        window_len=10,
        horizon=5,
    )
    encoded = encode_motion_array(
        qpos_b,
        modules,
        params["encoder"],
        params["bottleneck"],
        bn_buffers=params.get("bottleneck_buffers"),
        n_joints=67,
        use_qvel=False,
        window_len=10,
        horizon=5,
        deterministic=True,
    )
    t = 50
    w = 10
    z_adapter = adapter.encode_from_qpos(jnp.asarray(qpos[t : t + w]))
    z_emanate = jnp.asarray(encoded.mean[0, t + w - 1])

    np.testing.assert_allclose(
        np.asarray(z_adapter), np.asarray(z_emanate),
        rtol=5e-2, atol=5e-3,
    )


# ============================================================================
# Per-group lookup composition
# ============================================================================


def test_codebook_lookup_equals_per_group_concat():
    """``codebook_lookup(c) == concat(per_group_cb[g][c_g] for g)``
    where ``c_g`` is the unfold of ``c`` per the same place-value scheme
    as ``fold_to_product_index``."""
    _require_ckpt()
    adapter = _load_adapter()

    from c2a.codegen.kinematics_vae.ps_vqvae.tokenize import (
        fold_to_product_index,
    )

    per_group_cb = [np.asarray(cb) for cb in adapter._a.per_group_codebooks]
    K = list(adapter._a.group_codebook_sizes)
    G = len(K)

    # Spot-check several product indices, including corners and middle.
    cb_size = adapter.codebook_size
    test_indices = [0, 1, cb_size // 2, cb_size - 1]
    for c in test_indices:
        # Unfold c per place-value scheme.
        pv = [1] * G
        acc = 1
        for g in range(G - 1, -1, -1):
            pv[g] = acc
            acc *= K[g]
        c_per_group = np.array(
            [(c // pv[g]) % K[g] for g in range(G)], dtype=np.int32
        )
        # Sanity: re-fold should recover c.
        re_folded = int(fold_to_product_index(c_per_group, K))
        assert re_folded == c, (
            f"unfold/fold mismatch for c={c}: per_group={c_per_group}, "
            f"re_folded={re_folded}"
        )
        # Composed expected entry.
        expected = np.concatenate(
            [per_group_cb[g][c_per_group[g]] for g in range(G)], axis=-1,
        )
        got = np.asarray(adapter.codebook_lookup(jnp.int32(c)))
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-7)


def test_all_codebook_entries_matches_codebook_lookup():
    """``all_codebook_entries()[c] == codebook_lookup(c)`` for every c."""
    _require_ckpt()
    adapter = _load_adapter()
    cb = np.asarray(adapter.all_codebook_entries())
    for c in range(adapter.codebook_size):
        got = np.asarray(adapter.codebook_lookup(jnp.int32(c)))
        np.testing.assert_allclose(cb[c], got, rtol=1e-6, atol=1e-7)


# ============================================================================
# Distance properties
# ============================================================================


def test_self_distance_zero():
    _require_ckpt()
    adapter = _load_adapter()
    for c in [0, adapter.codebook_size // 2, adapter.codebook_size - 1]:
        target = adapter.codebook_lookup(jnp.int32(c))
        d_l2 = float(adapter.distance(target, jnp.int32(c), distance_form="l2"))
        d_cos = float(adapter.distance(target, jnp.int32(c), distance_form="cosine"))
        assert d_l2 < 1e-5
        assert d_cos < 1e-5


def test_distance_ranges():
    _require_ckpt()
    adapter = _load_adapter()
    rng = jax.random.PRNGKey(0)
    z = jax.random.normal(rng, (adapter.code_dim,))
    for c in [0, 5, adapter.codebook_size - 1]:
        d_l2 = float(adapter.distance(z, jnp.int32(c), distance_form="l2"))
        d_cos = float(adapter.distance(z, jnp.int32(c), distance_form="cosine"))
        assert d_l2 >= 0.0
        assert 0.0 - 1e-5 <= d_cos <= 2.0 + 1e-5


# ============================================================================
# Argmin agreement — fundamental product-quantization property
# ============================================================================


def test_argmin_l2_matches_per_group_argmins():
    """For L2 distance, the product-codebook argmin equals the
    folded-up concatenation of per-group argmins.

    This is a fundamental property of product quantization: since
    ``||z - cb[c]||^2 = sum_g ||z_g - cb_g[c_g]||^2`` and the groups
    are independent, the argmin factorizes. If this fails, the
    discrete-accuracy diagnostic is computing something other than
    what it claims.
    """
    _require_ckpt()
    adapter = _load_adapter()

    qpos = _load_one_clip_qpos(clip_length=250)
    z = adapter.encode_from_qpos(jnp.asarray(qpos[50:60]))

    # adapter argmin over the full product codebook
    dists = adapter.distances_to_all_codes(z, distance_form="l2")
    adapter_argmin = int(jnp.argmin(dists))

    # Per-group argmins
    per_group_cb = [np.asarray(cb) for cb in adapter._a.per_group_codebooks]
    group_dims = list(adapter._a.group_dims)
    z_np = np.asarray(z)
    offset = 0
    per_group_argmins: list[int] = []
    for g, (cb_g, D_g) in enumerate(zip(per_group_cb, group_dims)):
        z_g = z_np[offset:offset + D_g]
        d_g = ((z_g[None, :] - cb_g) ** 2).sum(axis=-1)
        per_group_argmins.append(int(np.argmin(d_g)))
        offset += D_g

    from c2a.codegen.kinematics_vae.ps_vqvae.tokenize import (
        fold_to_product_index,
    )
    expected_folded = int(fold_to_product_index(
        np.array(per_group_argmins, dtype=np.int32),
        list(adapter._a.group_codebook_sizes),
    ))

    assert adapter_argmin == expected_folded, (
        f"product-codebook argmin={adapter_argmin} but "
        f"folded per-group argmins={expected_folded} (per_group="
        f"{per_group_argmins}). Product quantization invariant broken."
    )


# ============================================================================
# JIT correctness
# ============================================================================


def test_ps_encode_under_jit():
    _require_ckpt()
    adapter = _load_adapter()

    @jax.jit
    def encode_jit(qpos):
        return adapter.encode_from_qpos(qpos)

    qpos = _load_one_clip_qpos(clip_length=250)
    window = jnp.asarray(qpos[50:60])

    z_eager = adapter.encode_from_qpos(window)
    z_jit = encode_jit(window)
    np.testing.assert_allclose(
        np.asarray(z_eager), np.asarray(z_jit), rtol=1e-2, atol=1e-3,
    )


def test_ps_codebook_lookup_under_jit():
    _require_ckpt()
    adapter = _load_adapter()

    @jax.jit
    def lookup_jit(c):
        return adapter.codebook_lookup(c)

    for c in [0, 5, adapter.codebook_size - 1]:
        eager = adapter.codebook_lookup(jnp.int32(c))
        jit_val = lookup_jit(jnp.int32(c))
        np.testing.assert_allclose(
            np.asarray(eager), np.asarray(jit_val), rtol=1e-6, atol=1e-7,
        )
