"""Tests for c2a.code2code.VAEAdapter.

K-Means centroids in the encoder's mean-space play the codebook role.
Verified in pre-design inspection that the current shipped VAE
tokenizer has ``pca=None``, so the adapter doesn't need to handle PCA
projection (and explicitly rejects ckpts that do).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_CKPT_DIR = Path(
    "/home/jovyan/vast/kaiwen/code2act/outputs/tokenizer/"
    "trained_kinematics_vae_ckpts/vae/step_0030000"
)
_TOKENIZER_PKL = Path(
    "/home/jovyan/vast/kaiwen/code2act/outputs/tokenizer/"
    "trained_kinematics_vae_ckpts/vae/tokenizer.pkl"
)
_REF_H5 = Path(
    "/home/jovyan/vast/kaiwen/code2act/data/rodent/rodent_reference_clips.h5"
)


def _require_ckpt():
    if not _CKPT_DIR.exists():
        pytest.skip(f"VAE ckpt not available: {_CKPT_DIR}")
    if not _TOKENIZER_PKL.exists():
        pytest.skip(f"VAE tokenizer.pkl not available: {_TOKENIZER_PKL}")
    if not _REF_H5.exists():
        pytest.skip(f"reference clips not available: {_REF_H5}")


def _load_adapter():
    from c2a.code2code import VAEAdapter
    return VAEAdapter.from_ckpt_dir(
        _CKPT_DIR,
        tokenizer_pkl=_TOKENIZER_PKL,
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


def test_vae_adapter_loads():
    _require_ckpt()
    adapter = _load_adapter()
    assert adapter.name == "vae"
    assert adapter.window_len == 10
    assert adapter.codebook_size > 0
    assert adapter.code_dim > 0
    cb = adapter.all_codebook_entries()
    assert cb.shape == (adapter.codebook_size, adapter.code_dim)


def test_vae_adapter_rejects_missing_pkl():
    from c2a.code2code import VAEAdapter
    if not _CKPT_DIR.exists():
        pytest.skip("ckpt missing")
    with pytest.raises(FileNotFoundError):
        VAEAdapter.from_ckpt_dir(_CKPT_DIR, tokenizer_pkl="/tmp/nope.pkl")


# ============================================================================
# Encoder equivalence (same as VQVAE; encoder + bottleneck behavior identical
# when deterministic and we read the mean directly)
# ============================================================================


def test_vae_encode_window_matches_encode_motion_array():
    _require_ckpt()
    adapter = _load_adapter()

    from c2a.codegen.kinematics_vae.backbone.codegen import (
        encode_motion_array, load_checkpoint,
    )
    from c2a.codegen.kinematics_vae.vae.bottleneck import GaussianBottleneck

    qpos = _load_one_clip_qpos(clip_length=250)
    qpos_b = qpos[None]

    bottleneck = GaussianBottleneck()
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
        n_joints=67,
        use_qvel=False,
        window_len=10,
        horizon=5,
        deterministic=True,
        bottleneck_kwargs={"sigma_max": 0.0},
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
# Codebook lookup matches K-Means centroids
# ============================================================================


def test_codebook_lookup_matches_kmeans_centroids():
    _require_ckpt()
    adapter = _load_adapter()

    import pickle

    with open(_TOKENIZER_PKL, "rb") as f:
        tok = pickle.load(f)
    centroids = np.asarray(tok["kmeans"].cluster_centers_, dtype=np.float32)

    for c in [0, 1, adapter.codebook_size // 2, adapter.codebook_size - 1]:
        entry = np.asarray(adapter.codebook_lookup(jnp.int32(c)))
        np.testing.assert_allclose(entry, centroids[c], rtol=1e-6, atol=1e-7)


# ============================================================================
# Distance properties
# ============================================================================


def test_vae_self_distance_zero():
    _require_ckpt()
    adapter = _load_adapter()
    for c in [0, adapter.codebook_size // 2, adapter.codebook_size - 1]:
        target = adapter.codebook_lookup(jnp.int32(c))
        d_l2 = float(adapter.distance(target, jnp.int32(c), distance_form="l2"))
        d_cos = float(adapter.distance(target, jnp.int32(c), distance_form="cosine"))
        assert d_l2 < 1e-5
        assert d_cos < 1e-5


def test_vae_distance_ranges():
    _require_ckpt()
    adapter = _load_adapter()
    rng = jax.random.PRNGKey(0)
    z = jax.random.normal(rng, (adapter.code_dim,))
    d_l2 = float(adapter.distance(z, jnp.int32(0), distance_form="l2"))
    d_cos = float(adapter.distance(z, jnp.int32(0), distance_form="cosine"))
    assert d_l2 >= 0.0
    assert 0.0 - 1e-5 <= d_cos <= 2.0 + 1e-5


# ============================================================================
# Argmin agreement
# ============================================================================


def test_vae_argmin_l2_matches_kmeans_predict():
    """``argmin(distances_to_all_codes(z, "l2")) == sklearn KMeans
    .predict(z.reshape(1,-1))[0]`` — the adapter's discrete-accuracy
    diagnostic must agree with the sklearn KMeans assignment.
    """
    _require_ckpt()
    adapter = _load_adapter()

    qpos = _load_one_clip_qpos(clip_length=250)
    z = adapter.encode_from_qpos(jnp.asarray(qpos[50:60]))

    dists = adapter.distances_to_all_codes(z, distance_form="l2")
    adapter_argmin = int(jnp.argmin(dists))

    import pickle

    with open(_TOKENIZER_PKL, "rb") as f:
        tok = pickle.load(f)
    sklearn_argmin = int(tok["kmeans"].predict(np.asarray(z)[None])[0])

    assert adapter_argmin == sklearn_argmin, (
        f"adapter L2 argmin={adapter_argmin} but sklearn KMeans "
        f".predict gives {sklearn_argmin}. Diagnostic and tokenize-time "
        f"clustering disagree."
    )


# ============================================================================
# JIT correctness
# ============================================================================


def test_vae_encode_under_jit():
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
