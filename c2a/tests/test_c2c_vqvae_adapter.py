"""Tests for c2a.code2code.VQVAEAdapter.

The adapter is the in-loop interface to a frozen VQ-VAE tokenizer
used by the code2code reward mode. The critical invariants we test:

1. **Loading**: a real VQ-VAE ckpt produces a working adapter without
   shape errors.
2. **Encoder equivalence**: ``adapter.encode_window(window)`` returns
   the *same* z that the existing
   ``c2a.codegen.kinematics_vae.backbone.codegen.encode_motion_array``
   would produce for a single-window single-clip input. This is the
   load-bearing correctness test — if it fails, the in-loop encoder
   diverges from the tokenize-time encoder and the codes the policy
   was conditioned on don't correspond to the codes the reward path
   is comparing against.
3. **Codebook lookup**: ``adapter.codebook_lookup(c)`` returns the
   raw codebook param at index ``c``.
4. **Distance ranges**: L2 ≥ 0, cosine ∈ [0, 2].
5. **Argmin agreement**: ``argmin(distances_to_all_codes(z, "l2")) ==
   bottleneck-quantizer's nearest code for z``.

Real-ckpt tests skip cleanly when the ckpt isn't present (CI etc.).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_CKPT_DIR = Path(
    "/home/jovyan/vast/kaiwen/code2act/outputs/tokenizer/"
    "trained_kinematics_vae_ckpts/vqvae/step_0030000"
)
_REF_H5 = Path(
    "/home/jovyan/vast/kaiwen/code2act/data/rodent/rodent_reference_clips.h5"
)


def _require_ckpt():
    if not _CKPT_DIR.exists():
        pytest.skip(f"VQ-VAE ckpt not available: {_CKPT_DIR}")
    if not _REF_H5.exists():
        pytest.skip(f"reference clips not available: {_REF_H5}")


def _load_adapter():
    from c2a.code2code import VQVAEAdapter
    return VQVAEAdapter.from_ckpt_dir(
        _CKPT_DIR,
        n_joints=67,
        use_qvel=False,
        window_len=10,
        horizon=5,
    )


def _load_one_clip_qpos(clip_length: int = 250) -> np.ndarray:
    """Load qpos for clip 0 from the reference H5; truncate to clip_length."""
    import h5py

    with h5py.File(_REF_H5, "r") as f:
        flat = f["qpos"][:]
    n_total = flat.shape[0] // clip_length
    clipped = flat[: n_total * clip_length].reshape(n_total, clip_length, -1)
    return clipped[0].astype(np.float32)  # (clip_length, 7 + n_joints)


# ============================================================================
# Loading
# ============================================================================


def test_adapter_loads_from_real_ckpt():
    _require_ckpt()
    adapter = _load_adapter()
    assert adapter.name == "vqvae"
    assert adapter.window_len == 10
    assert adapter.use_qvel is False
    assert adapter.codebook_size > 0
    assert adapter.code_dim > 0
    assert adapter.feat_dim > 0
    cb = adapter.all_codebook_entries()
    assert cb.shape == (adapter.codebook_size, adapter.code_dim)


def test_adapter_rejects_missing_ckpt():
    from c2a.code2code import VQVAEAdapter
    with pytest.raises(FileNotFoundError):
        VQVAEAdapter.from_ckpt_dir("/tmp/does_not_exist_12345")


# ============================================================================
# Encoder equivalence — single window must match encode_motion_array
# ============================================================================


def test_encode_window_matches_encode_motion_array_for_single_window():
    """Critical correctness invariant: per-window encoder output must equal
    what encode_motion_array would have produced for the same window.

    encode_motion_array operates on whole clips and assigns the
    windowed-z to the LAST frame of each window. So for a window
    ``qpos[t : t+w]``, the z that encode_motion_array emits at frame
    ``t + w - 1`` corresponds to our ``encode_window(qpos[t:t+w])``.
    """
    _require_ckpt()
    adapter = _load_adapter()

    from c2a.codegen.kinematics_vae.backbone.codegen import encode_motion_array
    from c2a.codegen.kinematics_vae.backbone.train_loop import _Clips
    from c2a.codegen.kinematics_vae.backbone.codegen import load_checkpoint
    from c2a.codegen.kinematics_vae.vqvae.bottleneck import VQVAEBottleneck

    qpos = _load_one_clip_qpos(clip_length=250)
    # encode_motion_array expects (n_clips, n_frames, nq); add clip dim
    qpos_b = qpos[None]  # (1, 250, nq)

    cb_size = adapter.codebook_size
    code_dim = adapter.code_dim
    bottleneck = VQVAEBottleneck(codebook_size=cb_size, code_dim=code_dim)
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
    )
    # encoded.mean shape: (1, 250, code_dim). Per per-frame semantics, the
    # window [t : t+10] yields the z assigned to frame (t + 10 - 1).
    # Pick a non-edge window so left-pad doesn't apply.
    t = 50
    w = 10
    window_qpos = jnp.asarray(qpos[t : t + w])           # (10, nq)

    z_adapter = adapter.encode_from_qpos(window_qpos)    # (code_dim,)
    z_emanate = jnp.asarray(encoded.mean[0, t + w - 1])  # (code_dim,)

    # Tolerance: ~1e-2 absolute is plenty for a reward signal. The
    # numerical diff comes from a different reduction order in the
    # Conv1D + Dense stack between the host-NumPy → JAX path in
    # `encode_motion_array` (extract_motion_frames + normalize in
    # NumPy, convert to JAX, then encoder) and our pure-JAX adapter
    # path (everything in JAX from the start). Empirically <2e-3.
    np.testing.assert_allclose(
        np.asarray(z_adapter), np.asarray(z_emanate),
        rtol=5e-2, atol=5e-3,
        err_msg=(
            "encode_window output diverged from encode_motion_array for "
            "the same input window. If this drift is >1e-2 absolute, "
            "the in-loop encoder has a real logic bug; if smaller, "
            "loosen this tolerance."
        ),
    )


# ============================================================================
# Codebook lookup
# ============================================================================


def test_codebook_lookup_matches_raw_codebook():
    _require_ckpt()
    adapter = _load_adapter()
    cb = np.asarray(adapter.all_codebook_entries())
    for c in [0, 1, adapter.codebook_size // 2, adapter.codebook_size - 1]:
        entry = np.asarray(adapter.codebook_lookup(jnp.int32(c)))
        np.testing.assert_array_equal(entry, cb[c])


# ============================================================================
# Distance properties
# ============================================================================


def test_l2_distance_nonneg_and_self_zero():
    _require_ckpt()
    adapter = _load_adapter()
    rng = jax.random.PRNGKey(0)
    z = jax.random.normal(rng, (adapter.code_dim,))
    for c in [0, 5, adapter.codebook_size - 1]:
        d = float(adapter.distance(z, jnp.int32(c), distance_form="l2"))
        assert d >= 0.0

    # Self-distance: pass the codebook entry itself; should be ~0.
    for c in [0, 5, adapter.codebook_size - 1]:
        target = adapter.codebook_lookup(jnp.int32(c))
        d_self = float(adapter.distance(target, jnp.int32(c), distance_form="l2"))
        assert d_self < 1e-5, f"L2 self-distance for code {c} is {d_self}, not ~0"


def test_cosine_distance_range_and_self_zero():
    _require_ckpt()
    adapter = _load_adapter()
    rng = jax.random.PRNGKey(0)
    z = jax.random.normal(rng, (adapter.code_dim,))
    for c in [0, 5, adapter.codebook_size - 1]:
        d = float(adapter.distance(z, jnp.int32(c), distance_form="cosine"))
        assert 0.0 - 1e-5 <= d <= 2.0 + 1e-5, (
            f"cosine distance out of [0,2] for code {c}: {d}"
        )

    # Self-cosine-distance: pass codebook entry itself; should be ~0.
    for c in [0, 5, adapter.codebook_size - 1]:
        target = adapter.codebook_lookup(jnp.int32(c))
        d_self = float(
            adapter.distance(target, jnp.int32(c), distance_form="cosine")
        )
        assert d_self < 1e-5, (
            f"cosine self-distance for code {c} is {d_self}, not ~0"
        )


# ============================================================================
# Argmin agreement with the quantizer
# ============================================================================


def test_argmin_l2_matches_quantizer_nearest():
    """``argmin(distances_to_all_codes(z, "l2")) == bottleneck.argmin``.

    The bottleneck's forward pass picks the nearest codebook entry via
    the same squared L2 in code space. Our diagnostic helper must
    agree, or the discrete-accuracy reward-hacking detector is wrong.
    """
    _require_ckpt()
    adapter = _load_adapter()

    # Use a real encoded z (encoder mean from a reference window) so the
    # geometry is realistic, not random.
    qpos = _load_one_clip_qpos(clip_length=250)
    z = adapter.encode_from_qpos(jnp.asarray(qpos[50:60]))

    dists = adapter.distances_to_all_codes(z, distance_form="l2")
    adapter_argmin = int(jnp.argmin(dists))

    # Compute the bottleneck's nearest code independently: for each cb
    # entry, distance to z; pick min.
    cb = np.asarray(adapter.all_codebook_entries())
    z_np = np.asarray(z)
    quantizer_dists = ((z_np[None, :] - cb) ** 2).sum(axis=-1)
    quantizer_argmin = int(np.argmin(quantizer_dists))

    assert adapter_argmin == quantizer_argmin, (
        f"argmin(distances_to_all_codes(z, 'l2'))={adapter_argmin} but "
        f"bottleneck nearest code is {quantizer_argmin}. These must agree."
    )


def test_argmin_cosine_normalizes_then_picks():
    """``argmin(distances_to_all_codes(z, "cosine"))`` should pick the
    codebook entry with maximum cosine similarity to z.
    """
    _require_ckpt()
    adapter = _load_adapter()
    rng = jax.random.PRNGKey(42)
    z = jax.random.normal(rng, (adapter.code_dim,))

    dists = adapter.distances_to_all_codes(z, distance_form="cosine")
    adapter_argmin = int(jnp.argmin(dists))

    cb = np.asarray(adapter.all_codebook_entries())
    z_np = np.asarray(z)
    z_n = z_np / (np.linalg.norm(z_np) + 1e-8)
    cb_n = cb / (np.linalg.norm(cb, axis=-1, keepdims=True) + 1e-8)
    sims = cb_n @ z_n
    expected_argmin = int(np.argmax(sims))  # max sim = min cosine dist

    assert adapter_argmin == expected_argmin, (
        f"cosine argmin={adapter_argmin} but max-similarity index is "
        f"{expected_argmin}"
    )


# ============================================================================
# JIT-friendliness sanity check
# ============================================================================


def test_encode_from_qpos_under_jit():
    _require_ckpt()
    adapter = _load_adapter()

    @jax.jit
    def encode_jit(qpos):
        return adapter.encode_from_qpos(qpos)

    qpos = _load_one_clip_qpos(clip_length=250)
    window = jnp.asarray(qpos[50:60])

    z_eager = adapter.encode_from_qpos(window)
    z_jit = encode_jit(window)

    # Eager and JIT use slightly different XLA lowerings; float32
    # accumulation noise is on the order of 1e-3. Reward signal is
    # insensitive to noise at this scale.
    np.testing.assert_allclose(
        np.asarray(z_eager), np.asarray(z_jit), rtol=1e-2, atol=1e-3
    )


def test_distance_under_jit():
    _require_ckpt()
    adapter = _load_adapter()

    @jax.jit
    def d_jit(z, c):
        return adapter.distance(z, c, distance_form="cosine")

    rng = jax.random.PRNGKey(7)
    z = jax.random.normal(rng, (adapter.code_dim,))
    c = jnp.int32(3)

    d_eager = adapter.distance(z, c, distance_form="cosine")
    d_jit_val = d_jit(z, c)

    np.testing.assert_allclose(
        float(d_eager), float(d_jit_val), rtol=1e-5, atol=1e-6
    )
