"""Pretrained-VQ-VAE artifacts + JIT-friendly motion-window plumbing for e2e mode.

The "e2e mode" of ``c2a.train_moseq_decoder`` loads a pretrained vanilla VQ-VAE
checkpoint, folds the encoder / bottleneck / decoder into the policy network
as live (trainable) Flax sub-modules, and routes the imitation + reconstruction
losses back through them. This module is the host-side glue:

    1. :func:`load_pretrained_vqvae` reads ``meta.json`` + ``*.msgpack`` +
       ``normalizer.npz`` + ``active_joints.npy`` from the ckpt and returns
       an :class:`E2EVQVAEContext` containing the per-module Flax param
       subtrees and the data needed to construct the env's motion buffer.
    2. :func:`build_motion_norm_padded` computes per-clip normalized motion
       features (under the VQ-VAE's ``active_joints`` mask) and left-pads by
       ``window_len - 1`` copies of frame 0 and right-pads by
       ``code_stack_size - 1`` copies of the last frame so every frame
       ``t in [0, n_frames - 1]`` admits a full ``code_stack_size``-stack of
       full ``window_len``-windows with no out-of-bounds indexing.
    3. :func:`make_window_stack_extractor` returns a JIT-friendly callable
       ``(motion_norm_padded, clip_idx, frame_t) -> (stack, w, feat_dim)``
       the env injects into ``obs["motion_window_stack"]`` in ``_get_obs``.
    4. :func:`assign_vqvae_to_policy_params` is a helper invoked from
       ``train_moseq_decoder``'s ``_post_init_params_fn`` to overwrite the
       freshly-Flax-init'd VQ-VAE subtrees with the pretrained values.

E2E is only legal when ``kpms_config.code_source == "vqvae"`` (the ckpt that
produced the codes is exactly the ckpt we load here). The caller enforces
that; this module assumes it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from c2a.codegen.kinematics_vae.backbone.codegen import load_checkpoint
from c2a.codegen.kinematics_vae.backbone.frame_features import (
    extract_motion_frames,
)
from c2a.codegen.kinematics_vae.vqvae.bottleneck import VQVAEBottleneck

log = logging.getLogger(__name__)


# Names of the Flax sub-module fields injected into the policy network when
# e2e_mode is on. Keep in sync with c2a.moseq_decoder_network where these
# modules are instantiated; the split-grad logger looks up these subtrees by
# exactly these keys.
VQVAE_PARAM_NAMES: tuple[str, ...] = (
    "vqvae_encoder",
    "vqvae_bottleneck",
    "vqvae_decoder",
)


@dataclass(frozen=True)
class E2EVQVAEContext:
    """All artifacts needed to wire the pretrained VQ-VAE into e2e training.

    Attributes:
        n_joints: Articulated-joint count for ``extract_motion_frames`` (env
            input). Must match the ckpt's training config.
        use_qvel: Whether root/joint velocity blocks were part of the motion
            feature vector during VQ-VAE pretraining. Must match.
        window_len: Encoder input window length ``w`` (10 for the default
            pretrain).
        horizon: Predictor horizon used during pretraining. We do not run
            the predictor in e2e, but the value is stored for provenance and
            because tokenize.py uses ``horizon`` for right-pad accounting.
        code_stack_size: How many consecutive windows the env packs into
            ``obs["motion_window_stack"]`` (= policy's ``code_stack_size``).
            Determines the right-pad applied to the motion buffer.
        branch_input_indices: Per-branch feat-vector indices used by the
            split decoder; loaded from the ckpt's ``meta.json``.
        branch_latent_dims: Per-branch latent dimensionalities. ``sum`` is
            the concatenated latent (= VQ ``code_dim``).
        branch_names: ``("root", "axial", "limb")`` for the v15 3-way split.
        feat_dim: Motion feature dimensionality (3 + 4 + n_active or similar).
        codebook_size: Number of codes ``K`` in the loaded bottleneck.
        code_dim: ``sum(branch_latent_dims)``; the per-code embedding dim.
        conv_channels / conv_kernel_size / conv_head_layers /
        decoder_layer_sizes: Encoder/decoder arch knobs (must match pretrain).
        encoder_params / bottleneck_params / decoder_params: Flax param
            subtrees suitable for direct assignment into
            ``training_state.params.policy["params"][prefix + name]``.
        normalizer_mean / normalizer_std: ``(feat_dim,)`` arrays used to
            normalize motion features before the encoder (same convention as
            ``c2a/codegen/kinematics_vae/backbone/codegen.py::encode_balanced_clips``).
        active_joints: Integer indices into ``[0, n_joints)`` of joints whose
            std exceeded the auto-detect threshold during pretraining; used
            to slice ``qpos[..., 7:]`` (and ``qvel[..., 6:]`` if ``use_qvel``).
        ckpt_dir: Resolved path the ckpt was loaded from.
    """

    n_joints: int
    use_qvel: bool
    window_len: int
    horizon: int
    code_stack_size: int

    branch_input_indices: tuple[tuple[int, ...], ...]
    branch_latent_dims: tuple[int, ...]
    branch_names: tuple[str, ...]
    feat_dim: int
    codebook_size: int
    code_dim: int

    conv_channels: tuple[int, ...]
    conv_kernel_size: int
    conv_head_layers: tuple[int, ...]
    decoder_layer_sizes: tuple[int, ...]

    encoder_params: dict[str, Any]
    bottleneck_params: dict[str, Any]
    decoder_params: dict[str, Any]

    normalizer_mean: np.ndarray
    normalizer_std: np.ndarray
    active_joints: np.ndarray

    ckpt_dir: Path

    def block_len(self) -> int:
        """``window_len + code_stack_size - 1`` — the length of one
        ``dynamic_slice`` per env step that the extractor consumes."""
        return int(self.window_len + self.code_stack_size - 1)


def load_pretrained_vqvae(
    ckpt_dir: str | Path,
    *,
    n_joints: int = 67,
    use_qvel: bool = False,
    window_len: int = 10,
    horizon: int = 5,
    code_stack_size: int = 5,
    conv_channels: tuple[int, ...] = (64, 128, 256),
    conv_kernel_size: int = 3,
    conv_head_layers: tuple[int, ...] = (256,),
    decoder_layer_sizes: tuple[int, ...] = (256, 256, 256),
) -> E2EVQVAEContext:
    """Load a vanilla-VQVAE checkpoint and return its e2e context.

    The msgpack params, normalizer, and active_joints come from
    :func:`c2a.codegen.kinematics_vae.backbone.codegen.load_checkpoint`
    (the same loader the tokenize script uses, so a successful load here
    guarantees the same encode behavior at e2e time as during the codegen
    that produced ``best_codes.npz``).

    The ``codebook_size`` is read directly from the bottleneck params
    (codebook table shape) rather than passed in, so a mismatch between
    config and ckpt is impossible by construction.
    """
    ckpt_dir = Path(ckpt_dir).resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"VQ-VAE checkpoint directory does not exist: {ckpt_dir}"
        )

    # Codebook size has to be passed to load_checkpoint via the bottleneck
    # instance, but the load expects the bottleneck shape to match what is on
    # disk. We bootstrap with the codebook size encoded in the bottleneck
    # msgpack itself (it's the first dim of the only param under "quantizer").
    cb_size_from_disk = _peek_codebook_size(ckpt_dir / "bottleneck.msgpack")
    code_dim_from_meta = _sum_branch_latent_dims(ckpt_dir / "meta.json")

    bottleneck = VQVAEBottleneck(
        codebook_size=int(cb_size_from_disk),
        code_dim=int(code_dim_from_meta),
    )
    modules, all_params = load_checkpoint(
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

    code_dim_from_params = int(
        np.asarray(
            all_params["bottleneck"]["params"]["quantizer"]["codebook"]
        ).shape[-1]
    )
    if code_dim_from_params != code_dim_from_meta:
        raise RuntimeError(
            f"VQ-VAE ckpt at {ckpt_dir} has codebook code_dim "
            f"{code_dim_from_params} but meta.json's sum(branch_latent_dims) "
            f"= {code_dim_from_meta}. Refusing to load — likely a mixed-arch "
            f"ckpt or a corrupted meta.json."
        )

    log.info(
        "Loaded VQ-VAE ckpt from %s: codebook_size=%d, code_dim=%d, "
        "feat_dim=%d, branch_latent_dims=%s, window_len=%d, horizon=%d, "
        "code_stack_size=%d, n_active_joints=%d",
        ckpt_dir,
        cb_size_from_disk,
        code_dim_from_meta,
        modules.feat_dim,
        modules.branch_latent_dims,
        window_len,
        horizon,
        code_stack_size,
        int(np.asarray(modules.active_joints).size),
    )

    return E2EVQVAEContext(
        n_joints=int(n_joints),
        use_qvel=bool(use_qvel),
        window_len=int(window_len),
        horizon=int(horizon),
        code_stack_size=int(code_stack_size),
        branch_input_indices=tuple(
            tuple(int(i) for i in ix) for ix in modules.branch_input_indices
        ),
        branch_latent_dims=tuple(int(d) for d in modules.branch_latent_dims),
        branch_names=tuple(str(s) for s in modules.branch_names),
        feat_dim=int(modules.feat_dim),
        codebook_size=int(cb_size_from_disk),
        code_dim=int(code_dim_from_params),
        conv_channels=tuple(int(c) for c in conv_channels),
        conv_kernel_size=int(conv_kernel_size),
        conv_head_layers=tuple(int(c) for c in conv_head_layers),
        decoder_layer_sizes=tuple(int(c) for c in decoder_layer_sizes),
        encoder_params=all_params["encoder"],
        bottleneck_params=all_params["bottleneck"],
        decoder_params=all_params["decoder"],
        normalizer_mean=np.asarray(modules.normalizer.mean, dtype=np.float32),
        normalizer_std=np.asarray(modules.normalizer.std, dtype=np.float32),
        active_joints=np.asarray(modules.active_joints, dtype=np.int64),
        ckpt_dir=ckpt_dir,
    )


def _peek_codebook_size(bottleneck_msgpack_path: Path) -> int:
    """Deserialize just the codebook leaf from the bottleneck msgpack.

    flax.serialization.from_bytes needs a template tree to align names against,
    but we only care about ``params/quantizer/codebook`` shape[0]. Cheapest:
    use msgpack directly to read the raw nested dict and look up the leaf.
    """
    import msgpack
    raw = bottleneck_msgpack_path.read_bytes()
    tree = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    # flax serializes leaves as {"__msgpack_ext_object_type__": "ndarray", "data": ...}
    # but for our purposes flax.serialization.msgpack_restore handles that.
    # Simpler: use flax's own restore, which preserves the structure as a
    # nested dict and ndarray leaves.
    import flax.serialization as fs
    restored = fs.msgpack_restore(raw)
    # restored is e.g. {"params": {"quantizer": {"codebook": np.ndarray(K, D)}}}
    cb = restored["params"]["quantizer"]["codebook"]
    return int(np.asarray(cb).shape[0])


def _sum_branch_latent_dims(meta_path: Path) -> int:
    """Read meta.json and return sum(branch_latent_dims) (= VQ code_dim)."""
    meta = json.loads(meta_path.read_text())
    return int(sum(int(d) for d in meta["branch_latent_dims"]))


def build_motion_norm_padded(
    qpos: np.ndarray,
    qvel: np.ndarray | None,
    ctx: E2EVQVAEContext,
) -> np.ndarray:
    """Construct the env's padded motion buffer for the given clip stack.

    Mirrors :func:`encode_balanced_clips`'s motion-feature pipeline
    (extract -> normalize) and then pads on the left by ``window_len - 1``
    copies of frame 0 and on the right by ``code_stack_size - 1`` copies of
    the last frame.

    Args:
        qpos: ``(n_clips, n_frames, 7 + n_joints)`` raw qpos from the
            ReferenceClips loader.
        qvel: ``(n_clips, n_frames, 6 + n_joints)`` raw qvel; required when
            ``ctx.use_qvel`` is True, ignored otherwise.
        ctx: Loaded e2e context.

    Returns:
        ``(n_clips, n_frames + w - 1 + stack - 1, feat_dim)`` float32 array.
        Indexing convention: padded index ``j`` for clip ``c`` corresponds
        to original frame ``max(0, min(n_frames-1, j - (w-1)))``.

    Padding rationale: any per-step query at original-frame ``t`` reads a
    contiguous block ``padded[t : t + w + stack - 1]`` (length 14 for the
    default w=10, stack=5). The block contains the union of all 5 windows
    ending at ``t, t+1, ..., t+stack-1``; sliding-window indexing on top
    yields ``(stack, w, feat_dim)`` with no OOB.
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 3 or qpos.shape[-1] != 7 + ctx.n_joints:
        raise ValueError(
            f"qpos shape {qpos.shape} != (n_clips, n_frames, 7 + n_joints={7 + ctx.n_joints})."
        )
    n_clips, n_frames, _ = qpos.shape

    if ctx.use_qvel:
        if qvel is None:
            raise ValueError("ctx.use_qvel=True but qvel was None.")
        qvel = np.asarray(qvel, dtype=np.float32)
        if qvel.ndim != 3 or qvel.shape != (n_clips, n_frames, 6 + ctx.n_joints):
            raise ValueError(
                f"qvel shape {qvel.shape} != (n_clips={n_clips}, "
                f"n_frames={n_frames}, 6 + n_joints={6 + ctx.n_joints})."
            )

    clips_like = _QposQvelHolder(qpos=qpos, qvel=qvel if ctx.use_qvel else qpos)
    motion = extract_motion_frames(
        clips_like,
        n_joints=ctx.n_joints,
        use_qvel=ctx.use_qvel,
        active_joints=ctx.active_joints,
    )
    if motion.shape != (n_clips, n_frames, ctx.feat_dim):
        raise RuntimeError(
            f"extract_motion_frames returned {motion.shape}, expected "
            f"{(n_clips, n_frames, ctx.feat_dim)}. The active_joints mask "
            f"and use_qvel flag must reproduce the ckpt's feat_dim."
        )

    motion_norm = (motion - ctx.normalizer_mean) / ctx.normalizer_std
    motion_norm = motion_norm.astype(np.float32, copy=False)

    left = np.repeat(motion_norm[:, :1, :], ctx.window_len - 1, axis=1)
    right = np.repeat(
        motion_norm[:, -1:, :], ctx.code_stack_size - 1, axis=1,
    )
    padded = np.concatenate([left, motion_norm, right], axis=1)
    expected_padded = n_frames + ctx.window_len - 1 + ctx.code_stack_size - 1
    if padded.shape != (n_clips, expected_padded, ctx.feat_dim):
        raise RuntimeError(
            f"padded shape {padded.shape} != expected "
            f"{(n_clips, expected_padded, ctx.feat_dim)}."
        )
    return padded


@dataclass(frozen=True)
class _QposQvelHolder:
    """Duck-typed ``_ClipsLike`` for :func:`extract_motion_frames`.

    The frame_features API expects ``.qpos`` and ``.qvel`` attributes. The
    real ReferenceClips object exposes both but its qpos / qvel are jax arrays;
    we want host-side numpy here so the windowing computation stays cheap.
    """

    qpos: np.ndarray
    qvel: np.ndarray


def make_window_stack_extractor(
    window_len: int,
    code_stack_size: int,
    feat_dim: int,
) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Return a JIT-friendly closure ``(padded, clip_idx, frame_t) -> windows``.

    The returned function:
        - Gathers ``padded[clip_idx]`` (axis-0 indexing; jax handles dynamic).
        - ``dynamic_slice`` extracts the contiguous block of length
          ``window_len + code_stack_size - 1`` starting at ``frame_t``.
        - Fancy-indexes the block with a precomputed ``(stack, w)`` offsets
          grid to produce the sliding-window tensor.

    Args:
        window_len: ``w``.
        code_stack_size: ``stack``. Together with ``w`` determines the
          length of the slice that ``dynamic_slice`` carves out per call.
        feat_dim: Trailing dim of the padded buffer; needed for the
          ``dynamic_slice`` size tuple at trace time.

    Returns:
        ``extract(padded, clip_idx, frame_t) -> jnp.ndarray`` of shape
        ``(code_stack_size, window_len, feat_dim)`` and dtype matching
        ``padded``.

    Edge safety: the caller is responsible for clamping ``frame_t`` to
    ``[0, n_frames_original - 1]``. The padding built by
    :func:`build_motion_norm_padded` guarantees no OOB for any
    ``frame_t`` in that range.
    """
    if window_len < 1 or code_stack_size < 1 or feat_dim < 1:
        raise ValueError(
            f"window_len={window_len}, code_stack_size={code_stack_size}, "
            f"feat_dim={feat_dim} must all be positive."
        )
    block_len = int(window_len + code_stack_size - 1)
    offsets = jnp.arange(code_stack_size)[:, None] + jnp.arange(window_len)[None, :]
    # Frozen int32 grid; promoted to padded.dtype's index axis at gather.
    offsets = offsets.astype(jnp.int32)

    def extract(
        padded: jnp.ndarray,
        clip_idx: jnp.ndarray,
        frame_t: jnp.ndarray,
    ) -> jnp.ndarray:
        clip_block = padded[clip_idx]
        block = jax.lax.dynamic_slice(
            clip_block,
            (frame_t.astype(jnp.int32), jnp.int32(0)),
            (block_len, feat_dim),
        )
        return block[offsets]

    return extract


def assign_vqvae_to_policy_params(
    policy_params: dict[str, Any],
    ctx: E2EVQVAEContext,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Overwrite the freshly-init'd VQ-VAE subtrees with the pretrained ones.

    Used inside ``train_moseq_decoder``'s ``_post_init_params_fn``. The
    policy's Flax-init produces empty/random VQ-VAE submodule params at
    ``policy_params["params"][prefix + "vqvae_encoder"]`` (etc.). We
    replace those subtrees with what we loaded from the ckpt.

    Mutates ``policy_params`` in place AND returns it for chaining.

    Raises:
        KeyError: if the policy network does not contain the expected
            sub-module fields under ``params``. That means the network
            module was constructed without ``e2e_mode=True`` — the caller
            wired things up incorrectly.
    """
    base = policy_params.get("params", policy_params)
    sources = {
        "vqvae_encoder": ctx.encoder_params,
        "vqvae_bottleneck": ctx.bottleneck_params,
        "vqvae_decoder": ctx.decoder_params,
    }
    for name, src in sources.items():
        key = prefix + name
        if key not in base:
            raise KeyError(
                f"policy params missing '{key}' subtree. The network was not "
                f"built with e2e_mode=True, or the prefix '{prefix}' is wrong."
            )
        # The msgpack loader returns each subtree wrapped in {"params": ...}
        # because that's the Flax variables structure. Strip the wrapper if
        # the destination is the inner params dict.
        if (
            isinstance(src, dict)
            and "params" in src
            and set(src.keys()) == {"params"}
        ):
            base[key] = src["params"]
        else:
            base[key] = src
    return policy_params


def vqvae_param_subtrees(
    policy_params: dict[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Return a dict ``{name: subtree}`` for the three VQ-VAE sub-modules.

    Used by the eval-only split-gradient logger
    (:func:`compute_e2e_split_grads`) to filter the flat gradient pytree
    down to "what the imitation/recon backward dropped into the VQ-VAE".
    """
    base = policy_params.get("params", policy_params)
    out: dict[str, Any] = {}
    for name in VQVAE_PARAM_NAMES:
        key = prefix + name
        if key not in base:
            raise KeyError(
                f"policy params missing '{key}' — e2e_mode is not wired."
            )
        out[name] = base[key]
    return out
