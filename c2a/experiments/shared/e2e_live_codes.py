"""Extract live (post-e2e-training) VQ-VAE code IDs from an e2e checkpoint.

The e2e moseq decoder ckpt carries the VQ-VAE encoder + bottleneck params
inside its policy_params subtree (see ``c2a.e2e_vqvae``). At inference
time we can re-tokenize any clip with these *current* params and compare
the result to the *fixed* codes that the pretrained VQ-VAE produced and
saved to ``best_codes.npz``. Difference = how the e2e training reshaped
the codebook (+ encoder) for the same input.

Public API:

* :func:`load_e2e_moseq_checkpoint` — Orbax-restores an e2e moseq ckpt
  (the shared ``load_moseq_checkpoint`` doesn't build the e2e network
  branch so its restore fails on the vqvae_* subtrees).
* :func:`extract_live_codes` — given an e2e ckpt path + pretrained VQ-VAE
  ckpt dir, run the e2e ckpt's current encoder+bottleneck over every
  balanced-split clip and return ``(n_clips, n_frames)`` int codes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Tuple

import jax
import numpy as np

# Monkeypatch for orbax-checkpoint that calls a removed
# jax.monitoring.record_scalar. Pure telemetry stub. See
# memory:project_c2a_orbax_jax_mismatch.
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

log = logging.getLogger(__name__)


def load_e2e_moseq_checkpoint(
    checkpoint_path: str | Path,
    pretrained_vqvae_dir: str | Path,
    *,
    step: int | None = None,
) -> Tuple[Any, Any, Any, Any]:
    """e2e-aware variant of ``load_moseq_checkpoint``.

    The shared loader in ``c2a.experiments.shared.checkpoint_utils`` builds
    the network factory WITHOUT ``e2e_mode=True`` (and the matching
    ``vqvae_*`` factory kwargs), so the abstract template fed to Orbax
    lacks the ``vqvae_encoder`` / ``vqvae_bottleneck`` / ``vqvae_decoder``
    subtrees that an e2e ckpt actually stores. Restore then fails with a
    pytree-mismatch ``Diff(lhs=None, rhs=...)``.

    Here we mirror what ``c2a.train_moseq_decoder`` does at training time:
    load the pretrained VQ-VAE ctx, derive ``e2e_motion_window_shape`` +
    ``vqvae_*`` factory kwargs from it, and pass them into the network
    factory. The resulting abstract policy matches the saved structure.

    Returns the same tuple as the shared loader:
    ``(cfg, normalizer_state, policy_params, ppo_networks)``.
    """
    import functools
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from omegaconf import DictConfig, OmegaConf

    from track_mjx.agent import checkpointing
    from track_mjx.agent.observation_utils import init_dict_normalizer
    from c2a.moseq_ppo_networks import make_moseq_recurrent_decoder_ppo_networks
    from c2a.e2e_vqvae import load_pretrained_vqvae

    STEP_PREFIX = "MoSeqPPONetwork"
    checkpoint_path = str(checkpoint_path)

    cfg = checkpointing.load_config_from_checkpoint(
        checkpoint_path, step_prefix=STEP_PREFIX, step=step,
    )
    cfg = OmegaConf.create(cfg) if not isinstance(cfg, DictConfig) else cfg

    net_cfg = cfg.network_config
    e2e_cfg = cfg.get("e2e_config", None)
    e2e_mode = bool(e2e_cfg is not None and e2e_cfg.get("e2e_mode", False))
    if not e2e_mode:
        raise ValueError(
            f"ckpt {checkpoint_path} is not e2e_mode; use the standard "
            "load_moseq_checkpoint instead."
        )

    # Load pretrained VQ-VAE ctx — gives us all the vqvae_* factory kwargs
    # that train_moseq_decoder.py would have used.
    vqvae_arch = e2e_cfg.get("vqvae_arch", {})
    ctx = load_pretrained_vqvae(
        pretrained_vqvae_dir,
        n_joints=int(vqvae_arch.get("n_joints", 67)),
        use_qvel=bool(vqvae_arch.get("use_qvel", False)),
        window_len=int(vqvae_arch.get("window_len", 10)),
        horizon=int(vqvae_arch.get("horizon", 5)),
        code_stack_size=int(net_cfg.get("code_stack_size", 5)),
        conv_channels=tuple(vqvae_arch.get("conv_channels", [64, 128, 256])),
        conv_kernel_size=int(vqvae_arch.get("conv_kernel_size", 3)),
        conv_head_layers=tuple(vqvae_arch.get("conv_head_layers", [256])),
        decoder_layer_sizes=tuple(vqvae_arch.get("decoder_layer_sizes", [256, 256, 256])),
    )

    e2e_factory_kwargs = {
        "e2e_mode": True,
        "e2e_motion_window_shape": (ctx.code_stack_size, ctx.window_len, ctx.feat_dim),
        "vqvae_branch_input_indices": ctx.branch_input_indices,
        "vqvae_branch_latent_dims": ctx.branch_latent_dims,
        "vqvae_codebook_size": ctx.codebook_size,
        "vqvae_code_dim": ctx.code_dim,
        "vqvae_conv_channels": ctx.conv_channels,
        "vqvae_conv_kernel_size": ctx.conv_kernel_size,
        "vqvae_conv_head_layers": ctx.conv_head_layers,
        "vqvae_decoder_layer_sizes": ctx.decoder_layer_sizes,
    }

    obs_sizes = dict(net_cfg.obs_sizes)
    ppo_networks = make_moseq_recurrent_decoder_ppo_networks(
        obs_sizes=obs_sizes,
        action_size=int(net_cfg.action_size),
        num_codes=int(net_cfg.num_codes),
        code_embed_dim=int(net_cfg.code_embed_dim),
        value_hidden_layer_sizes=tuple(net_cfg.critic_layer_sizes),
        use_continuous_encoder=bool(net_cfg.get("use_continuous_encoder", False)),
        encoder_layer_sizes=tuple(net_cfg.get("encoder_layer_sizes", [256, 128])),
        continuous_latent_dim=int(net_cfg.get("continuous_latent_dim", 16)),
        z_e_dropout_rate=float(net_cfg.get("z_e_dropout_rate", 0.0)),
        rnn_hidden_sizes=tuple(net_cfg.get("rnn_hidden_sizes", [256])),
        rnn_cell_type=str(net_cfg.get("rnn_cell_type", "gru")),
        z_e_at_action_head=bool(net_cfg.get("z_e_at_action_head", False)),
        reinit_hidden_on_code=bool(net_cfg.get("reinit_hidden_on_code", False)),
        learned_hidden_init=bool(net_cfg.get("learned_hidden_init", False)),
        use_distillation_head=bool(net_cfg.get("use_distillation_head", False)),
        distill_head_layer_sizes=tuple(
            net_cfg.get("distillation_head_layer_sizes", [256, 128])
        ),
        distill_logvar_min=net_cfg.get("distill_logvar_min", None),
        distill_logvar_max=net_cfg.get("distill_logvar_max", None),
        use_pretrained_decoder=bool(net_cfg.get("use_pretrained_decoder", False)),
        decoder_layer_sizes_vae=tuple(
            net_cfg.get("decoder_layer_sizes_vae", [512, 256, 256, 256])
        ),
        encoder_uses_proprio=bool(net_cfg.get("encoder_uses_proprio", False)),
        **e2e_factory_kwargs,
    )

    # Abstract template. The e2e ckpt's normalizer also tracks
    # ``motion_window_stack`` as a separate key (track-mjx's dict-based
    # normalizer iterates over every obs key it sees), so add it here.
    key_policy, key_value = jax.random.split(jax.random.key(1))
    dummy_obs = {
        "task_obs": jnp.zeros((1, obs_sizes["task_obs"])),
        "proprioception": jnp.zeros((1, obs_sizes["proprioception"])),
    }
    if "motion_window_stack" in obs_sizes:
        dummy_obs["motion_window_stack"] = jnp.zeros(
            (1, int(obs_sizes["motion_window_stack"]))
        )
    normalizer_state = init_dict_normalizer(dummy_obs)
    init_policy = ppo_networks.policy_network.init(key_policy)
    init_value = ppo_networks.value_network.init(key_value)
    from brax.training.agents.ppo import losses as ppo_losses
    init_params = ppo_losses.PPONetworkParams(policy=init_policy, value=init_value)
    abstract_policy = (normalizer_state, init_params.policy)

    mgr_options = ocp.CheckpointManagerOptions(create=False, step_prefix=STEP_PREFIX)
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)
    restore_step = step if step is not None else ckpt_mgr.latest_step()
    normalizer_state, policy_params = ckpt_mgr.restore(
        restore_step,
        args=ocp.args.Composite(
            policy=ocp.args.StandardRestore(abstract_policy),
        ),
    )["policy"]
    log.info(
        f"Loaded e2e MoSeq ckpt {checkpoint_path} step {restore_step}: "
        f"num_codes={net_cfg.num_codes}, codebook_size={ctx.codebook_size}"
    )
    return cfg, normalizer_state, policy_params, ppo_networks


def extract_live_codes(
    e2e_ckpt_dir: str | Path,
    pretrained_vqvae_dir: str | Path,
    reference_data_path: str | Path,
    balanced_split_path: str | Path,
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
) -> tuple[np.ndarray, np.ndarray]:
    """Re-tokenize every balanced-split clip with the e2e ckpt's live VQ-VAE.

    Args:
        e2e_ckpt_dir: moseq-decoder e2e checkpoint root (with
            ``MoSeqPPONetwork_*`` subdirs).
        pretrained_vqvae_dir: standalone VQ-VAE ckpt used to seed e2e
            training (provides the module wiring + normalizer + active
            joints). Only the architecture pieces are taken from here;
            the e2e ckpt's live encoder/bottleneck params override.
        reference_data_path: ``rodent_reference_clips.h5``.
        balanced_split_path: ``rodent_balanced_splits.json``.

    Returns:
        ``(codes, clip_ids)``: ``codes`` is ``(n_clips, n_frames)`` int32
        of per-frame live codes; ``clip_ids`` is the matching sorted
        train ∪ test clip-id array.
    """
    from c2a.codegen.kinematics_vae.backbone.codegen import encode_balanced_clips
    from c2a.e2e_vqvae import load_pretrained_vqvae

    e2e_ckpt_dir = Path(e2e_ckpt_dir).resolve()
    pretrained_vqvae_dir = Path(pretrained_vqvae_dir).resolve()

    # 1. Get the standalone-VQVAE module wiring + normalizer + active joints
    #    (we need these because the e2e ckpt only carries params, not the
    #    Flax Module class hierarchy).
    log.info(f"Loading pretrained VQ-VAE wiring from {pretrained_vqvae_dir}")
    ctx = load_pretrained_vqvae(
        pretrained_vqvae_dir,
        n_joints=n_joints, use_qvel=use_qvel,
        window_len=window_len, horizon=horizon,
        code_stack_size=code_stack_size,
        conv_channels=conv_channels,
        conv_kernel_size=conv_kernel_size,
        conv_head_layers=conv_head_layers,
        decoder_layer_sizes=decoder_layer_sizes,
    )

    # 2. Load e2e ckpt's policy_params and extract the live VQ-VAE subtrees.
    log.info(f"Loading e2e ckpt {e2e_ckpt_dir}")
    _ckpt_cfg, _norm_state, policy_params, _ppo_networks = load_e2e_moseq_checkpoint(
        str(e2e_ckpt_dir),
        pretrained_vqvae_dir=str(pretrained_vqvae_dir),
    )
    base = policy_params.get("params", policy_params)
    if "vqvae_encoder" not in base or "vqvae_bottleneck" not in base:
        raise KeyError(
            f"e2e ckpt {e2e_ckpt_dir} policy params missing 'vqvae_encoder' / "
            "'vqvae_bottleneck' subtrees — checkpoint may not be e2e-mode."
        )
    live_enc_params = {"params": base["vqvae_encoder"]}
    live_bn_params = {"params": base["vqvae_bottleneck"]}

    # 3. Need a BackboneModules-shaped object for encode_balanced_clips. The
    #    one from load_pretrained_vqvae is the right shape; we just swap
    #    the params we'll pass in.
    from c2a.codegen.kinematics_vae.backbone.codegen import load_checkpoint
    modules, _all_params = load_checkpoint(
        pretrained_vqvae_dir,
        bottleneck=_make_bottleneck_from_ctx(ctx),
        n_joints=n_joints, use_qvel=use_qvel,
        window_len=window_len, horizon=horizon,
        conv_channels=conv_channels,
        conv_kernel_size=conv_kernel_size,
        conv_head_layers=conv_head_layers,
        decoder_layer_sizes=decoder_layer_sizes,
    )

    # 4. Encode with live params over the full balanced split. We get back
    #    one row per clip in the sorted train ∪ test order.
    log.info("Encoding all balanced clips with live VQ-VAE params ...")
    encoded = encode_balanced_clips(
        reference_data_path=str(reference_data_path),
        balanced_split_path=str(balanced_split_path),
        modules=modules,
        enc_params=live_enc_params,
        bn_params=live_bn_params,
        n_joints=n_joints, use_qvel=use_qvel,
        window_len=window_len, horizon=horizon,
        deterministic=True,
    )
    if "codes" not in encoded.aux:
        raise RuntimeError(
            "VQVAEBottleneck did not emit 'codes' in aux_metrics; "
            "is the bottleneck still VQ-VAE flavor?"
        )
    codes = np.asarray(encoded.aux["codes"], dtype=np.int32)

    # Reconstruct the matching clip_ids in the sorted-union convention.
    import json
    with open(balanced_split_path) as f:
        splits = json.load(f)
    train = list(map(int, splits["balanced"]["train_indices"]))
    test = list(map(int, splits["balanced"]["test_indices"]))
    clip_ids = np.asarray(sorted(set(train) | set(test)), dtype=np.int64)
    if codes.shape[0] != clip_ids.shape[0]:
        raise RuntimeError(
            f"encoded codes shape {codes.shape} doesn't match "
            f"clip_ids shape {clip_ids.shape}"
        )
    log.info(
        f"Live codes: shape {codes.shape}, unique used {int(np.unique(codes).size)}"
    )
    return codes, clip_ids


def _make_bottleneck_from_ctx(ctx: Any) -> Any:
    """Re-instantiate VQVAEBottleneck with the codebook size + code dim
    that were baked into the loaded pretrained ckpt."""
    from c2a.codegen.kinematics_vae.vqvae.bottleneck import VQVAEBottleneck
    return VQVAEBottleneck(
        codebook_size=int(ctx.codebook_size),
        code_dim=int(ctx.code_dim),
    )
