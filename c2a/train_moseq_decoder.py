"""Entry point for MoSeq decoder-only RL training (Pipeline A).

Usage:
    python -m c2a.train_moseq_decoder

    # Override config values:
    python -m c2a.train_moseq_decoder network_config.num_codes=64

    # Use random codes for smoke testing (no KPMS sweep needed):
    python -m c2a.train_moseq_decoder kpms_config.codes_path=null
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import functools
import json
import logging
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np

# Workaround for venv's orbax-checkpoint (0.11.24) calling jax.monitoring.record_scalar,
# which was removed from JAX. Pure telemetry call — safe to stub. Without this, the
# async checkpoint save thread crashes at step 0 and kills training.
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

import orbax.checkpoint as ocp
import wandb
from mujoco_playground import wrapper as playground_wrappers
from omegaconf import DictConfig, OmegaConf
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from track_mjx.config import utils
from track_mjx.agent import checkpointing, wandb_logging
from track_mjx.agent.domain_randomization import domain_randomization_maker

from c2a.paths import get_repo_root
from c2a.moseq_ppo_networks import (
    make_moseq_decoder_ppo_networks,
    make_moseq_recurrent_decoder_ppo_networks,
)
from c2a.moseq_ppo import train as moseq_train
from c2a.moseq_env_wrapper import MoSeqImitation
from c2a.codegen.code_sources import (
    SOURCE_KINEMATICS_VAE, load_codes,
)


def _setup_environment() -> None:
    """Configure environment variables for JAX."""
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_gpu_triton_gemm_any=True"
    os.environ["XLA_FLAGS"] = xla_flags
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _run_eval_rollouts(
    env,
    jit_reset,
    jit_step,
    jit_logging_inference_fn,
    params,
    key,
    episode_length,
    n_rollouts,
    use_rnn,
    ppo_network,
):
    """Run eval rollouts and return raw data (no logging).

    Returns:
        ``(all_indices, all_states, all_rewards, final_key)``.
    """
    all_rollout_indices: list[np.ndarray] = []
    all_rollout_states: list[list] = []
    all_rollout_rewards: list[list] = []
    # Per-frame "reconstructed" codes — what the frozen tokenizer's
    # argmin says about the agent's actual motion at each step. Empty
    # array per rollout when c2c is off (info field absent).
    all_rollout_reconstructed: list[np.ndarray] = []

    for rollout_i in range(n_rollouts):
        key, subkey = jax.random.split(key)
        state = jit_reset(subkey)

        rollout_states = [state]
        rollout_indices = []
        rollout_rewards = []
        rollout_reconstructed: list[int] = []

        if use_rnn and ppo_network is not None:
            hidden = ppo_network.policy_network.init_hidden(1)
        else:
            hidden = None

        for _ in range(episode_length):
            key, subkey = jax.random.split(key)

            if hidden is not None:
                batched_obs = jax.tree_util.tree_map(lambda x: x[None], state.obs)
                action, extras, hidden = jit_logging_inference_fn(
                    params, batched_obs, hidden, subkey
                )
                action = jax.tree_util.tree_map(lambda x: x[0], action)
                extras = jax.tree_util.tree_map(
                    lambda x: x[0] if hasattr(x, "shape") else x, extras
                )
            else:
                action, extras = jit_logging_inference_fn(
                    params, state.obs, subkey, None
                )

            if "code_idx" in extras:
                rollout_indices.append(int(extras["code_idx"]))
            elif "indices" in extras:
                rollout_indices.append(int(extras["indices"]))

            state = jit_step(state, action)
            rollout_states.append(state)
            rollout_rewards.append(float(state.reward))
            # When c2c is active the env writes ``c2c_reconstructed_code``
            # — the frozen tokenizer's nearest codebook entry for the
            # agent's actual motion at this step. Capture aligned with
            # ``rollout_indices`` for the side-by-side timeline bar.
            if "c2c_reconstructed_code" in state.info:
                rollout_reconstructed.append(
                    int(state.info["c2c_reconstructed_code"])
                )

            if state.done:
                if hidden is not None and ppo_network is not None:
                    hidden = ppo_network.policy_network.init_hidden(1)
                break

        all_rollout_indices.append(np.array(rollout_indices))
        all_rollout_states.append(rollout_states)
        all_rollout_rewards.append(rollout_rewards)
        all_rollout_reconstructed.append(
            np.array(rollout_reconstructed, dtype=np.int64)
        )

    return (
        all_rollout_indices, all_rollout_states, all_rollout_rewards,
        all_rollout_reconstructed, key,
    )


def _log_eval_metrics(
    all_rollout_indices,
    all_rollout_states,
    all_rollout_rewards,
    env,
    cfg,
    model_path,
    current_step,
    num_codes,
    render_video,
    metric_prefix="moseq",
    video_prefix="videos",
    all_reconstructed=None,
):
    """Log metrics, code plots, and videos from eval rollouts.

    Returns:
        ``episode_reward_mean`` (float) for reward gap computation.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Episode reward stats
    episode_rewards = [sum(r) for r in all_rollout_rewards if len(r) > 0]
    episode_lengths = [len(r) for r in all_rollout_rewards if len(r) > 0]

    reward_mean = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    reward_std = float(np.std(episode_rewards)) if episode_rewards else 0.0
    length_mean = float(np.mean(episode_lengths)) if episode_lengths else 0.0

    wandb.log(
        {
            f"{metric_prefix}/episode_reward_mean": reward_mean,
            f"{metric_prefix}/episode_reward_std": reward_std,
            f"{metric_prefix}/episode_length_mean": length_mean,
        },
        commit=False,
    )

    # Code metrics from first rollout
    indices_array = all_rollout_indices[0] if all_rollout_indices else None

    if indices_array is not None and len(indices_array) > 0:
        code_counts = np.bincount(indices_array, minlength=num_codes)
        probs = code_counts / (code_counts.sum() + 1e-8)
        perplexity = float(np.exp(-np.sum(probs * np.log(probs + 1e-8))))
        codes_used = int(np.sum(code_counts > 0))
        utilization = codes_used / num_codes

        code_transitions = int(np.sum(indices_array[1:] != indices_array[:-1]))
        transition_rate = code_transitions / max(len(indices_array) - 1, 1)

        wandb.log(
            {
                f"{metric_prefix}/perplexity": perplexity,
                f"{metric_prefix}/codebook_utilization": utilization,
                f"{metric_prefix}/codes_used": codes_used,
                f"{metric_prefix}/eval_transition_rate": transition_rate,
                f"{metric_prefix}/eval_transitions": code_transitions,
                f"{metric_prefix}/eval_steps": len(indices_array),
            },
            commit=False,
        )

        # Code sequence plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 4), height_ratios=[1, 2])

        axes[0].bar(range(num_codes), code_counts, edgecolor="none")
        axes[0].set_xlabel("Code Index")
        axes[0].set_ylabel("Count")
        axes[0].set_title(
            f"Code Usage (perplexity={perplexity:.2f}, "
            f"used={codes_used}/{num_codes})"
        )
        axes[0].set_xlim(-0.5, num_codes - 0.5)

        timesteps = np.arange(len(indices_array))
        for i in range(len(indices_array) - 1):
            axes[1].axvspan(timesteps[i], timesteps[i + 1], alpha=0.8)
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("Code")
        axes[1].set_title(
            f"Code Sequence (transitions={code_transitions}, "
            f"rate={transition_rate:.2%})"
        )
        axes[1].set_xlim(0, len(indices_array))
        axes[1].set_ylim(-0.5, num_codes - 0.5)

        plt.tight_layout()
        wandb.log({f"{metric_prefix}/code_sequence": wandb.Image(fig)}, commit=False)
        plt.close(fig)

    # Render video
    if render_video and all_rollout_states:
        try:
            from c2a.rendering_utils import render_rollout_to_video

            render_fps = cfg.render_config.render_fps
            n_rollouts = len(all_rollout_states)
            num_videos = min(
                int(cfg.render_config.get("num_eval_rollout_videos", 1)),
                n_rollouts,
            )

            # MLB override: behavior/direction-colored bar + code-name text
            # overlay so the viewer can read which steering code is active.
            # No-op for other code sources.
            _code_colors_override = None
            _code_names = None
            if str(cfg.kpms_config.get("code_source", "")) == "mlb":
                try:
                    from c2a.codegen.shared.mlb_labels import (
                        MLB_LABELS, mlb_color_array,
                    )
                    _code_colors_override = mlb_color_array()
                    _code_names = list(MLB_LABELS)
                except Exception as e:
                    logging.warning("mlb overlay disabled: %s", e)

            for vid_i in range(num_videos):
                vid_states = all_rollout_states[vid_i]
                vid_indices = (
                    all_rollout_indices[vid_i]
                    if vid_i < len(all_rollout_indices)
                    else None
                )
                vid_reconstructed = (
                    all_reconstructed[vid_i]
                    if (
                        all_reconstructed is not None
                        and vid_i < len(all_reconstructed)
                        and len(all_reconstructed[vid_i]) > 0
                    )
                    else None
                )
                video_path = (
                    f"{model_path}/{current_step}_{metric_prefix}_vid{vid_i}.mp4"
                )

                # When the c2c env is active we have BOTH the target
                # code per frame (vid_indices, from the policy's obs)
                # AND the reconstructed code per frame (vid_reconstructed,
                # the frozen tokenizer's nearest entry for the agent's
                # actual motion). Pass both as stacked bars so the
                # rendered video shows the per-step alignment / mismatch
                # between what was asked and what was produced.
                if vid_reconstructed is not None and vid_indices is not None:
                    # Align lengths (rollout may terminate before the
                    # full episode_length; trim to the shorter).
                    n_aligned = min(len(vid_indices), len(vid_reconstructed))
                    # Recompute num_codes to span BOTH bars so the color
                    # palette is consistent across them.
                    nc_for_bars = max(
                        int(num_codes) if num_codes is not None else 1,
                        int(np.max(vid_indices[:n_aligned])) + 1
                        if n_aligned > 0 else 1,
                        int(np.max(vid_reconstructed[:n_aligned])) + 1
                        if n_aligned > 0 and (vid_reconstructed[:n_aligned] >= 0).any()
                        else 1,
                    )
                    # Replace sentinel -1 (no-code-yet) with 0 so the
                    # color modulo math doesn't wrap to a wrong color.
                    rec_clean = np.where(
                        vid_reconstructed[:n_aligned] < 0, 0,
                        vid_reconstructed[:n_aligned],
                    ).astype(vid_indices.dtype)
                    render_rollout_to_video(
                        env=env,
                        rollout_states=vid_states,
                        output_path=video_path,
                        camera=f"{cfg.render_config.render_camera_name}"
                               f"{env._suffix}",
                        width=640,
                        height=480,
                        fps=render_fps,
                        indices_per_depth=[
                            np.asarray(vid_indices[:n_aligned]),
                            rec_clean,
                        ],
                        bar_row_labels=("target", "reconstructed"),
                        num_codes=nc_for_bars,
                        code_bar_height=30,
                    )
                else:
                    render_rollout_to_video(
                        env=env,
                        rollout_states=vid_states,
                        output_path=video_path,
                        camera=f"{cfg.render_config.render_camera_name}"
                               f"{env._suffix}",
                        width=640,
                        height=480,
                        fps=render_fps,
                        indices=vid_indices,
                        num_codes=num_codes,
                        code_bar_height=30,
                        code_colors_override=_code_colors_override,
                        code_names=_code_names,
                    )

                wandb.log(
                    {
                        f"{video_prefix}/{metric_prefix}_rollout_{vid_i}": wandb.Video(
                            video_path, format="mp4"
                        )
                    },
                    commit=False,
                )

        except Exception as e:
            logging.warning(f"Failed to render {metric_prefix} video: {e}")

    return reward_mean


def moseq_rollout_logging_fn(
    env,
    jit_reset,
    jit_step,
    cfg,
    model_path,
    current_step,
    jit_logging_inference_fn,
    params,
    policy_params_fn_key,
    render_video=True,
    ppo_network=None,
    jit_decoder_only_inference_fn=None,
    num_codes=None,
):
    """Rollout logging with MoSeq code metrics and video rendering.

    Runs full eval (z_e=1) and optionally decoder-only eval (z_e=0).

    Args:
        num_codes: Codebook size threaded explicitly from the loader. When
            ``None``, falls back to ``cfg.network_config.num_codes`` for
            backward compatibility, but callers should pass the data-derived
            value so plots and video overlays match the actual embedding.
    """
    if num_codes is None:
        num_codes = int(cfg.network_config.num_codes)

    physics_steps_per_ctrl = cfg.env_config.ctrl_dt / cfg.env_config.sim_dt
    steps_per_mocap_frame = (1 / cfg.env_config.mocap_hz) / (
        cfg.env_config.sim_dt * physics_steps_per_ctrl
    )
    episode_length = int(cfg.env_config.clip_length * steps_per_mocap_frame)
    n_rollouts = cfg.render_config.get("eval_rollouts_for_transition", 16)
    use_rnn = bool(cfg.network_config.get("use_rnn_decoder", False))

    key = policy_params_fn_key

    # --- Full eval (z_e=1.0) ---
    (
        all_indices, all_states, all_rewards, all_reconstructed, key,
    ) = _run_eval_rollouts(
        env,
        jit_reset,
        jit_step,
        jit_logging_inference_fn,
        params,
        key,
        episode_length,
        n_rollouts,
        use_rnn,
        ppo_network,
    )
    full_reward = _log_eval_metrics(
        all_indices,
        all_states,
        all_rewards,
        env,
        cfg,
        model_path,
        current_step,
        num_codes,
        render_video,
        metric_prefix="moseq",
        video_prefix="videos",
        all_reconstructed=all_reconstructed,
    )

    # --- Decoder-only eval (z_e=0.0) ---
    if jit_decoder_only_inference_fn is not None:
        (
            do_indices, do_states, do_rewards, do_reconstructed, key,
        ) = _run_eval_rollouts(
            env,
            jit_reset,
            jit_step,
            jit_decoder_only_inference_fn,
            params,
            key,
            episode_length,
            n_rollouts,
            use_rnn,
            ppo_network,
        )
        decoder_only_reward = _log_eval_metrics(
            do_indices,
            do_states,
            do_rewards,
            env,
            cfg,
            model_path,
            current_step,
            num_codes,
            render_video,
            metric_prefix="decoder_only",
            video_prefix="videos",
            all_reconstructed=do_reconstructed,
        )

        # Reward gap: how much performance is lost without z_e
        wandb.log(
            {"decoder_only/reward_gap": full_reward - decoder_only_reward},
            commit=False,
        )


# ---------------------------------------------------------------------------
# code2code reward-mode plumbing
# ---------------------------------------------------------------------------


def _build_c2c_env_kwargs(cfg, payload, codes_path) -> dict:
    """Build the c2c_* kwargs for :class:`MoSeqImitation`.

    Loads the configured tokenizer adapter, validates code-source
    compatibility, auto-calibrates τ if requested, and returns a dict
    of kwargs to splat into ``MoSeqImitation(**c2c_kwargs)``. When
    ``code2code.reward_mode == "off"``, returns an empty dict — env
    behavior is bit-equivalent to the pre-c2c default.

    Args:
        cfg: Full Hydra cfg.
        payload: ``CodesPayload`` from :func:`load_codes`.
        codes_path: Resolved path to the codes npz.

    Returns:
        Dict of c2c_* kwargs. Empty when c2c is disabled.

    Raises:
        ValueError: incompatible code source for c2c, missing tokenizer
            ckpt, codebook-size mismatch.
        NotImplementedError: tokenizer_kind not in {vae}.
    """
    c2c_cfg = cfg.get("code2code", None)
    if c2c_cfg is None:
        return {}
    reward_mode = str(c2c_cfg.get("reward_mode", "off"))
    if reward_mode not in ("off", "additive"):
        # Common pitfall: YAML/Hydra parses unquoted `on`/`true` as the
        # boolean True; str(True) -> "True" silently passes through.  Reject
        # at the config-load site rather than letting the env constructor
        # raise a less-informative error.
        raise ValueError(
            f"code2code.reward_mode must be 'off' or 'additive'; "
            f"got {reward_mode!r}. If you wrote `reward_mode=on` on the "
            f"CLI, quote it: `code2code.reward_mode=additive` or "
            f"`code2code.reward_mode='\"off\"'`."
        )
    if reward_mode == "off":
        return {}

    # code2code only supports the neural kinematics-VAE tokenizer.
    # KPMS / gt_directional_turn can't run inside the env step
    # (see scratch/2026-05-26-code2code-mode/README.md §5).
    if payload.code_source != SOURCE_KINEMATICS_VAE:
        raise ValueError(
            f"code2code.reward_mode={reward_mode!r} requires the neural "
            f"kinematics_vae tokenizer code source. "
            f"Got code_source={payload.code_source!r}. Either set "
            f"code2code.reward_mode: off or switch to a neural code source."
        )

    tokenizer_kind = c2c_cfg.get("tokenizer_kind", None)
    if tokenizer_kind is None:
        raise ValueError(
            "code2code.reward_mode != 'off' requires "
            "code2code.tokenizer_kind to be 'vae'."
        )
    tokenizer_ckpt = c2c_cfg.get("tokenizer_ckpt", None)
    if tokenizer_ckpt is None:
        raise ValueError(
            "code2code.reward_mode != 'off' requires "
            "code2code.tokenizer_ckpt to point to the tokenizer step dir."
        )
    ckpt_p = Path(str(tokenizer_ckpt))
    if not ckpt_p.is_absolute():
        ckpt_p = get_repo_root() / ckpt_p
    if not ckpt_p.exists():
        raise FileNotFoundError(
            f"code2code.tokenizer_ckpt={tokenizer_ckpt} does not resolve "
            f"to an existing directory (tried {ckpt_p})."
        )

    arch_cfg = c2c_cfg.get("tokenizer_arch", {})
    arch_kwargs = dict(
        n_joints=int(arch_cfg.get("n_joints", 67)),
        use_qvel=bool(arch_cfg.get("use_qvel", False)),
        window_len=int(arch_cfg.get("window_len", 10)),
        horizon=int(arch_cfg.get("horizon", 5)),
        conv_channels=tuple(arch_cfg.get("conv_channels", [64, 128, 256])),
        conv_kernel_size=int(arch_cfg.get("conv_kernel_size", 3)),
        conv_head_layers=tuple(arch_cfg.get("conv_head_layers", [256])),
        decoder_layer_sizes=tuple(arch_cfg.get("decoder_layer_sizes", [256, 256, 256])),
    )

    if tokenizer_kind == "vae":
        from c2a.code2code import VAEAdapter
        vae_pkl = c2c_cfg.get("vae_tokenizer_pkl", None)
        if vae_pkl is None:
            raise ValueError(
                "code2code.tokenizer_kind='vae' requires "
                "code2code.vae_tokenizer_pkl to be set."
            )
        vae_pkl_p = Path(str(vae_pkl))
        if not vae_pkl_p.is_absolute():
            vae_pkl_p = get_repo_root() / vae_pkl_p
        if not vae_pkl_p.exists():
            raise FileNotFoundError(
                f"code2code.vae_tokenizer_pkl={vae_pkl} does not resolve "
                f"(tried {vae_pkl_p})."
            )
        adapter = VAEAdapter.from_ckpt_dir(
            ckpt_p, tokenizer_pkl=vae_pkl_p, **arch_kwargs,
        )

    else:
        raise NotImplementedError(
            f"code2code.tokenizer_kind={tokenizer_kind!r} not implemented; "
            f"expected: vae"
        )

    # Codebook-size consistency: the tokenizer's K must match what the
    # codes npz reports (silent mismatch = garbage targets).
    if adapter.codebook_size != int(payload.num_codes):
        raise RuntimeError(
            f"code2code adapter codebook_size={adapter.codebook_size} "
            f"!= payload.num_codes={payload.num_codes}. The tokenizer "
            f"ckpt and codes_path must come from the same training run."
        )

    # Load cal_qpos/cal_codes always — needed both for optional temperature
    # auto-calibration AND for the integrity check below.  (Previously only
    # set inside the auto-calibration branch, which caused an UnboundLocalError
    # when temperature was set to a fixed float — Phase 3 config bump exposed
    # this latent bug.)
    temp_cfg = c2c_cfg.get("temperature", "auto")
    distance_form = str(c2c_cfg.get("distance_form", "cosine"))
    ref_h5 = cfg.env_config.reference_data_path
    clip_length = int(cfg.env_config.clip_length)
    import h5py

    with h5py.File(ref_h5, "r") as f:
        qpos_flat = f["qpos"][:]
    n_total = qpos_flat.shape[0] // clip_length
    clipped = qpos_flat[: n_total * clip_length].reshape(
        n_total, clip_length, -1,
    ).astype(np.float32)
    # payload.train_codes is aligned to train_clips row order; we
    # can use it directly. clipped is in original H5 order, so we
    # need to subset by train_indices to align.
    try:
        with open(cfg.env_config.balanced_split_path) as f:
            splits = json.load(f)
        train_inds = np.array(
            splits["balanced"]["train_indices"], dtype=np.int64,
        )
    except Exception:
        train_inds = np.arange(payload.train_codes.shape[0])
    cal_qpos = clipped[train_inds[: int(c2c_cfg.get("calibrate_n_clips", 50))]]
    cal_codes = payload.train_codes[
        : int(c2c_cfg.get("calibrate_n_clips", 50))
    ]

    # Temperature: auto-calibrate, or use the user-specified scalar.
    if str(temp_cfg) == "auto":
        from c2a.code2code.calibrate import calibrate_temperature

        _, tau = calibrate_temperature(
            adapter,
            ref_qpos=cal_qpos,
            target_codes=cal_codes,
            distance_form=distance_form,  # type: ignore[arg-type]
            target_initial_reward=float(
                c2c_cfg.get("calibrate_target_reward", 0.3)
            ),
            n_clips_sample=int(c2c_cfg.get("calibrate_n_clips", 50)),
            n_windows_per_clip=int(c2c_cfg.get("calibrate_n_windows_per_clip", 8)),
        )
        temperature = float(tau)
    else:
        temperature = float(temp_cfg)

    # ─── Reference-data round-trip integrity check ─────────────────────────
    # Verify that the runtime encode/lookup pipeline reproduces the
    # calibration on reference data. Catches: wrong normalizer loaded,
    # mis-indexed codebook, wrong window-len/rate convention, wrong
    # encoder weights, etc. This is fail-fast: a misaligned pipeline
    # cannot be detected during PPO (rewards just look noisy), so we
    # gate startup on it.
    #
    # The check assumes auto-calibrated temperature → expected reward of
    # `calibrate_target_reward` (default 0.3). When the user sets a
    # manual temperature (Phase 3 soft-basin bump), expected reward
    # shifts so we skip the check with a warning rather than failing
    # spuriously.
    from c2a.code2code.calibrate import validate_reference_round_trip

    if str(temp_cfg) == "auto":
        validate_reference_round_trip(
            adapter,
            ref_qpos=cal_qpos,
            target_codes=cal_codes,
            tau=temperature,
            distance_form=distance_form,  # type: ignore[arg-type]
            reward_form=str(c2c_cfg.get("reward_form", "exp_neg_distance")),
            expected_reward=float(c2c_cfg.get("calibrate_target_reward", 0.3)),
            tolerance=float(c2c_cfg.get("integrity_check_tolerance", 0.08)),
            n_clips_sample=int(c2c_cfg.get("calibrate_n_clips", 50)),
            n_windows_per_clip=int(c2c_cfg.get("calibrate_n_windows_per_clip", 8)),
        )
    else:
        logging.warning(
            "Skipping c2c reference round-trip integrity check because "
            "code2code.temperature is manually set (%.4f rather than 'auto'). "
            "The check expects auto-calibrated reward ≈ %.3f and would trip "
            "on the soft-basin temperature bump.",
            temperature,
            float(c2c_cfg.get("calibrate_target_reward", 0.3)),
        )

    logging.info(
        "code2code reward mode: %s, tokenizer_kind=%s, ckpt=%s, "
        "K=%d, code_dim=%d, distance=%s, reward_form=%s, tau=%.4f, "
        "alpha=%.3f, track_weight=%.3f",
        reward_mode, tokenizer_kind, ckpt_p,
        adapter.codebook_size, adapter.code_dim,
        distance_form, str(c2c_cfg.get("reward_form", "exp_neg_distance")),
        temperature,
        float(c2c_cfg.get("alpha", 0.5)),
        float(c2c_cfg.get("track_weight", 1.0)),
    )

    return dict(
        c2c_adapter=adapter,
        c2c_alpha=float(c2c_cfg.get("alpha", 0.5)),
        c2c_track_weight=float(c2c_cfg.get("track_weight", 1.0)),
        c2c_temperature=temperature,
        c2c_distance_form=distance_form,  # type: ignore[arg-type]
        c2c_reward_form=str(c2c_cfg.get("reward_form", "exp_neg_distance")),  # type: ignore[arg-type]
        c2c_reward_mode=reward_mode,  # type: ignore[arg-type]
    )


@hydra.main(version_base=None, config_path="configs", config_name="moseq_decoder")
def main(cfg: DictConfig) -> None:
    """Main MoSeq decoder training entry point."""
    _setup_environment()

    try:
        n_devices = jax.device_count(backend="gpu")
        logging.info(f"Using {n_devices} GPUs")
    except RuntimeError:
        n_devices = 1
        logging.info("Not using GPUs")

    # Checkpoint path
    run_id, checkpoint_path, existing_run_state = checkpointing.load_from_run_state(cfg)

    # Prepare config
    cfg, cfg_dict, env_cfg_ml = utils.prepare_config(cfg)

    # Reroute reference_data_path to c2a's local data dir. prepare_config sets
    # it via Path(track_mjx.__file__).parent.parent.parent / "data/..." which
    # points inside whatever track_mjx clone was pip-installed from. In c2a
    # deployments where track_mjx lives as a sibling clone inside code2act/,
    # that resolver lands in an empty `code2act/track-mjx/data/` instead of
    # the real `code2act/data/`. Always prefer the c2a-local file when it exists.
    _c2a_ref = get_repo_root() / "data" / "rodent" / "rodent_reference_clips.h5"
    if _c2a_ref.exists():
        if str(_c2a_ref) != cfg.env_config.reference_data_path:
            logging.info(
                f"Overriding reference_data_path: "
                f"{cfg.env_config.reference_data_path} -> {_c2a_ref}"
            )
        OmegaConf.update(
            cfg.env_config, "reference_data_path", str(_c2a_ref), merge=False
        )
        cfg_dict["env_config"]["reference_data_path"] = str(_c2a_ref)

    # Checkpoint manager
    mgr_options = ocp.CheckpointManagerOptions(
        create=True,
        step_prefix="MoSeqPPONetwork",
    )
    ckpt_mgr = ocp.CheckpointManager(checkpoint_path, options=mgr_options)

    # Load balanced splits
    logging.info(f"Loading data: {cfg.env_config.reference_data_path}")
    balanced_split_path = cfg.env_config.get("balanced_split_path", None)
    if balanced_split_path:
        bsp = Path(balanced_split_path)
        if not bsp.is_absolute():
            bsp = get_repo_root() / bsp
        if not bsp.exists():
            raise FileNotFoundError(
                f"env_config.balanced_split_path={balanced_split_path} does not "
                f"resolve (tried {bsp}). Set balanced_split_path=null to fall "
                "back to a seeded random split."
            )
        with open(bsp) as f:
            splits = json.load(f)
        train_indices = splits["balanced"]["train_indices"]
        test_indices = splits["balanced"]["test_indices"]

        train_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=np.array(train_indices),
        )
        test_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=np.array(test_indices),
        )
        logging.info(
            f"Loaded balanced splits: {len(train_indices)} train, "
            f"{len(test_indices)} test"
        )
    else:
        reference_clips = ReferenceClips(
            data_path=cfg.env_config.reference_data_path,
            n_frames_per_clip=cfg.env_config.clip_length,
            keep_clips_idx=cfg.env_config.keep_clips_idx,
        )
        key_split, _ = jax.random.split(
            jax.random.PRNGKey(cfg.train_setup.train_config.seed)
        )
        train_clips, test_clips = reference_clips.split(
            train_ratio=cfg.train_setup.train_subset_ratio,
            seed=key_split,
        )
        train_indices = list(range(train_clips.qpos.shape[0]))
        test_indices = list(range(test_clips.qpos.shape[0]))

    # Load codes from the configured source (kpms_raw / kpms_merged /
    # gt_directional_turn / random for null codes_path). All validation,
    # provenance, and source-specific quirks live in c2a.codegen.code_sources.
    payload = load_codes(
        cfg=cfg,
        train_indices=train_indices,
        test_indices=test_indices,
        n_frames=int(cfg.env_config.clip_length),
    )
    train_codes = payload.train_codes
    test_codes = payload.test_codes
    num_codes = payload.num_codes
    codes_path = payload.provenance.get("codes_path")  # None for GT/random

    # Patch the checkpoint-metadata dict (NOT the live cfg) so saved
    # checkpoints record the codebook size that was actually used. We do
    # NOT mutate cfg.network_config.num_codes because:
    #   (a) checkpointing.load_from_run_state hashes cfg before this point
    #       and save_run_state hashes it after — mutating cfg here would
    #       create an asymmetric hash and silently drop run-state on
    #       preemption resume for any source whose K != YAML default.
    #   (b) The rollout-logging closure reads num_codes through an
    #       explicit kwarg (see _policy_params_fn_wrapper below), not via
    #       cfg, so cfg's value can stay equal to the YAML.
    cfg_dict["network_config"]["num_codes"] = int(num_codes)

    # Code stack size: how many consecutive codes to give the decoder
    code_stack_size = int(cfg.network_config.get("code_stack_size", 1))

    # ------------------------------------------------------------------
    # code2code reward mode: optional frozen-tokenizer reward shaping
    # ------------------------------------------------------------------
    # When enabled, an instance of c2a.code2code.C2CTokenizerAdapter is
    # constructed from the configured tokenizer ckpt and threaded into
    # MoSeqImitation. The env computes the code-match reward at every
    # step. The adapter is NEVER part of the policy params and PPO
    # never updates it — see scratch/2026-05-26-code2code-mode/
    # design-A-reward-shaping.md for the architectural rationale.
    c2c_kwargs = _build_c2c_env_kwargs(cfg, payload, codes_path)

    # MoSeqImitation overrides _get_obs to inject kpms_code, flatten nested
    # obs, and strip the "state" hierarchy — all inline. No wrapper chain
    # needed, since BraxDomainRandomizationVmapWrapper bypasses wrappers
    # via env.unwrapped.
    env = MoSeqImitation(
        config=env_cfg_ml, clips=train_clips, kpms_codes=train_codes,
        code_stack_size=code_stack_size,
        **c2c_kwargs,
    )
    test_env = MoSeqImitation(
        config=env_cfg_ml, clips=test_clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
        **c2c_kwargs,
    )

    logging.info(f"Environment config: {cfg.env_config}")

    # Episode length
    steps_per_frame = (1 / cfg.env_config.mocap_hz) / cfg.env_config.ctrl_dt
    episode_length = (
        cfg.env_config.clip_length
        - cfg.env_config.start_frame_range[-1]
        - cfg.env_config.reference_length
    ) * steps_per_frame
    logging.info(f"episode_length {episode_length}")

    logging.info("Using MoSeq Decoder PPO Pipeline")

    # Continuous encoder config
    use_continuous_encoder = bool(
        cfg.network_config.get("use_continuous_encoder", False)
    )
    encoder_layer_sizes = tuple(
        cfg.network_config.get("encoder_layer_sizes", [256, 128])
    )
    continuous_latent_dim = int(cfg.network_config.get("continuous_latent_dim", 16))
    kl_weight = float(cfg.network_config.get("kl_weight", 0.0))
    z_e_dropout_rate = float(cfg.network_config.get("z_e_dropout_rate", 0.0))
    z_e_at_action_head = bool(cfg.network_config.get("z_e_at_action_head", False))
    reinit_hidden_on_code = bool(cfg.network_config.get("reinit_hidden_on_code", False))
    learned_hidden_init = bool(cfg.network_config.get("learned_hidden_init", False))
    encoder_uses_proprio = bool(
        cfg.network_config.get("encoder_uses_proprio", False)
    )

    # RNN decoder config
    use_rnn_decoder = bool(cfg.network_config.get("use_rnn_decoder", False))
    rnn_hidden_sizes = tuple(cfg.network_config.get("rnn_hidden_sizes", [256]))
    rnn_cell_type = str(cfg.network_config.get("rnn_cell_type", "gru"))

    # Distillation head config
    use_distillation_head = bool(
        cfg.network_config.get("use_distillation_head", False)
    )
    distill_head_layer_sizes = tuple(
        cfg.network_config.get("distillation_head_layer_sizes", [256, 128])
    )
    distill_kl_weight = float(cfg.network_config.get("distill_kl_weight", 1.0))
    distillation_encoder_checkpoint = cfg.network_config.get(
        "distillation_encoder_checkpoint", None
    )
    distill_logvar_min = cfg.network_config.get("distill_logvar_min", None)
    distill_logvar_max = cfg.network_config.get("distill_logvar_max", None)
    if distill_logvar_min is not None:
        distill_logvar_min = float(distill_logvar_min)
    if distill_logvar_max is not None:
        distill_logvar_max = float(distill_logvar_max)
    use_pretrained_decoder = bool(
        cfg.network_config.get("use_pretrained_decoder", False)
    )
    decoder_layer_sizes_vae = tuple(
        cfg.network_config.get("decoder_layer_sizes_vae", [512, 256, 256, 256])
    )

    # Auto-disable / auto-override when distillation head is enabled
    if use_distillation_head:
        if not use_rnn_decoder:
            raise ValueError(
                "use_distillation_head=True requires use_rnn_decoder=True "
                "(distill head reads from RNN hidden state)"
            )
        if distillation_encoder_checkpoint is None:
            logging.warning(
                "use_distillation_head=True without encoder checkpoint — "
                "using randomly initialized encoder (for testing only)"
            )
        # Force continuous encoder ON (needed to produce distillation targets)
        use_continuous_encoder = True
        # z_e must NOT enter the action path
        z_e_at_action_head = False
        z_e_dropout_rate = 0.0
        # Disable z_e action-path KL schedule (z_e is detached from actions)
        kl_weight = 0.0

        logging.info(
            f"Distillation mode enabled: z_e detached from action path, "
            f"KL schedule disabled, distill_kl_weight={distill_kl_weight}, "
            f"encoder_checkpoint={distillation_encoder_checkpoint}"
        )

    # KL schedule config
    kl_schedule = str(cfg.network_config.get("kl_schedule", "none"))
    if use_distillation_head:
        kl_schedule = "none"  # Force disable when distilling
    kl_sched_cfg = cfg.network_config.get("kl_schedule_config", {})
    kl_sched_start = float(kl_sched_cfg.get("start_value", kl_weight))
    kl_sched_end = float(kl_sched_cfg.get("end_value", 0.5))
    kl_sched_start_frac = float(kl_sched_cfg.get("start_frac", 0.3))
    kl_sched_end_frac = float(kl_sched_cfg.get("end_frac", 0.7))
    kl_sched_num_cycles = int(kl_sched_cfg.get("num_cycles", 4))

    num_evals_total = int(
        cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
    )

    z_e_scale_fn = None  # No z_e_scale scheduling — use KL schedule instead
    kl_weight_fn = None

    if kl_schedule != "none" and use_continuous_encoder:
        _start = kl_sched_start_frac * num_evals_total
        _end = kl_sched_end_frac * num_evals_total
        _range = max(_end - _start, 1.0)
        _sv = kl_sched_start
        _ev = kl_sched_end

        if kl_schedule == "ramp":
            def kl_weight_fn(step, _s=_start, _r=_range, _e=_end, sv=_sv, ev=_ev):
                frac = jnp.clip((step - _s) / _r, 0.0, 1.0)
                w = sv + (ev - sv) * frac
                return jnp.where(step < _s, sv, jnp.where(step > _e, ev, w))

            logging.info(
                f"KL ramp: {_sv}→{_ev} over steps {_start:.0f}-{_end:.0f} "
                f"(of {num_evals_total} evals)"
            )

        elif kl_schedule == "cosine_anneal":
            _nc = kl_sched_num_cycles

            def kl_weight_fn(step, _s=_start, _n=num_evals_total, sv=_sv, ev=_ev, nc=_nc):
                progress = jnp.clip((step - _s) / max(_n - _s, 1.0), 0.0, 1.0)
                cos_val = 0.5 * (1.0 + jnp.cos(2.0 * jnp.pi * nc * progress))
                # cos_val=1 at trough (low KL), cos_val=0 at peak (high KL)
                w = ev + (sv - ev) * cos_val
                return jnp.where(step < _s, sv, w)

            logging.info(
                f"KL cosine anneal: {_sv}↔{_ev}, {_nc} cycles, "
                f"starting at step {_start:.0f} (of {num_evals_total} evals)"
            )
        else:
            logging.warning(f"Unknown kl_schedule '{kl_schedule}', using constant kl_weight")

    # Network factory
    if use_rnn_decoder:
        network_factory = functools.partial(
            make_moseq_recurrent_decoder_ppo_networks,
            num_codes=num_codes,
            code_embed_dim=int(cfg.network_config.code_embed_dim),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            use_continuous_encoder=use_continuous_encoder,
            encoder_layer_sizes=encoder_layer_sizes,
            continuous_latent_dim=continuous_latent_dim,
            z_e_dropout_rate=z_e_dropout_rate,
            rnn_hidden_sizes=rnn_hidden_sizes,
            rnn_cell_type=rnn_cell_type,
            z_e_at_action_head=z_e_at_action_head,
            reinit_hidden_on_code=reinit_hidden_on_code,
            learned_hidden_init=learned_hidden_init,
            use_distillation_head=use_distillation_head,
            distill_head_layer_sizes=distill_head_layer_sizes,
            distill_logvar_min=distill_logvar_min,
            distill_logvar_max=distill_logvar_max,
            use_pretrained_decoder=use_pretrained_decoder,
            decoder_layer_sizes_vae=decoder_layer_sizes_vae,
            encoder_uses_proprio=encoder_uses_proprio,
        )
        logging.info(
            f"Using RNN decoder: cell={rnn_cell_type}, hidden={rnn_hidden_sizes}, "
            f"z_e_at_action_head={z_e_at_action_head}, reinit_hidden={reinit_hidden_on_code}, "
            f"learned_init={learned_hidden_init}, "
            f"distillation_head={use_distillation_head}, "
            f"encoder_uses_proprio={encoder_uses_proprio}"
        )
    else:
        network_factory = functools.partial(
            make_moseq_decoder_ppo_networks,
            num_codes=num_codes,
            code_embed_dim=int(cfg.network_config.code_embed_dim),
            decoder_hidden_layer_sizes=tuple(cfg.network_config.decoder_layer_sizes),
            value_hidden_layer_sizes=tuple(cfg.network_config.critic_layer_sizes),
            use_continuous_encoder=use_continuous_encoder,
            encoder_layer_sizes=encoder_layer_sizes,
            continuous_latent_dim=continuous_latent_dim,
            z_e_dropout_rate=z_e_dropout_rate,
            encoder_uses_proprio=encoder_uses_proprio,
        )

    # Encoder loading: inject pre-trained encoder params from a VAE
    # checkpoint trained with train_mimic_encoder.py.  Both the VAE and
    # the distillation model use IntentionEncoder, so the encoder subtree
    # at params["encoder_module"] has identical structure — one-line drop-in.
    #
    # IMPORTANT — resume safety: when ``train_setup.checkpoint_to_restore``
    # is set, track-mjx's ppo.train loads the FULL training state (including
    # any trained VQ-VAE / encoder subtrees) BEFORE calling
    # `_post_init_params_fn`. Overwriting them with fresh-from-disk
    # pretrained values would silently nuke the live training progress.
    # We therefore skip both subtree assignments on resume (the resumed
    # checkpoint already carries the correct current values for both).
    _resume_in_progress = cfg.train_setup.get("checkpoint_to_restore", None) is not None

    _post_init_params_fn = None
    if use_distillation_head and distillation_encoder_checkpoint is not None:
        # Orbax requires absolute paths. Resolve relatives against the repo
        # root (same rule as codes_path / balanced_split_path above). Without
        # this, a config like `${oc.env:C2A_ROOT,.}/trained_mimic_ckpts/...`
        # evaluates to a bare relative path whenever C2A_ROOT is unset, and
        # Hydra's cwd change makes the directory unreachable by the time
        # orbax tries to read it.
        _enc_p = Path(distillation_encoder_checkpoint)
        if not _enc_p.is_absolute():
            _enc_p = get_repo_root() / _enc_p
        if not _enc_p.exists():
            raise FileNotFoundError(
                f"network_config.distillation_encoder_checkpoint="
                f"{distillation_encoder_checkpoint} does not resolve to an "
                f"existing directory (tried {_enc_p})."
            )
        _encoder_ckpt_path = str(_enc_p.resolve())
        _encoder_step_prefix = str(
            cfg.network_config.get("distillation_encoder_step_prefix", "MimicEncoder")
        )

        def _post_init_params_fn(training_state):
            """Load encoder from VAE checkpoint (direct subtree assignment)."""
            if _resume_in_progress:
                # Resumed run: the restored training_state already has the
                # correct (current) values for both subtrees. Don't clobber.
                logging.info(
                    "Resume detected (checkpoint_to_restore is set) — "
                    "skipping pretrained subtree assignment in _post_init_params_fn."
                )
                return training_state

            mgr_options = ocp.CheckpointManagerOptions(
                create=False, step_prefix=_encoder_step_prefix,
            )
            with ocp.CheckpointManager(_encoder_ckpt_path, options=mgr_options) as mgr:
                restored = mgr.restore(
                    mgr.latest_step(),
                    args=ocp.args.Composite(policy=ocp.args.StandardRestore()),
                )
            _, loaded_policy_params = restored["policy"]

            # The VAE checkpoint stores encoder under params["encoder"].
            # Our model stores it under params["encoder_module"].
            # Both are the same IntentionEncoder Flax module.
            loaded_encoder = loaded_policy_params["params"]["encoder"]
            training_state.params.policy["params"]["encoder_module"] = loaded_encoder
            logging.info(
                f"Loaded encoder from {distillation_encoder_checkpoint} "
                f"(step_prefix={_encoder_step_prefix})"
            )

            # Also load decoder if use_pretrained_decoder is enabled
            if use_pretrained_decoder and "decoder" in loaded_policy_params["params"]:
                loaded_decoder = loaded_policy_params["params"]["decoder"]
                training_state.params.policy["params"]["decoder_module"] = loaded_decoder
                logging.info(
                    f"Loaded decoder from {distillation_encoder_checkpoint}"
                )

            return training_state

    # WandB
    wandb_logging.initialize_wandb_logging(
        logging_cfg=cfg.logging_config,
        cfg=cfg,
        run_id=run_id,
        existing_run_state=existing_run_state,
    )

    if use_distillation_head:
        arch_name = "moseq_rnn_distill"
    elif use_rnn_decoder:
        arch_name = (
            "moseq_rnn_encoder_decoder"
            if use_continuous_encoder
            else "moseq_rnn_decoder"
        )
    else:
        arch_name = (
            "moseq_encoder_decoder" if use_continuous_encoder else "moseq_decoder"
        )

    # KPMS provenance is now carried in payload.provenance (loader-supplied)
    # for kpms_raw / kpms_merged. For gt_directional_turn the loader supplies
    # GT-specific metadata (class names, counts). Fall back to sweep_results.json
    # alongside a KPMS .npz only when the .npz itself didn't carry kappa.
    kpms_kappa = payload.provenance.get("kpms_kappa")
    kpms_num_states = payload.provenance.get("kpms_num_states")
    kpms_model_type = payload.provenance.get("kpms_model_type")
    kpms_mean_duration = payload.provenance.get("kpms_mean_duration")
    if codes_path and kpms_kappa is None:
        sweep_results_path = Path(codes_path).parent / "sweep_results.json"
        if sweep_results_path.exists():
            try:
                with open(sweep_results_path) as f:
                    sweep_results = json.load(f)
                best = sweep_results.get("best_model", {})
                kpms_kappa = best.get("kappa")
                kpms_num_states = best.get("n_states")
                kpms_model_type = best.get("model_type")
                kpms_mean_duration = best.get("mean_duration")
                logging.info(
                    f"KPMS provenance (from sweep_results.json): "
                    f"kappa={kpms_kappa}, n_states={kpms_num_states}, "
                    f"model={kpms_model_type}, mean_duration={kpms_mean_duration}"
                )
            except Exception as e:
                logging.warning(f"Could not read KPMS sweep results: {e}")

    wandb.config.update(
        {
            "arch": arch_name,
            "num_codes": num_codes,
            "code_embed_dim": int(cfg.network_config.code_embed_dim),
            "code_stack_size": code_stack_size,
            "code_source": payload.code_source,
            "codes_path": str(codes_path) if codes_path else "n/a",
            "is_merged": payload.provenance.get("is_merged"),
            "gt_class_names": payload.provenance.get("gt_class_names"),
            "gt_train_counts": payload.provenance.get("gt_train_counts"),
            "gt_test_counts": payload.provenance.get("gt_test_counts"),
            "labels_csv_path": payload.provenance.get("labels_csv_path"),
            "kpms_kappa": kpms_kappa,
            "kpms_num_states": kpms_num_states,
            "kpms_model_type": kpms_model_type,
            "kpms_mean_duration": kpms_mean_duration,
            "use_continuous_encoder": use_continuous_encoder,
            "continuous_latent_dim": continuous_latent_dim,
            "kl_weight": kl_weight,
            "kl_schedule": kl_schedule,
            "use_rnn_decoder": use_rnn_decoder,
            "rnn_hidden_sizes": list(rnn_hidden_sizes) if use_rnn_decoder else None,
            "rnn_cell_type": rnn_cell_type if use_rnn_decoder else None,
            "z_e_dropout_rate": z_e_dropout_rate,
            "z_e_at_action_head": z_e_at_action_head,
            "reinit_hidden_on_code": reinit_hidden_on_code,
            "learned_hidden_init": learned_hidden_init,
            "use_distillation_head": use_distillation_head,
            "encoder_uses_proprio": encoder_uses_proprio,
            "distill_kl_weight": distill_kl_weight if use_distillation_head else None,
            "distillation_encoder_checkpoint": (
                str(distillation_encoder_checkpoint) if distillation_encoder_checkpoint else None
            ),
        }
    )

    if existing_run_state is None:
        checkpointing.save_run_state(
            cfg=cfg,
            run_id=run_id,
            checkpoint_path=checkpoint_path,
            wandb_run_id=wandb.run.id,
        )

    checkpoint_callback = checkpointing.create_checkpoint_callback(
        cfg=cfg,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        wandb_run_id=wandb.run.id,
    )

    # Training
    train_fn = functools.partial(
        moseq_train,
        **cfg.train_setup.train_config,
        num_evals=int(
            cfg.train_setup.train_config.num_timesteps / cfg.train_setup.eval_every
        ),
        num_resets_per_eval=cfg.train_setup.eval_every // cfg.train_setup.reset_every,
        episode_length=episode_length,
        network_factory=network_factory,
        ckpt_mgr=ckpt_mgr,
        checkpoint_to_restore=cfg.train_setup.checkpoint_to_restore,
        config_dict=cfg_dict,
        eval_env_test_set=test_env,
        freeze_decoder=cfg.train_setup.get("freeze_decoder", False),
        checkpoint_callback=checkpoint_callback,
        wrap_for_training=functools.partial(
            playground_wrappers.wrap_for_brax_training, full_reset=False
        ),
        randomization_fn=(
            domain_randomization_maker(
                floor_friction=cfg.env_config.domain_randomization.floor_friction,
                static_friction_scale=cfg.env_config.domain_randomization.static_friction_scale,
                armature_scale=cfg.env_config.domain_randomization.armature_scale,
                com_jitter=cfg.env_config.domain_randomization.com_jitter,
                link_mass_scale=cfg.env_config.domain_randomization.link_mass_scale,
                torso_mass_jitter=cfg.env_config.domain_randomization.torso_mass_jitter,
                qpos0_jitter=cfg.env_config.domain_randomization.qpos0_jitter,
            )
            if cfg.env_config.domain_randomization.use_domain_randomization
            else None
        ),
        num_codes=num_codes,
        code_embed_dim=int(cfg.network_config.code_embed_dim),
        kl_weight=kl_weight,
        use_rnn_decoder=use_rnn_decoder,
        rnn_hidden_sizes=rnn_hidden_sizes,
        z_e_scale_fn=z_e_scale_fn,
        kl_weight_fn=kl_weight_fn,
        distill_kl_weight=distill_kl_weight if use_distillation_head else 0.0,
        post_init_params_fn=_post_init_params_fn,
    )

    # Rollout env for logging
    rollout_cfg = env_cfg_ml.copy_and_resolve_references()
    rollout_cfg.start_frame_range = [0, 0]
    rollout_env = MoSeqImitation(
        config=rollout_cfg, clips=test_clips, kpms_codes=test_codes,
        code_stack_size=code_stack_size,
        **c2c_kwargs,
    )

    jit_reset = jax.jit(rollout_env.reset)
    jit_step = jax.jit(rollout_env.step)

    # Cache for decoder-only inference fn (created lazily once ppo_network available)
    _decoder_only_fn_cache = {}

    def _policy_params_fn_wrapper(
        current_step,
        jit_logging_inference_fn,
        params,
        policy_params_fn_key,
        render_video=True,
        ppo_network=None,
    ):
        # Create decoder-only inference fn (z_e=0) from ppo_network
        # Skip in distillation mode (z_e never enters action path, so z_e=0 is
        # identical to z_e=1)
        jit_decoder_only_fn = None
        if use_continuous_encoder and not use_distillation_head and ppo_network is not None:
            if "fn" not in _decoder_only_fn_cache:
                from c2a.moseq_ppo_networks import (
                    make_moseq_recurrent_logging_inference_fn,
                    make_moseq_logging_inference_fn,
                )

                if use_rnn_decoder:
                    make_logging = make_moseq_recurrent_logging_inference_fn(
                        ppo_network
                    )
                else:
                    make_logging = make_moseq_logging_inference_fn(ppo_network)

                _decoder_only_fn_cache["fn"] = jax.jit(
                    make_logging(deterministic=True, z_e_scale=0.0)
                )
            jit_decoder_only_fn = _decoder_only_fn_cache["fn"]

        return moseq_rollout_logging_fn(
            rollout_env,
            jit_reset,
            jit_step,
            cfg,
            checkpoint_path,
            current_step=current_step,
            jit_logging_inference_fn=jit_logging_inference_fn,
            params=params,
            policy_params_fn_key=policy_params_fn_key,
            render_video=render_video,
            ppo_network=ppo_network,
            jit_decoder_only_inference_fn=jit_decoder_only_fn,
            num_codes=int(num_codes),
        )

    # Wrap wandb_progress to add `/`-prefixed metric groups for WandB panels.
    # Six grouping panels:
    #   losses/         PPO scalar losses
    #   z_e/            encoder (μ, log σ) magnitudes + KL
    #   codes/          codebook utilization / perplexity / transition rate
    #   rnn/            RNN-decoder hidden state norm
    #   distillation/   distillation head KL
    #   c2c/            code2code reward composition + diagnostics
    # The `training/` prefix from track-mjx's ppo.train is stripped before
    # branch matching; the original `training/*` key is preserved at the
    # end of the loop for any consumer that reads it directly.
    def _grouped_wandb_progress(num_steps: int, metrics: dict) -> None:
        grouped = {}
        for k, v in metrics.items():
            leaf = k.split("/", 1)[1] if k.startswith("training/") else k

            if leaf in ("total_loss", "policy_loss", "v_loss", "entropy_loss"):
                grouped[f"losses/{leaf}"] = v
            elif leaf in ("kl_loss", "scaled_kl_loss", "z_e_norm", "z_e_std", "z_e_scale"):
                grouped[f"z_e/{leaf}"] = v
            elif leaf in ("transition_rate", "perplexity", "codebook_utilization", "codes_used"):
                grouped[f"codes/{leaf}"] = v
            elif leaf in ("hidden_state_norm",):
                grouped[f"rnn/{leaf}"] = v
            elif leaf in (
                "distill_kl_loss", "scaled_distill_kl_loss",
                "distill_mean_norm", "distill_logvar_mean",
            ):
                grouped[f"distillation/{leaf}"] = v
            elif (
                k.startswith("c2c_")
                or k.startswith("c2c/")
                or "episode_c2c_" in k
            ):
                # code2code reward diagnostics: the env writes c2c_* into
                # state.metrics and the eval aggregator surfaces them as
                # eval/episode_c2c_* (train rollouts) and
                # eval/test_set/episode_c2c_* (test rollouts).  All route
                # into the single c2c/ panel.
                if "episode_c2c_" in k:
                    short = k.split("episode_c2c_", 1)[1]
                    if k.startswith("eval/test_set/"):
                        key = f"c2c/eval_test/{short}"
                    elif k.startswith("eval/"):
                        key = f"c2c/eval/{short}"
                    else:
                        key = f"c2c/{short}"
                else:
                    key = k if k.startswith("c2c/") else f"c2c/{k[4:]}"
                grouped[key] = v
            # Keep originals for backward compat with shared code
            grouped[k] = v
        wandb_logging.wandb_progress(num_steps, grouped)

    make_inference_fn, params, _ = train_fn(
        environment=env,
        progress_fn=_grouped_wandb_progress,
        policy_params_fn=_policy_params_fn_wrapper,
    )

    # Log final summary for sweep filtering
    try:
        wandb.run.summary.update({
            "training_completed": True,
            "kpms_kappa": kpms_kappa,
            "kpms_num_states": kpms_num_states,
        })
    except Exception:
        pass

    try:
        checkpointing.cleanup_run_state(cfg)
        logging.info("Training completed successfully, cleaned up run state")
    except Exception as e:
        logging.warning(f"Failed to cleanup run state: {e}")

    wandb.finish()


if __name__ == "__main__":
    main()
