"""Experiment 3: Generative models of KPMS codes + free-loop rollouts.

Inline (no checkpoint needed):
    transition_matrix, arhmm_level1, arhmm_level2, controls.

AR / trainable (loaded via ``ar_checkpoints.<name>``; produced by
``python -m c2a.ar_codegen.train model=<name>``):
    gpt2, mamba, s5, hmm_pretrained.

The inline ``hmm_dynamax`` branch moved to
``c2a.ar_codegen.models.hmm.HMMGenerator``.

Usage:
    python -m c2a.experiments.run_code_generation \\
        generation.methods=[transition_matrix,arhmm_level2,gpt2] \\
        ar_checkpoints.gpt2=outputs/ar_codegen/gpt2/<dir>
"""

import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import logging
import sys
from pathlib import Path


from c2a.paths import get_c2a_root, get_repo_root
import h5py
import hydra
import jax
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy.special import softmax

MOSEQ_DIR = get_c2a_root()
REPO_ROOT = get_repo_root()
from track_mjx.config import utils
from vnl_playground.tasks.rodent.imitation import ReferenceClips
from c2a.moseq_env_wrapper import MoSeqImitation

from c2a.codegen.code_sources import (
    SOURCE_GT_DIRECTIONAL_TURN,
    SOURCE_KPMS_MERGED,
    SOURCE_KPMS_RAW,
    SOURCE_RANDOM,
)
from c2a.experiments.shared.checkpoint_utils import (
    load_moseq_checkpoint,
    load_mimic_checkpoint,
    make_inference_fn,
    make_mimic_inference_fn,
    run_rollout,
)
from c2a.experiments.shared.metrics import compute_transition_matrix, plot_transition_matrix
from c2a.experiments.shared.plotting import (
    set_nature_style,
    get_code_colormap,
    NATURE_COLORS,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


CONTROL_METHODS = {"uniform_random", "single_code", "reversed"}


def _method_subdir(output_dir: Path, method_name: str) -> Path:
    """Return the subfolder for a given method name, creating it if needed."""
    if method_name.startswith("tm_"):
        sub = output_dir / "tm"
    elif method_name.startswith("hmm"):
        sub = output_dir / "hmm"
    elif method_name.startswith("arhmm"):
        sub = output_dir / "arhmm"
    elif method_name in CONTROL_METHODS or "mimic" in method_name:
        sub = output_dir / "baselines"
    else:
        sub = output_dir / "other"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


# ---------------------------------------------------------------------------
# Method A: Empirical Transition Matrix
# ---------------------------------------------------------------------------


def sample_from_transition_matrix(
    T: np.ndarray,
    length: int,
    temperature: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Sample a code sequence from the empirical transition matrix.

    Args:
        T: Row-normalized transition matrix ``[K, K]``.
        length: Sequence length.
        temperature: Sampling temperature (1.0 = original, <1 = greedy).
        seed: Random seed.

    Returns:
        ``[length]`` int array of code indices.
    """
    rng = np.random.RandomState(seed)
    K = T.shape[0]

    # Stationary distribution
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi) / np.abs(pi).sum()

    # Temperature-scaled transition
    log_T = np.log(T + 1e-12)
    T_temp = softmax(log_T / max(temperature, 1e-6), axis=1)

    seq = np.zeros(length, dtype=np.int32)
    seq[0] = rng.choice(K, p=pi)
    for t in range(1, length):
        seq[t] = rng.choice(K, p=T_temp[seq[t - 1]])
    return seq


def generate_empirical_tm_sequences(
    all_codes: np.ndarray,
    num_codes: int,
    cfg: DictConfig,
) -> dict[str, list[np.ndarray]]:
    """Generate sequences via empirical transition matrix."""
    log.info("  Method A: Empirical Transition Matrix")
    T = compute_transition_matrix(list(all_codes), num_codes).astype(float)
    T_norm = T / (T.sum(axis=1, keepdims=True) + 1e-12)

    results = {}
    seed = int(cfg.generation.seed)
    length = int(cfg.generation.sequence_length)
    n_seq = int(cfg.generation.num_sequences)

    for temp in cfg.generation.transition_matrix.temperatures:
        key = f"tm_T{temp}"
        seqs = [
            sample_from_transition_matrix(T_norm, length, temperature=float(temp), seed=seed + i)
            for i in range(n_seq)
        ]
        results[key] = seqs
        log.info(f"    T={temp}: generated {n_seq} sequences")

    return results


# ---------------------------------------------------------------------------
# Method B: HMM via dynamax  →  moved to c2a/ar_codegen/models/hmm.py
# ---------------------------------------------------------------------------
#
# HMM fitting used to happen inline here. It now lives in
# ``c2a.ar_codegen.models.hmm.HMMGenerator`` so that the HMM, GPT-2,
# Mamba, and S5 share one unified train/save/load pipeline. To use the
# HMM in this experiment:
#
#   1) Fit:    ``python -m c2a.ar_codegen.train model=hmm``
#   2) Point:  pass ``ar_checkpoints.hmm=<output_dir>`` to this script
#   3) Enable: include ``hmm_pretrained`` in ``generation.methods``
#
# The pretrained-AR loader below picks it up via
# ``CodeGenerator.from_checkpoint``.


# ---------------------------------------------------------------------------
# Method C: ARHMM Level 1 (discrete only)
# ---------------------------------------------------------------------------


def load_arhmm_params(cfg: DictConfig) -> dict[str, np.ndarray]:
    """Load ARHMM parameters from keypoint-moseq checkpoint."""
    model_path = cfg.generation.arhmm.model_path
    snapshot = int(cfg.generation.arhmm.snapshot_idx)

    with h5py.File(model_path, "r") as f:
        params = f[f"model_snapshots/{snapshot}/params"]
        result = {
            "pi": params["pi"][()],
            "betas": params["betas"][()],
            "Ab": params["Ab"][()],
            "Q": params["Q"][()],
        }
    log.info(f"  Loaded ARHMM params: pi={result['pi'].shape}, Ab={result['Ab'].shape}")
    return result


def _assert_decoder_compatible_with_arhmm(
    ckpt_cfg: DictConfig, methods: list[str],
) -> None:
    """Refuse to run AR-HMM sampling against a non-KPMS decoder.

    The ARHMM (loaded from a keypoint-moseq snapshot in
    ``cfg.generation.arhmm.model_path``) is a generative model over the
    K-class KPMS state space — its transition matrix ``pi``, AR
    weights ``Ab``, and noise ``Q`` are indexed by syllable id in that
    KPMS state space. Feeding those samples into a decoder whose code
    embedding table was trained on a DIFFERENT label set (e.g.
    ``gt_directional_turn``, where ints 0..4 mean
    Immobile/Rear/Walk/Turn_Left/Turn_Right rather than KPMS syllables)
    just routes ARHMM-sampled integers through unrelated embeddings —
    the rollouts are garbage, the FID/KID numbers are meaningless.

    The merged variant (``kpms_merged``) is the only label set whose
    integer ids line up with the AR-HMM's state space AND have a
    principled K (after Bhattacharyya collapse). The unmerged raw
    KPMS source (``kpms_raw``) lives in the same KPMS state space too,
    but the merged variant is the one the unconditioned-generation
    figures expect; gate raw with a clear warning rather than a hard
    refuse so a user can knowingly run on raw if needed.

    Args:
        ckpt_cfg: The decoder checkpoint's resolved Hydra config.
        methods: The list of generation methods this run will execute
            (from ``cfg.generation.methods``). The check only fires
            when an AR-HMM method is actually requested — other
            methods (uniform_random, transition_matrix, hmm_dynamax,
            …) don't rely on KPMS state semantics.

    Raises:
        RuntimeError: ``code_source`` is ``gt_directional_turn`` /
            ``random`` (or any other source whose ints don't index the
            AR-HMM's state space).
    """
    arhmm_methods = {m for m in methods if m.startswith("arhmm")}
    if not arhmm_methods:
        return

    code_source = str(ckpt_cfg.kpms_config.get("code_source", ""))
    if code_source == SOURCE_KPMS_MERGED:
        log.info(
            "  ARHMM compatibility check: decoder trained on "
            "kpms_merged (OK)."
        )
        return
    if code_source == SOURCE_KPMS_RAW:
        log.warning(
            "  ARHMM compatibility check: decoder trained on kpms_raw, "
            "NOT kpms_merged. The AR-HMM lives in the raw KPMS state "
            "space, so integer ids do align — but the unconditioned-"
            "generation figures (FID/KID, behavior-parade panels) are "
            "built around the merged variant. Numbers will not be "
            "directly comparable to those reported in the paper. "
            "Re-train the decoder with code_source=kpms_merged for the "
            "intended comparison."
        )
        return

    raise RuntimeError(
        f"AR-HMM unconditioned generation requires a decoder trained on "
        f"KPMS codes (code_source={SOURCE_KPMS_MERGED!r}, or "
        f"{SOURCE_KPMS_RAW!r} with a warning). The loaded checkpoint "
        f"reports code_source={code_source!r} — the AR-HMM at "
        f"cfg.generation.arhmm.model_path was fit on the KPMS state "
        f"space, so its sampled integer ids do not correspond to this "
        f"decoder's code embedding table (e.g. gt_directional_turn ids "
        f"index the 5-class behavior enum, not KPMS syllables). "
        f"Requested AR-HMM methods: {sorted(arhmm_methods)}. "
        f"Either drop the arhmm_* methods from "
        f"cfg.generation.methods, or point cfg.checkpoint.path at a "
        f"decoder trained with kpms_config.code_source={SOURCE_KPMS_MERGED!r}."
    )


def generate_arhmm_level1_sequences(
    arhmm_params: dict[str, np.ndarray],
    cfg: DictConfig,
) -> list[np.ndarray]:
    """Sample discrete-only sequences from ARHMM transition matrix."""
    log.info("  Method C: ARHMM Level 1 (discrete only)")
    pi = arhmm_params["pi"]
    betas = arhmm_params["betas"]
    pi0 = softmax(betas)

    seed = int(cfg.generation.seed)
    length = int(cfg.generation.sequence_length)
    n_seq = int(cfg.generation.num_sequences)

    rng = np.random.RandomState(seed)
    seqs = []
    for i in range(n_seq):
        seq = np.zeros(length, dtype=np.int32)
        seq[0] = rng.choice(len(pi0), p=pi0)
        for t in range(1, length):
            seq[t] = rng.choice(pi.shape[1], p=pi[seq[t - 1]])
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Method D: ARHMM Level 2 (discrete + AR latent)
# ---------------------------------------------------------------------------


def generate_arhmm_level2_sequences(
    arhmm_params: dict[str, np.ndarray],
    cfg: DictConfig,
) -> list[np.ndarray]:
    """Forward-simulate ARHMM with AR latent dynamics."""
    log.info("  Method D: ARHMM Level 2 (discrete + AR latent)")
    pi = arhmm_params["pi"]
    betas = arhmm_params["betas"]
    Ab = arhmm_params["Ab"]  # [K, latent_dim, nlags*latent_dim+1]
    Q = arhmm_params["Q"]  # [K, latent_dim, latent_dim]

    pi0 = softmax(betas)
    K, latent_dim, ar_input_dim = Ab.shape
    nlags = (ar_input_dim - 1) // latent_dim

    seed = int(cfg.generation.seed)
    length = int(cfg.generation.sequence_length)
    n_seq = int(cfg.generation.num_sequences)

    rng = np.random.RandomState(seed + 5000)
    seqs = []

    for si in range(n_seq):
        seq = np.zeros(length, dtype=np.int32)
        x_history = [np.zeros(latent_dim) for _ in range(nlags)]

        # Initial state
        seq[0] = rng.choice(K, p=pi0)
        x_t = rng.multivariate_normal(np.zeros(latent_dim), Q[seq[0]])
        x_history.append(x_t)

        for t in range(1, length):
            # Build AR input: [x_{t-1}, ..., x_{t-nlags}, 1]
            x_lags = np.concatenate(
                [x_history[-(l + 1)] for l in range(nlags)] + [np.ones(1)]
            )

            # Compute log-posterior for each candidate state
            log_probs = np.log(pi[seq[t - 1]] + 1e-30)
            for k in range(K):
                mu_k = Ab[k] @ x_lags
                try:
                    Q_reg = Q[k] + 1e-6 * np.eye(latent_dim)
                    diff = x_t - mu_k
                    Q_inv = np.linalg.solve(Q_reg, diff)
                    sign, logdet = np.linalg.slogdet(Q_reg)
                    log_lik = -0.5 * (diff @ Q_inv + logdet)
                    log_probs[k] += log_lik
                except np.linalg.LinAlgError:
                    pass

            # Sample state from posterior
            probs = softmax(log_probs)
            seq[t] = rng.choice(K, p=probs)

            # Sample latent
            mu = Ab[seq[t]] @ x_lags
            x_t = rng.multivariate_normal(mu, Q[seq[t]] + 1e-6 * np.eye(latent_dim))
            x_history.append(x_t)

        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Control sequences
# ---------------------------------------------------------------------------


def generate_control_sequences(
    all_codes: np.ndarray,
    num_codes: int,
    cfg: DictConfig,
) -> dict[str, list[np.ndarray]]:
    """Generate control code sequences."""
    seed = int(cfg.generation.seed)
    length = int(cfg.generation.sequence_length)
    n_seq = int(cfg.generation.num_sequences)
    rng = np.random.RandomState(seed + 9000)
    controls = {}

    if cfg.controls.get("uniform_random", True):
        controls["uniform_random"] = [
            rng.randint(0, num_codes, size=length) for _ in range(n_seq)
        ]
    if cfg.controls.get("single_code", True):
        codes_to_use = rng.choice(num_codes, size=n_seq, replace=True)
        controls["single_code"] = [
            np.full(length, c, dtype=np.int32) for c in codes_to_use
        ]
    if cfg.controls.get("reversed", True):
        real_seqs = all_codes[rng.choice(len(all_codes), size=n_seq)]
        controls["reversed"] = [seq[::-1].copy() for seq in real_seqs]
        # Pad to length
        controls["reversed"] = [
            np.pad(s, (0, max(0, length - len(s))), mode="edge")[:length]
            for s in controls["reversed"]
        ]

    return controls


# ---------------------------------------------------------------------------
# Free-loop rollout with generated codes
# ---------------------------------------------------------------------------


def run_free_loop_rollouts(
    method_name: str,
    code_sequences: list[np.ndarray],
    env,
    inf_fn,
    params: tuple,
    ppo_networks,
    use_rnn: bool,
    max_steps: int,
    seed: int,
    num_codes: int,
    output_dir: Path,
    jit_reset=None,
    jit_step=None,
) -> dict:
    """Run free-loop rollouts with generated code sequences."""
    log.info(f"  Free-loop rollouts for {method_name} ({len(code_sequences)} sequences)")

    code_colors = get_code_colormap(num_codes)
    survivals = []
    all_qpos = []

    for si, seq in enumerate(code_sequences):
        key = jax.random.PRNGKey(seed + si)
        result = run_rollout(
            env, inf_fn, params, ppo_networks, use_rnn, key,
            max_steps=min(max_steps, len(seq)),
            code_override=seq,
            jit_reset=jit_reset, jit_step=jit_step,
        )
        survivals.append(result["survival"])
        all_qpos.append(result["qpos"][:-1])

        if si < 3:  # Render first 3 as solo videos
            try:
                from experiments.shared.ghost_rendering import render_solo_video
                sub = _method_subdir(output_dir, method_name)
                vid_path = sub / f"{method_name}_solo_{si}.mp4"
                render_solo_video(
                    env, result["qpos"][:-1], result["code_indices"], vid_path,
                    fps=50, num_codes=num_codes, title=f"{method_name} #{si}",
                )
            except Exception as e:
                log.warning(f"    Solo video {si} failed: {e}")

    mean_surv = np.mean(survivals)
    log.info(f"    Mean survival: {mean_surv:.1f} steps")

    return {"survivals": survivals, "qpos": all_qpos}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="code_generation_exp")
def main(cfg: DictConfig) -> None:
    log.info("=== Code Generation Experiment ===")

    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load code2act (MoSeq decoder) checkpoint
    ckpt_cfg, norm_state, policy_params, ppo_networks = load_moseq_checkpoint(cfg.checkpoint.path)
    use_rnn = bool(ckpt_cfg.network_config.get("use_rnn_decoder", False))
    num_codes = int(ckpt_cfg.network_config.num_codes)
    code2act_params = (norm_state, policy_params)

    # Load mimic-mjx (oracle VAE) checkpoint
    mimic_cfg, mimic_norm, mimic_policy, mimic_ppo = load_mimic_checkpoint(
        cfg.mimic_checkpoint.path, step=cfg.mimic_checkpoint.get("step"),
    )
    mimic_params = (mimic_norm, mimic_policy)

    # Load KPMS codes
    codes_data = np.load(cfg.data.codes_path)
    all_codes = codes_data["all_codes"]

    # Create env for free-loop
    splits_json = cfg.data.balanced_split_path
    ref_path = cfg.data.reference_data_path
    import json
    with open(splits_json) as f:
        splits = json.load(f)
    test_indices = splits["balanced"]["test_indices"]
    test_codes = codes_data["test_codes"]
    test_clips = ReferenceClips(
        data_path=ref_path,
        n_frames_per_clip=int(ckpt_cfg.env_config.clip_length),
        keep_clips_idx=np.array(test_indices),
    )
    _, _, env_cfg = utils.prepare_config(ckpt_cfg)
    env_cfg.start_frame_range = [0, 0]
    env_cfg.domain_randomization.use_domain_randomization = False
    code_stack_size = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = MoSeqImitation(config=env_cfg, clips=test_clips, kpms_codes=test_codes,
                         code_stack_size=code_stack_size)

    # Pre-compile JIT functions ONCE (critical for performance)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Code2Act inference fn for free-loop
    inf_fn = make_inference_fn(ppo_networks, use_rnn=use_rnn, deterministic=True)

    max_steps = int(cfg.free_loop.max_steps)
    seed = int(cfg.free_loop.seed)

    # -------------------------------------------------------------------
    # Generate sequences
    # -------------------------------------------------------------------
    all_generated: dict[str, list[np.ndarray]] = {}

    methods = list(cfg.generation.methods)

    if "transition_matrix" in methods:
        tm_results = generate_empirical_tm_sequences(all_codes, num_codes, cfg)
        all_generated.update(tm_results)

    # ---------------------------------------------------------------
    # Trainable generators (gpt2 / mamba / s5 / hmm_pretrained):
    # load from c2a/ar_codegen checkpoints and sample.
    # See c2a.ar_codegen.train for how to produce these checkpoints.
    # ---------------------------------------------------------------
    n_seq_ar = int(cfg.generation.num_sequences)
    length_ar = int(cfg.generation.sequence_length)
    seed_ar = int(cfg.generation.seed)
    ar_method_to_key = {
        "gpt2": "gpt2",
        "mamba": "mamba",
        "s5": "s5",
        "hmm_pretrained": "hmm",
    }
    ar_ckpt_cfg = cfg.get("ar_checkpoints", None)
    for ar_method, ckpt_key in ar_method_to_key.items():
        if ar_method not in methods:
            continue
        ckpt_path = None if ar_ckpt_cfg is None else ar_ckpt_cfg.get(ckpt_key, None)
        if ckpt_path is None:
            log.warning(
                "  Skipping %s: no ar_checkpoints.%s configured. Train one "
                "with `python -m c2a.ar_codegen.train model=%s`.",
                ar_method, ckpt_key, ckpt_key,
            )
            continue
        # Resolve relative checkpoint paths against the repo root, NOT
        # Hydra's run-dir CWD. Without this, `ar_checkpoints.gpt2=outputs/...`
        # would silently look under outputs/<this_run>/outputs/... and fail
        # in a confusing way.
        from c2a.paths import get_repo_root
        ckpt_p = Path(ckpt_path)
        if not ckpt_p.is_absolute():
            ckpt_p = get_repo_root() / ckpt_p
        from c2a.ar_codegen.models.base import CodeGenerator
        gen = CodeGenerator.from_checkpoint(ckpt_p)
        if gen.num_codes != num_codes:
            raise RuntimeError(
                f"{ar_method} checkpoint at {ckpt_path} has num_codes="
                f"{gen.num_codes} but the decoder was trained with "
                f"num_codes={num_codes}. Refusing to sample — the decoder "
                "embedding table is incompatible. Re-train the AR model "
                "on the same code source as the decoder."
            )
        seqs = gen.sample(
            n_seq=n_seq_ar, length=length_ar, seed=seed_ar,
        )
        all_generated[ar_method] = [seqs[i] for i in range(n_seq_ar)]
        log.info(
            "  %s: sampled %d sequences of length %d from %s",
            ar_method, n_seq_ar, length_ar, ckpt_path,
        )

    arhmm_params = None
    if "arhmm_level1" in methods or "arhmm_level2" in methods:
        # Hard gate: the AR-HMM was fit on KPMS syllable ids; if the
        # decoder's embedding table was trained on a different label
        # set (e.g. gt_directional_turn) the sampled ids index unrelated
        # embeddings and the experiment produces meaningless output.
        _assert_decoder_compatible_with_arhmm(ckpt_cfg, methods)
        arhmm_params = load_arhmm_params(cfg)

    if "arhmm_level1" in methods and arhmm_params is not None:
        l1_seqs = generate_arhmm_level1_sequences(arhmm_params, cfg)
        all_generated["arhmm_level1"] = l1_seqs

    if "arhmm_level2" in methods and arhmm_params is not None:
        l2_seqs = generate_arhmm_level2_sequences(arhmm_params, cfg)
        all_generated["arhmm_level2"] = l2_seqs

    # Controls
    ctrl_results = generate_control_sequences(all_codes, num_codes, cfg)
    all_generated.update(ctrl_results)

    # Save all generated sequences for post-hoc analysis
    for method_name, seqs in all_generated.items():
        sub = _method_subdir(output_dir, method_name)
        np.savez_compressed(
            sub / f"generated_{method_name}.npz",
            sequences=np.array(seqs),
        )
    log.info(f"  Saved generated sequences for {len(all_generated)} methods")

    # -------------------------------------------------------------------
    # Diagnostics: transition matrices per method
    # -------------------------------------------------------------------
    log.info("\n--- Transition matrix diagnostics ---")
    for method_name, seqs in all_generated.items():
        sub = _method_subdir(output_dir, method_name)
        T = compute_transition_matrix(seqs, num_codes)
        fig = plot_transition_matrix(T, title=f"TM: {method_name}")
        fig.savefig(sub / f"tm_{method_name}.png", dpi=300)
        np.save(sub / f"tm_{method_name}.npy", T)
        plt.close(fig)

    # -------------------------------------------------------------------
    # Free-loop rollouts per method
    # -------------------------------------------------------------------
    log.info("\n--- Free-loop rollouts ---")
    survival_summary = {}
    all_rollout_data = {}
    for method_name, seqs in all_generated.items():
        result = run_free_loop_rollouts(
            method_name, seqs[:int(cfg.generation.num_sequences)],
            env, inf_fn, code2act_params, ppo_networks, use_rnn,
            max_steps, seed, num_codes, output_dir,
            jit_reset=jit_reset, jit_step=jit_step,
        )
        survival_summary[method_name] = np.mean(result["survivals"])
        all_rollout_data[method_name] = result
        # Save per-method rollout data
        sub = _method_subdir(output_dir, method_name)
        np.savez_compressed(
            sub / f"rollouts_{method_name}.npz",
            survivals=np.array(result["survivals"]),
            qpos=np.array(result["qpos"], dtype=object),
        )

    # --- Mimic-MJX oracle baseline (survival upper bound) ---
    log.info("\n--- Mimic-MJX oracle baseline ---")
    mimic_inf_fn = make_mimic_inference_fn(mimic_ppo, deterministic=True)
    n_seq = int(cfg.generation.num_sequences)
    mimic_survivals = []
    for si in range(n_seq):
        key = jax.random.PRNGKey(seed + si)
        result = run_rollout(
            env, mimic_inf_fn, mimic_params, mimic_ppo, False, key,
            max_steps=max_steps,
            jit_reset=jit_reset, jit_step=jit_step,
            model_type="mimic_mjx",
        )
        mimic_survivals.append(result["survival"])
    survival_summary["mimic_mjx (oracle)"] = np.mean(mimic_survivals)
    log.info(f"  Mimic-MJX mean survival: {np.mean(mimic_survivals):.1f}")

    # Save survival summary for all methods
    import json as _json
    with open(output_dir / "survival_summary.json", "w") as f:
        _json.dump(survival_summary, f, indent=2)
    mimic_sub = _method_subdir(output_dir, "mimic_mjx")
    np.savez_compressed(
        mimic_sub / "rollouts_mimic_mjx.npz",
        survivals=np.array(mimic_survivals),
    )

    # -------------------------------------------------------------------
    # Survival comparison plot
    # -------------------------------------------------------------------
    set_nature_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    methods_sorted = sorted(survival_summary.keys(), key=lambda x: survival_summary[x], reverse=True)
    bar_colors = [
        NATURE_COLORS["orange"] if "mimic" in m.lower() else NATURE_COLORS["blue"]
        for m in methods_sorted
    ]
    bars = ax.barh(
        range(len(methods_sorted)),
        [survival_summary[m] for m in methods_sorted],
        color=bar_colors,
        edgecolor="none",
    )
    ax.set_yticks(range(len(methods_sorted)))
    ax.set_yticklabels(methods_sorted)
    ax.set_xlabel("Mean survival (steps)")
    ax.set_title("Free-loop survival by generation method")
    plt.tight_layout()
    fig.savefig(output_dir / "survival_comparison.png", dpi=300)
    plt.close(fig)

    log.info("=== Code Generation Experiment Complete ===")


if __name__ == "__main__":
    main()
