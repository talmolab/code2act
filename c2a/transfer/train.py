"""Option-B task-transfer training (categorical code policy + frozen decoder/prior).

Trains a small high-level **categorical** policy with brax PPO. The frozen GRU
decoder and the frozen Mamba (or TM) code prior live inside
``CodePriorDecoderWrapper`` (baked in by closure) — PPO only trains the policy +
value nets.

Run:
    python3.12 bash_scripts/_pyrun.py c2a.transfer.train --smoke
    python3.12 bash_scripts/_pyrun.py c2a.transfer.train \
        --num_timesteps 50000000 --num_envs 2048 --run_name walkrear_mamba
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import pickle
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import c2a.transfer  # noqa: F401  (record_scalar shim before orbax/brax)

import jax
from brax.training.agents.ppo import train as brax_ppo
from mujoco_playground._src import wrapper as mjxw

from c2a.paths import get_repo_root
from c2a.transfer.decoder_adapter import load_frozen_decoder
from c2a.transfer.networks import make_categorical_ppo_networks
from c2a.transfer.priors import (
    mamba_prior_from_checkpoint,
    transition_matrix_prior_from_payload,
)
from c2a.transfer.wrappers import CodePriorDecoderWrapper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_DECODER = "model_checkpoints/260603_180852_140475"
DEFAULT_MAMBA = "outputs/ar_codegen/mamba/260611_081253_devuser"
DEFAULT_SPLIT = "data/rodent/rodent_balanced_splits.json"


def build_prior(kind: str, frozen, repo: Path, mamba_path: str | None = None):
    if kind == "mamba":
        return mamba_prior_from_checkpoint(str(repo / (mamba_path or DEFAULT_MAMBA)))
    if kind == "tm":
        return transition_matrix_prior_from_payload(frozen.cfg, str(repo / DEFAULT_SPLIT))
    raise ValueError(f"unknown prior kind {kind!r}")


def make_env(decoder_ckpt: str, prior_kind: str, dwell: int, repo: Path,
             head_dense_weight: float = 5.0, task: str = "rearing",
             scent_weight: float = 1.0, forage_overrides: dict | None = None,
             occupancy_weight: float = 1.0, harvest_weight: float = 10.0,
             mamba_path: str | None = None):
    from vnl_playground.tasks.rodent import consts

    frozen = load_frozen_decoder(str(repo / decoder_ckpt))
    prior = build_prior(prior_kind, frozen, repo, mamba_path=mamba_path)
    # All tasks must use the decoder's EXACT training body (rodent.xml): the
    # default box_feet body's foot-collision dynamics destabilize the frozen
    # torques. Same nu/nbody/proprio, ctrl_dt/sim_dt=0.01/0.002 (match decoder).
    if task == "rearing":
        from vnl_playground.tasks.rodent import rearing

        cfg = rearing.default_config()
        cfg.walker_xml_path = consts.RODENT_XML_PATH
        # Dense head-height shaping (default weight 0 = sparse-only = unlearnable).
        cfg.reward_terms.head_height_dense.weight = float(head_dense_weight)
        base = rearing.Rearing(config=cfg)
    elif task == "forage":
        from c2a.transfer.envs import arena_forage

        cfg = arena_forage.default_config()
        cfg.walker_xml_path = consts.RODENT_XML_PATH
        # Enable potential-based scent shaping (registered but off by default):
        # F = γ·Φ(s')−Φ(s), Φ=total_scent. Without it, treat_collection is fully
        # sparse → unlearnable. Policy-invariant shaping. Scent is 3-D, so rearing
        # toward a HIGH treat raises the skull → scent ↑ → rear-to-reach is rewarded.
        cfg.reward_terms["scent_proximity"] = {"weight": float(scent_weight)}
        for k, v in (forage_overrides or {}).items():
            if k == "treat_weight":
                # Boost the TRUE objective so scent shaping can't be hacked
                # (e.g. rear-in-place to raise skull→scent without collecting).
                cfg.reward_terms["treat_collection"]["weight"] = float(v)
            else:
                setattr(cfg, k, v)
        base = arena_forage.ArenaForage(config=cfg)
    elif task == "forage_dwell":
        from c2a.transfer.envs import arena_forage

        cfg = arena_forage.default_config()
        cfg.walker_xml_path = consts.RODENT_XML_PATH
        cfg.scent_at_torso = True  # anti rear-hack: rearing can't inflate scent
        # Reward = dense sustained-occupancy + harvest bonus + torso-scent for
        # navigation. treat_collection (instant) is dropped — harvest is the event.
        cfg.reward_terms = {
            "dwell_occupancy": {"weight": float(occupancy_weight)},
            "dwell_harvest": {"weight": float(harvest_weight)},
            "scent_proximity": {"weight": float(scent_weight)},
        }
        for k, v in (forage_overrides or {}).items():
            if k == "treat_weight":
                continue  # not used in dwell mode
            setattr(cfg, k, v)
        if cfg.dwell_steps <= 0:
            raise ValueError("forage_dwell needs dwell_steps>0")
        base = arena_forage.ArenaForage(config=cfg)
    else:
        raise ValueError(f"unknown task {task!r}")
    env = CodePriorDecoderWrapper(base, frozen, prior, dwell=dwell)
    return env, frozen, prior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_ckpt", default=DEFAULT_DECODER)
    p.add_argument("--mamba_path", default=None,
                   help="Mamba prior dir relative to repo root (default: DEFAULT_MAMBA constant)")
    p.add_argument("--prior", default="mamba", choices=["mamba", "tm"])
    p.add_argument("--dwell", type=int, default=2)
    p.add_argument("--num_timesteps", type=int, default=50_000_000)
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--episode_length", type=int, default=250)  # high-level steps
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_minibatches", type=int, default=16)
    p.add_argument("--num_updates_per_batch", type=int, default=4)
    p.add_argument("--unroll_length", type=int, default=20)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--entropy_cost", type=float, default=1e-2)
    p.add_argument("--discounting", type=float, default=0.97)
    p.add_argument("--num_evals", type=int, default=10)
    p.add_argument("--num_eval_envs", type=int, default=64)
    p.add_argument("--head_dense_weight", type=float, default=5.0)
    p.add_argument("--task", default="rearing", choices=["rearing", "forage", "forage_dwell"])
    p.add_argument("--dwell_steps", type=int, default=0, help="forage_dwell: control steps to sustain before harvest")
    p.add_argument("--immobile_thresh", type=float, default=0.06, help="torso speed below which = immobile (low patches)")
    p.add_argument("--occupancy_weight", type=float, default=1.0, help="dwell: per-step sustain reward")
    p.add_argument("--harvest_weight", type=float, default=10.0, help="dwell: harvest-completion bonus")
    p.add_argument("--scent_weight", type=float, default=1.0)
    # forage TRAIN arena
    p.add_argument("--treat_radius", type=float, default=0.08)
    p.add_argument("--num_treats", type=int, default=6)
    p.add_argument("--spawn_radius", type=float, default=0.6)
    p.add_argument("--min_spawn_radius", type=float, default=0.25)
    p.add_argument("--treat_height", type=float, default=0.04)
    p.add_argument("--high_treat_height", type=float, default=0.16)
    p.add_argument("--frac_high_treats", type=float, default=0.5)
    p.add_argument("--treat_weight", type=float, default=1.0, help="treat_collection reward weight (raise to beat scent-hacking)")
    p.add_argument("--high_treat_mult", type=float, default=1.0, help=">1 pays more for high (rear-required) treats")
    p.add_argument("--scent_sigma", type=float, default=0.5, help="scent Gaussian width; small=local (anti rear-hack)")
    # forage held-out TEST arena (rendered in the wandb video). Harder/different
    # from train → generalization check, not a replay of the train env.
    # 0 / -1 = fall back to the train value.
    p.add_argument("--test_num_treats", type=int, default=0)
    p.add_argument("--test_spawn_radius", type=float, default=0.0)
    p.add_argument("--test_frac_high", type=float, default=-1.0)
    p.add_argument("--log_video", action="store_true", help="log composite wandb.Video every --video_every evals")
    p.add_argument("--video_every", type=int, default=3)
    p.add_argument("--video_top_n", type=int, default=6)
    p.add_argument("--video_steps", type=int, default=1000, help="frames in wandb video (length = video_steps/50 s; 1000 = 20 s). rollout resets on done so it can exceed one episode")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", default="walkrear_transfer")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--wandb_project", default="code2act-transfer")
    args = p.parse_args()

    if args.smoke:
        args.num_timesteps = 200_000
        args.num_envs = 64
        args.batch_size = 64
        args.num_minibatches = 4
        args.num_evals = 2
        args.episode_length = 60
        args.num_eval_envs = 8

    repo = Path(get_repo_root())
    out_dir = repo / "outputs" / "transfer" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_forage = dict(
        treat_sphere_radius=args.treat_radius,
        num_treats=args.num_treats,
        spawn_radius=args.spawn_radius,
        min_spawn_radius=args.min_spawn_radius,
        treat_height=args.treat_height,
        high_treat_height=args.high_treat_height,
        frac_high_treats=args.frac_high_treats,
        treat_weight=args.treat_weight,
        scent_sigma=args.scent_sigma,
        high_treat_reward_mult=args.high_treat_mult,
        dwell_steps=args.dwell_steps,
        immobile_speed_thresh=args.immobile_thresh,
    )
    mk = lambda ov: make_env(args.decoder_ckpt, args.prior, args.dwell, repo,
                             args.head_dense_weight, args.task, args.scent_weight, ov,
                             args.occupancy_weight, args.harvest_weight,
                             mamba_path=args.mamba_path)
    env, frozen, prior = mk(train_forage)
    eval_env, _, _ = mk(train_forage)

    # Held-out TEST arena for the wandb video (generalization, NOT the train env).
    test_forage = dict(train_forage)
    if args.test_num_treats > 0:
        test_forage["num_treats"] = args.test_num_treats
    if args.test_spawn_radius > 0:
        test_forage["spawn_radius"] = args.test_spawn_radius
    if args.test_frac_high >= 0:
        test_forage["frac_high_treats"] = args.test_frac_high
    if args.task in ("forage", "forage_dwell") and test_forage != train_forage:
        test_env, _, _ = mk(test_forage)
        print(f"[video] held-out TEST arena: {test_forage}", flush=True)
    else:
        test_env = eval_env
    print(
        f"transfer env ready: num_codes={frozen.num_codes} proprio={frozen.proprio_size} "
        f"action_size={env.action_size} prior={args.prior} dwell={args.dwell}",
        flush=True,
    )

    # ---- wandb ----
    use_wandb = os.environ.get("WANDB_MODE", "") != "disabled"
    wb = None
    if use_wandb:
        try:
            import wandb

            wb = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
                dir=str(out_dir),
            )
            # Plot everything against env_step (a real x-axis) while letting the
            # wandb global step auto-increment — otherwise the media/Step slider
            # shows raw env-steps (e.g. 8,355,840), which is unreadable.
            wb.define_metric("env_step")
            wb.define_metric("*", step_metric="env_step")
        except Exception as e:  # noqa: BLE001
            log.warning("wandb init failed (%s); continuing without wandb.", e)
            wb = None

    def progress(step, metrics):
        scalars = {k: float(v) for k, v in metrics.items() if hasattr(v, "__float__")}
        g = scalars.get
        # print (not logging) so progress is visible regardless of root-logger level.
        # torso_z + head_height catch the "rodent flies" failure mode; rearing_success
        # is the actual task signal; entropy shows the policy isn't collapsing.
        # task-agnostic line: rearing (rear_success/head_h) + forage (treats/scent).
        print(
            f"[transfer] step={int(step):>10d} "
            f"reward={g('eval/episode_reward', float('nan')):.3f} "
            f"treats={g('eval/episode_mean_collected_treats', float('nan')):.3f} "
            f"harvest={g('eval/episode_mean_harvested', float('nan')):.3f} "
            f"high_h={g('eval/episode_high_harvested', float('nan')):.3f} "
            f"sustain_hi={g('eval/episode_sustained_high_steps', float('nan')):.2f} "
            f"scent={g('eval/episode_rewards/scent_proximity', float('nan')):.3f} "
            f"rear_success={g('eval/episode_rewards/rearing_success', float('nan')):.3f} "
            f"head_h={g('eval/episode_relative_head_height', float('nan')):.3f} "
            f"torso_z={g('eval/episode_torso_z', float('nan')):.3f} "
            f"entropy={g('training/entropy_loss', float('nan')):.3f} "
            f"sps={g('training/sps', float('nan')):.0f}",
            flush=True,
        )
        if wb is not None:
            scalars["env_step"] = int(step)
            wb.log(scalars)  # auto-increment global step; env_step is the x-axis

    network_factory = functools.partial(
        make_categorical_ppo_networks,
        policy_hidden_layer_sizes=(256, 256, 128),
        value_hidden_layer_sizes=(256, 256, 128),
    )

    # ---- optional live composite-video logging to wandb ----
    video_fn = (lambda *a: None)
    if args.log_video:
        try:
            from c2a.transfer.visualize import make_video_logger, popular_codes

            code_list = popular_codes(frozen, repo, args.video_top_n)
            video_fn = make_video_logger(
                args.decoder_ckpt, test_env, frozen, repo,
                tiles_npz=str(out_dir / "code_tiles.npz"), code_list=code_list,
                wb=wb, out_dir=str(out_dir), every=args.video_every,
                max_steps=args.video_steps,  # resets on done, so not capped by episode_length
            )
            print(f"[video] live composite logging ON (every {args.video_every} evals)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[video] setup failed ({e}); training without video.", flush=True)
            video_fn = (lambda *a: None)

    train_fn = functools.partial(
        brax_ppo.train,
        num_timesteps=args.num_timesteps,
        num_envs=args.num_envs,
        episode_length=args.episode_length,
        batch_size=args.batch_size,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        unroll_length=args.unroll_length,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        discounting=args.discounting,
        gae_lambda=0.95,
        clipping_epsilon=0.2,
        normalize_observations=True,
        reward_scaling=1.0,
        num_evals=args.num_evals,
        num_eval_envs=args.num_eval_envs,
        seed=args.seed,
        network_factory=network_factory,
        # full_reset=True so the auto-reset wrapper re-inits our t_* recurrent
        # carries (GRU hidden + Mamba state) on EVERY done, including
        # episode-length truncation (inner-env done alone misses truncation).
        wrap_env_fn=functools.partial(mjxw.wrap_for_brax_training, full_reset=True),
        progress_fn=progress,
        policy_params_fn=video_fn,
        save_checkpoint_path=str(out_dir / "brax_ckpt"),
    )

    print(f"starting PPO: timesteps={args.num_timesteps} envs={args.num_envs}", flush=True)
    make_policy, params, metrics = train_fn(environment=env, eval_env=eval_env)

    ckpt = out_dir / "ppo_params.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(jax.device_get(params), f)
    final = {k: float(v) for k, v in metrics.items() if hasattr(v, "__float__")}
    print(f"saved policy params -> {ckpt}", flush=True)
    print(f"final metrics: {final}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
