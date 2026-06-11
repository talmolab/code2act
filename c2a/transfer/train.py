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
DEFAULT_MAMBA = "outputs/ar_codegen/mamba/260610_071355_devuser"
DEFAULT_SPLIT = "data/rodent/rodent_balanced_splits.json"


def build_prior(kind: str, frozen, repo: Path):
    if kind == "mamba":
        return mamba_prior_from_checkpoint(str(repo / DEFAULT_MAMBA))
    if kind == "tm":
        return transition_matrix_prior_from_payload(frozen.cfg, str(repo / DEFAULT_SPLIT))
    raise ValueError(f"unknown prior kind {kind!r}")


def make_env(decoder_ckpt: str, prior_kind: str, dwell: int, repo: Path,
             head_dense_weight: float = 5.0):
    from vnl_playground.tasks.rodent import consts, rearing

    frozen = load_frozen_decoder(str(repo / decoder_ckpt))
    prior = build_prior(prior_kind, frozen, repo)
    cfg = rearing.default_config()
    # Match the decoder's EXACT training body (rodent.xml). Rearing defaults to
    # rodent_box_feet.xml whose different foot-collision dynamics destabilize the
    # frozen decoder's torques (rodent launches; torso_z ~4 m). Same nu/nbody/proprio
    # (verified), so the task is unchanged. ctrl_dt/sim_dt already match (0.01/0.002).
    cfg.walker_xml_path = consts.RODENT_XML_PATH
    # Enable DENSE head-height shaping. Default rearing has head_height_dense
    # weight=0, so the only rearing signal is the sparse 500-pt hold-1s bonus —
    # unlearnable from scratch. A dense gradient toward raising the skull makes
    # the transfer demo learnable (policy must select the rear code to climb it).
    cfg.reward_terms.head_height_dense.weight = float(head_dense_weight)
    base = rearing.Rearing(config=cfg)
    env = CodePriorDecoderWrapper(base, frozen, prior, dwell=dwell)
    return env, frozen, prior


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decoder_ckpt", default=DEFAULT_DECODER)
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

    env, frozen, prior = make_env(args.decoder_ckpt, args.prior, args.dwell, repo, args.head_dense_weight)
    eval_env, _, _ = make_env(args.decoder_ckpt, args.prior, args.dwell, repo, args.head_dense_weight)
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
        except Exception as e:  # noqa: BLE001
            log.warning("wandb init failed (%s); continuing without wandb.", e)
            wb = None

    def progress(step, metrics):
        scalars = {k: float(v) for k, v in metrics.items() if hasattr(v, "__float__")}
        g = scalars.get
        # print (not logging) so progress is visible regardless of root-logger level.
        # torso_z + head_height catch the "rodent flies" failure mode; rearing_success
        # is the actual task signal; entropy shows the policy isn't collapsing.
        print(
            f"[transfer] step={int(step):>10d} "
            f"reward={g('eval/episode_reward', float('nan')):.3f} "
            f"rear_success={g('eval/episode_rewards/rearing_success', float('nan')):.3f} "
            f"head_h={g('eval/episode_relative_head_height', float('nan')):.3f} "
            f"torso_z={g('eval/episode_torso_z', float('nan')):.3f} "
            f" entropy={g('training/entropy_loss', float('nan')):.3f} "
            f"sps={g('training/sps', float('nan')):.0f}",
            flush=True,
        )
        if wb is not None:
            wb.log(scalars, step=int(step))

    network_factory = functools.partial(
        make_categorical_ppo_networks,
        policy_hidden_layer_sizes=(256, 256, 128),
        value_hidden_layer_sizes=(256, 256, 128),
    )

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
