"""Signature-parity test for the track_mjx ff_ppo monkey-patch targets.

Background: ``c2a.moseq_ppo`` replaces
``track_mjx.agent.ff_ppo.losses.compute_ppo_loss`` at runtime with a MoSeq
loss that must expose the same kwargs the ``ppo.train`` loop passes. If an
upstream track_mjx change adds a new kwarg (e.g. a new scheduler), the
wrapper silently crashes with ``unexpected keyword argument`` — or, worse,
the monkey-patch stops working (if ppo.py starts binding the function at
import time rather than re-reading the module attribute every call).

This test fires when that drift happens so we notice before a training run.
"""

import inspect

import pytest

pytest.importorskip("track_mjx.agent.ff_ppo")


def _kwargs(fn):
    sig = inspect.signature(fn)
    return {
        name
        for name, p in sig.parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def test_compute_ppo_loss_signature_is_stable():
    """Moseq loss wrappers replicate the kwargs of the upstream loss. If
    upstream adds a kwarg, our wrapper needs updating."""
    from track_mjx.agent.ff_ppo import losses as upstream_losses
    from c2a import moseq_losses

    upstream = _kwargs(upstream_losses.compute_ppo_loss)
    moseq = _kwargs(moseq_losses.compute_moseq_ppo_loss)
    moseq_rnn = _kwargs(moseq_losses.compute_moseq_recurrent_ppo_loss)

    # Upstream kwargs the train loop passes — these MUST be accepted by both
    # moseq variants. Using set-intersection here is too strict (moseq adds
    # its own kwargs); we assert set-containment.
    required = upstream - {"self", "cls"}
    assert required.issubset(moseq | {"use_rnn_decoder"}), (
        f"compute_moseq_ppo_loss missing kwargs: {required - moseq}"
    )
    assert required.issubset(moseq_rnn | {"use_rnn_decoder"}), (
        f"compute_moseq_recurrent_ppo_loss missing kwargs: {required - moseq_rnn}"
    )


def test_make_inference_fn_signature_is_stable():
    """Moseq inference-fn factories mirror the upstream factory — same call
    site, same return shape."""
    from track_mjx.agent.ff_ppo import ppo_networks as upstream_pn
    from c2a import moseq_ppo_networks as moseq_pn

    upstream = _kwargs(upstream_pn.make_inference_fn)
    moseq = _kwargs(moseq_pn.make_moseq_inference_fn)

    assert upstream.issubset(moseq), (
        f"make_moseq_inference_fn missing kwargs: {upstream - moseq}"
    )


def test_ppo_train_has_required_kwargs():
    """``c2a.moseq_ppo.train`` passes several kwargs to ``ppo.train``; if the
    upstream signature loses any of them, the monkey-patch breaks."""
    from track_mjx.agent.ff_ppo import ppo

    required = {
        "generate_unroll_fn",
        "init_carry_state_fn",
        "make_rollout_policy_fn",
        "post_init_params_fn",
        "get_activation",
        "use_kl_schedule",
        "latent_kl_weight",
        "latent_ar1_weight",
        "wrap_for_training",
        "checkpoint_callback",
        "freeze_decoder",
        "eval_env_test_set",
    }
    actual = _kwargs(ppo.train)
    missing = required - actual
    assert not missing, f"ppo.train missing expected kwargs: {missing}"
