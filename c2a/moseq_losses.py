"""PPO loss computation for MoSeq decoder training.

Provides two loss functions:

* ``compute_moseq_ppo_loss`` — feedforward decoder (original).
* ``compute_moseq_recurrent_ppo_loss`` — RNN decoder with scan-based
  forward pass, hidden-state carry, and deterministic z_e replay.

Standard PPO losses (clipped surrogate, value, entropy). When ``e2e_mode``
is enabled the loss additionally consumes the policy network's ``e2e_aux``
dict (live VQ-VAE forward) to add a window-reconstruction loss plus the
van den Oord commit + codebook terms; the imitation pathway flows back to
the VQ-VAE encoder via the straight-through z_q inside the policy.

The :func:`compute_e2e_recon_only_loss` and :func:`compute_e2e_split_grads`
helpers below provide the eval-time snapshot of imitation vs reconstruction
gradient magnitudes into the VQ-VAE subtree, per the spec.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.types import Params

from c2a.e2e_vqvae import VQVAE_PARAM_NAMES


@flax.struct.dataclass
class PPONetworkParams:
    """Container for PPO network parameters.

    Attributes:
        policy: Decoder policy parameters.
        value: Value function parameters.
    """

    policy: Params
    value: Params


def _compute_e2e_aux_losses(
    e2e_aux: Any,
    observation: Any,
    *,
    e2e_mode: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pull recon, commit, and codebook loss tensors out of the e2e_aux dict.

    Returns ``(recon, commit, codebook)``. When ``e2e_mode=False`` or the
    aux dict is empty, returns zero scalars (so the caller can fold the
    products with ``e2e_w_*`` weights into the total loss unconditionally).

    For the recurrent loss, ``e2e_aux`` is a per-step dict whose tensor
    leaves carry a leading ``T`` axis; ``commit`` and ``codebook`` are
    already mean-reduced across the encoded batch inside the bottleneck so
    we just average over ``T``. ``recon_pred`` has shape
    ``(T, B, stack, w, feat_dim)`` and we compare it against
    ``observation["motion_window_stack"]`` of the same shape via MSE.

    For the feedforward loss, the same structure holds without the ``T``
    axis — ``commit`` / ``codebook`` are scalars, ``recon_pred`` is
    ``(B, stack, w, feat_dim)``. The ``jnp.mean`` calls are no-ops in that
    case so this helper covers both call sites.
    """
    if not e2e_mode or not e2e_aux:
        zero = jnp.array(0.0)
        return zero, zero, zero
    recon_pred = e2e_aux["recon_pred"]
    recon_target = observation["motion_window_stack"]
    recon_loss = jnp.mean((recon_pred - recon_target) ** 2)
    commit_loss = jnp.mean(e2e_aux["commit"])
    codebook_loss = jnp.mean(e2e_aux["codebook"])
    return recon_loss, commit_loss, codebook_loss


def _e2e_metrics(
    e2e_aux: Any,
    observation: Any,
    *,
    e2e_mode: bool,
    e2e_w_recon: float,
    e2e_w_commit: float,
    e2e_w_codebook: float,
) -> dict[str, jnp.ndarray]:
    """Compute the e2e/* metric dict (recon, commit, codebook, perplexity, ...).

    Always returns the same keys so WandB panels don't disappear when
    e2e_mode flips between runs. Zero-valued scalars are used when off.
    """
    recon, commit, codebook = _compute_e2e_aux_losses(
        e2e_aux, observation, e2e_mode=e2e_mode,
    )
    if e2e_mode and e2e_aux:
        # Aux dict has scalar `perplexity` and `utilization` per step
        # (for recurrent) or single scalars (for FF). Mean over leading
        # axes works for both.
        e2e_perplexity = jnp.mean(e2e_aux["perplexity"])
        e2e_utilization = jnp.mean(e2e_aux["utilization"])
        live_codes = e2e_aux["live_codes"]
        # live vs pretrained agreement (a quick drift signal).
        pretrained_codes = jnp.round(observation["kpms_code"]).astype(jnp.int32)
        # live_codes shape matches (..., stack); pretrained_codes shape
        # is also (..., stack). Both share leading axes — direct compare.
        agreement = jnp.mean((live_codes == pretrained_codes).astype(jnp.float32))
    else:
        e2e_perplexity = jnp.array(0.0)
        e2e_utilization = jnp.array(0.0)
        agreement = jnp.array(0.0)
    return {
        "e2e/recon_loss": recon,
        "e2e/scaled_recon_loss": e2e_w_recon * recon,
        "e2e/commit_loss": commit,
        "e2e/scaled_commit_loss": e2e_w_commit * commit,
        "e2e/codebook_loss": codebook,
        "e2e/scaled_codebook_loss": e2e_w_codebook * codebook,
        "e2e/live_perplexity": e2e_perplexity,
        "e2e/live_codebook_utilization": e2e_utilization,
        "e2e/live_vs_pretrained_agreement": agreement,
    }


def compute_gae(
    truncation: jnp.ndarray,
    termination: jnp.ndarray,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    bootstrap_value: jnp.ndarray,
    lambda_: float = 1.0,
    discount: float = 0.99,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation.

    Args:
        truncation: Truncation signal, shape ``[T, B]``.
        termination: Termination signal, shape ``[T, B]``.
        rewards: Rewards, shape ``[T, B]``.
        values: Value estimates V(s_t), shape ``[T, B]``.
        bootstrap_value: Value at time T, shape ``[B]``.
        lambda_: GAE lambda.
        discount: Discount factor gamma.

    Returns:
        ``(value_targets, advantages)`` each shape ``[T, B]``.
    """
    truncation_mask = 1 - truncation

    values_tp1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
    )
    deltas = rewards + discount * (1 - termination) * values_tp1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)

    def _scan_fn(carry, target_t):
        lam, acc = carry
        trunc_mask, delta, term = target_t
        acc = delta + discount * (1 - term) * trunc_mask * lam * acc
        return (lam, acc), acc

    (_, _), vs_minus_v = jax.lax.scan(
        _scan_fn,
        (lambda_, acc),
        (truncation_mask, deltas, termination),
        length=int(truncation_mask.shape[0]),
        reverse=True,
    )

    vs = jnp.add(vs_minus_v, values)
    vs_tp1 = jnp.concatenate([vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (
        rewards + discount * (1 - termination) * vs_tp1 - values
    ) * truncation_mask

    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def compute_moseq_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: Any,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    kl_weight: float = 0.0,
    *,
    e2e_mode: bool = False,
    e2e_w_recon: float = 0.0,
    e2e_w_commit: float = 0.0,
    e2e_w_codebook: float = 0.0,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute standard PPO loss for the MoSeq decoder policy.

    Args:
        params: Policy and value network parameters.
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch, shape ``[B, T, ...]``.
        rng: JAX PRNG key.
        step: Current training step (unused, kept for interface).
        ppo_network: ``MoSeqPPONetworks`` container.
        entropy_cost: Entropy bonus coefficient.
        discounting: Discount factor gamma.
        reward_scaling: Reward multiplier.
        gae_lambda: GAE lambda.
        clipping_epsilon: PPO clipping range.
        normalize_advantage: Whether to normalize advantages.
        vf_coefficient: Value loss coefficient.

    Returns:
        ``(total_loss, metrics_dict)``.
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    action_dist = ppo_network.parametric_action_distribution
    policy_network = ppo_network.policy_network
    value_apply = ppo_network.value_network.apply

    # [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Forward pass through decoder policy (now returns e2e_aux as 5th element).
    policy_logits, code_idx, cont_mean, cont_logvar, e2e_aux = policy_network.apply(
        normalizer_params, params.policy, data.observation, policy_key
    )

    # Value function
    baseline = value_apply(normalizer_params, params.value, data.observation)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    # Standard PPO losses
    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_log_probs = action_dist.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rho = jnp.exp(target_log_probs - behaviour_log_probs)

    surrogate1 = rho * advantages
    surrogate2 = jnp.clip(rho, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    entropy = jnp.mean(action_dist.entropy(policy_logits, entropy_key))
    entropy_loss = entropy_cost * -entropy

    # KL loss for continuous encoder
    if cont_mean is not None:
        kl_loss = 0.5 * jnp.mean(
            jnp.exp(cont_logvar) + cont_mean**2 - 1.0 - cont_logvar
        )
        scaled_kl_loss = kl_weight * kl_loss
    else:
        kl_loss = jnp.array(0.0)
        scaled_kl_loss = jnp.array(0.0)

    # e2e: live VQ-VAE recon + VQ regularizers
    e2e_recon_loss, e2e_commit_loss, e2e_codebook_loss = _compute_e2e_aux_losses(
        e2e_aux, data.observation, e2e_mode=e2e_mode,
    )
    scaled_e2e_recon = e2e_w_recon * e2e_recon_loss
    scaled_e2e_commit = e2e_w_commit * e2e_commit_loss
    scaled_e2e_codebook = e2e_w_codebook * e2e_codebook_loss

    total_loss = (
        policy_loss + v_loss + entropy_loss + scaled_kl_loss
        + scaled_e2e_recon + scaled_e2e_commit + scaled_e2e_codebook
    )

    # Code transition rate (diagnostic only)
    if code_idx.shape[0] > 1:
        valid_mask = data.discount[:-1] * (1 - truncation[:-1])
        num_valid = jnp.sum(valid_mask) + 1e-8
        changed = (code_idx[1:] != code_idx[:-1]).astype(jnp.float32)
        transition_rate = jnp.sum(changed * valid_mask) / num_valid
    else:
        transition_rate = jnp.array(0.0)

    # Code utilization
    num_codes = ppo_network.num_codes
    code_flat = code_idx.reshape(-1)
    counts = jnp.bincount(code_flat, length=num_codes).astype(jnp.float32)
    probs = counts / (counts.sum() + 1e-8)
    perplexity = jnp.exp(-jnp.sum(probs * jnp.log(probs + 1e-8)))
    codes_used = jnp.sum(counts > 0).astype(jnp.float32)
    utilization = codes_used / num_codes

    # z_e diagnostics
    if cont_mean is not None:
        z_e_norm = jnp.mean(jnp.sqrt(jnp.sum(cont_mean**2, axis=-1)))
        z_e_std = jnp.mean(jnp.std(cont_mean, axis=0))
    else:
        z_e_norm = jnp.array(0.0)
        z_e_std = jnp.array(0.0)

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "transition_rate": transition_rate,
        "perplexity": perplexity,
        "codebook_utilization": utilization,
        "codes_used": codes_used,
        "z_e_norm": z_e_norm,
        "z_e_std": z_e_std,
        # Placeholders for VQ metrics (enables shared logging code)
        "vq_loss": jnp.array(0.0),
        "commitment_loss": jnp.array(0.0),
        "codebook_loss": jnp.array(0.0),
        "scaled_vq_loss": jnp.array(0.0),
        "vq_weight": jnp.array(1.0),
        "ce_stickiness_loss": jnp.array(0.0),
        "scaled_ce_stickiness_loss": jnp.array(0.0),
        "prob_of_prev_code": jnp.array(0.0),
        "scaled_codebook_entropy_reg": jnp.array(0.0),
        "kl_loss": kl_loss,
        "scaled_kl_loss": scaled_kl_loss,
    }
    metrics.update(_e2e_metrics(
        e2e_aux, data.observation,
        e2e_mode=e2e_mode,
        e2e_w_recon=e2e_w_recon,
        e2e_w_commit=e2e_w_commit,
        e2e_w_codebook=e2e_w_codebook,
    ))

    return total_loss, metrics


def compute_distill_kl_loss(
    encoder_mean: jnp.ndarray,
    encoder_logvar: jnp.ndarray,
    distill_mean: jnp.ndarray,
    distill_logvar: jnp.ndarray,
) -> jnp.ndarray:
    """KL divergence between frozen encoder and distill head distributions.

    Computes ``KL(q_encoder_sg || p_distill)`` with stop-gradient on the
    encoder outputs so that this loss only trains the distill head (and
    the RNN via h_t backprop), not the encoder.

    Args:
        encoder_mean: Encoder mean ``[T, B, D]``.
        encoder_logvar: Encoder log-variance ``[T, B, D]``.
        distill_mean: Distill head mean ``[T, B, D]``.
        distill_logvar: Distill head log-variance ``[T, B, D]``.

    Returns:
        Scalar KL loss (sum over latent dims, mean over samples).
    """
    enc_mean_sg = jax.lax.stop_gradient(encoder_mean)
    enc_logvar_sg = jax.lax.stop_gradient(encoder_logvar)

    # KL(N(mu_e, sigma_e^2) || N(mu_d, sigma_d^2)) per latent dim:
    # = 0.5 * [ log(sigma_d^2 / sigma_e^2) + sigma_e^2 / sigma_d^2
    #           + (mu_e - mu_d)^2 / sigma_d^2 - 1 ]
    log_var_diff = distill_logvar - enc_logvar_sg
    var_ratio = jnp.exp(enc_logvar_sg - distill_logvar)
    mean_diff_sq = jnp.square(enc_mean_sg - distill_mean) / jnp.exp(distill_logvar)
    element_kl = 0.5 * (log_var_diff + var_ratio + mean_diff_sq - 1)

    # Sum over latent dims, mean over (T, B)
    kl_per_sample = jnp.sum(element_kl, axis=-1)
    return jnp.mean(kl_per_sample)


def compute_moseq_recurrent_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: Any,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    gae_lambda: float = 0.95,
    clipping_epsilon: float = 0.3,
    normalize_advantage: bool = True,
    vf_coefficient: float = 0.5,
    kl_weight: float = 0.0,
    z_e_scale: float = 1.0,
    distill_kl_weight: float = 0.0,
    *,
    e2e_mode: bool = False,
    e2e_w_recon: float = 0.0,
    e2e_w_commit: float = 0.0,
    e2e_w_codebook: float = 0.0,
) -> tuple[jnp.ndarray, types.Metrics]:
    """Compute PPO loss for the recurrent MoSeq decoder.

    Uses ``apply_sequence`` (jax.lax.scan) to recompute the policy forward
    pass over time, threading hidden state through timesteps and replaying
    stored PRNG keys for deterministic z_e samples.

    Args:
        params: Policy and value network parameters.
        normalizer_params: Running statistics for observation normalization.
        data: Transition batch, shape ``[B, T, ...]``.  Must contain
            ``data.extras["policy_extras"]["initial_carry_state"]`` (RNN
            hidden state at the start of the unroll) and optionally
            ``data.extras["policy_extras"]["policy_rng"]`` (stored PRNG
            keys for deterministic z_e replay).
        rng: JAX PRNG key.
        step: Current training step (unused, kept for interface).
        ppo_network: ``MoSeqRecurrentPPONetworks`` container.
        entropy_cost: Entropy bonus coefficient.
        discounting: Discount factor gamma.
        reward_scaling: Reward multiplier.
        gae_lambda: GAE lambda.
        clipping_epsilon: PPO clipping range.
        normalize_advantage: Whether to normalize advantages.
        vf_coefficient: Value loss coefficient.
        kl_weight: KL regularization weight for continuous encoder.
        z_e_scale: Multiplier on z_e (1.0 = full, 0.0 = decoder-only).
        distill_kl_weight: Weight for distillation KL loss
            ``KL(encoder_sg || distill_head)``.

    Returns:
        ``(total_loss, metrics_dict)``.
    """
    _, policy_key, entropy_key = jax.random.split(rng, 3)
    action_dist = ppo_network.parametric_action_distribution
    policy_network = ppo_network.policy_network
    value_apply = ppo_network.value_network.apply

    # --- Extract initial_carry_state BEFORE swapaxes (it has no T dim) ---
    initial_carry = data.extras["policy_extras"]["initial_carry_state"]

    # [B, T, ...] -> [T, B, ...]
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)

    # Restore initial_carry (swapaxes would have corrupted it)
    policy_extras = dict(data.extras["policy_extras"])
    policy_extras["initial_carry_state"] = initial_carry
    data = types.Transition(
        observation=data.observation,
        action=data.action,
        reward=data.reward,
        discount=data.discount,
        next_observation=data.next_observation,
        extras={
            "policy_extras": policy_extras,
            "state_extras": data.extras["state_extras"],
        },
    )

    # Extract stored PRNG keys for deterministic z_e replay. These MUST be
    # present when the recurrent MoSeq policy uses a continuous encoder —
    # silently falling back to fresh keys de-correlates z_e noise between
    # rollout and loss and produces a different training distribution than
    # advertised. Raise loudly.
    if "policy_rng" not in data.extras["policy_extras"]:
        raise KeyError(
            "policy_rng missing from data.extras['policy_extras']. "
            "Recurrent MoSeq loss requires per-step PRNG replay keys; check "
            "that make_moseq_rnn_rollout_policy_fn threads policy_rng into "
            "transition extras (c2a/moseq_ppo_networks.py)."
        )
    stored_keys = data.extras["policy_extras"]["policy_rng"]

    # Done flags for hidden state reset
    done = data.discount < 0.5

    # Forward pass via scan over time (apply_sequence returns 7 values now
    # — `e2e_aux_seq` is a per-step dict that's empty when e2e_mode is off).
    (
        policy_logits, cont_mean, cont_logvar,
        distill_means, distill_logvars, final_hidden,
        e2e_aux_seq,
    ) = policy_network.apply_sequence(
        normalizer_params,
        params.policy,
        data.observation,
        initial_carry,
        done,
        policy_key,
        deterministic=False,
        stored_keys=stored_keys,
        z_e_scale=z_e_scale,
    )

    # Value function (feedforward — no hidden state)
    baseline = value_apply(normalizer_params, params.value, data.observation)
    last_next_obs = jax.tree_util.tree_map(lambda x: x[-1], data.next_observation)
    bootstrap_value = value_apply(normalizer_params, params.value, last_next_obs)

    # Standard PPO losses
    rewards = data.reward * reward_scaling
    truncation = data.extras["state_extras"]["truncation"]
    termination = (1 - data.discount) * (1 - truncation)

    target_log_probs = action_dist.log_prob(
        policy_logits, data.extras["policy_extras"]["raw_action"]
    )
    behaviour_log_probs = data.extras["policy_extras"]["log_prob"]

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=gae_lambda,
        discount=discounting,
    )

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    rho = jnp.exp(target_log_probs - behaviour_log_probs)

    surrogate1 = rho * advantages
    surrogate2 = jnp.clip(rho, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * vf_coefficient

    entropy = jnp.mean(action_dist.entropy(policy_logits, entropy_key))
    entropy_loss = entropy_cost * -entropy

    # KL loss for continuous encoder
    if cont_mean is not None:
        kl_loss = 0.5 * jnp.mean(
            jnp.exp(cont_logvar) + cont_mean**2 - 1.0 - cont_logvar
        )
        scaled_kl_loss = kl_weight * kl_loss
    else:
        kl_loss = jnp.array(0.0)
        scaled_kl_loss = jnp.array(0.0)

    # Distillation KL loss (distill head predicts frozen encoder distribution)
    if distill_means is not None and distill_kl_weight > 0:
        distill_kl_loss = compute_distill_kl_loss(
            cont_mean, cont_logvar, distill_means, distill_logvars,
        )
        scaled_distill_kl_loss = distill_kl_weight * distill_kl_loss
    else:
        distill_kl_loss = jnp.array(0.0)
        scaled_distill_kl_loss = jnp.array(0.0)

    # e2e: live VQ-VAE recon + VQ regularizers, weighted small per spec.
    e2e_recon_loss, e2e_commit_loss, e2e_codebook_loss = _compute_e2e_aux_losses(
        e2e_aux_seq, data.observation, e2e_mode=e2e_mode,
    )
    scaled_e2e_recon = e2e_w_recon * e2e_recon_loss
    scaled_e2e_commit = e2e_w_commit * e2e_commit_loss
    scaled_e2e_codebook = e2e_w_codebook * e2e_codebook_loss

    total_loss = (
        policy_loss + v_loss + entropy_loss
        + scaled_kl_loss + scaled_distill_kl_loss
        + scaled_e2e_recon + scaled_e2e_commit + scaled_e2e_codebook
    )

    # --- z_e and hidden state diagnostics ---
    if cont_mean is not None:
        z_e_norm = jnp.mean(jnp.sqrt(jnp.sum(cont_mean**2, axis=-1)))
        z_e_std = jnp.mean(jnp.std(cont_mean, axis=0))
    else:
        z_e_norm = jnp.array(0.0)
        z_e_std = jnp.array(0.0)

    # Hidden state norm (mean across batch and layers)
    hidden_norms = [jnp.mean(jnp.sqrt(jnp.sum(h**2, axis=-1))) for h in final_hidden]
    hidden_state_norm = jnp.mean(jnp.stack(hidden_norms))

    # Code metrics (from observation, not from scan — codes don't change)
    kpms_code = data.observation.get("kpms_code")
    if kpms_code is not None:
        code_idx = jnp.round(kpms_code[..., 0]).astype(jnp.int32)
        num_codes = ppo_network.num_codes
        code_flat = code_idx.reshape(-1)
        counts = jnp.bincount(code_flat, length=num_codes).astype(jnp.float32)
        probs = counts / (counts.sum() + 1e-8)
        perplexity = jnp.exp(-jnp.sum(probs * jnp.log(probs + 1e-8)))
        codes_used = jnp.sum(counts > 0).astype(jnp.float32)
        utilization = codes_used / num_codes

        if code_idx.shape[0] > 1:
            valid_mask = data.discount[:-1] * (1 - truncation[:-1])
            num_valid = jnp.sum(valid_mask) + 1e-8
            changed = (code_idx[1:] != code_idx[:-1]).astype(jnp.float32)
            transition_rate = jnp.sum(changed * valid_mask) / num_valid
        else:
            transition_rate = jnp.array(0.0)
    else:
        perplexity = jnp.array(0.0)
        codes_used = jnp.array(0.0)
        utilization = jnp.array(0.0)
        transition_rate = jnp.array(0.0)
        num_codes = ppo_network.num_codes

    # Distill head diagnostics
    if distill_means is not None:
        distill_mean_norm = jnp.mean(jnp.sqrt(jnp.sum(distill_means**2, axis=-1)))
        distill_logvar_mean = jnp.mean(distill_logvars)
    else:
        distill_mean_norm = jnp.array(0.0)
        distill_logvar_mean = jnp.array(0.0)

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "transition_rate": transition_rate,
        "perplexity": perplexity,
        "codebook_utilization": utilization,
        "codes_used": codes_used,
        "kl_loss": kl_loss,
        "scaled_kl_loss": scaled_kl_loss,
        "z_e_norm": z_e_norm,
        "z_e_std": z_e_std,
        "z_e_scale": jnp.array(z_e_scale),
        "hidden_state_norm": hidden_state_norm,
        "distill_kl_loss": distill_kl_loss,
        "scaled_distill_kl_loss": scaled_distill_kl_loss,
        "distill_mean_norm": distill_mean_norm,
        "distill_logvar_mean": distill_logvar_mean,
        # Placeholders for VQ metrics (shared logging code)
        "vq_loss": jnp.array(0.0),
        "commitment_loss": jnp.array(0.0),
        "codebook_loss": jnp.array(0.0),
        "scaled_vq_loss": jnp.array(0.0),
        "vq_weight": jnp.array(1.0),
        "ce_stickiness_loss": jnp.array(0.0),
        "scaled_ce_stickiness_loss": jnp.array(0.0),
        "prob_of_prev_code": jnp.array(0.0),
        "scaled_codebook_entropy_reg": jnp.array(0.0),
    }
    metrics.update(_e2e_metrics(
        e2e_aux_seq, data.observation,
        e2e_mode=e2e_mode,
        e2e_w_recon=e2e_w_recon,
        e2e_w_commit=e2e_w_commit,
        e2e_w_codebook=e2e_w_codebook,
    ))

    return total_loss, metrics


# =============================================================================
# E2E split-gradient logging (eval-only snapshot)
# =============================================================================


def compute_e2e_recon_only_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: Any,
    *,
    z_e_scale: float = 1.0,
    e2e_w_recon: float = 1.0,
    e2e_w_commit: float = 0.0,
    e2e_w_codebook: float = 0.0,
) -> jnp.ndarray:
    """Reconstruction-only loss (no PPO terms) — used by the split-grad helper.

    Re-runs ``policy_network.apply_sequence`` on the same data, ignores
    every output except ``e2e_aux_seq``, and returns
    ``w_recon * recon + w_commit * commit + w_codebook * codebook``. The
    caller takes ``jax.grad`` of this w.r.t. ``params.policy`` and reads
    the gradient norm into the VQ-VAE subtree.

    Assumes ``data`` was collected with an e2e-mode policy (the rollout
    transitions therefore contain ``motion_window_stack`` in obs). If
    e2e_mode was off, ``e2e_aux_seq`` is an empty dict and this returns 0,
    so ``jax.grad`` is also identically zero.
    """
    initial_carry = data.extras["policy_extras"]["initial_carry_state"]
    data_t = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    policy_extras = dict(data_t.extras["policy_extras"])
    policy_extras["initial_carry_state"] = initial_carry
    data_t = types.Transition(
        observation=data_t.observation,
        action=data_t.action,
        reward=data_t.reward,
        discount=data_t.discount,
        next_observation=data_t.next_observation,
        extras={
            "policy_extras": policy_extras,
            "state_extras": data_t.extras["state_extras"],
        },
    )
    done = data_t.discount < 0.5
    stored_keys = data_t.extras["policy_extras"].get("policy_rng")
    (
        _logits, _mean, _logvar, _dm, _dl, _final_h, e2e_aux_seq,
    ) = ppo_network.policy_network.apply_sequence(
        normalizer_params,
        params.policy,
        data_t.observation,
        initial_carry,
        done,
        rng,
        deterministic=False,
        stored_keys=stored_keys,
        z_e_scale=z_e_scale,
    )
    recon, commit, codebook = _compute_e2e_aux_losses(
        e2e_aux_seq, data_t.observation, e2e_mode=True,
    )
    return e2e_w_recon * recon + e2e_w_commit * commit + e2e_w_codebook * codebook


def compute_e2e_split_grads(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    step: int,
    ppo_network: Any,
    *,
    imitation_loss_kwargs: dict[str, Any],
    e2e_w_recon: float,
    e2e_w_commit: float,
    e2e_w_codebook: float,
    z_e_scale: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Eval-only: ``||grad VQ-VAE||`` from imitation vs reconstruction pathways.

    The actual PPO update still uses the combined loss; this helper exists
    solely to log the relative magnitudes of the two gradient pathways into
    the VQ-VAE param subtree (per the spec, snapshot at every eval boundary).

    Imitation pathway: ``compute_moseq_recurrent_ppo_loss`` with all e2e
    weights set to zero — gradient into VQ-VAE flows only through the
    straight-through z_q used inside the policy.

    Recon pathway: :func:`compute_e2e_recon_only_loss` — gradient into
    VQ-VAE flows from MSE(recon_pred, motion_window_stack) plus the
    weighted commit / codebook terms.

    Args:
        params: Current full PPO param container.
        normalizer_params: Running stats.
        data: Mini-batch of transitions ``(B, T, ...)``. Must come from an
            e2e-mode rollout (obs contains ``motion_window_stack``).
        rng: PRNG key.
        step: Training step (forwarded into the imitation loss).
        ppo_network: ``MoSeqRecurrentPPONetworks`` container.
        imitation_loss_kwargs: Same kwargs passed to the per-minibatch
            ``compute_moseq_recurrent_ppo_loss`` during training (entropy_cost,
            discounting, etc.). The helper overrides the e2e_w_* keys to 0.
        e2e_w_recon / e2e_w_commit / e2e_w_codebook: Weights used by the
            ACTUAL training loss; used here to scale the recon pathway grad
            so the logged magnitudes reflect what the optimizer sees.
        z_e_scale: Forwarded to apply_sequence.

    Returns:
        Dict ``{"imit/<subtree>": norm, "recon/<subtree>": norm,
        "imit/total": norm, "recon/total": norm}`` of L2 norms over the
        VQ-VAE param subtree leaves. Subtree names come from
        :data:`c2a.e2e_vqvae.VQVAE_PARAM_NAMES`.
    """
    imit_kwargs = dict(imitation_loss_kwargs)
    imit_kwargs["e2e_w_recon"] = 0.0
    imit_kwargs["e2e_w_commit"] = 0.0
    imit_kwargs["e2e_w_codebook"] = 0.0
    imit_kwargs.setdefault("e2e_mode", True)
    imit_kwargs.setdefault("z_e_scale", z_e_scale)

    def imit_loss(policy_params):
        new_params = params.replace(policy=policy_params)
        loss, _ = compute_moseq_recurrent_ppo_loss(
            params=new_params,
            normalizer_params=normalizer_params,
            data=data,
            rng=rng,
            step=step,
            ppo_network=ppo_network,
            **imit_kwargs,
        )
        return loss

    def recon_loss(policy_params):
        new_params = params.replace(policy=policy_params)
        return compute_e2e_recon_only_loss(
            params=new_params,
            normalizer_params=normalizer_params,
            data=data,
            rng=rng,
            ppo_network=ppo_network,
            z_e_scale=z_e_scale,
            e2e_w_recon=e2e_w_recon,
            e2e_w_commit=e2e_w_commit,
            e2e_w_codebook=e2e_w_codebook,
        )

    g_imit = jax.grad(imit_loss)(params.policy)
    g_recon = jax.grad(recon_loss)(params.policy)
    out: dict[str, jnp.ndarray] = {}
    out.update(_grad_norms_by_subtree(g_imit, prefix="imit"))
    out.update(_grad_norms_by_subtree(g_recon, prefix="recon"))
    return out


def _grad_norms_by_subtree(
    g_policy: dict[str, Any], *, prefix: str,
) -> dict[str, jnp.ndarray]:
    """L2 norms of gradient leaves restricted to the VQ-VAE subtrees.

    Args:
        g_policy: Gradient pytree with the same structure as
            ``params.policy`` — top-level dict with ``"params"`` key and a
            nested dict of submodule names below it.
        prefix: WandB metric prefix (e.g. ``"imit"``, ``"recon"``). The
            returned keys are ``f"{prefix}/{subtree}"`` plus
            ``f"{prefix}/total"``.

    Subtrees absent from the gradient (e.g. e2e_mode was off) contribute
    zero — the metric is always emitted so panels stay stable.
    """
    base = g_policy.get("params", g_policy)
    norms: dict[str, jnp.ndarray] = {}
    total_sq = jnp.array(0.0)
    for name in VQVAE_PARAM_NAMES:
        sub = base.get(name)
        if sub is None:
            n2 = jnp.array(0.0)
        else:
            leaves = jax.tree_util.tree_leaves(sub)
            if not leaves:
                n2 = jnp.array(0.0)
            else:
                n2 = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
        norms[f"{prefix}/{name}"] = jnp.sqrt(n2)
        total_sq = total_sq + n2
    norms[f"{prefix}/total"] = jnp.sqrt(total_sq)
    return norms
