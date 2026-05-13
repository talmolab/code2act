"""Motion predictor: ``z_t -> next horizon motion frames``.

Ported from ``origin/scott_claude/latent-mimic-handoff:track_mjx/agent/
latent_ppo/networks/predictor.py``. The predictor is an auxiliary
dynamics-prediction loss on the same latent ``z`` the decoder
reconstructs from — it's NOT a recurrent rollout.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from c2a.codegen.kinematics_vae.backbone.mlp import Mlp


class MotionPredictor(nn.Module):
    """``f(z_t) = Ŝ^motion_{t+1..t+horizon}`` (a window of length ``horizon``)."""

    layer_sizes: Sequence[int] = (256, 128)
    horizon: int = 5
    feat_dim: int = 77

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch = z.shape[0]
        h = Mlp(layer_sizes=self.layer_sizes, activate_final=True, name="trunk")(z)
        flat_out = nn.Dense(self.horizon * self.feat_dim, name="pred_head")(h)
        return flat_out.reshape(batch, self.horizon, self.feat_dim)


class MotionPredictorSplit(nn.Module):
    """Body-part-split predictor: each branch's z slice predicts its
    own feats. Outputs are scattered back into a ``(B, horizon,
    full_feat_dim)`` tensor by ``branch_output_indices``."""

    branch_output_indices: Tuple[Tuple[int, ...], ...]
    branch_latent_dims: Tuple[int, ...]
    full_feat_dim: int
    horizon: int = 5
    layer_sizes: Sequence[int] = (256, 128)

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        batch = z.shape[0]
        out = jnp.zeros((batch, self.horizon, self.full_feat_dim), dtype=z.dtype)
        offset = 0
        for i, (idx, ld) in enumerate(
            zip(self.branch_output_indices, self.branch_latent_dims)
        ):
            # Plain Python slice — offset/length are concrete Python ints
            # (Flax iteration over module config). Avoids static-shape issues.
            z_branch = z[:, offset:offset + int(ld)]
            offset += int(ld)
            pred_branch = MotionPredictor(
                layer_sizes=tuple(self.layer_sizes),
                horizon=self.horizon,
                feat_dim=len(idx),
                name=f"branch_{i}",
            )(z_branch)
            idx_arr = jnp.asarray(idx, dtype=jnp.int32)
            out = out.at[:, :, idx_arr].set(pred_branch)
        return out
