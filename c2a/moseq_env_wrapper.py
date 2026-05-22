"""Imitation environment subclass that includes pre-computed KPMS codes.

Subclasses ``Imitation`` to inject ``kpms_code`` directly in ``_get_obs``,
and flattens the v0.0.13 nested obs structure to a flat dict. This avoids
the pitfalls of an env *wrapper* approach where:

* ``BraxDomainRandomizationVmapWrapper`` bypasses wrappers via ``env.unwrapped``
* ``EpisodeWrapper.step``'s ``jax.lax.scan`` requires carry-in / carry-out to
  have identical pytree structure, but the base env's ``_get_obs`` would strip
  the extra key.

By producing the flat ``{"task_obs": ..., "proprioception": ..., "kpms_code": ...}``
inside ``_get_obs``, every wrapper (Vmap, DR, Episode, AutoReset) sees a consistent
obs structure at all times.

When ``e2e_motion_norm_padded`` is supplied the env additionally injects
``obs["motion_window_stack"]`` of shape ``(code_stack_size, window_len, feat_dim)``
so a live VQ-VAE folded into the policy can re-encode the reference motion at
every step. The buffer's per-clip ordering must match ``kpms_codes`` row order.
"""

import collections
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jp
import jax.numpy as jnp
from mujoco import mjx
from vnl_playground.tasks.rodent.imitation import Imitation
from vnl_playground.tasks.rodent.imitation import ReferenceClips

from c2a.e2e_vqvae import make_window_stack_extractor


class MoSeqImitation(Imitation):
    """Imitation env whose observations include pre-computed KPMS syllable codes.

    Overrides ``_get_obs`` to:
    1. Inject ``kpms_code`` into the obs dict.
    2. Flatten nested obs values to 1D arrays.
    3. Strip the ``state`` hierarchy (v0.0.13 nesting).
    4. (Optional, e2e mode) Inject ``motion_window_stack`` for the live VQ-VAE.

    This produces a flat dict ``{"task_obs": 1D, "proprioception": 1D, "kpms_code": 1D}``
    (plus ``"motion_window_stack": 3D`` in e2e mode) at all times, avoiding
    wrapper-chain pitfalls.

    Attributes:
        _kpms_codes: Pre-computed code array ``[n_clips, n_frames]``.
        _code_stack_size: Number of consecutive codes to stack (1 = current only,
            5 = current + 4 future codes).
        _e2e_motion_norm_padded: Optional pre-padded normalized motion features
            of shape ``[n_clips, n_frames + window_len - 1 + code_stack_size - 1,
            feat_dim]``. When provided, e2e mode is active and the env emits a
            ``motion_window_stack`` obs entry.
        _e2e_window_len: VQ-VAE encoder window length ``w``. Must match the
            value used when ``_e2e_motion_norm_padded`` was built.
        _e2e_feat_dim: Trailing dim of the padded buffer; cached for the
            JIT-friendly extractor.
    """

    def __init__(
        self,
        config: Any,
        clips: ReferenceClips,
        kpms_codes: jnp.ndarray,
        code_stack_size: int = 1,
        *,
        e2e_motion_norm_padded: Optional[jnp.ndarray] = None,
        e2e_window_len: int = 10,
    ):
        super().__init__(config=config, clips=clips)
        self._kpms_codes = jnp.asarray(kpms_codes, dtype=jnp.int32)
        self._code_stack_size = int(code_stack_size)

        if e2e_motion_norm_padded is None:
            self._e2e_motion_norm_padded = None
            self._e2e_window_len = 0
            self._e2e_feat_dim = 0
            self._e2e_extract = None
        else:
            padded = jnp.asarray(e2e_motion_norm_padded, dtype=jnp.float32)
            if padded.ndim != 3:
                raise ValueError(
                    f"e2e_motion_norm_padded must be 3D (n_clips, n_padded, "
                    f"feat_dim); got shape {tuple(padded.shape)}."
                )
            n_clips_codes = int(self._kpms_codes.shape[0])
            n_clips_padded = int(padded.shape[0])
            if n_clips_codes != n_clips_padded:
                raise ValueError(
                    f"e2e_motion_norm_padded leading dim {n_clips_padded} != "
                    f"kpms_codes leading dim {n_clips_codes}. The per-clip "
                    f"ordering of the two arrays must match position by position."
                )
            n_frames_codes = int(self._kpms_codes.shape[1])
            expected_padded = (
                n_frames_codes + e2e_window_len - 1 + self._code_stack_size - 1
            )
            if int(padded.shape[1]) != expected_padded:
                raise ValueError(
                    f"e2e_motion_norm_padded second dim {int(padded.shape[1])} "
                    f"!= expected {expected_padded} = n_frames "
                    f"({n_frames_codes}) + window_len-1 ({e2e_window_len - 1}) "
                    f"+ code_stack_size-1 ({self._code_stack_size - 1}). The "
                    f"padding must match the env's code_stack_size."
                )
            self._e2e_motion_norm_padded = padded
            self._e2e_window_len = int(e2e_window_len)
            self._e2e_feat_dim = int(padded.shape[-1])
            self._e2e_extract = make_window_stack_extractor(
                window_len=self._e2e_window_len,
                code_stack_size=self._code_stack_size,
                feat_dim=self._e2e_feat_dim,
            )

    # ------------------------------------------------------------------
    # Override _get_obs to inject kpms_code and (optionally) motion windows
    # ------------------------------------------------------------------

    def _get_obs(self, data: mjx.Data, info: Mapping[str, Any]) -> Mapping[str, Any]:
        obs = super()._get_obs(data, info)
        frame = self._get_cur_frame(data, info)
        n_frames = self._kpms_codes.shape[1]
        clip_idx = info["reference_clip"]

        # Inject kpms_code (pretrained tokenizer codes — frozen even in e2e mode)
        if self._code_stack_size <= 1:
            frame = jnp.clip(frame, 0, n_frames - 1)
            code = self._kpms_codes[clip_idx, frame]
            kpms_code = code[..., None].astype(jnp.float32)
        else:
            offsets = jnp.arange(self._code_stack_size)
            frames = jnp.clip(frame[..., None] + offsets, 0, n_frames - 1)
            codes = self._kpms_codes[clip_idx[..., None], frames]
            kpms_code = codes.astype(jnp.float32)

        # Flatten nested obs and strip "state" hierarchy:
        # v0.0.13 returns {"state": OrderedDict(task_obs=nested, proprioception=nested)}
        # We produce {"task_obs": 1D, "proprioception": 1D, "kpms_code": 1D}
        state_obs = obs["state"]
        flat = collections.OrderedDict()
        for k, v in state_obs.items():
            flat[k] = jnp.nan_to_num(jax.flatten_util.ravel_pytree(v)[0])
        flat["kpms_code"] = kpms_code

        # e2e mode: inject the VQ-VAE encoder's motion-window stack. The stack
        # uses the same `frame` base + code_stack_size convention as
        # kpms_code, so window k (k=0..stack-1) ends at original-frame
        # frame+k under the same edge-clamp policy that kpms_code uses.
        if self._e2e_motion_norm_padded is not None:
            base_frame = jnp.clip(frame, 0, n_frames - 1)
            windows = self._e2e_extract(
                self._e2e_motion_norm_padded,
                clip_idx,
                base_frame,
            )
            flat["motion_window_stack"] = windows.astype(jnp.float32)

        return flat

    @property
    def proprioceptive_obs_size(self) -> int:
        obs_size = self.non_flattened_observation_size
        return jp.sum(
            jax.flatten_util.ravel_pytree(obs_size["proprioception"])[0]
        )

    @property
    def observation_size(self):
        obs = self.non_flattened_observation_size
        return jp.sum(jax.flatten_util.ravel_pytree(obs)[0])

    # ------------------------------------------------------------------
    # e2e-mode introspection (consumed by the network factory at init)
    # ------------------------------------------------------------------

    @property
    def e2e_mode(self) -> bool:
        """True iff this env was constructed with a motion-window buffer."""
        return self._e2e_motion_norm_padded is not None

    @property
    def e2e_motion_window_shape(self) -> tuple[int, int, int]:
        """``(code_stack_size, window_len, feat_dim)`` — the per-step shape of
        ``obs["motion_window_stack"]``.

        Used by ``make_moseq_*_ppo_networks`` to build a correctly-shaped
        ``dummy_obs`` for Flax param init. Conv1D kernel shapes depend on
        ``feat_dim``, so the dummy shape must be correct or the init traces
        wrong-shape kernels.

        Raises:
            RuntimeError: e2e mode is off; the caller should not be inspecting
                this property.
        """
        if not self.e2e_mode:
            raise RuntimeError(
                "e2e_motion_window_shape requested but env was constructed "
                "without e2e_motion_norm_padded — e2e_mode is off."
            )
        return (
            self._code_stack_size,
            self._e2e_window_len,
            self._e2e_feat_dim,
        )
