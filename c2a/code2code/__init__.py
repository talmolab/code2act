"""In-training-loop tokenizer adapters for the code2code reward mode.

The code2code mode (see `scratch/2026-05-26-code2code-mode/`) adds a
per-step reward signal computed by feeding the policy's rollout motion
back through a frozen tokenizer's encoder and comparing the result to
the codebook entry for the target code:

    r_c2c = exp(-distance(z_rollout, codebook[c_target]) / τ)

The encoder + codebook stay **frozen** throughout RL — this is the
critical difference from the (removed) e2e_mode design. The tokenizer's
discrete codes therefore stay anchored to their pretraining-time
kinematic meaning instead of drifting into a private control basis.

This package provides adapters that expose the in-loop interface:

* :class:`C2CTokenizerAdapter` — abstract base.
* :class:`VQVAEAdapter` — concrete adapter for vanilla VQ-VAE ckpts.

PS-VQ-VAE and VAE adapters land in follow-up stages.

KPMS is not supported by code2code mode because it needs
``jax_enable_x64=True`` and runs Gibbs sampling in a separate process
— see the round-trip experiment for the offline x64-subprocess pattern.
The code2code config validator raises ``NotSupportedError`` if the
caller sets ``code2code.reward_mode != "off"`` with a KPMS code source.
"""

from c2a.code2code.adapter_base import C2CTokenizerAdapter
from c2a.code2code.ps_vqvae_adapter import PSVQVAEAdapter
from c2a.code2code.vae_adapter import VAEAdapter
from c2a.code2code.vqvae_adapter import VQVAEAdapter

__all__ = ["C2CTokenizerAdapter", "PSVQVAEAdapter", "VAEAdapter", "VQVAEAdapter"]
