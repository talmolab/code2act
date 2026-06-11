"""Option-B interpretable task transfer.

A high-level **categorical** policy emits a discrete code per decision step; a
frozen ar_codegen prior supplies ``prior_logits``; the two combine via
**policy-gated additive logits** ``final_logits = sigmoid(beta)*prior_logits +
delta`` and one code is sampled. The sampled code drives the **frozen GRU
decoder** (decoder-only, ``z_e_scale=0``) which renders it into torques for a
``vnl_playground`` task env (walk_rear / navigation / forage).

This is the discrete, interpretable analog of SCAMPER's continuous residual
transfer: the task solution is a readable sequence of named behavioral codes.

See ``scratch/2026-06-09-option-b-task-adaptation/{plan,design}.md``.

Modules:
    * ``_compat`` — orbax/jax ``record_scalar`` shim (imported for side effects).
    * ``decoder_adapter`` — load + drive the frozen GRU decoder from a code stream.
"""

from c2a.transfer import _compat  # noqa: F401  (applies the record_scalar shim)
