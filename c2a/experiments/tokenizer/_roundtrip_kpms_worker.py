"""x64 subprocess worker: forward kinematics + KPMS re-segmentation.

``keypoint_moseq`` forces ``jax_enable_x64=True`` irreversibly per process,
which would corrupt the float32 decoder rollouts in the round-trip driver.
This worker is therefore a *separate process*: it sets ``JAX_ENABLE_X64=1``
before importing JAX, runs FK (qpos -> 3D keypoints) and KPMS ``apply_model``
for one or more Gibbs seeds, and writes the per-frame codes to an ``.npz``.

It is invoked by :class:`c2a.experiments.shared.segmenters.KPMSWorkerSegmenter`
— not run directly by users.

Usage::

    python -m c2a.experiments.tokenizer._roundtrip_kpms_worker \\
        --qpos-npz IN.npz --out-npz OUT.npz \\
        --kpms-model-dir DIR --kpms-model-name NAME --num-iters 200 \\
        --fk-h5 H5 --stac-xml XML --seeds 0 1 2 3 4

``IN.npz`` must contain ``qpos`` of shape ``(n_clips, n_frames, nq)``.
``OUT.npz`` gets ``codes`` ``(n_seeds, n_clips, n_frames)`` int32 (``-1`` for
frames KPMS did not assign), ``seeds`` and ``num_codes``.
"""

from __future__ import annotations

import os

# Must precede any JAX import — keypoint_moseq and FK both need float64.
os.environ["JAX_ENABLE_X64"] = "1"

import argparse
import logging

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from c2a.experiments.shared.keypoint_fk import (
    qpos_to_keypoints_fk,
    setup_stac_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("roundtrip_kpms_worker")

# keypoint_moseq still references the long-removed ``np.bool8`` alias.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

# Config keys ``apply_model`` needs threaded back in (mirrors the original
# KPMS apply path in run_generalization / the old run_roundtrip).
_INIT_KEYS = (
    "anterior_idxs", "posterior_idxs", "fix_heading", "whiten",
    "error_estimator", "conf_threshold", "PCA_fitting_num_frames",
    "trans_hypparams", "ar_hypparams", "obs_hypparams", "cen_hypparams",
)


def _fk(qpos: np.ndarray, fk_h5: str, stac_xml: str) -> np.ndarray:
    """qpos ``(n_clips, n_frames, nq)`` -> keypoints ``(n_clips, n_frames, K, 3)``."""
    n_clips, n_frames = qpos.shape[:2]
    qpos_flat = qpos.reshape(-1, qpos.shape[-1]).astype(np.float32)
    log.info("FK on %d frames (%d clips)...", qpos_flat.shape[0], n_clips)
    mj_model, mj_data, site_ids, kp_names = setup_stac_model(fk_h5, stac_xml)
    kps = qpos_to_keypoints_fk(qpos_flat, mj_model, mj_data, site_ids)
    return kps.reshape(n_clips, n_frames, -1, 3)


def _kpms_apply(
    keypoints: np.ndarray,
    model_dir: str,
    model_name: str,
    num_iters: int,
    seeds: list[int],
) -> tuple[np.ndarray, int]:
    """Run KPMS ``apply_model`` once per seed on the FK keypoints.

    Returns ``(codes, num_codes)`` where ``codes`` is
    ``(n_seeds, n_clips, n_frames)`` int32 (``-1`` for unassigned frames).
    """
    import keypoint_moseq as kpms

    n_clips, n_frames = keypoints.shape[:2]
    model, _, _, _ = kpms.load_checkpoint(
        project_dir=model_dir, model_name=model_name,
    )
    config = kpms.load_config(model_dir)
    num_codes = int(config["trans_hypparams"]["num_states"])

    coordinates = {f"clip_{i}": keypoints[i] for i in range(n_clips)}
    fmt_config = dict(config)
    fmt_config["added_noise_level"] = 0.0
    data, metadata = kpms.format_data(
        coordinates, confidences=None, **fmt_config,
    )
    init_kwargs = {k: v for k, v in config.items() if k in _INIT_KEYS}

    codes = np.full((len(seeds), n_clips, n_frames), -1, dtype=np.int32)
    for si, seed in enumerate(seeds):
        log.info(
            "KPMS apply_model: seed=%d (%d/%d), %d iters, %d clips",
            seed, si + 1, len(seeds), num_iters, n_clips,
        )
        seeded = dict(model)  # shallow copy — never mutate the cached ckpt
        seeded["seed"] = jax.random.PRNGKey(int(seed))
        results = kpms.apply_model(
            model=seeded, data=data, metadata=metadata,
            num_iters=int(num_iters), ar_only=False,
            return_model=False, save_results=False, verbose=True,
            **init_kwargs,
        )
        for i in range(n_clips):
            syl = np.asarray(results[f"clip_{i}"]["syllable"])
            usable = min(syl.shape[0], n_frames)
            if usable < n_frames:
                log.warning(
                    "seed %d clip %d: KPMS returned only %d/%d frames "
                    "— trailing frames left as -1.",
                    seed, i, usable, n_frames,
                )
            codes[si, i, :usable] = syl[:usable].astype(np.int32)
    return codes, num_codes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FK + KPMS re-segmentation worker (x64 subprocess)."
    )
    parser.add_argument("--qpos-npz", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--kpms-model-dir", required=True)
    parser.add_argument("--kpms-model-name", required=True)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--fk-h5", required=True)
    parser.add_argument("--stac-xml", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    qpos = np.asarray(np.load(args.qpos_npz)["qpos"], dtype=np.float32)
    if qpos.ndim != 3:
        raise ValueError(
            f"qpos must be (n_clips, n_frames, nq); got {qpos.shape}."
        )

    keypoints = _fk(qpos, args.fk_h5, args.stac_xml)
    codes, num_codes = _kpms_apply(
        keypoints,
        args.kpms_model_dir,
        args.kpms_model_name,
        args.num_iters,
        list(args.seeds),
    )

    np.savez(
        args.out_npz,
        codes=codes,
        seeds=np.asarray(args.seeds, dtype=np.int32),
        num_codes=np.int32(num_codes),
    )
    log.info(
        "Wrote %s: codes %s, num_codes=%d",
        args.out_npz, codes.shape, num_codes,
    )


if __name__ == "__main__":
    main()
