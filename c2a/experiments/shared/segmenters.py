"""Unified segmenter abstraction for the round-trip experiment.

A :class:`Segmenter` maps a kinematics trajectory (per-clip ``qpos``) to one
discrete code per frame. Concrete segmenters wrap the project's four motion
tokenizers — ``vae``, ``vqvae``, ``ps_vqvae`` (kinematics autoencoders) and
``kpms`` (Keypoint-MoSeq).

Determinism
-----------
The three autoencoders are deterministic functions: ``vae`` discretizes the
posterior mean ``μ`` with a frozen K-Means, ``vqvae``/``ps_vqvae`` do hard
``argmin`` codebook lookup. KPMS applies via Gibbs sampling, so its ``encode``
is seed-dependent — stage 1 of the round-trip experiment measures exactly that.

KPMS process isolation
----------------------
``keypoint_moseq`` forces ``jax_enable_x64=True`` irreversibly per process,
which would corrupt the float32 decoder rollouts. This module therefore never
imports ``keypoint_moseq``: :class:`KPMSWorkerSegmenter` shells out to
``c2a.experiments.tokenizer._roundtrip_kpms_worker`` (a separate x64 process).
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class Segmenter:
    """Maps a kinematics trajectory to one discrete code per frame.

    Attributes:
        name: Short identifier (``vae`` / ``vqvae`` / ``ps_vqvae`` / ``kpms``).
        num_codes: Size of the discrete code alphabet.
        boundary_pad: ``(left, right)`` count of per-frame codes at each clip
            edge that are *replicated* from the nearest window rather than
            independently observed. Round-trip metrics can exclude these so
            edge frames are not double-counted. ``(0, 0)`` for KPMS.
    """

    name: str = "segmenter"
    num_codes: int = 0
    boundary_pad: tuple[int, int] = (0, 0)

    def encode(self, qpos: np.ndarray, *, seed: int = 0) -> np.ndarray:
        """Segment a batch of clips.

        Args:
            qpos: ``(n_clips, n_frames, 7 + n_joints)`` joint positions.
            seed: PRNG seed (ignored by the deterministic AE segmenters;
                used by KPMS Gibbs inference).

        Returns:
            ``(n_clips, n_frames)`` int32 per-frame codes.
        """
        raise NotImplementedError

    def encode_repeats(
        self, qpos: np.ndarray, seeds: list[int]
    ) -> np.ndarray:
        """Encode the same clips once per seed.

        Returns:
            ``(n_seeds, n_clips, n_frames)`` int32 codes.
        """
        return np.stack(
            [self.encode(qpos, seed=s) for s in seeds], axis=0
        )


# ---------------------------------------------------------------------------
# Kinematics-autoencoder segmenters (vae / vqvae / ps_vqvae)
# ---------------------------------------------------------------------------


class AESegmenter(Segmenter):
    """Wraps a frozen kinematics autoencoder tokenizer.

    Loads the encoder + bottleneck (+ for ``vae`` the frozen PCA/K-Means
    ``tokenizer.pkl``) and never re-fits anything. ``encode`` reproduces the
    code path of the matching ``c2a.codegen.kinematics_vae.<kind>.tokenize``
    script via :func:`encode_motion_array`.
    """

    def __init__(
        self,
        name: str,
        kind: str,
        ckpt_dir: str | Path,
        *,
        num_codes: int,
        n_joints: int = 67,
        use_qvel: bool = False,
        window_len: int = 10,
        horizon: int = 5,
        code_dim: int | None = None,
        group_dims: tuple[int, ...] | None = None,
        group_codebook_sizes: tuple[int, ...] | None = None,
        tokenizer_pkl: str | Path | None = None,
    ) -> None:
        from c2a.codegen.kinematics_vae.backbone.codegen import load_checkpoint

        self.name = name
        self.kind = kind
        self.num_codes = int(num_codes)
        self.n_joints = int(n_joints)
        self.use_qvel = bool(use_qvel)
        self.window_len = int(window_len)
        self.horizon = int(horizon)
        # encode_motion_array assigns one code per window then replicates
        # the edge windows over window_len-1 left / horizon right frames.
        self.boundary_pad = (self.window_len - 1, self.horizon)

        if kind == "vae":
            from c2a.codegen.kinematics_vae.vae.bottleneck import (
                GaussianBottleneck,
            )

            bottleneck = GaussianBottleneck()
        elif kind == "vqvae":
            from c2a.codegen.kinematics_vae.vqvae.bottleneck import (
                VQVAEBottleneck,
            )

            if code_dim is None:
                raise ValueError("vqvae requires code_dim.")
            bottleneck = VQVAEBottleneck(
                codebook_size=int(num_codes), code_dim=int(code_dim)
            )
        elif kind == "ps_vqvae":
            from c2a.codegen.kinematics_vae.ps_vqvae.bottleneck import (
                PSVQVAEBottleneck,
            )

            if group_dims is None or group_codebook_sizes is None:
                raise ValueError(
                    "ps_vqvae requires group_dims and group_codebook_sizes."
                )
            self.group_codebook_sizes = tuple(
                int(k) for k in group_codebook_sizes
            )
            bottleneck = PSVQVAEBottleneck(
                group_dims=tuple(int(d) for d in group_dims),
                group_codebook_sizes=self.group_codebook_sizes,
            )
        else:
            raise ValueError(f"unknown AE segmenter kind {kind!r}")

        self.modules, self.params = load_checkpoint(
            ckpt_dir,
            bottleneck=bottleneck,
            n_joints=self.n_joints,
            use_qvel=self.use_qvel,
            window_len=self.window_len,
            horizon=self.horizon,
        )

        self._pca = None
        self._kmeans = None
        if kind == "vae":
            if tokenizer_pkl is None:
                raise ValueError("vae requires tokenizer_pkl (PCA + K-Means).")
            with open(tokenizer_pkl, "rb") as f:
                tok = pickle.load(f)
            self._pca = tok["pca"]
            self._kmeans = tok["kmeans"]

    def encode(self, qpos: np.ndarray, *, seed: int = 0) -> np.ndarray:
        from c2a.codegen.kinematics_vae.backbone.codegen import (
            encode_motion_array,
        )

        qpos = np.asarray(qpos, dtype=np.float32)
        # vae's GaussianBottleneck reparameterizes unless sigma is clamped;
        # codegen pins sigma_max=0.0 so z == mean. VQ bottlenecks ignore it.
        bn_kwargs = {"sigma_max": 0.0} if self.kind == "vae" else None
        encoded = encode_motion_array(
            qpos,
            self.modules,
            self.params["encoder"],
            self.params["bottleneck"],
            bn_buffers=self.params.get("bottleneck_buffers"),
            n_joints=self.n_joints,
            use_qvel=self.use_qvel,
            window_len=self.window_len,
            horizon=self.horizon,
            deterministic=True,
            bottleneck_kwargs=bn_kwargs,
        )

        if self.kind == "vae":
            z = encoded.mean  # (n_clips, n_frames, latent_dim) — μ
            n_clips, n_frames, latent_dim = z.shape
            z_flat = z.reshape(-1, latent_dim)
            feat = (
                self._pca.transform(z_flat)
                if self._pca is not None
                else z_flat
            )
            codes = self._kmeans.predict(feat).astype(np.int32)
            codes = codes.reshape(n_clips, n_frames)
        elif self.kind == "vqvae":
            if "codes" not in encoded.aux:
                raise RuntimeError("VQVAEBottleneck emitted no 'codes' aux.")
            codes = encoded.aux["codes"].astype(np.int32)
        else:  # ps_vqvae
            from c2a.codegen.kinematics_vae.ps_vqvae.tokenize import (
                fold_to_product_index,
            )

            if "codes" not in encoded.aux:
                raise RuntimeError("PSVQVAEBottleneck emitted no 'codes' aux.")
            codes = fold_to_product_index(
                encoded.aux["codes"], self.group_codebook_sizes
            )
        return np.asarray(codes, dtype=np.int32)


# ---------------------------------------------------------------------------
# KPMS segmenter (subprocess proxy — runs in a separate x64 process)
# ---------------------------------------------------------------------------


class KPMSWorkerSegmenter(Segmenter):
    """Driver-side proxy for KPMS segmentation.

    Holds only the KPMS model paths. ``encode`` writes the qpos batch to a
    temp ``.npz``, runs ``c2a.experiments.tokenizer._roundtrip_kpms_worker`` as a
    subprocess (which sets ``JAX_ENABLE_X64=1`` and runs FK + KPMS
    ``apply_model``), then reads the codes back. KPMS is a Gibbs sampler, so
    the result depends on ``seed``.
    """

    def __init__(
        self,
        name: str,
        *,
        model_dir: str | Path,
        model_name: str,
        num_iters: int,
        num_codes: int,
        fk_h5: str | Path,
        stac_xml: str | Path,
        tmp_dir: str | Path,
    ) -> None:
        self.name = name
        self.num_codes = int(num_codes)
        self.model_dir = str(model_dir)
        self.model_name = str(model_name)
        self.num_iters = int(num_iters)
        self.fk_h5 = str(fk_h5)
        self.stac_xml = str(stac_xml)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self._calls = 0

    def encode(self, qpos: np.ndarray, *, seed: int = 0) -> np.ndarray:
        return self.encode_repeats(qpos, [int(seed)])[0]

    def encode_repeats(
        self, qpos: np.ndarray, seeds: list[int]
    ) -> np.ndarray:
        from c2a.paths import get_repo_root

        qpos = np.asarray(qpos, dtype=np.float32)
        self._calls += 1
        tag = f"{self.name}_{os.getpid()}_{self._calls}"
        in_npz = self.tmp_dir / f"kpms_in_{tag}.npz"
        out_npz = self.tmp_dir / f"kpms_out_{tag}.npz"
        np.savez(in_npz, qpos=qpos)
        # Launch through bash_scripts/_pyrun.py so the x64 worker subprocess
        # gets the same venv site-packages + repo-root sys.path the driver
        # has (a bare ``python -m`` child would not inherit it).
        pyrun = Path(get_repo_root()) / "bash_scripts" / "_pyrun.py"
        cmd = [
            sys.executable,
            str(pyrun),
            "c2a.experiments.tokenizer._roundtrip_kpms_worker",
            "--qpos-npz", str(in_npz),
            "--out-npz", str(out_npz),
            "--kpms-model-dir", self.model_dir,
            "--kpms-model-name", self.model_name,
            "--num-iters", str(self.num_iters),
            "--fk-h5", self.fk_h5,
            "--stac-xml", self.stac_xml,
            "--seeds", *[str(int(s)) for s in seeds],
        ]
        log.info("KPMS worker (%d seeds): %s", len(seeds), " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"KPMS worker failed for segmenter {self.name!r} "
                f"(seeds={seeds}, exit code {e.returncode}) — see the "
                f"worker traceback above."
            ) from e
        if not out_npz.exists():
            raise RuntimeError(
                f"KPMS worker for {self.name!r} exited 0 but produced no "
                f"output at {out_npz}."
            )

        out = np.load(out_npz)
        codes = np.asarray(out["codes"], dtype=np.int32)  # (n_seeds, n_clip, T)
        if "num_codes" in out:
            worker_nc = int(out["num_codes"])
            if worker_nc != self.num_codes:
                log.warning(
                    "KPMS worker reported num_codes=%d but config said %d "
                    "— using the worker value.",
                    worker_nc, self.num_codes,
                )
                self.num_codes = worker_nc
        return codes


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def load_segmenter(
    name: str,
    seg_cfg,
    *,
    fk_h5: str | Path | None = None,
    stac_xml: str | Path | None = None,
    tmp_dir: str | Path | None = None,
) -> Segmenter:
    """Build a :class:`Segmenter` from a config block.

    Args:
        name: Segmenter id used for logs and temp filenames.
        seg_cfg: A mapping with at least ``type`` (``"ae"`` or ``"kpms"``)
            and the type-specific fields (see ``configs/roundtrip.yaml``).
        fk_h5, stac_xml, tmp_dir: Required for ``kpms`` (FK + subprocess).
    """
    seg_type = str(seg_cfg["type"])
    if seg_type == "ae":
        return AESegmenter(
            name,
            str(seg_cfg["kind"]),
            seg_cfg["ckpt_dir"],
            num_codes=int(seg_cfg["num_codes"]),
            n_joints=int(seg_cfg.get("n_joints", 67)),
            use_qvel=bool(seg_cfg.get("use_qvel", False)),
            window_len=int(seg_cfg.get("window_len", 10)),
            horizon=int(seg_cfg.get("horizon", 5)),
            code_dim=seg_cfg.get("code_dim"),
            group_dims=seg_cfg.get("group_dims"),
            group_codebook_sizes=seg_cfg.get("group_codebook_sizes"),
            tokenizer_pkl=seg_cfg.get("tokenizer_pkl"),
        )
    if seg_type == "kpms":
        if fk_h5 is None or stac_xml is None or tmp_dir is None:
            raise ValueError(
                "kpms segmenter requires fk_h5, stac_xml and tmp_dir."
            )
        return KPMSWorkerSegmenter(
            name,
            model_dir=seg_cfg["model_dir"],
            model_name=str(seg_cfg["model_name"]),
            num_iters=int(seg_cfg.get("num_iters", 200)),
            num_codes=int(seg_cfg["num_codes"]),
            fk_h5=fk_h5,
            stac_xml=stac_xml,
            tmp_dir=tmp_dir,
        )
    raise ValueError(f"unknown segmenter type {seg_type!r}")
