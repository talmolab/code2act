"""Experiment: label-free code distinctness (P1 + P2 + P3) + video features.

Post-processes a ``run_code_grid_rollouts`` output directory plus the
tokenizer codes npz.

Numeric panels (always produced):

* **P1** — within-code vs between-code MMD vs a code-shuffled null.
* **P2** — behavioral-phylogeny dendrogram + reordered MMD heatmap, with
  the within-code reproducibility noise floor drawn on the dendrogram.
* **P3** — NMI(code ; 4-class ethogram), effective code count, behavior
  -> code spread matrix.

Video features (``render_videos=true`` only; need a MuJoCo env):

* **Feature 1 — video dendrogram.** The P2 tree, but every node is a
  rendered video of the agents under it. At each above-floor merge the two
  children are colored orange (left) and light blue (right); subtrees
  entirely below the noise floor are collapsed into a single, single-color
  video node. Two MP4s per pose: ``F1`` (full tree) and ``F1b`` (pruned to
  the above-noise-floor merges only — narrower).
* **Feature 2 — distinct-codes-by-behavior grid.** A 4-column grid (one
  column per behavior), rows = the distinct codes (noise-floor cluster
  representatives), each cell the code's rollout video, placed in the
  column of its dominant behavior.

Usage:
    cd code2act
    python -m c2a.experiments.run_code_distinctness                # numeric
    python -m c2a.experiments.run_code_distinctness render_videos=true
"""

import json
import logging
from pathlib import Path

import hydra
import jax
import matplotlib
import numpy as np
from omegaconf import DictConfig

# Workaround for the venv's orbax-checkpoint (0.11.24) calling
# jax.monitoring.record_scalar, which was removed from current JAX. Must run
# before any Orbax restore (load_moseq_checkpoint in _build_render_env). Pure
# telemetry; stubbing is safe. Same patch as run_code_grid_rollouts.py.
if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *a, **k: None

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

from c2a.codegen.shared.label_alignment import (  # noqa: E402
    BEHAVIORS, UNLABELED, load_label_frames,
)
from c2a.experiments.shared.code_grid import (  # noqa: E402
    add_banner, build_uniform_walkers, mmd_squared, render_panel_frames,
    stack_panels_into_grid, write_mp4,
)
from c2a.paths import get_repo_root  # noqa: E402

log = logging.getLogger(__name__)

REPO_ROOT = get_repo_root()
MAX_POOL = 1500  # cap pooled qpos vectors per MMD argument, for speed

# Per-behavior fill colors for the Feature-2 grid (order = BEHAVIORS).
_BEHAVIOR_COLORS = [
    [0.55, 0.57, 0.62, 1.0],  # Immobile — slate
    [0.62, 0.40, 0.80, 1.0],  # Rear — purple
    [0.95, 0.60, 0.15, 1.0],  # Turn — orange
    [0.30, 0.70, 0.42, 1.0],  # Walk — green
]


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Pooling / MMD helpers
# ---------------------------------------------------------------------------


def _pool(bodies: list[np.ndarray], strip_root: bool, rng) -> np.ndarray:
    """Pool a list of ``(T, nq)`` trajectories into ``(<=MAX_POOL, feat)``."""
    flat = np.concatenate(
        [b.reshape(b.shape[0], -1) for b in bodies], axis=0,
    ).astype(np.float32)
    if strip_root and flat.shape[1] > 3:
        flat = flat[:, 3:]
    if flat.shape[0] > MAX_POOL:
        flat = flat[rng.choice(flat.shape[0], MAX_POOL, replace=False)]
    return flat


def _split_half_mmd(
    bodies: list[np.ndarray], sigma: float, strip_root: bool, rng,
) -> float:
    """MMD between the two halves of a list of K body trajectories."""
    half = len(bodies) // 2
    if half < 1 or len(bodies) - half < 1:
        return float("nan")
    a = _pool(bodies[:half], strip_root, rng)
    b = _pool(bodies[half:], strip_root, rng)
    return float(np.sqrt(max(0.0, mmd_squared(a, b, sigma))))


# ---------------------------------------------------------------------------
# P1 — within / between / null
# ---------------------------------------------------------------------------


def _p1_distances(
    qpos_per_code: dict[int, np.ndarray],
    D: np.ndarray,
    code_ids: list[int],
    sigma: float,
    strip_root: bool,
    n_null: int,
    rng,
) -> dict[str, np.ndarray]:
    """Return within-code, between-code, and shuffled-null MMD arrays."""
    within = []
    for c in code_ids:
        q = qpos_per_code[c]  # (K, T, nq)
        bodies = [q[k] for k in range(q.shape[0])]
        within.append(_split_half_mmd(bodies, sigma, strip_root, rng))
    within = np.asarray(within, dtype=np.float32)

    iu = np.triu_indices(D.shape[0], k=1)
    between = D[iu].astype(np.float32)

    all_bodies: list[np.ndarray] = []
    K = None
    for c in code_ids:
        q = qpos_per_code[c]
        K = q.shape[0]
        all_bodies.extend(q[k] for k in range(q.shape[0]))
    null = []
    for _ in range(n_null):
        idx = rng.choice(len(all_bodies), size=K, replace=False)
        grp = [all_bodies[i] for i in idx]
        null.append(_split_half_mmd(grp, sigma, strip_root, rng))
    null = np.asarray(null, dtype=np.float32)

    return {"within": within, "between": between, "null": null}


def _plot_p1(dists: dict[str, np.ndarray], pose: str, path: Path) -> dict:
    within = dists["within"][~np.isnan(dists["within"])]
    between = dists["between"]
    null = dists["null"][~np.isnan(dists["null"])]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = [within, null, between]
    labels = [
        f"within-code\n(n={within.size})",
        f"shuffled null\n(n={null.size})",
        f"between-code\n(n={between.size})",
    ]
    colors = ["#2c7fb8", "#888888", "#d95f0e"]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    ax.set_ylabel("MMD (posture, root-stripped)")
    ax.set_title(
        f"P1 - {pose}: code reproducibility vs distinctness\n"
        f"within << between and within << null  =>  reproducible codes"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return {
        "within_median": float(np.median(within)),
        "between_median": float(np.median(between)),
        "null_median": float(np.median(null)),
        "within_over_between": float(
            np.median(within) / max(np.median(between), 1e-9)
        ),
        "within_over_null": float(
            np.median(within) / max(np.median(null), 1e-9)
        ),
    }


# ---------------------------------------------------------------------------
# P2 — behavioral phylogeny
# ---------------------------------------------------------------------------


def _plot_p2(
    D: np.ndarray, code_ids: list[int], floor: float, pose: str, path: Path,
) -> None:
    """Dendrogram + reordered MMD heatmap, with the within-code floor line."""
    cond = squareform(D, checks=False)
    Z = hierarchy.linkage(cond, method="average")  # UPGMA

    fig, (ax_d, ax_h) = plt.subplots(
        1, 2, figsize=(13, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.15]},
    )

    dend = hierarchy.dendrogram(
        Z, labels=[str(c) for c in code_ids], ax=ax_d,
        color_threshold=0, above_threshold_color="#444444",
    )
    ax_d.axhline(
        floor, color="#d95f0e", ls="--", lw=1.6,
        label=f"within-code noise floor ({floor:.3f})",
    )
    ax_d.set_title(
        f"P2 - {pose}: behavioral phylogeny\n"
        f"merges far above the floor = real distinctions"
    )
    ax_d.set_xlabel("code id")
    ax_d.set_ylabel("MMD merge height")
    ax_d.legend(loc="upper right", fontsize=8)

    order = dend["leaves"]
    D_re = D[np.ix_(order, order)]
    im = ax_h.imshow(D_re, cmap="magma", aspect="auto")
    ax_h.set_xticks(range(len(order)))
    ax_h.set_yticks(range(len(order)))
    ax_h.set_xticklabels([code_ids[i] for i in order], fontsize=6, rotation=90)
    ax_h.set_yticklabels([code_ids[i] for i in order], fontsize=6)
    ax_h.set_title(f"{pose}: MMD matrix (dendrogram-ordered)")
    fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="MMD")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# P3 — NMI vs ethogram
# ---------------------------------------------------------------------------


def _nmi(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Normalized mutual information I(x;y)/sqrt(H(x)H(y)), plus MI, H(x), H(y)."""
    cx = np.unique(x, return_inverse=True)[1]
    cy = np.unique(y, return_inverse=True)[1]
    n = cx.size
    joint = np.zeros((cx.max() + 1, cy.max() + 1), dtype=np.float64)
    np.add.at(joint, (cx, cy), 1.0)
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    nz = joint > 0
    outer = px[:, None] * py[None, :]
    mi = float((joint[nz] * np.log(joint[nz] / outer[nz])).sum())
    hx = float(-(px[px > 0] * np.log(px[px > 0])).sum())
    hy = float(-(py[py > 0] * np.log(py[py > 0])).sum())
    nmi = mi / np.sqrt(hx * hy) if hx > 0 and hy > 0 else 0.0
    return nmi, mi, hx, hy


def _p3(cfg: DictConfig, out_dir: Path, rng) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (summary, spread_counts[n_beh,num_codes], usage_counts[num_codes])."""
    codes_npz = np.load(_abs(cfg.codes_path), allow_pickle=True)
    num_codes = int(codes_npz["num_states"])
    clip_len = int(codes_npz["train_codes"].shape[1])

    behavior = load_label_frames(
        str(_abs(cfg.data.labels_path)),
        clip_range=tuple(cfg.data.get("labeled_clip_range", [0, 841])),
        n_frames=clip_len,
        csv_to_h5_subsample=int(cfg.data.get("csv_to_h5_subsample", 2)),
    ).behavior  # (N_clips, clip_len)

    code_flat, beh_flat = [], []
    for codes_key, idx_key in [
        ("train_codes", "train_indices"), ("test_codes", "test_indices"),
    ]:
        codes_arr = codes_npz[codes_key]
        idx_arr = codes_npz[idx_key]
        for row, clip in zip(codes_arr, idx_arr):
            clip = int(clip)
            if clip >= behavior.shape[0]:
                continue
            code_flat.append(np.asarray(row, dtype=np.int64))
            beh_flat.append(behavior[clip].astype(np.int64))
    code_flat = np.concatenate(code_flat)
    beh_flat = np.concatenate(beh_flat)

    mask = (beh_flat != UNLABELED) & (code_flat >= 0)
    code_lab = code_flat[mask]
    beh_lab = beh_flat[mask]

    nmi, mi, h_code, h_beh = _nmi(code_lab, beh_lab)
    null_nmi = []
    for _ in range(int(cfg.get("n_null_samples", 200))):
        null_nmi.append(_nmi(rng.permutation(code_lab), beh_lab)[0])
    null_nmi = np.asarray(null_nmi)

    counts = np.bincount(code_flat[code_flat >= 0], minlength=num_codes)
    p = counts / counts.sum()
    h_usage = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    eff_codes = float(np.exp(h_usage))

    n_beh = len(BEHAVIORS)
    spread = np.zeros((n_beh, num_codes), dtype=np.float64)
    np.add.at(spread, (beh_lab, code_lab), 1.0)
    col = spread.sum(axis=1, keepdims=True)
    spread_norm = spread / np.maximum(col, 1e-9)

    fig, (ax_t, ax_h) = plt.subplots(
        2, 1, figsize=(11, 5.6),
        gridspec_kw={"height_ratios": [1, 3.2]},
    )
    ax_t.axis("off")
    ax_t.text(
        0.0, 0.5,
        f"P3 - codes are finer than the ethogram\n"
        f"NMI(code ; 4-class behavior) = {nmi:.3f}   "
        f"(shuffled-code null = {null_nmi.mean():.3f} +/- {null_nmi.std():.3f})\n"
        f"effective code count (perplexity) = {eff_codes:.1f}  "
        f"vs {n_beh} behavior classes\n"
        f"partial NMI with effective codes >> {n_beh}  =>  codes resolve "
        f"sub-structure the ethogram cannot",
        va="center", ha="left", fontsize=11, family="monospace",
    )
    im = ax_h.imshow(spread_norm, cmap="viridis", aspect="auto")
    ax_h.set_yticks(range(n_beh))
    ax_h.set_yticklabels(BEHAVIORS)
    ax_h.set_xlabel("code id")
    ax_h.set_ylabel("behavior")
    ax_h.set_title("each behavior's frame mass spreads across many codes")
    fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02, label="P(code | behavior)")
    fig.tight_layout()
    fig.savefig(out_dir / "P3_nmi_spread.png", dpi=150)
    plt.close(fig)

    summary = {
        "nmi_code_ethogram": float(nmi),
        "nmi_shuffled_null_mean": float(null_nmi.mean()),
        "nmi_shuffled_null_std": float(null_nmi.std()),
        "mutual_information_nats": float(mi),
        "entropy_code_nats": float(h_code),
        "entropy_ethogram_nats": float(h_beh),
        "effective_code_count": eff_codes,
        "num_behavior_classes": n_beh,
        "num_codes": num_codes,
        "n_labeled_frames": int(code_lab.size),
    }
    return summary, spread, counts


# ---------------------------------------------------------------------------
# Video features — env build
# ---------------------------------------------------------------------------


def _build_render_env(cfg: DictConfig):
    """Build a MoSeqImitation env (arbitrary clips) for ghost-model rendering."""
    from c2a.experiments.run_labeled_hidden_dynamics_controlled import (
        _build_pose_env,
    )
    from c2a.experiments.shared.checkpoint_utils import load_moseq_checkpoint

    ckpt_path = _abs(cfg.checkpoint.path)
    ckpt_cfg = load_moseq_checkpoint(
        str(ckpt_path), step=cfg.checkpoint.get("step", None),
    )[0]
    # The checkpoint's baked codes_path may be stale; point it at the
    # analysis codes npz so prepare_config resolves cleanly.
    ckpt_cfg.kpms_config.codes_path = str(_abs(cfg.codes_path))

    clip_len = int(ckpt_cfg.env_config.clip_length)
    codes_npz = np.load(_abs(cfg.codes_path), allow_pickle=True)
    keep = np.asarray(codes_npz["test_indices"][:4], dtype=np.int64)
    test_codes = codes_npz["test_codes"][:4]
    code_stack = int(ckpt_cfg.network_config.get("code_stack_size", 1))
    env = _build_pose_env(
        ckpt_cfg, str(_abs(cfg.data.reference_data_path)), clip_len,
        keep_clips_idx=keep, test_codes=test_codes,
        code_stack_size=code_stack, probe_frame=0,
    )
    return env


# ---------------------------------------------------------------------------
# Feature 1 — video dendrogram
# ---------------------------------------------------------------------------


def _build_displayed_tree(
    Z: np.ndarray, code_ids: list[int], floor: float,
) -> dict:
    """Collapse below-floor subtrees; return the displayed tree root node.

    Node dict keys: kind ('merge'/'collapsed'/'leaf'), codes (list[int]),
    children (list of 0 or 2 nodes), height, x (leaf-order coord), y.
    """
    n = len(code_ids)

    def height(nid: int) -> float:
        return 0.0 if nid < n else float(Z[nid - n, 2])

    def kids(nid: int) -> tuple[int, int]:
        return int(Z[nid - n, 0]), int(Z[nid - n, 1])

    def leafset(nid: int) -> list[int]:
        if nid < n:
            return [nid]
        l, r = kids(nid)
        return leafset(l) + leafset(r)

    leaf_x = [0]

    def build(nid: int) -> dict:
        h = height(nid)
        if nid < n:
            x = leaf_x[0]
            leaf_x[0] += 1
            return {"kind": "leaf", "codes": [int(code_ids[nid])],
                    "children": [], "height": 0.0, "x": float(x), "y": 0.0}
        if h < floor:
            codes = [int(code_ids[l]) for l in leafset(nid)]
            x = leaf_x[0]
            leaf_x[0] += 1
            return {"kind": "collapsed", "codes": codes, "children": [],
                    "height": h, "x": float(x), "y": 0.0}
        l, r = kids(nid)
        cl, cr = build(l), build(r)
        return {"kind": "merge", "codes": cl["codes"] + cr["codes"],
                "children": [cl, cr], "height": h,
                "x": 0.5 * (cl["x"] + cr["x"]), "y": h}

    return build(2 * n - 2)


def _node_panel(env, node: dict, qpos_per_code: dict, vcfg: DictConfig) -> np.ndarray:
    """Render one displayed node as a (T, h, w, 3) video panel."""
    cap = int(vcfg.bodies_per_code)
    col_a = list(vcfg.child_color_a)
    col_b = list(vcfg.child_color_b)
    col_noise = list(vcfg.noise_color)

    agents: list[np.ndarray] = []
    colors: list[list[float]] = []

    def add(codes: list[int], color: list[float]) -> None:
        for c in codes:
            q = qpos_per_code[int(c)]
            for k in range(min(cap, q.shape[0])):
                agents.append(q[k])
                colors.append(color)

    if node["kind"] == "merge":
        add(node["children"][0]["codes"], col_a)
        add(node["children"][1]["codes"], col_b)
    else:
        add(node["codes"], col_noise)

    ghost_model, base_nq = build_uniform_walkers(
        env, walker_colors=colors,
        camera_distance=float(vcfg.camera_distance),
        camera_elevation=float(vcfg.camera_elevation),
        camera_azimuth=float(vcfg.camera_azimuth),
    )
    return render_panel_frames(
        ghost_model, base_nq, agents, camera_name="divergent_cam",
        width=int(vcfg.panel_w), height=int(vcfg.panel_h),
    )


def _composite_dendrogram(
    render_nodes: list[dict], panels: dict[int, np.ndarray], link_fn,
    vcfg: DictConfig, title: str, out_path: Path,
) -> None:
    """Composite node panels into one MP4 as a tidy, non-overlapping tree.

    ``render_nodes[0]`` must be the tree root. Layout is a uniform grid:
    row = depth from the root, column = subtree leaf-mean. Because every
    node occupies a distinct grid cell sized exactly one panel + gap, panels
    never overlap and space is used as tightly as the tree shape allows.
    Parent->child edges are drawn as orthogonal "elbow" connectors. Frames
    are streamed one canvas at a time to bound memory.
    """
    idx_set = {nd["idx"] for nd in render_nodes}
    kids = {
        nd["idx"]: [c for c in link_fn(nd) if c["idx"] in idx_set]
        for nd in render_nodes
    }
    root = render_nodes[0]

    depth: dict[int, int] = {}
    col: dict[int, float] = {}
    leafn = [0]

    def assign(nid: int, d: int) -> None:
        depth[nid] = d
        ch = kids[nid]
        if not ch:
            col[nid] = float(leafn[0])
            leafn[0] += 1
        else:
            for c in ch:
                assign(c["idx"], d + 1)
            col[nid] = sum(col[c["idx"]] for c in ch) / len(ch)

    assign(root["idx"], 0)

    T = min(panels[nd["idx"]].shape[0] for nd in render_nodes)
    ph, pw = panels[root["idx"]].shape[1:3]
    xgap, ygap, margin, hdr = 26, 30, 40, 26
    n_cols = max(leafn[0], 1)
    n_rows = max(depth.values()) + 1
    canvas_w = 2 * margin + n_cols * (pw + xgap) - xgap
    canvas_h = 2 * margin + hdr + n_rows * (ph + ygap) - ygap

    def center(nid: int) -> tuple[int, int]:
        cx = margin + int(col[nid] * (pw + xgap)) + pw // 2
        cy = margin + hdr + depth[nid] * (ph + ygap) + ph // 2
        return cx, cy

    base = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(base)
    draw.text((8, 8), title, fill=(20, 20, 20))
    # Orthogonal elbow connectors: parent bottom -> mid-gap -> child top.
    for nd in render_nodes:
        ch = kids[nd["idx"]]
        if not ch:
            continue
        pcx, pcy = center(nd["idx"])
        mid = pcy + ph // 2 + ygap // 2
        draw.line([(pcx, pcy + ph // 2), (pcx, mid)], fill=(110, 110, 110), width=3)
        for c in ch:
            ccx, ccy = center(c["idx"])
            draw.line([(pcx, mid), (ccx, mid)], fill=(110, 110, 110), width=3)
            draw.line([(ccx, mid), (ccx, ccy - ph // 2)],
                      fill=(110, 110, 110), width=3)
    base_arr = np.asarray(base, dtype=np.uint8)
    placed = [(center(nd["idx"]), nd["idx"]) for nd in render_nodes]

    def frames():
        for t in range(T):
            canvas = base_arr.copy()
            for (cx, cy), idx in placed:
                x0, y0 = cx - pw // 2, cy - ph // 2
                canvas[y0:y0 + ph, x0:x0 + pw] = panels[idx][t]
            yield canvas

    write_mp4(frames(), out_path, fps=int(vcfg.fps))
    log.info(f"  wrote {out_path} ({n_rows} rows x {n_cols} cols)")


def _render_video_dendrograms(
    env, pose: str, D: np.ndarray, code_ids: list[int], floor: float,
    qpos_per_code: dict, vcfg: DictConfig, out_full: Path, out_pruned: Path,
) -> None:
    """Feature 1: render the dendrogram with a video at each node.

    Produces two MP4s from the same rendered node panels:
      * ``out_full``   — every node, incl. below-floor noise clusters.
      * ``out_pruned`` — only the above-floor merge structure (narrower).
    Both use the tidy-tree grid layout (no overlaps).
    """
    Z = hierarchy.linkage(squareform(D, checks=False), method="average")
    root = _build_displayed_tree(Z, code_ids, floor)

    nodes: list[dict] = []

    def collect(nd: dict) -> None:
        nd["idx"] = len(nodes)
        nodes.append(nd)
        for ch in nd["children"]:
            collect(ch)

    collect(root)
    merges = [nd for nd in nodes if nd["kind"] == "merge"]
    log.info(
        f"  [{pose}] dendrogram: {len(nodes)} nodes, {len(merges)} merges, "
        f"floor={floor:.3f}"
    )

    panels: dict[int, np.ndarray] = {}
    for nd in nodes:
        frames = _node_panel(env, nd, qpos_per_code, vcfg)
        if nd["kind"] == "merge":
            txt = f"merge h={nd['height']:.2f} ({len(nd['codes'])} codes)"
        elif nd["kind"] == "collapsed":
            head = ",".join(str(c) for c in nd["codes"][:6])
            txt = f"noise cluster: {head}{'+' if len(nd['codes']) > 6 else ''}"
        else:
            txt = f"code {nd['codes'][0]}"
        panels[nd["idx"]] = np.stack([
            add_banner(f, txt, banner_h=24, bg=(245, 245, 245),
                       fg=(30, 30, 30), font_size=13)
            for f in frames
        ], axis=0)
        log.info(f"    [{pose}] node {nd['idx'] + 1}/{len(nodes)} rendered")

    # Full: every node (tidy tree, depth rows).
    _composite_dendrogram(
        nodes, panels, lambda nd: nd["children"], vcfg,
        f"{pose}: video dendrogram (full)", out_full,
    )

    # Pruned: only above-floor merge nodes.
    if merges:
        _composite_dendrogram(
            merges, panels,
            lambda nd: [c for c in nd["children"] if c["kind"] == "merge"],
            vcfg, f"{pose}: video dendrogram (above noise floor)", out_pruned,
        )
    else:
        log.warning(
            f"  [{pose}] no above-floor merges; skipping pruned dendrogram."
        )


# ---------------------------------------------------------------------------
# Feature 2 — distinct codes by behavior grid
# ---------------------------------------------------------------------------


def _render_behavior_grid(
    env, pose: str, D: np.ndarray, code_ids: list[int], floor: float,
    qpos_per_code: dict, spread_counts: np.ndarray, usage_counts: np.ndarray,
    vcfg: DictConfig, out_path: Path,
) -> None:
    """Feature 2: 4-column (behavior) grid of distinct-code rollout videos."""
    Z = hierarchy.linkage(squareform(D, checks=False), method="average")
    flat = hierarchy.fcluster(Z, t=floor, criterion="distance")

    # One representative (highest usage) per noise-floor cluster.
    clusters: dict[int, list[int]] = {}
    for row, cl in enumerate(flat):
        clusters.setdefault(int(cl), []).append(int(code_ids[row]))
    reps = [
        max(codes, key=lambda c: usage_counts[c]) for codes in clusters.values()
    ]

    n_beh = len(BEHAVIORS)
    # Base-rate correction: assign each code to the behavior it is most
    # ENRICHED for (lift = P(behavior|code) / P(behavior)), not raw argmax.
    # Immobile dominates the corpus frame count, so raw argmax would funnel
    # nearly every code into the Immobile column and leave the others empty.
    beh_p = spread_counts.sum(axis=1).astype(np.float64)
    beh_p = beh_p / max(beh_p.sum(), 1.0)
    cols: list[list[int]] = [[] for _ in range(n_beh)]
    for c in reps:
        cc = spread_counts[:, c].astype(np.float64)
        if cc.sum() <= 0:
            continue  # unlabeled — skip
        lift = (cc / cc.sum()) / np.maximum(beh_p, 1e-9)
        cols[int(np.argmax(lift))].append(c)
    for b in range(n_beh):
        cols[b].sort(key=lambda c: -usage_counts[c])
    max_rows = max((len(c) for c in cols), default=0)
    log.info(
        f"  [{pose}] behavior grid: {len(reps)} distinct codes -> "
        f"{[ (BEHAVIORS[b], len(cols[b])) for b in range(n_beh) ]}"
    )
    if max_rows == 0:
        log.warning(f"  [{pose}] no distinct codes to grid; skipping.")
        return

    cap = max(1, int(vcfg.bodies_per_code) + 2)
    panels: list[np.ndarray | None] = []
    for r in range(max_rows):
        for b in range(n_beh):
            if r >= len(cols[b]):
                panels.append(None)
                continue
            c = cols[b][r]
            q = qpos_per_code[c]
            agents = [q[k] for k in range(min(cap, q.shape[0]))]
            colors = [_BEHAVIOR_COLORS[b]] * len(agents)
            ghost_model, base_nq = build_uniform_walkers(
                env, walker_colors=colors,
                camera_distance=float(vcfg.camera_distance),
                camera_elevation=float(vcfg.camera_elevation),
                camera_azimuth=float(vcfg.camera_azimuth),
            )
            frames = render_panel_frames(
                ghost_model, base_nq, agents, camera_name="divergent_cam",
                width=int(vcfg.panel_w), height=int(vcfg.panel_h),
            )
            fg = tuple(int(255 * x * 0.6) for x in _BEHAVIOR_COLORS[b][:3])
            panels.append(np.stack([
                add_banner(f, f"code {c} [{BEHAVIORS[b]}]", banner_h=20,
                           bg=(245, 245, 245), fg=fg, font_size=11)
                for f in frames
            ], axis=0))

    real = [p for p in panels if p is not None]
    T = min(p.shape[0] for p in real)
    h, w = real[0].shape[1:3]
    blank = np.full((T, h, w, 3), 255, dtype=np.uint8)
    panels = [p[:T] if p is not None else blank for p in panels]

    grid = stack_panels_into_grid(
        panels, rows=max_rows, cols=n_beh,
        grid_title=f"{pose}: distinct codes by behavior  "
                   f"(cols: {' | '.join(BEHAVIORS)})",
        grid_banner_h=40,
    )
    write_mp4(grid, out_path, fps=int(vcfg.fps))
    log.info(f"  [{pose}] wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="code_distinctness_exp",
)
def main(cfg: DictConfig) -> None:
    log.info("=== Code Distinctness experiment (P1 + P2 + P3) ===")
    rng = np.random.RandomState(int(cfg.get("seed", 0)))

    out_base = _abs(cfg.output.base_dir) / str(cfg.output.run_name)
    data_dir = out_base / "data"
    merged_dir = out_base / "merged"
    dist_dir = out_base / "distinctness"
    dist_dir.mkdir(parents=True, exist_ok=True)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"No rollout data dir at {data_dir} — run run_code_grid_rollouts first."
        )

    strip_root = bool(cfg.get("strip_root_pos", True))
    n_null = int(cfg.get("n_null_samples", 200))
    summary: dict = {"run": str(out_base), "poses": {}}
    pose_cache: dict[str, dict] = {}

    for pose in list(cfg.poses):
        dist_path = merged_dir / f"distance_{pose}.npz"
        if not dist_path.exists():
            log.warning(f"[{pose}] no {dist_path.name}; skipping P1/P2.")
            continue
        d = np.load(dist_path, allow_pickle=True)
        D = np.asarray(d["distance"], dtype=np.float32)
        all_code_ids = [int(c) for c in d["code_ids"].tolist()]
        sigma = float(d["sigma"])

        qpos_per_code: dict[int, np.ndarray] = {}
        missing = []
        for c in all_code_ids:
            f = data_dir / f"rollout_{pose}_code{c:03d}.npz"
            if not f.exists():
                missing.append(c)
                continue
            qpos_per_code[c] = np.load(f)["qpos"]
        if missing:
            log.warning(
                f"[{pose}] missing {len(missing)} rollout npz(s): {missing[:8]}"
            )
        code_ids = [c for c in all_code_ids if c in qpos_per_code]
        if len(code_ids) < 2:
            log.warning(f"[{pose}] <2 codes with rollouts; skipping.")
            continue
        keep = [i for i, c in enumerate(all_code_ids) if c in qpos_per_code]
        D = D[np.ix_(keep, keep)]

        log.info(f"[{pose}] P1: {len(code_ids)} codes, sigma={sigma:.4g}")
        dists = _p1_distances(
            qpos_per_code, D, code_ids, sigma, strip_root, n_null, rng,
        )
        p1 = _plot_p1(dists, pose, dist_dir / f"P1_within_between_{pose}.png")

        floor = float(np.nanmedian(dists["within"]))
        _plot_p2(D, code_ids, floor, pose, dist_dir / f"P2_phylogeny_{pose}.png")
        log.info(
            f"[{pose}] within/between={p1['within_over_between']:.3f} "
            f"within/null={p1['within_over_null']:.3f}"
        )
        summary["poses"][pose] = p1
        pose_cache[pose] = {
            "D": D, "code_ids": code_ids, "floor": floor,
            "qpos": qpos_per_code,
        }

    log.info("P3: NMI(code ; ethogram) ...")
    p3_summary, spread_counts, usage_counts = _p3(cfg, dist_dir, rng)
    summary["P3"] = p3_summary
    log.info(
        f"P3: NMI={p3_summary['nmi_code_ethogram']:.3f} "
        f"(null {p3_summary['nmi_shuffled_null_mean']:.3f}), "
        f"effective codes={p3_summary['effective_code_count']:.1f}"
    )

    with (dist_dir / "distinctness_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # --- Video features (Features 1 & 2) ---
    if bool(cfg.get("render_videos", False)):
        if not pose_cache:
            log.warning("render_videos set but no pose data; skipping videos.")
        else:
            log.info("\n=== Rendering video features ===")
            vcfg = cfg.video
            env = _build_render_env(cfg)
            vid_dir = dist_dir / "videos"
            vid_dir.mkdir(parents=True, exist_ok=True)

            feats = list(vcfg.get("features", ["dendrogram", "grid"]))
            want = str(vcfg.get("pose", "walk"))
            video_poses = (
                ["walk", "rear"] if want == "both" else [want]
            )
            if "dendrogram" in feats:
                for pose in video_poses:
                    if pose not in pose_cache:
                        log.warning(f"[{pose}] no pose data; skipping dendrogram.")
                        continue
                    pc = pose_cache[pose]
                    _render_video_dendrograms(
                        env, pose, pc["D"], pc["code_ids"], pc["floor"],
                        pc["qpos"], vcfg,
                        vid_dir / f"F1_video_dendrogram_{pose}.mp4",
                        vid_dir / f"F1b_video_dendrogram_pruned_{pose}.mp4",
                    )

            if "grid" in feats:
                for pose in video_poses:
                    if pose not in pose_cache:
                        log.warning(f"[{pose}] no pose data; skipping grid.")
                        continue
                    pc = pose_cache[pose]
                    _render_behavior_grid(
                        env, pose, pc["D"], pc["code_ids"], pc["floor"],
                        pc["qpos"], spread_counts, usage_counts, vcfg,
                        vid_dir / f"F2_distinct_by_behavior_{pose}.mp4",
                    )

    log.info(f"=== Done. Artifacts under {dist_dir}/ ===")


if __name__ == "__main__":
    main()
