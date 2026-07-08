# Presentation demo videos (`c2a/demos/`)

Video renders for talks/presentations, distinct from the paper figures
(`c2a/figures/`) and the analysis experiments (`c2a/experiments/`). Every demo
is parameterized by **steering strategy** via `c2a/demos/strategies.py`
(`STRATEGIES` / `STRATEGY_ORDER`), so **mlb drops in automatically** once its
checkpoint is trained (until then its data paths are `None` and the demos
TBD-placeholder / skip it). All outputs go to `outputs/demos/<demo>/`.

Run from the repo root with the venv shim, e.g.:

```bash
C2A_ROOT="$PWD" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python3.12 bash_scripts/_pyrun.py c2a.demos.<module> --strategy kpms
```

## Aesthetic standard

Body-grid demos render in the **usual environment background** — the default
dark-blue checkerboard arena `vnl_playground.tasks.rodent.consts.ARENA_XML_PATH`
(the one `build_parade_trail_model` uses, as seen in the prior project videos
like `behavior_parade_side.mp4`). The pure-white `white_arena.xml` was tried and
rejected (flat/weird). Bodies are overlaid walker copies (the single arena
choice lives in `demo_render.build_white_overlay_model`). Body color is the
grid_gt **teal** `[30,128,146]` (clearly visible on the dark floor); coloring is
demo-specific (teal→red→teal phase for kick-recovery; 9-class for the parade /
hidden-dynamics).

## Demos

1. **`kick_recovery_demo.py`** — per strategy, a 2-panel video (recover-from-rear
   | recover-from-walk), each panel **5 bodies** animated through the velocity
   impulse, **teal→red→teal** phase coloring (held/recovered = grid_gt teal so the
   body is visible on white; impulse = red), white arena. Window the clip wide
   (~300 frames of the held code before the kick, ~500 after) so the behavior is
   visible at the start and end. Data: `perturbation_dir`.
2. **`parade_motion_demo.py`** — kpms + kvae **side by side**, the body executing
   walk→immobile→rear, showing **both the motion and the gait-stance strip**,
   9-class colored. Data: `parade_dir`.
3. **`hidden_dynamics_demo.py`** — per strategy, the **most-common code per
   directional category**, shown two ways together: (a) a **20-body** white-arena
   panel per code (20 instantiations of that code from different initial
   conditions, via `demo_render`), and (b) all those codes' within-syllable
   hidden-state trajectories in **one shared PCA space** (single joint basis, same
   plot), colored by **9-class directional category** (Walk/Rear L/R/S, Turn L/R,
   Immobile) so directionality is visible in the hidden dynamics. Reuses
   `fit_joint_pca` / `render_grouped_rotating_pca_video` from
   `c2a/experiments/shared/rotating_pca_video.py`. Data: `dynamics_dir` +
   `rollouts_dir`.
4. **`code_atlas_demo.py`** — video version of the code-atlas figures (Figs
   14/15/16): per strategy, a grid of codes, each cell **5 bodies** holding that
   code simultaneously, clean arena. Data: `rollouts_dir`.

## Status

kpms + kvae implemented; **mlb pending its checkpoint** (add nothing — the
scripts pick it up from `strategies.py` once `policy_ckpt`/data dirs are set).
