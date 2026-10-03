# Project-page assets (`c2a/website/`)

Scripts that produce every video and data file on the Code2Act project page
(`Code2Act-website/`). Videos are rendered from the rollouts behind the
manuscript's figures; only the sequence viewer runs new rollouts, with the
paper's MLB controller (step 208) and the §4.4 rollout function.

| Script | Produces | Paper figure |
|---|---|---|
| `render_videos.py atlas` | `public/media/atlas/<strategy>/codeNNN.mp4`, `src/data/atlas.json` | Fig 2c, 11, 12 |
| `render_videos.py roundtrip` | `public/media/roundtrip/<symbol>.mp4`, `src/data/roundtrip.json` | Fig 3a |
| `render_videos.py kick` | `public/media/kick/<strategy>_<behavior>.mp4`, `src/data/kick.json` | Fig 8 |
| `render_videos.py pairs` | `public/media/pairs/<A>__<B>.mp4`, `src/data/pairs.json` | Fig 13 |
| `run_sequences.py` | `public/media/sequences/*.mp4`, `public/data/sequence_*.json` | sequence viewer |
| `export_data.py` | `src/data/{roundtrip_bars,perturbation,transitions}.json`, `public/data/hidden3d.json` | Fig 3a/6, 3b,c/7, 13, 2b |

`render_lib.py` holds the shared renderer: the paper's arena, palette and
lighting, a live body with a fading ghost trail and root trace, a camera that
follows the body, and H.264 encoding.

## Running

```bash
C2A_ROOT=$PWD python -m c2a.website.render_videos atlas roundtrip kick pairs streams
C2A_ROOT=$PWD python -m c2a.website.run_sequences
C2A_ROOT=$PWD python -m c2a.website.export_data
```

Rendering needs MuJoCo's EGL backend. If `import mujoco.egl` fails with
`'NoneType' object has no attribute 'eglQueryString'`, the NVIDIA EGL driver is
installed but the GLVND dispatcher (`libEGL.so.1`) is not on the library path.
Point `LD_LIBRARY_PATH` at a directory containing `libEGL.so.1`,
`libGLdispatch.so.0`, `libGL.so.1` and `libOpenGL.so.0` from a GLVND build
(e.g. a conda env with `libglvnd`).

## Body selection

Bodies are chosen by rule, never by eye:

- atlas: among bodies whose labeler readout matches the code's modal label,
  the one with the median path length;
- round trip: a seeded random draw from the 60 scored bodies, with readouts
  computed by `run_code_roundtrip`'s own rule;
- kick (MLB only): per impulse size, the best-recovering of 48 bodies. Rear: among
  bodies that took at least a median-sized shove, the smallest final posture gap to
  the unperturbed twin relative to its peak. Walk: among bodies (and twins) moving in
  at least 80% of the last 3 s, the closest final pace to the twin;
- pairs: among bodies the labeler reads as A then B with no bridge behavior,
  the best summed rank of per-phase clarity (rear height, walk speed, turn yaw,
  immobile stillness);
- sequences: the body with the highest command/readout agreement of 48 (the
  median is recorded alongside).
