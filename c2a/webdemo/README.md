# In-browser control demo (`c2a/webdemo/`)

Exports the paper's MLB controller (`model_checkpoints/260629_212713_669229_step208`)
and its scene so it can run live in a browser (MuJoCo WASM + onnxruntime-web).
The web app is a separate repo, `code2act-control-website`, cloned into
`code2act/code2act-control-website/` (gitignored here). It is deployed as a
static Cloudflare Worker.

All outputs go to `outputs/webdemo/` (gitignored).

## Pipeline

```bash
# 1. Project venv (JAX): scene, weights, start poses, recorded MJX rollout, metadata.
C2A_ROOT=$PWD python -m c2a.webdemo.export_assets

# 2. Light venv (no JAX). One-time setup:
#    uv venv .venv-webdemo --python 3.12
#    uv pip install --python .venv-webdemo/bin/python mujoco==3.13.0 onnx==1.17.0 onnxruntime==1.20.1 "numpy<2.3"
.venv-webdemo/bin/python -m c2a.webdemo.build_onnx        # controller.onnx + manifest.json
.venv-webdemo/bin/python -m c2a.webdemo.parity            # ONNX / MuJoCo-C vs recorded MJX
.venv-webdemo/bin/python -m c2a.webdemo.sim_gap           # website presets x 48 starts in MuJoCo C
C2A_ROOT=$PWD python -m c2a.webdemo.score_sim_gap         # labeler agreement + falls vs MJX
.venv-webdemo/bin/python -m c2a.webdemo.sim_gap --set demo   # the demo's own presets
C2A_ROOT=$PWD python -m c2a.webdemo.score_sim_gap --set demo
.venv-webdemo/bin/python -m c2a.webdemo.export_fixtures   # fixture for the web app's tests

# 3. Copy into the web app.
cd code2act-control-website && node scripts/sync_assets.mjs ../outputs/webdemo
```

| File | Role |
|---|---|
| `export_assets.py` | Builds the eval env exactly as `c2a.website.run_sequences` does, then writes:<br>• the compiled scene XML, with solver options baked in and checked field by field against the env model<br>• the action-path weights and proprio normalizer<br>• the 48 start poses<br>• a recorded MJX rollout<br>• the PCA basis<br>• per-symbol metadata |
| `build_onnx.py` | Writes the ONNX graph node by node from the Flax params: code embedding, 4 `GRUCell`s, VAE head (z = mu) and frozen MIMIC decoder. Action = `tanh(loc)`. |
| `c_sim.py` | **Executable spec for the browser.** It builds the 277-d proprioception from plain `MjData` and runs the rollout loop. `src/sim/observation.ts` and `src/sim/runner.ts` in the web app port it line for line. |
| `parity.py` | Checks the network against JAX and the proprioception at reset. Runs one control step and a closed loop against the recorded MJX rollout. |
| `sim_gap.py`, `score_sim_gap.py` | The go/no-go behavioral comparison between MuJoCo C and MJX. |

## Demo palette and presets

The demo offers 6 of the 9 MLB symbols (`PALETTE` in `export_assets.py`):
Walk L, Walk R, Rear L, Rear R, Turn R and Immobile. The other three are left
out based on the round-trip summary:
- Walk S and Rear S are read as Walk L and Rear R (exact recall 0.00 and 0.07).
- Turn L is almost never performed (0.02).

`DEMO_PRESETS` uses only these six symbols and only switches the pair
experiment reaches reliably. The scores below are from MuJoCo C, 48 start
poses, median / mean labeler agreement:

| Preset | Agreement | Falls |
|---|---|---|
| tour | 0.83 / 0.71 | 0 |
| unseen (4 switches with zero corpus count) | 1.00 / 0.88 | 0 |
| quick (1 s holds) | 0.75 / 0.67 | 0 |

The project page's presets, which use all 9 symbols, scored 0.50 / 0.57 / 0.50
medians under the same conditions.

## Fragile switches (2026-10-10, MuJoCo C, 48 start poses)

These numbers back the timeline warnings in `src/switches.ts` of the web app.
The scripts were ad hoc; each test replays `c_sim.rollout` with the listed
blocks.

| Sequence | Outcome |
|---|---|
| Walk R → Rear R (direct) | 42% of walking bodies rear |
| Walk R → Immobile 0.5 / 1 / 2 s → Rear R | 84 / 89 / 82% rear |
| Walk L → Rear L, direct vs. via Immobile 1 s | 27% vs. 89% rear |
| Rear R → Walk R / Rear L → Walk L / Rear R → Walk L | 96 / 88 / 69% walk |
| Rear R → Turn R 1 s → Walk L | 100% walk |
| Walk R → Immobile → Rear R → Walk L (rear began from a stop) | 38% walk, 33% frozen |
| …same with Turn R 1 s before the walk | 83% walk |
| Walk L → Immobile 0.5–3 s → Walk R | 73–77% walk, independent of hold length |

Two further findings:
- **Freezing is sticky.** Bodies already stalled during a walk walk again only
  7–14% of the time afterwards.
- **Turn R is the best unsticker.** Branching the 19 frozen bodies of the demo
  tour at 15 s, Turn R gets 79% moving again, against 16–32% for the walks and
  21–26% for the rears.

## Env details the browser must reproduce

- **Actuator slot:** the "actuator_ctrl" proprio slot is `qfrc_actuator` (nv = 73), not `ctrl`.
- **Actuator dynamics:** actuators have filter dynamics (`dyntype=filter`, na = 38), so the force depends on `data.act`.
- **Lagged action:** the observation's "prev_action" is the action from **two** control steps ago. The env sets `prev_action = info["action"]` before storing the new action.
- **No extra `mj_forward`:** observations are read straight after `mj_step`, as MJX does.
- **Code window:** the next 5 *control steps*, padded with the last symbol. This is `windows_for` / `run_rollout`, the convention behind every paper result. Training stacked the next 5 *50 Hz frames* instead; both behave the same in `sim_gap`.
- **Reset:** reference qpos, qvel = 0, zero carry, zero actions. There is no termination; every paper rollout uses `ignore_done=True`.

## Verification (2026-10-09)

**Network**
- ONNX vs JAX on CPU at `highest` matmul precision: max |Δaction| 1.3e-6.
- The 3e-3 gap against the GPU-recorded rollout is TF32 on the A40.

**Engine**
- MuJoCo WASM 3.13.0 vs native MuJoCo C 3.13.0: |Δqpos| ≤ 3e-14 over 200 control steps.
- Cost is 0.31 ms per `mj_step` in Node.

**MuJoCo C vs MJX** (48 start poses, website presets)

| Preset | MJX agreement, median / mean | C agreement, median / mean | Falls ≥ 1 s |
|---|---|---|---|
| tour | 0.50 / 0.43 | 0.50 / 0.45 | 0 / 0 |
| unseen | 0.57 / 0.49 | 0.57 / 0.49 | 0 / 0 |
| quick | 0.50 / 0.49 | 0.50 / 0.48 | 0 / 0 |

**Browser**
- The TS port vs `c_sim` over the first 0.5 s: |Δqpos| 4e-6, |Δaction| 4e-5. Contact chaos then amplifies the ~1e-7 kernel rounding.
- In headless Chromium with 4 threads it runs in real time at 2.4–6 ms per 10 ms control step.
