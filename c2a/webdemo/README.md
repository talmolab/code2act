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

# Verified sequences (see below). Results are already in verified_chains.json;
# rerun only if the controller, palette or start poses change.
# Start-pose screen: all 9 symbols, depth 3, all 48 clips (scr9_0, scr9_201, scr9_202).
.venv-webdemo/bin/python -m c2a.webdemo.chain_tree scr9_$s --noise-seed $s --phys --hold 400 \
    --depth 3 --palette Walk_L,Walk_R,Walk_S,Rear_L,Rear_R,Rear_S,Turn_L,Turn_R,Immobile
CL=20,4,44,24,18,7,30,41,25,17   # the 10 clips with the most passing 3-symbol trees
# Per-clip trees: 17 replicas per clip, pruned, allowed switches only (~50 min on 64 cores).
.venv-webdemo/bin/python -m c2a.webdemo.chain_tree srch8 --search --hold 400 --depth 8 --clips $CL
#   + held-out TS replay of 3,000 offered pairs ({"hold": 400, "pairs": ...} job):
#   PHYS=1 VERIFY_JOB=.../ts_srch8/job.json SHARD=i NSHARD=48 OUT=... npx vitest run tests/verify_chains.test.ts
C2A_ROOT=$PWD python -m c2a.webdemo.chain_select phys-ts ts_srch8
C2A_ROOT=$PWD python -m c2a.webdemo.chain_select verify-clip srch8 --clips $CL --held ts_srch8
C2A_ROOT=$PWD python -m c2a.webdemo.export_assets --sequences-only     # presets + verified chains -> symbols.json

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
| `chain_tree.py`, `chain_select.py`, `verified_chains.json` | The exhaustive chain search behind the demo's verified sequences. |

## Demo palette and presets

The demo offers 6 of the 9 MLB symbols (`PALETTE` in `export_assets.py`):
Walk L, Walk R, Rear L, Rear R, Turn R and Immobile. A symbol can fail by
losing control or by producing a different behavior. Per symbol, 4 s holds,
alone or after any other symbol, 48 clips, MuJoCo C (runs `scr9_*` for
control, `beh9` + TopoMimic behavior labels for behavior):

| Symbol | Loses control | Commanded behavior |
|---|---|---|
| Walk L / R / S | 2.3 / 2.3 / 1.6% | 82 / 90 / 84% |
| Rear L / R / S | 2.6 / 2.1 / 3.0% | 49 / 64 / 59% |
| Turn L / R | 2.8 / 2.8% | **6%** (walks left 88%) / 54% |
| Immobile | 3.9% | 51% (walks 44%) |

- Turn L is left out: it is performed as Walk L. Walk S and Rear S would
  duplicate Walk L/R and Rear L/R (the labeler cannot separate rear direction).
- Physically the symbols are alike; held alone for 20 s (`long20`), Rear R
  never lost control (0/48).
- **Most wrong behaviors come from the switch.** Held alone, rears rear 85–90%
  and Turn R turns 100%; right after a walk, 20–29% and 22%. Rear needs the rat
  nearly still at the switch: 83–89% below 0.1 m/s, 8–13% above 0.2 m/s
  (after a walk: 67% if slow, 11% if moving). An Immobile in between does not
  fix it: Walk → Immobile → Rear rears 31–46% (Immobile itself often keeps
  walking), and it hurts where Rear already works.

`DEMO_PRESETS` are 4 five-symbol chains, each with the start pose it plays
from. Each is verified from that pose and was picked because every block showed
the commanded behavior in 17 fresh MuJoCo C replicas (15/17 for "Rears, then
zigzag") and in a replay through the TS runner. The tour was chosen from 360
candidates that use all four behaviors. Longer examples were tried: no
8-symbol chain had every block right in more than 9 of 17 replicas, and
Walk → Immobile → Rear chains reared in 12–29%.

## Verified sequences (2026-10-15, 4 s holds, up to 8 symbols)

The demo only lets users build sequences that were checked, **separately for
each start pose**. The palette lights a symbol only if the sequence so far plus
that symbol is in `verified_chains.json` for the chosen clip.

**Switches.** The 9 switches in `chain_select.WRONG_BEHAVIOR` are never
offered: the commanded behavior showed up in fewer than half of the runs
(Walk L/R → Rear L/R, Walk L/R → Turn R, Walk L → Immobile, Rear L →
Immobile, Turn R → Rear L; 6–44%). They are also physically riskier. Once
the rat walks, only the other walk (or Walk R → Immobile) follows.

**Control.** A chain × replica fails if, at any point of the whole chain
(`physical_failure` in `chain_select.py`):
- **crazy:** root speed over one control step exceeds 2.5 m/s, or the state
  is not finite. Normal motion stays under ~1 m/s. Rendered examples above
  5 m/s were the body flying or tumbling; 1.5–2 m/s were the rat toppling out
  of a tall rear, which is not counted (if it then lies frozen, sinking
  catches it). A 1.5 m/s cutoff made Rear look like the weakest symbol;
- **sinking:** over some 1 s window, mean root height < -1 cm and mean joint
  speed < 0.1 rad/s (a frozen, hunched crouch near the floor; 2% of ordinary
  still Immobile windows are that low);
- **flipped:** upside down (up-axis z < -0.5) below 3 cm for 0.25 s (never
  triggered).

**Switching, not holding, is what breaks control.** Per 2.5 s window, among
runs still fine at its start, holding one symbol fails 0.3–2.1% for the first
12.5 s, while switching every 2.5 s rises to 6.1% in 10–12.5 s (after 4
switches). The risk builds over the chain; this is why each symbol is held 4 s.

1. **Search** (`chain_tree.py --search`): per start pose (20, 4, 44, 24, 18, 7,
   30, 41, 25, 17: most passing 3-symbol trees in the 48-clip screen), a batch
   of 17 replicas (clean + 16 with N(0, 1e-4) action noise, seeded by clip,
   chain and segment, so a prefix replays identically in any job; checked).
   A chain is kept if no replica loses control, and only then extended, over
   the 21 allowed switches, up to 8 symbols (32 s). Prefixes are shared
   through bit-exact state copies.
2. **Per-clip trees** (`chain_select verify-clip`): the kept chains, prefix
   closed. 219–645 chains per clip (4,283 pairs); all 6 first symbols and
   20–21 of the 21 two-symbol chains on every clip; 3–123 eight-symbol chains
   per clip. Rear after a walk is reachable only via Walk R → Immobile.
3. **Held-out estimate:** the TS runner replayed 3,000 random offered pairs
   (`ts_srch8`, not used for selection): 95 lost control, 3.2%, about 1 in 32.
   By length: 0% up to 3 symbols, 1.9–3.1% at 4–6, 6.6–7.3% at 7–8.
   Chains that just pass 17 replicas are selected for luck, so deep ones fail
   more in fresh runs.

Earlier versions: the labeler-only criterion (2026-10-12) kept 93–169 chains
per clip; physical-only at 2.5 s holds (`phys_*`) 2,320–2,866 chains, 1.4%
held-out failure, too fast to watch; physical-only at 4 s with all switches
(`h400_*`, 17 separate runs) 399–725 chains, 3.1%. Runs are in
`outputs/webdemo/chain_tree/`.

## Fragile switches (2026-10-10, MuJoCo C, 48 start poses)

Labeler-based. These numbers backed the timeline warnings of an earlier web
app version, and the switches left out of the labeler-based chain search.
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
