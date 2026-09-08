# Unitree G1 Tabletop Teleoperation in MuJoCo

A fixed-base (welded) Unitree G1 humanoid standing next to a table in MuJoCo, teleoperated
from a Meta Quest 3S to pick, place, and stack bricks — with the first-person head camera
streamed to the headset and per-episode demo recording. Both arms following the controllers
(position) is confirmed on real hardware, along with a scripted pick-lift-hold sequence;
faithful 6-DOF orientation follow (via the mink weighted IK) and brick stacking are in
progress.

<p float="left">
  <img src="https://img.shields.io/badge/MuJoCo-%3E%3D3.2-blue" alt="mujoco>=3.2">
</p>

## What's here

| File | Purpose |
|---|---|
| [make_fixed_base.py](make_fixed_base.py) | Generates a fixed-base G1 MJCF from a stock Menagerie G1 (welds the pelvis, trims the keyframe, adds an FPV camera). |
| [g1_fixed_upper.xml](g1_fixed_upper.xml) | Reference copy of the generated fixed-base G1 (currently: Dex3-1 3-finger hands). |
| [scene_fixed_table.xml](scene_fixed_table.xml) | Full scene: robot + floor + table + three bricks, plus the `stand_at_table` reset keyframe. |
| [stand_next_to_table.py](stand_next_to_table.py) | Launcher — opens the viewer next to the table. `--pick {brick1,brick2,brick3,none}` drives a live scripted pick-lift-hold sequence (default `brick1`); `--pick none` just holds stance; `--teleop` drives both arms live from a VR headset instead (`--teleop-weighted-ik` + `--teleop-orientation` for faithful 6-DOF follow via the **mink** weighted IK, `--display-mode ego\|immersive` to stream the head camera to the headset, `--record-episodes DIR` + `s` in the viewer to record demos). See "Teleop loop" below. |
| [actuator_groups.py](actuator_groups.py) | Shared `LEG`/`WAIST`/`LEFT_ARM`/`LEFT_HAND`/`RIGHT_ARM`/`RIGHT_HAND`/`UPPER_BODY` qpos/ctrl slice constants. |
| [arm_ik.py](arm_ik.py) | Damped-least-squares position (+ optional orientation) IK for one arm, with a nullspace posture bias. Used by everything scripted. |
| [weighted_arm_ik.py](weighted_arm_ik.py) | `WeightedArmIK` — a **parallel, teleop-only** weighted IK for one arm, default backend **mink** ([kevinzakka/mink](https://github.com/kevinzakka/mink), a MuJoCo-native differential-IK QP solved against the same `MjModel`). Same `sync()`/`solve()`/`site_pos()` surface as `ArmIK`; `TeleopController` opts in via `--teleop-weighted-ik` (default off — `arm_ik.py` and every scripted path untouched). The fix for `--teleop-orientation`: faithful gripper rotation on all three axes with a stable non-moving arm, where the DLS solver overshoots wildly. |
| [approach_path.py](approach_path.py) | Re-targetable up/over/down approach path (`ApproachPath`) — any start → any goal, advanced one control step at a time. |
| [verify_arm_ik.py](verify_arm_ik.py) | Headless verification for the IK solver + approach path. |
| [grasp_primitive.py](grasp_primitive.py) | Dex3-1 hand shape (open/pre-grasp/closed via a grip scalar) and the grasp approach orientation. |
| [verify_grasp_primitive.py](verify_grasp_primitive.py) | Headless check that approach + orient + close-on-empty-space doesn't blow up, for all three pick targets. |
| [verify_grasp_hold.py](verify_grasp_hold.py) | The real grasp test: approach + orient + close + lift 20cm + hold 3-5s. Passes for `brick1`/`brick2`; `brick3` is a known open issue. |
| [grasp_state.py](grasp_state.py) | Grasp-state detection — is the object actually grasped, from aperture stall + gripper-to-object proximity (no privileged contact reads). |
| [verify_grasp_state.py](verify_grasp_state.py) | Calibrates and verifies grasp detection against grasped/empty/dropped/long-hold reference scenarios. |
| [verify_approach_obstacles.py](verify_approach_obstacles.py) | Verifies obstacle-aware `safe_z` geometry, and honestly reports (doesn't assert) how well it holds up in real simulation — see CLAUDE.md's "Collision Avoidance". |
| [pick_sequence.py](pick_sequence.py) | `PickSequence` — assembles arm IK + approach path + grasp primitive + grasp-state detection into one incremental, per-frame-advanceable pick-lift-hold state machine, plus an optional pick-and-**place** extension. This is what `stand_next_to_table.py` now drives live. |
| [verify_pick_sequence.py](verify_pick_sequence.py) | Headless regression check that `PickSequence`, driven one `.step()` per physics step (matching the live loop), reproduces `verify_grasp_hold.py`'s results. |
| [stack_sequence.py](stack_sequence.py) | `StackSequence` — orchestrates pick-and-place cycles to build a stack on `brick1`. **In progress — major placement-collision cause found and fixed, a smaller landing-precision issue remains** — see CLAUDE.md's "Stacking Sequencing". |
| [verify_stack_sequence.py](verify_stack_sequence.py) | Headless regression check for `StackSequence`; currently documents (and is expected to hit) the open placement failure rather than hiding it. `--record [OUTPUT.mp4]` saves an offscreen video of the run. |
| [teleop_control.py](teleop_control.py) | `TeleopController` — continuous, per-frame two-arm VR-follow control (Phase 2's live-input counterpart to `PickSequence`), via the `televuer` package. Also `FpvStreamer` (streams the `fpv_teleop` camera to the headset) and `HandRetargeter` (real per-finger control via `dex_retargeting`, `--hand-tracking`). **Position tracking + FPV streaming confirmed on a real Quest 3S**; the mink weighted IK for orientation is sim-verified only — see CLAUDE.md's "Teleop Input (Phase 2)". |
| [verify_teleop_control.py](verify_teleop_control.py) | Headless regression check for `TeleopController` against a synthetic input source — verifies the control-loop wiring, not the coordinate mapping against a real headset. |
| [dex3_retargeting/](dex3_retargeting/) | Vendored, adapted Dex3-1 retargeting config + URDFs for `HandRetargeter` (see CLAUDE.md's "Teleop Input" for what was adapted and why). |
| [verify_hand_retargeting.py](verify_hand_retargeting.py) | Headless regression check for hand-tracking mode against synthetic keypoints; skips gracefully if `dex_retargeting` isn't installed. |

## Requirements

- Python 3
- MuJoCo ≥ 3.2:
  ```bash
  pip install -U "mujoco>=3.2" numpy imageio
  ```
- For `--teleop` only (a real VR headset, e.g. Meta Quest 3S):
  ```bash
  pip install televuer   # see https://github.com/unitreerobotics/televuer for SSL cert setup
  ```
  On macOS the teleop launcher must run under **`mjpython`** (not `python3`) — a
  `launch_passive` / Cocoa main-thread requirement.
- For `--teleop-weighted-ik` only (the mink weighted IK that makes `--teleop-orientation`
  faithful):
  ```bash
  pip install mink
  ```
- For `--teleop --hand-tracking` only (real per-finger control instead of controller
  triggers) — the **upstream** `dex_retargeting`, not `unitreerobotics/xr_teleoperate`'s
  forked pin (see CLAUDE.md's "Teleop Input" for why):
  ```bash
  pip install dex_retargeting
  pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, no GPU needed
  ```

## Setup

The G1 meshes come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
and are **not** vendored in this repo (they're large binaries). Clone the `unitree_g1/` folder
as a **sibling directory** of this repo:

```bash
cd ..   # parent directory of this repo
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set unitree_g1 && cd -
```

You should end up with:

```
<parent>/
├── UNITREE_TELEOP_MUJOCO/     (this repo)
└── mujoco_menagerie/
    └── unitree_g1/
```

Generate the fixed-base robot (uses `g1_with_hands.xml` for the Dex3-1 dexterous hand):

```bash
cd ../mujoco_menagerie/unitree_g1
python3 /path/to/UNITREE_TELEOP_MUJOCO/make_fixed_base.py g1_with_hands.xml
```

Then copy the scene file (and this repo's brick mesh, if you don't already have it) into the
same directory, next to `assets/`:

```bash
cp /path/to/UNITREE_TELEOP_MUJOCO/scene_fixed_table.xml .
```

## Usage

```bash
python3 stand_next_to_table.py                       # third-person view, picks brick1 by default
python3 stand_next_to_table.py --view egocentric      # robot head POV
python3 stand_next_to_table.py --pick brick2          # pick brick2 instead (also right arm)
python3 stand_next_to_table.py --pick none            # just hold stance, no pick sequence
mjpython stand_next_to_table.py --teleop --cert-file cert.pem --key-file key.pem
                                                       # live VR teleop, both arms — position
                                                       # follow (needs a headset + televuer;
                                                       # macOS: run under mjpython, not python3)
mjpython stand_next_to_table.py --teleop --teleop-weighted-ik --teleop-orientation \
         --display-mode immersive --record-episodes recordings/demo1 \
         --cert-file cert.pem --key-file key.pem
                                                       # faithful 6-DOF follow via the mink
                                                       # weighted IK, head camera streamed to
                                                       # the headset; press 's' in the viewer
                                                       # to start a demo take, 's' again to save
python3 stand_next_to_table.py --teleop --cert-file cert.pem --key-file key.pem --hand-tracking
                                                       # real per-finger control instead of
                                                       # controller triggers (needs
                                                       # `pip install dex_retargeting` + CPU torch)
python3 stand_next_to_table.py --view third_person /path/to/scene.xml   # explicit scene
```

Run from any directory — the launcher resolves the Menagerie clone relative to its own file
location (override with the `MODEL_DIR` env var if your clone lives somewhere other than as a
sibling of this repo).

To check the arm IK solver headlessly (no GUI needed):

```bash
python3 verify_arm_ik.py
```

## Teleop loop

`--teleop` runs `TeleopController` ([teleop_control.py](teleop_control.py)) — a continuous
per-frame two-arm follow driven by a Meta Quest 3S over the
[`televuer`](https://github.com/unitreerobotics/televuer) package. Each frame it reads both
controller poses, applies their delta *since calibration* on top of a fixed `TELEOP_HOME`
anchor pose, solves IK toward that target for each arm, and sets grip from the trigger. Legs
and waist stay pinned to stance every step — only the arms/hands move.

- **Calibration is gated** — at startup hold both controllers still in a relaxed, *symmetric*
  pose (elbows ~90°, hands ~30 cm apart) until `teleop: calibrated` prints; the arms then
  ramp up to `TELEOP_HOME` and start following. Press `c` in the viewer to recalibrate.
- **Raw controller input is filtered** (single-frame glitch reject + EMA smoothing) and the
  IK target is rate-limited, so a tracking dropout/snap can't slam the arm; a sustained
  out-of-reach target makes the arm hold position instead of diverging. Operator range is
  compressed onto the robot by `--teleop-scale` (default 0.5).
- **IK: `--teleop-weighted-ik`** swaps the default DLS solver for `WeightedArmIK` with the
  **mink** backend ([kevinzakka/mink](https://github.com/kevinzakka/mink), a MuJoCo-native
  differential-IK QP). This is what makes **`--teleop-orientation`** (6-DOF follow) usable —
  a 45° controller rotation maps to ~45° of gripper rotation on every axis with the
  non-moving arm staying put, where the DLS solver overshoots wildly and drags the other arm
  10+ cm. It also beats DLS on pure-position tracking. `arm_ik.py` and every scripted path
  are untouched. (The G1's wrist actuators are torque-limited, so large wrist
  reconfigurations still fall a few cm short — an arm-hardware limit no solver removes.)
- **`--display-mode ego\|immersive`** streams the `fpv_teleop` camera to the headset
  (`FpvStreamer`, mono). **`--record-episodes DIR`** samples frames + arm/hand joint
  states/actions into per-episode datasets — press `s` in the viewer to start a take, `s`
  again to stop and save (repeat for more takes).

**Confirmed on a real Quest 3S:** position tracking of both arms and FPV streaming. The mink
weighted IK for orientation is verified in sim, not yet on the headset. On macOS the launcher
must run under `mjpython`. See CLAUDE.md's "Teleop Input (Phase 2)" for the full record.

## Scene contents

- **Robot**: Unitree G1, base welded to the world (no balance controller — it's not needed
  since the pelvis can't move), holding a stance pose. 43 actuators: legs, waist, arms, and a
  3-finger Dex3-1 dexterous hand per wrist.
- **Table**: wooden tabletop with top surface at `z = 0.74 m`.
- **Bricks**: three `yellowPLAEXLong` bricks, resting on the table at different orientations
  at `x=0.28` (within confirmed arm reach — see `verify_arm_ik.py`), ready to be picked up.
  Enlarged from the original ~7.9 × 3.75 × 3.45 cm to **~13.1 × 6.25 × 5.75 cm** (L × W × H
  for an un-yawed brick; mesh `scale` 0.015 → 0.025, ~283 g at `density=600`) to test
  graspability under VR teleop. `brick1` sits at `y=-0.20` (moved from `-0.15`) so its
  collision box clears the 90°-yawed `brick2` at the larger size.
- **Camera**: `fpv_teleop`, mounted on the torso and tilted down toward the tabletop — pass
  `--view egocentric` to see through it.

## Status

- [x] Fixed-base robot compiles and holds its stance
- [x] Egocentric camera aimed at the tabletop
- [x] Bricks placed on the table
- [x] Switched to the Dex3-1 dexterous hand end-effector
- [x] Arm IK solver, with optional orientation control, reaching a pre-grasp point above a brick
- [x] Grasp primitive (hand open/pre-grasp/closed pose, verified approach + orient + close)
- [x] Contact-only grasp holding, verified to survive a real lift — `brick1`/`brick2` (right arm)
- [x] Grasp-state detection (aperture stall + gripper-to-object proximity, no privileged reads)
- [~] Collision avoidance against other bricks/partial stack — geometry verified correct, but real-simulation testing found it unreliable (can even worsen collisions); see CLAUDE.md's "Collision Avoidance"
- [x] Grasp control loop wired into `stand_next_to_table.py` — `PickSequence` drives approach → orient → close → lift → hold live, per frame (`--pick`); see CLAUDE.md's "Wired-In Control Loop"
- [ ] `brick3` (left arm) grasp-approach geometry — known open issue, now also confirmed unable to reach the chosen stack location; see CLAUDE.md
- [~] Stacking sequencing — location/order/arm assignment decided, `StackSequence` built; the major collision cause (the hand's own fingers hitting `brick1`, fixed via release-from-hover) dropped base-brick displacement from ~80cm to ~11cm, but a smaller horizontal landing-precision issue remains — see CLAUDE.md's "Stacking Sequencing"
- [~] Teleop input — `TeleopController` (via `televuer`, Quest 3S controllers): **position tracking of both arms and FPV streaming confirmed on a real Quest 3S**. `--teleop-orientation` via the **mink** weighted IK (`--teleop-weighted-ik`) gives faithful 6-DOF follow — verified in sim, headset run pending. Hand-tracking (`HandRetargeter`, upstream `dex_retargeting`) exists but is untested with real keypoints. See CLAUDE.md's "Teleop Input (Phase 2)"

## Verifying a change

Before opening the GUI (which needs a real display), sanity-check any scene edit headlessly:

```python
import mujoco
m = mujoco.MjModel.from_xml_path("scene_fixed_table.xml")
d = mujoco.MjData(m)
key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
mujoco.mj_resetDataKeyframe(m, d, key_id)
d.ctrl[:] = m.key_ctrl[key_id]
for _ in range(int(3.0 / m.opt.timestep)):
    mujoco.mj_step(m, d)
# then check: pelvis didn't drift, no NaNs, bricks still resting on the table
```

See [CLAUDE.md](CLAUDE.md) for the full set of invariants (freejoint removal, keyframe
trimming, the `stand`/`stand_at_table` split, actuator group layout) that any change here
needs to respect.
