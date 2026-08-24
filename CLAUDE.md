# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A scaffold for a fixed-base (welded) Unitree G1 humanoid standing next to a table in MuJoCo,
holding a stance pose, with a working egocentric head camera and three manipulable bricks
resting on the tabletop. It's the setup step before teleoperated pick-and-place / stacking
control. Phase 1 of that control loop is in progress: a working arm IK solver with optional
orientation control (`arm_ik.py`), a Dex3-1 hand grasp primitive (`grasp_primitive.py`), and
tuned fingertip/brick contact all exist — **verified to survive a real lift** for two of the
three assigned pick targets (`brick1`, `brick2` via the right arm), with the third
(`brick3`, left arm) a known, documented open issue distinct from contact tuning — see "Arm
IK" and "Grasp Primitive" below. Not yet wired into the live control loop, no grasp-state
detection, no teleop input mapping (see "Not yet built").

## Repository Layout

- **[make_fixed_base.py](make_fixed_base.py)** — generator. Takes a stock Menagerie G1 MJCF
  (`g1.xml` or `g1_with_hands.xml`) and produces `g1_fixed_upper.xml`: removes the pelvis
  `<freejoint>` (welds the base), trims the `stand` keyframe's qpos to match the new (smaller)
  `nq`, injects the `fpv_teleop` camera into `torso_link`, and injects `left_gripper_site`/
  `right_gripper_site` into each wrist (the IK target frame, see "Arm IK" below). Regex-based,
  not literal-string matching — `g1.xml` and `g1_with_hands.xml` format the `imu_in_torso` site
  with different attribute order, which a literal match silently fails on.
- **[g1_fixed_upper.xml](g1_fixed_upper.xml)** — a *reference copy* of the generator's output
  (currently generated from `g1_with_hands.xml`). The copy that's actually loaded at runtime
  lives inside the Menagerie clone, not here — see "Two-location split" below.
- **[scene_fixed_table.xml](scene_fixed_table.xml)** — includes `g1_fixed_upper.xml`, adds
  floor + table + three `yellowPLAEXLong` bricks, and defines the `stand_at_table` keyframe
  (see "Two keyframes" below).
- **[stand_next_to_table.py](stand_next_to_table.py)** — launcher. Resolves `MODEL_DIR`,
  loads the scene, resets to `stand_at_table`, holds the stance every step, opens the viewer.
  `--view {third_person,egocentric}` switches the camera.
- **[actuator_groups.py](actuator_groups.py)** — the `LEG`/`WAIST`/`LEFT_ARM`/`LEFT_HAND`/
  `RIGHT_ARM`/`RIGHT_HAND`/`UPPER_BODY` qpos/ctrl slice constants, shared by
  `stand_next_to_table.py` and `verify_arm_ik.py` so they can't drift apart.
- **[grasp_primitive.py](grasp_primitive.py)** — Dex3-1 hand shape (open/pre-grasp/closed,
  driven by a single grip scalar) and the grasp approach orientation helper
  (`grasp_target_quat`), see "Grasp Primitive" below.
- **[verify_grasp_primitive.py](verify_grasp_primitive.py)** — headless verification
  combining `arm_ik.py` + `approach_path.py` + `grasp_primitive.py`: approach, orient, close
  on empty space at the pregrasp height, for all three assigned pick targets. Checks the
  mechanism doesn't blow up — not whether a grasp actually holds (see
  `verify_grasp_hold.py`).
- **[verify_grasp_hold.py](verify_grasp_hold.py)** — the real grasp test: approach → orient
  → descend to grasp height → close → lift 20cm → hold 3-5s → confirm the brick rose with
  the hand instead of slipping back down. Passes for `brick1`/`brick2` (right arm); `brick3`
  (left arm) is a known, documented open issue — see "Fingertip/Brick Contact Tuning".
- **[arm_ik.py](arm_ik.py)** — damped-least-squares position IK for one arm, solved against
  `left_gripper_site`/`right_gripper_site` (see "Arm IK" below).
- **[approach_path.py](approach_path.py)** — the re-targetable up/over/down approach path
  (`ApproachPath`): any start → any goal, advanced one control step at a time so it composes
  with a real control loop instead of blocking it. Promoted out of `verify_arm_ik.py` and
  `verify_grasp_primitive.py`, where this logic used to be duplicated per test case.
- **[verify_arm_ik.py](verify_arm_ik.py)** — headless verification for `arm_ik.py` +
  `approach_path.py` (see "Arm IK").

## Critical External Dependency (not in this repo)

The Menagerie G1 assets (meshes, stock MJCF) are **not vendored** — they're large binaries.
`unitree_g1/` is expected as a **sibling directory** of this repo:

```
<parent>/
├── UNITREE_TELEOP_MUJOCO/     (this repo)
└── mujoco_menagerie/
    └── unitree_g1/            (g1.xml, g1_with_hands.xml, assets/, ...)
```

```bash
cd ..   # parent of this repo
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set unitree_g1
```

`stand_next_to_table.py` resolves this path relative to `__file__` (not cwd), overridable via
the `MODEL_DIR` env var.

### Two-location split

`g1_fixed_upper.xml`, `scene_fixed_table.xml`, and the brick mesh (`yellowPLAEXLong.obj`) exist
in **two places**: a reference copy in this repo, and the copy actually loaded at runtime inside
`<MODEL_DIR>` (the Menagerie clone's `unitree_g1/`), because MJCF `<include>`/`<mesh file=...>`
paths and `<compiler meshdir="assets"/>` all resolve relative to that directory. There is no
symlink or build step — **after editing any of these three files in this repo, copy them into
`<MODEL_DIR>` for the change to take effect**:

```bash
cp g1_fixed_upper.xml scene_fixed_table.xml "$MODEL_DIR/"
```

## Generation Pipeline & Hard Invariants

- **Never restore the pelvis freejoint or hold the base with a balance/hold controller.** The
  base is fixed by *removing* the freejoint in `make_fixed_base.py`. "Fixed-base" here means
  literally welded, not actively balanced.
- **Keyframe qpos length must always equal `nq`.** The trim logic reads the source keyframe's
  actual length rather than hardcoding 29 or 43, so it works for either source MJCF.
- **Current source is `g1_with_hands.xml`** (Dex3-1 3-finger dexterous hand: thumb/index/middle,
  7 DoF/hand) → **43 actuators**. The bare `g1.xml` (`rubber_hand` — a cosmetic, non-articulated
  mesh cap with zero joints, can't grasp anything) still works as a generator input if a
  fingerless variant is ever needed, giving 29 actuators instead.
- **`fpv_teleop` camera tilt (49° down)** is solved against this scene's specific stance +
  table position; re-derive if either moves (see the derivation comment in
  `make_fixed_base.py`). The correction is expressed as a `quat`, never `euler=` — this
  document's `<compiler angle="radian"/>` makes any `euler=` attribute in the merged scene
  radians, not degrees, which is an easy silent-bug trap.
- **Actuator/ctrl groups** (in `actuator_groups.py`, shared by every consumer):
  `LEG(0:12) WAIST(12:15) LEFT_ARM(15:22) LEFT_HAND(22:29) RIGHT_ARM(29:36) RIGHT_HAND(36:43)`;
  `UPPER_BODY(15:43)` covers all four contiguously. **Never broadcast ctrl across all 43** —
  legs/waist must stay pinned to the stance target every step; only `UPPER_BODY` is for teleop.
  The two hands are **not internally symmetric**: left hand joint order is
  thumb, middle, index; right hand is thumb, index, middle.
- **Gripper sites** (`left_gripper_site`/`right_gripper_site`, in each wrist's local frame,
  `pos="0.1117 ±0.0402 0"`) are the point where the thumb tip and index/middle tip midpoint
  actually converge at full curl, measured via forward-kinematics sweep (see "Corrected
  gripper site" in "Grasp Primitive" below) — not the original `(0.12, 0, 0)` estimate, which
  was off by ~4cm on the axis that matters and is why early grasp attempts only ever grazed
  the brick with one finger.

### Two keyframes — `stand` vs `stand_at_table`

`g1_fixed_upper.xml`'s own `stand` keyframe is robot-only (43 values). Since
`scene_fixed_table.xml` adds three free-body bricks (nq 43 → 64), MuJoCo silently
zero/identity-pads `stand` to the larger `nq` — which resets all three bricks to the world
origin, not the table. `scene_fixed_table.xml` defines a second keyframe, `stand_at_table`,
with the full correct qpos (robot block copied verbatim from `g1_fixed_upper.xml`'s `stand`,
plus 7 qpos values per brick). `stand_next_to_table.py` resolves it by name with a fallback
chain (`stand_at_table` → `stand` → index 0). **Whenever a body is added to or removed from
the scene, `stand_at_table`'s qpos needs updating by hand** — MuJoCo won't do it for you, and
won't warn you either.

## Brick Assets

`yellowPLAEXLong.obj` is borrowed from the sibling project `BRICK-SIMULATOR-3D-MUJOCO`
(`BlenderFiles/`), copied into `<MODEL_DIR>/assets/`. It's a raw Blender export (Y-up); MuJoCo
needs Z-up, so each brick geom applies a fixed correction quat `0.7071068 0.7071068 0 0`
(equivalent to `euler="90 0 0"` in degrees — again, don't write it as `euler=` in this
document). Scale is `0.015`, **not** the sibling project's own `WORLD_UNITS_PER_METER=10`
(0.1 m/unit) dam-building convention — that would make bricks absurdly large for a tabletop
scene. At 0.015 each brick is ~7.9 × 3.75 × 3.45 cm. Each brick has a visual mesh geom
(`density="0"`, `group="2"`) and a separate box collision geom (`density="600"`, `group="3"`) —
same visual/collision split convention the G1 model itself uses.

Bricks sit at `x=0.28`, not the table's visual center (`x=0.55`) — `verify_arm_ik.py` found the
arm's true max reach is ~0.499 m (joint-limited, measured; less than the naive fully-outstretched
estimate), and `x=0.45` put targets ~0.56 m from the shoulder, physically outside the arm's
workspace. Re-run `verify_arm_ik.py` if the bricks or table move again.

`brick2`'s `y` is `-0.07`, not `0` (dead center). At `y=0` it was equidistant from both
shoulders, which looked like the natural "either arm can take it" case — but reaching dead
center requires the shoulder to rotate across the body's own centerline, and that produced a
real torso/shoulder self-collision for **both** arms, leaving the physically-simulated gripper
site ~4.4-4.8cm off target (vs. a 2cm tolerance) even though the IK solve itself converged
fine. A taller pregrasp height didn't help (the self-collision persisted up to 20cm above the
brick) — it's a lateral geometry problem, not a vertical clearance one. Nudging the target
sideways did: at `y=-0.07` the right arm clears it cleanly (matching brick1/brick3's ~1.6cm
margin) while the left arm now clearly can't (10.5cm) — confirmed no `brick1`↔`brick2`
collision-geom overlap at rest. **Pick assignment: right arm takes `brick1` and `brick2`, left
arm takes `brick3`.**

## Arm IK

`arm_ik.py`'s `ArmIK` solves damped-least-squares position IK for one arm against its gripper
site, plus a **nullspace posture bias** toward the stance pose. The bias is load-bearing, not
cosmetic: a 7-DOF arm solving a 3-DOF position target has 4 redundant DOF, and plain DLS has no
preference among the solutions that satisfy the primary task. Verified empirically
(`verify_arm_ik.py`) that without the bias, the solver finds a valid-in-position but physically
awkward "elbow-down, wrist-low" configuration that grazes the table even when the target itself
sits comfortably inside the arm's reach envelope. `posture_gain=0.12` was picked empirically as
the largest gain that still converges within 2 cm for the tested targets while reliably avoiding
that configuration — retune if the targets change materially. The bias trades a small amount of
steady-state accuracy for it (see the tradeoff note in `arm_ik.py`'s docstring).

**IK correctness alone does not mean a target is safely reachable.** `verify_arm_ik.py`'s
driving loop went through three failed approaches before landing on a working one — worth
reading before building any real teleop/grasp control loop on top of `ArmIK`:

1. Commanding the fully-solved joint target instantly from `t=0`: makes the position actuators
   demand a large instantaneous joint change, swinging the arm through the table along the way.
   An IK solver only promises the *destination* is valid, not the path to it.
2. Solving once, then ramping each joint toward its solved value at an independent bounded rate:
   joints of different magnitude finish at different times, producing an uncoordinated
   intermediate posture that can still collide even though both endpoints are individually fine.
3. A straight-line **Cartesian** interpolation from the arm's current position to the target: the
   stance/resting hand position sits *below* table height, so a straight 3D line from there up to
   a point above the target sweeps through the table's front edge/underside partway through.

The fix — and what `verify_arm_ik.py` now does, matching `BRICK-SIMULATOR-3D-MUJOCO/animation.py`'s
own kinematic brick moves — is a simple **up → over → down** waypoint path: lift straight up to a
safe height clear of the table first, move horizontally at that height, then descend onto the
target, re-solving IK fresh each step (`ArmIK.solve()`'s warm-start is designed for exactly this:
small target motion per call, not one big jump).

One more trap already hit once: **don't reuse the same `ArmIK` instance for both a diagnostic
"does this converge" check and the real driving loop without re-`sync()`ing in between** — a
`solve()` call advances the instance's internal scratch state, so a diagnostic solve leaves the
"current position" the driving loop starts from already at the target, silently collapsing a
carefully-built gradual approach back into an instant jump.

### Orientation control

`ArmIK.solve()` takes an optional `target_quat` — position-only by default (unchanged
behavior), 6-DOF (position + orientation, via `mju_subQuat` against the stacked
position+rotation Jacobian) when given one. **Necessary, not optional, for grasping**:
verified that position-only IK's natural solution for a tabletop pre-grasp point has the
reach axis nearly horizontal (dot product with world -Z of only ~0.03) and the
finger-spread axis almost perpendicular to the brick's short axis (dot product ~0.13) —
nothing in a 3-DOF task constrains orientation at all, so it settles wherever the nullspace
posture bias happens to leave it.

**Full alignment (fingers exactly straddling the brick's short axis) is not reliably
achievable.** Tried first and found to require the elbow to sit at its exact upper joint
limit (2.0944 rad); traced per-iteration and confirmed it doesn't settle there, it
oscillates in and out of the limit indefinitely. `grasp_primitive.py`'s `grasp_target_quat`
instead blends 60% of the way from the arm's natural (position-only) orientation toward the
ideal one — picked empirically as the largest blend that still converges reliably (position
error <2cm, orientation error small) for all three assigned pick targets. It's a soft
improvement, not a guarantee: alignment quality varies by target (brick1 0.13→0.83, brick2
— tighter reach geometry, see above — only 0.06→0.27, brick3 -0.13→0.64).

Adding an orientation target during descend also costs a little position accuracy (same
dq_task-vs-dq_posture tradeoff as the posture bias, just with an orientation term added) —
`verify_grasp_primitive.py` uses a 2.5cm tolerance rather than `verify_arm_ik.py`'s 2cm for
exactly this reason; confirmed via tracing that the left-arm/brick3 case's ~2.2cm residual
is a stable equilibrium (flat for 1.4s of settle time), not still converging.

## Grasp Primitive

`grasp_primitive.py`'s `hand_ctrl(grip, side)` drives the Dex3-1 hand through three named
poses — open (grip=0) → pre-grasp (0.5) → closed (1) — via a single scalar, piecewise-linear
through the three waypoints, not full per-finger IK. CLOSED drives every joint to (near) its
declared range limit, with `thumb_0=0` (see "Corrected gripper site" below for why).

**Known, harmless artifact**: `*_hand_thumb_1_link` overlaps its own `*_wrist_yaw_link` by
~0.27mm at *any* thumb angle, including 0 — a static geometric quirk of the stock Menagerie
collision meshes, confirmed present regardless of grip state, not something introduced here
or worth fixing at this magnitude. A second, larger self-contact appears specifically when
CLOSED curls on **empty space** (nothing to stop the thumb early): the thumb tip grazes its
own wrist by ~1.6cm. Also harmless and expected — confirmed via `d.contact` that it's
self-contact, not brick contact, and that it disappears once a real object (which stops the
thumb well before its joint limit) is actually being grasped.

### Corrected gripper site (the real story behind Steps 1-2's estimate)

The original gripper site position (`pos="0.12 0 0"`, picked in Step 1 before any grasp
testing existed) was **wrong by about 4cm along the axis that matters most**. It assumed the
hand's pinch point sat on the wrist's local Y=0 plane; it doesn't. Verified by a pure
forward-kinematics sweep (no arm, no brick -- just the hand's own kinematic chain): with
`thumb_1`/`thumb_2` and `index`/`middle` all driven to their curled joint limits, sweeping
`thumb_0` (abduction) to find where the thumb tip and the index/middle tip midpoint converge
gives `thumb_0=0` and a pinch point at local `(0.1117, ±0.0402, 0)` in the wrist frame (sign
flips between hands, verified symmetric) -- **not** `(0.12, 0, 0)`. Residual gap at full
closure: 1.5cm (the hand's closed aperture -- objects narrower than that would slip through
fingers closing on empty space, objects wider get caught by real contact first, which is
what you want).

This explains a mystery from the original grasp-primitive work: closing the CLOSED-at-the-time
(~70%-curl) hand at the 8cm pregrasp point only ever grazed the brick with one fingertip,
with the thumb consistently 5-7cm away *regardless of its own abduction angle*. The site
was aiming the wrist at the wrong local target, so the fingers were never actually
positioned to converge on the object in the first place -- no amount of finger-curl tuning
could have fixed that. Fixing the site position (`make_fixed_base.py`) and driving CLOSED to
the joint limit instead of ~70% (`grasp_primitive.py`) both had to change together.

**The pregrasp height also had to change.** 8cm above the brick was tuned for the *old,
wrong* site position; with the corrected site, closing at 8cm makes no brick contact at all
(the wrist pose that satisfies the new site target is different from before). A fresh height
sweep (checking real hand-brick contacts, not just IK convergence) found **4cm above brick
center** gives genuine opposing contact -- thumb vs. middle finger -- for the right arm.
Position-only proximity is not sufficient evidence of a working grasp; always check
`d.contact` for actual opposing multi-point contact, not just how close the IK converged.

## Fingertip/Brick Contact Tuning

Contact-only holding (the decided strategy -- no kinematic weld) needed real friction/solref
tuning to survive a lift; MuJoCo's default contact (`friction="1 0.005 0.0001"`,
`solref="0.02 1"`) let the brick slip back down onto the table within ~0.5s of holding at a
lifted height, even though it looked like a successful lift for the first instant (momentum
carried it briefly during the lift motion itself -- a transient, not a real hold; always test
sustained holding at a fixed height, not just the moment right after lifting).

**The fix that actually mattered was contact stiffness, not friction magnitude.** Both
`right_hand_{thumb_1,thumb_2,index_0,index_1,middle_0,middle_1}_link`'s collision geoms
(injected by `make_fixed_base.py`) and each brick's collision box (`scene_fixed_table.xml`)
now carry `friction="3 0.5 0.3" solref="0.005 1"` (vs. MuJoCo's defaults above -- higher
friction across all three components, and a solver time constant 4x shorter/stiffer).
Verified by direct comparison: high friction (`3 0.5 0.3`) **with the default, softer
solref** held *worse* than the untouched baseline -- the contact was too springy/compliant to
develop real friction force before the grip slipped. Only once solref was stiffened did the
same friction values hold reliably. Both sides of the same contact pair (fingertip geoms and
brick geom) are tuned together in lockstep, not independently -- change one without the
other and they'll fight (MuJoCo mixes contact parameters from both geoms in a pair; keeping
them matched avoids one side dominating unpredictably).

Verified via `verify_grasp_hold.py`: approach → orient → descend to grasp height → close →
lift 20cm → **hold for 3-5 seconds** (not just an instant check) → confirm the brick rose
with the hand instead of settling back down. **Passes for `brick1` and `brick2`** (right
arm) with the brick staying rock-steady at the lifted height for the full hold. **Does not
pass for `brick3`** (left arm) -- a separate, unresolved problem: the left wrist collides
with `brick3` at every grasp height and lateral offset tried (confirmed independent of
`grasp_target_quat`'s orientation blend, down to blend=0/no orientation constraint at all),
so no real opposing finger contact ever forms for that specific arm/target combination. This
is a grasp-*approach*-geometry problem, not a contact-tuning problem -- flagged as an open
item rather than worked around.

## Verification Pattern

Always run a headless physics smoke test before touching the GUI (this environment sometimes
has no interactive display available):

1. Load the scene, reset to `stand_at_table`, hold `ctrl` at `key_ctrl` for ~3 s.
2. Assert: pelvis drift ≈ 0, `qpos` error vs. keyframe small, no NaNs, each brick resting on the
   table (`0 < z - table_top_z < ~0.06`).

For visual evidence, offscreen rendering with `MUJOCO_GL=glfw` works fine on macOS without a
real display (`egl`/`osmesa` are **not** valid backends there, unlike Linux) — use
`mujoco.Renderer` to save PNGs rather than assuming nothing can be verified visually.

## Not Yet Built

`arm_ik.py` (with orientation control), `approach_path.py`, `grasp_primitive.py`, and tuned
fingertip/brick contact give a working, **lift-verified** grasp for two of the three
assigned pick targets (`brick1`/`brick2`, right arm — see `verify_grasp_hold.py`), driven by
scripted targets — not yet wired into `stand_next_to_table.py`'s live control loop, which
still just holds `UPPER_BODY` at the stance target every step (see the commented-out line in
its loop). Still missing/open:
- **`brick3` (left arm) grasp-approach geometry** — the left wrist collides with the brick at
  every height/offset tried; a distinct, unresolved problem from contact tuning (see
  "Fingertip/Brick Contact Tuning").
- No grasp-state detection (contact-only holding needs something like fingertip-to-brick
  proximity, not a simple weld-on-detect flag).
- No collision avoidance against the *other* bricks/partial stack during transit
  (`ApproachPath` only knows about its own start/goal/safe height — only the table has been
  verified as an obstacle, not neighboring bricks or a growing stack).
- No teleop input mapping (VR or keyboard/mouse) feeding IK targets.
- No stacking sequencing.

See the memory system's project notes for the fuller roadmap.
