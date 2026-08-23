# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A scaffold for a fixed-base (welded) Unitree G1 humanoid standing next to a table in MuJoCo,
holding a stance pose, with a working egocentric head camera and three manipulable bricks
resting on the tabletop. It's the setup step before teleoperated pick-and-place / stacking
control — no grasp or stacking controller exists yet (see "Not yet built" below).

## Repository Layout

- **[make_fixed_base.py](make_fixed_base.py)** — generator. Takes a stock Menagerie G1 MJCF
  (`g1.xml` or `g1_with_hands.xml`) and produces `g1_fixed_upper.xml`: removes the pelvis
  `<freejoint>` (welds the base), trims the `stand` keyframe's qpos to match the new (smaller)
  `nq`, and injects the `fpv_teleop` camera into `torso_link`. Regex-based, not literal-string
  matching — `g1.xml` and `g1_with_hands.xml` format the `imu_in_torso` site with different
  attribute order, which a literal match silently fails on.
- **[g1_fixed_upper.xml](g1_fixed_upper.xml)** — a *reference copy* of the generator's output
  (currently generated from `g1_with_hands.xml`). The copy that's actually loaded at runtime
  lives inside the Menagerie clone, not here — see "Two-location split" below.
- **[scene_fixed_table.xml](scene_fixed_table.xml)** — includes `g1_fixed_upper.xml`, adds
  floor + table + three `yellowPLAEXLong` bricks, and defines the `stand_at_table` keyframe
  (see "Two keyframes" below).
- **[stand_next_to_table.py](stand_next_to_table.py)** — launcher. Resolves `MODEL_DIR`,
  loads the scene, resets to `stand_at_table`, holds the stance every step, opens the viewer.
  `--view {third_person,egocentric}` switches the camera.

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
- **Actuator/ctrl groups** (in `stand_next_to_table.py`):
  `LEG(0:12) WAIST(12:15) LEFT_ARM(15:22) LEFT_HAND(22:29) RIGHT_ARM(29:36) RIGHT_HAND(36:43)`;
  `UPPER_BODY(15:43)` covers all four contiguously. **Never broadcast ctrl across all 43** —
  legs/waist must stay pinned to the stance target every step; only `UPPER_BODY` is for teleop.
  The two hands are **not internally symmetric**: left hand joint order is
  thumb, middle, index; right hand is thumb, index, middle.

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

No grasp or stacking control. `UPPER_BODY` is currently just held at the stance target every
step (see the commented-out line in `stand_next_to_table.py`'s loop). No IK, no hand-closing
logic, no teleop input mapping. This is the natural next step — the actuator groups above
(`LEFT_HAND`/`RIGHT_HAND`) are there specifically to be driven once it's built.
