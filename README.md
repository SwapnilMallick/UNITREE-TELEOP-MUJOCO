# Unitree G1 Tabletop Teleop Scaffold

A fixed-base (welded) Unitree G1 humanoid standing next to a table in MuJoCo, holding a
stance pose, with a working first-person head camera and three bricks resting on the
tabletop — the setup step before teleoperated pick-and-place / stacking control.

<p float="left">
  <img src="https://img.shields.io/badge/MuJoCo-%3E%3D3.2-blue" alt="mujoco>=3.2">
</p>

## What's here

| File | Purpose |
|---|---|
| [make_fixed_base.py](make_fixed_base.py) | Generates a fixed-base G1 MJCF from a stock Menagerie G1 (welds the pelvis, trims the keyframe, adds an FPV camera). |
| [g1_fixed_upper.xml](g1_fixed_upper.xml) | Reference copy of the generated fixed-base G1 (currently: Dex3-1 3-finger hands). |
| [scene_fixed_table.xml](scene_fixed_table.xml) | Full scene: robot + floor + table + three bricks, plus the `stand_at_table` reset keyframe. |
| [stand_next_to_table.py](stand_next_to_table.py) | Launcher — opens the viewer with the robot holding its stance next to the table. |

## Requirements

- Python 3
- MuJoCo ≥ 3.2:
  ```bash
  pip install -U "mujoco>=3.2" numpy imageio
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
python3 stand_next_to_table.py                       # third-person view (default)
python3 stand_next_to_table.py --view egocentric      # robot head POV
python3 stand_next_to_table.py --view third_person /path/to/scene.xml   # explicit scene
```

Run from any directory — the launcher resolves the Menagerie clone relative to its own file
location (override with the `MODEL_DIR` env var if your clone lives somewhere other than as a
sibling of this repo).

## Scene contents

- **Robot**: Unitree G1, base welded to the world (no balance controller — it's not needed
  since the pelvis can't move), holding a stance pose. 43 actuators: legs, waist, arms, and a
  3-finger Dex3-1 dexterous hand per wrist.
- **Table**: wooden tabletop with top surface at `z = 0.74 m`.
- **Bricks**: three `yellowPLAEXLong` bricks (~7.9 × 3.75 × 3.45 cm), resting on the table at
  different orientations, ready to be picked up.
- **Camera**: `fpv_teleop`, mounted on the torso and tilted down toward the tabletop — pass
  `--view egocentric` to see through it.

## Status

- [x] Fixed-base robot compiles and holds its stance
- [x] Egocentric camera aimed at the tabletop
- [x] Bricks placed on the table
- [x] Switched to the Dex3-1 dexterous hand end-effector
- [ ] Grasp control
- [ ] Stacking control
- [ ] Teleop input mapping

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
