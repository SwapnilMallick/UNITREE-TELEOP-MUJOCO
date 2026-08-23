"""
Step 1: G1 standing fixed next to a table (no balance control needed).

The base is welded (freejoint removed) and the legs/waist are held at the
'stand' keyframe by their position actuators. Everything above the waist
(arms + Dex3-1 3-finger hands, indices 15..42) is left for you to drive in
the teleop step.

    python stand_next_to_table.py                       # third-person view (default)
    python stand_next_to_table.py --view egocentric      # robot head POV, for teleoperation
    python stand_next_to_table.py --view third_person /path/to/scene.xml
"""
import argparse, time, pathlib, os
import numpy as np
import mujoco, mujoco.viewer

# Resolved relative to this file (not the caller's cwd) so the script runs
# from anywhere. The Menagerie clone is expected as a sibling of this repo:
#   <parent>/mujoco_menagerie/unitree_g1
# Override with the MODEL_DIR env var, or pass an explicit scene path as argv[1].
MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
DEFAULT_SCENE = MODEL_DIR / "scene_fixed_table.xml"

# the fixed camera baked into g1_fixed_upper.xml by make_fixed_base.py
EGOCENTRIC_CAMERA = "fpv_teleop"

# Actuator groups, verified against g1_fixed_upper.xml generated from
# g1_with_hands.xml (43 actuators: 29 body + Dex3-1 hands, 7 DoF/hand --
# thumb x3, index x2, middle x2). Order is declaration order in the source
# MJCF's <actuator> block, which for the arm+hand subtrees also matches qpos
# order (fingers are nested under wrist_yaw_link with no sibling interleaving).
# Quirk: the two hands are NOT internally symmetric -- left is
# thumb,middle,index (22:25, 25:27, 27:29) but right is thumb,index,middle
# (36:39, 39:41, 41:43). Doesn't affect the slices below, but matters if you
# later index a specific finger.
LEG        = slice(0, 12)    # hips / knees / ankles
WAIST      = slice(12, 15)   # yaw / roll / pitch
LEFT_ARM   = slice(15, 22)   # shoulder x3 / elbow / wrist x3
LEFT_HAND  = slice(22, 29)   # Dex3-1: thumb x3, middle x2, index x2
RIGHT_ARM  = slice(29, 36)   # shoulder x3 / elbow / wrist x3
RIGHT_HAND = slice(36, 43)   # Dex3-1: thumb x3, index x2, middle x2
UPPER_BODY = slice(15, 43)   # LEFT_ARM+LEFT_HAND+RIGHT_ARM+RIGHT_HAND (contiguous)

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", nargs="?", type=pathlib.Path, default=DEFAULT_SCENE,
                         help=f"path to scene xml (default: {DEFAULT_SCENE})")
    parser.add_argument("--view", choices=["third_person", "egocentric"], default="third_person",
                         help="camera view: 'third_person' (default, free orbit camera) or "
                              "'egocentric' (robot head POV, for teleoperation)")
    return parser.parse_args()

def main():
    args = parse_args()
    scene = args.scene
    if not scene.exists():
        raise SystemExit(f"scene file not found: {scene}\n"
                          f"set MODEL_DIR or pass the scene path as an argument")
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    # Prefer "stand_at_table" (full-scene keyframe incl. brick poses, defined in
    # scene_fixed_table.xml) over index 0: the robot-only "stand" keyframe from
    # g1_fixed_upper.xml gets zero/identity-padded by the compiler for any bodies
    # (e.g. bricks) it doesn't know about, which resets them to the world origin.
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    if key_id < 0:
        key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        key_id = 0

    # start in the stance pose and hold it
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()  # legs/waist -> 0, arms -> raised stance
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        if args.view == "egocentric":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = m.camera(EGOCENTRIC_CAMERA).id

        while viewer.is_running():
            t0 = time.time()
            # keep lower body pinned every step; leave UPPER_BODY for teleop later
            d.ctrl[LEG]   = hold[LEG]
            d.ctrl[WAIST] = hold[WAIST]
            # d.ctrl[UPPER_BODY] = <your retargeted arm+hand targets>   # <-- next step
            d.ctrl[UPPER_BODY] = hold[UPPER_BODY]
            mujoco.mj_step(m, d)
            viewer.sync()
            dt = m.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

if __name__ == "__main__":
    main()
