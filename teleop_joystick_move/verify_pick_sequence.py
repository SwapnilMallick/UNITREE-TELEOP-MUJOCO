"""
Headless verification for pick_sequence.py: drives PickSequence exactly the
way stand_next_to_table.py's live viewer loop will -- one .step() call per
mj_step(), never a blocking `while` loop internally -- and confirms it
reaches HOLD with grasp_confidence().grasped == True, for the same cases
verify_grasp_hold.py already validated with its blocking reference
implementation. This is a regression check that the incremental,
per-frame-advanceable version behaves the same as the proven one, not a
re-derivation of the grasp physics itself.

Run: python verify_pick_sequence.py
"""
import pathlib, os
import numpy as np
import mujoco

from actuator_groups import LEFT_ARM, LEFT_HAND, RIGHT_ARM, RIGHT_HAND
from pick_sequence import PickSequence, PHASES

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
SCENE = MODEL_DIR / "scene_fixed_table.xml"

MAX_TIME = 12.0  # s -- generous upper bound; the full sequence takes ~5-6s

# brick3/left is the same known, documented open issue as verify_grasp_hold.py
# -- included so this reports it honestly rather than hiding it, not asserted.
CASES = [
    ("right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick1", True),
    ("right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick2", True),
    ("left",  "left_gripper_site",  LEFT_ARM,  LEFT_HAND,  "brick3", False),
]


def run_case(m, d, key_id, side, site_name, arm_slice, hand_slice, brick_name):
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    hold[LEFT_HAND] = 0.0
    hold[RIGHT_HAND] = 0.0
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    seq = PickSequence(m, side, site_name, arm_slice, hand_slice, brick_name)
    seq.start(m, d)

    phases_seen = []
    steps = 0
    max_steps = int(MAX_TIME / m.opt.timestep)
    hold_steps = 0
    while steps < max_steps:
        phase = seq.step(m, d, hold)
        mujoco.mj_step(m, d)
        steps += 1
        if not phases_seen or phases_seen[-1] != phase:
            phases_seen.append(phase)
        if phase == "HOLD":
            hold_steps += 1
            if hold_steps >= int(2.0 / m.opt.timestep):  # 2s of settled holding
                break

    has_nan = bool(np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)))
    return dict(phases_seen=phases_seen, reached_hold=(seq.phase == "HOLD"),
                confidence=seq.confidence, has_nan=has_nan, steps=steps)


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")

    all_ok = True
    for side, site_name, arm_slice, hand_slice, brick_name, must_pass in CASES:
        print(f"\n=== {side} arm -> {brick_name}: PickSequence, driven per-frame ===")
        r = run_case(m, d, key_id, side, site_name, arm_slice, hand_slice, brick_name)
        print(f"  phases: {' -> '.join(r['phases_seen'])}")
        print(f"  reached_hold={r['reached_hold']}  confidence={r['confidence']}  "
              f"NaN={r['has_nan']}  steps={r['steps']}")
        assert set(r["phases_seen"]) <= set(PHASES), "unexpected phase name -- typo in pick_sequence.py?"
        ok = (r["reached_hold"] and r["confidence"] is not None
              and r["confidence"]["grasped"] and not r["has_nan"])
        status = "PASS" if ok else "FAIL"
        print(f"  {status}" + ("" if must_pass else " (known open issue, not required to pass)"))
        if must_pass:
            all_ok = all_ok and ok

    print("\n" + ("ALL REQUIRED CASES PASSED" if all_ok else "SOME REQUIRED CASES FAILED"))
    assert all_ok


if __name__ == "__main__":
    main()
