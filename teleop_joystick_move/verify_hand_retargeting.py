"""
Headless verification for teleop_control.py's HandRetargeter and
TeleopController's hand-tracking mode -- against SYNTHETIC hand keypoints,
not real hand-tracking data (none is reachable from this environment). This
confirms the WIRING (config loads, retarget() runs every step without
crashing, output lands within/near real joint limits, LEG/WAIST stay
pinned, no NaN) -- it does NOT confirm the retargeted grasp shape is
actually good against a real human hand. See HandRetargeter's docstring in
teleop_control.py for the full fork-vs-upstream story and what's verified.

Skips (not a failure) if dex_retargeting isn't installed -- it's an optional,
heavier dependency (torch/pinocchio/nlopt) only needed for hand-tracking
mode; the controller-trigger path (verify_teleop_control.py) doesn't need it
at all.

Run: python verify_hand_retargeting.py
"""
import pathlib, os, sys
import numpy as np
import mujoco

from actuator_groups import LEG, WAIST, LEFT_HAND, RIGHT_HAND
from teleop_control import TeleopController

try:
    from teleop_control import HandRetargeter
    HandRetargeter()  # cheap existence probe -- raises ImportError early if missing
except ImportError as e:
    print(f"SKIPPED: {e}")
    sys.exit(0)

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
SCENE = MODEL_DIR / "scene_fixed_table.xml"

READY_AFTER_FRAMES = 5
STEPS = 300
KEYPOINT_SEED = 0


class FakeHandTeleData:
    def __init__(self, left_wrist_pose, right_wrist_pose,
                 left_hand_pos, right_hand_pos, motion_data_ready):
        self.left_wrist_pose = left_wrist_pose
        self.right_wrist_pose = right_wrist_pose
        self.left_hand_pos = left_hand_pos
        self.right_hand_pos = right_hand_pos
        self.motion_data_ready = motion_data_ready


class FakeHandTrackingSource:
    """Both wrists stay put (isolates this test to the hand-retargeting path,
    not the arm-delta-mapping path already covered by verify_teleop_control.py).
    Hand keypoints are fixed random points in a small box around the origin --
    not a realistic hand shape, just enough to exercise retarget() every step
    with SOME variation (a slightly different draw each call, like real
    tracking jitter would look, not a frozen constant)."""

    def __init__(self):
        self._frame = 0
        self._rng = np.random.default_rng(KEYPOINT_SEED)

    def get_tele_data(self):
        ready = self._frame >= READY_AFTER_FRAMES
        self._frame += 1
        left_pose = np.eye(4)
        right_pose = np.eye(4)
        left_hand = self._rng.uniform(-0.08, 0.08, size=(25, 3))
        right_hand = self._rng.uniform(-0.08, 0.08, size=(25, 3))
        return FakeHandTeleData(left_pose, right_pose, left_hand, right_hand, ready)


def joint_ranges(m, slice_):
    ids = [j for j in range(m.njnt) if slice_.start <= m.jnt_qposadr[j] < slice_.stop]
    return m.jnt_range[ids]


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    retargeter = HandRetargeter()
    ctrl = TeleopController(m, hand_retargeter=retargeter)
    source = FakeHandTrackingSource()

    left_ranges = joint_ranges(m, LEFT_HAND)
    right_ranges = joint_ranges(m, RIGHT_HAND)
    max_overshoot = 0.0
    hand_ctrl_changed = {"left": False, "right": False}
    stance_hand = {"left": hold[LEFT_HAND].copy(), "right": hold[RIGHT_HAND].copy()}

    calibrated_at_step = None
    for i in range(STEPS):
        data = source.get_tele_data()
        if not ctrl.calibrated:
            if data.motion_data_ready:
                ctrl.calibrate(m, d, data)
                calibrated_at_step = i
        else:
            ctrl.step(m, d, hold, data)
            for side, ranges, slc in [("left", left_ranges, LEFT_HAND),
                                        ("right", right_ranges, RIGHT_HAND)]:
                q = d.ctrl[slc]
                below = np.maximum(ranges[:, 0] - q, 0)
                above = np.maximum(q - ranges[:, 1], 0)
                max_overshoot = max(max_overshoot, below.max(), above.max())
                if np.linalg.norm(q - stance_hand[side]) > 0.05:
                    hand_ctrl_changed[side] = True

        mujoco.mj_step(m, d)
        assert np.allclose(d.ctrl[LEG], hold[LEG]), f"LEG ctrl drifted at step {i}"
        assert np.allclose(d.ctrl[WAIST], hold[WAIST]), f"WAIST ctrl drifted at step {i}"
        assert not np.any(np.isnan(d.qpos)) and not np.any(np.isnan(d.qvel)), \
            f"NaN at step {i}"

    assert calibrated_at_step is not None, "never calibrated"
    print(f"calibrated at step {calibrated_at_step}")
    print(f"max joint-limit overshoot over {STEPS - calibrated_at_step} retarget() calls, "
          f"both hands: {max_overshoot:.6f} rad ({np.degrees(max_overshoot):.4f} deg)")
    print(f"hand ctrl actually changed from stance: {hand_ctrl_changed}")

    # generous overshoot tolerance -- see teleop_control.py's manual sweep,
    # which measured ~0.001 rad as pure optimizer/numerical boundary noise
    # on unrealistic random input; this just guards against something far
    # worse (a real reindexing bug would blow well past this)
    ok = (max_overshoot < 0.05) and all(hand_ctrl_changed.values())
    print("PASS" if ok else "FAIL")
    print("\nReminder: this only verifies the wiring (retarget() runs, output is in-range,")
    print("hand ctrl actually changes) against synthetic keypoints. It does NOT verify the")
    print("retargeted grasp shape is good against a real human hand.")
    assert ok


if __name__ == "__main__":
    main()
