"""
Headless verification for teleop_control.py, against a SYNTHETIC input
source, not a real headset -- there is no VR hardware reachable from this
environment. This is the "headless-testable input stub" from the Phase 2
plan: it can confirm the CONTROL LOOP wiring is correct (pinning, no NaN,
target tracking, grip response, calibration timing) without ever touching
televuer or real hardware, but it cannot confirm the coordinate-mapping
math is actually correct against a real headset's output -- that's a real,
separate, on-hardware check (see teleop_control.py's module docstring,
"First real-hardware checks").

FakeTeleopSource fabricates TeleData-shaped frames: motion_data_ready is
False for the first few calls (mimicking real startup, where the headset
connection isn't live instantly), then True. Once "live", the right
controller's wrist_pose ramps a straight-line +5cm move in x over
RAMP_TIME seconds (left stays put, isolating the test to one arm) and its
trigger value ramps released(10.0) -> fully pressed(0.0) -> released again,
so both the position-delta and grip mapping get exercised.

Run: python verify_teleop_control.py
"""
import pathlib, os
import numpy as np
import mujoco

from actuator_groups import LEG, WAIST
from teleop_control import TeleopController, trigger_to_grip

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
SCENE = MODEL_DIR / "scene_fixed_table.xml"

READY_AFTER_FRAMES = 5     # startup delay before motion_data_ready goes True
RAMP_TIME = 2.0            # s, right wrist's synthetic +5cm move in x
MOVE_DISTANCE = 0.05       # m
TRIGGER_RAMP_TIME = 1.0    # s, released -> fully pressed
ROTATE_RAMP_TIME = 2.0     # s, small synthetic rotation on the right wrist,
                            # just to exercise the orientation path without NaN


class FakeTeleData:
    def __init__(self, left_wrist_pose, right_wrist_pose,
                 left_ctrl_triggerValue, right_ctrl_triggerValue, motion_data_ready):
        self.left_wrist_pose = left_wrist_pose
        self.right_wrist_pose = right_wrist_pose
        self.left_ctrl_triggerValue = left_ctrl_triggerValue
        self.right_ctrl_triggerValue = right_ctrl_triggerValue
        self.motion_data_ready = motion_data_ready


class FakeTeleopSource:
    def __init__(self, dt):
        self.dt = dt
        self._t = 0.0
        self._frame = 0

    def get_tele_data(self):
        ready = self._frame >= READY_AFTER_FRAMES
        self._frame += 1
        t_live = max(self._t - READY_AFTER_FRAMES * self.dt, 0.0) if ready else 0.0
        self._t += self.dt

        left_pose = np.eye(4)  # left stays put for the whole run

        right_pose = np.eye(4)
        move_alpha = min(t_live / RAMP_TIME, 1.0)
        right_pose[0, 3] = move_alpha * MOVE_DISTANCE  # +x translation

        rot_alpha = min(t_live / ROTATE_RAMP_TIME, 1.0)
        angle = rot_alpha * 0.2  # small rotation about z, radians
        c, s = np.cos(angle), np.sin(angle)
        right_pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # trigger: released -> fully pressed -> released, over TRIGGER_RAMP_TIME each way
        tri_alpha = min(t_live / TRIGGER_RAMP_TIME, 1.0)
        if t_live < TRIGGER_RAMP_TIME:
            right_trigger = 10.0 - tri_alpha * 10.0     # 10 -> 0
        elif t_live < 2 * TRIGGER_RAMP_TIME:
            back_alpha = (t_live - TRIGGER_RAMP_TIME) / TRIGGER_RAMP_TIME
            right_trigger = back_alpha * 10.0            # 0 -> 10
        else:
            right_trigger = 10.0

        return FakeTeleData(left_pose, right_pose, 10.0, right_trigger, ready)


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    dt = m.opt.timestep
    source = FakeTeleopSource(dt)
    ctrl = TeleopController(m)

    right_site_start = None
    calibrated_at_step = None
    min_grip_seen = 1.0
    max_grip_seen = 0.0
    steps = int(3 * TRIGGER_RAMP_TIME / dt) + int(RAMP_TIME / dt) + 500

    for i in range(steps):
        data = source.get_tele_data()
        if not ctrl.calibrated:
            assert not np.any(np.isnan(d.qpos)), "NaN before calibration even happened"
            if data.motion_data_ready:
                ctrl.calibrate(m, d, data)
                calibrated_at_step = i
                right_site_start = ctrl._iks["right"].site_pos().copy()
                print(f"calibrated at step {i} (t={i*dt:.2f}s)")
        else:
            ctrl.step(m, d, hold, data)
            min_grip_seen = min(min_grip_seen, ctrl.last_grip["right"])
            max_grip_seen = max(max_grip_seen, ctrl.last_grip["right"])

        mujoco.mj_step(m, d)

        # LEG/WAIST must stay pinned every step, calibrated or not
        assert np.allclose(d.ctrl[LEG], hold[LEG]), f"LEG ctrl drifted at step {i}"
        assert np.allclose(d.ctrl[WAIST], hold[WAIST]), f"WAIST ctrl drifted at step {i}"
        assert not np.any(np.isnan(d.qpos)) and not np.any(np.isnan(d.qvel)), \
            f"NaN at step {i}"

    assert calibrated_at_step is not None, "never calibrated -- motion_data_ready logic broken"

    right_site_end = ctrl._iks["right"].site_pos()
    tracked_delta = right_site_end - right_site_start
    print(f"right site moved {tracked_delta} (commanded +x move was {MOVE_DISTANCE} m)")
    print(f"grip range observed on the right hand: [{min_grip_seen:.3f}, {max_grip_seen:.3f}]")

    # sanity check the delta-mapping wiring, NOT real-world coordinate
    # correctness (that needs a real headset) -- the site should have moved
    # noticeably, and mostly along +x since that's the only axis the fake
    # input moved on and SCALE=1.0
    moved_enough = np.linalg.norm(tracked_delta) > 0.02
    mostly_x = abs(tracked_delta[0]) > 0.5 * np.linalg.norm(tracked_delta)
    grip_responded = (min_grip_seen < 0.1) and (max_grip_seen > 0.9)

    print(f"\nmoved_enough={moved_enough}  mostly_x={mostly_x}  "
          f"grip_fully_closed_and_opened={grip_responded}")
    ok = moved_enough and mostly_x and grip_responded
    print("PASS" if ok else "FAIL")
    print("\nReminder: this only verifies the control-loop wiring in simulation.")
    print("It does NOT verify the coordinate mapping against a real headset --")
    print("see teleop_control.py's 'First real-hardware checks' before trusting it live.")
    assert ok


if __name__ == "__main__":
    main()
