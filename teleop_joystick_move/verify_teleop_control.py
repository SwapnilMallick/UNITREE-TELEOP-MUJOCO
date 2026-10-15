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
connection isn't live instantly), then True but with the wrist poses still
at identity for a few more frames (mimicking "connection up, controller
tracking not locked on yet" -- the exact stale-frame window that used to
poison calibration). Once genuinely live, the controllers sit at a
realistic non-identity resting pose and the right one's wrist_pose ramps a
straight-line +5cm move in x over RAMP_TIME seconds (left stays put,
isolating the test to one arm); its trigger value ramps released(10.0) ->
fully pressed(0.0) -> released again, so both the position-delta and grip
mapping get exercised. The test asserts calibration does NOT fire during
the stale window.

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
ROBOT_XML = MODEL_DIR / "g1_fixed_upper.xml"   # Pinocchio model for --teleop-weighted-ik

READY_AFTER_FRAMES = 5     # startup delay before motion_data_ready goes True
POSE_STALE_FRAMES = 3      # extra frames where it's "ready" but poses are still identity
RAMP_TIME = 2.0           # s, right wrist's synthetic +5cm move in x
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
    # realistic resting controller poses -- a real headset never reports a
    # controller exactly at the tracking origin with identity rotation, and the
    # calibration gate now rejects identity poses as "tracking not locked yet"
    RIGHT_BASE = np.array([0.35, -0.25, 1.05])
    LEFT_BASE = np.array([0.35, 0.25, 1.05])
    POSE_LOCKED_FRAME = READY_AFTER_FRAMES + POSE_STALE_FRAMES

    def __init__(self, dt):
        self.dt = dt
        self._frame = 0

    def get_tele_data(self):
        f = self._frame
        self._frame += 1
        ready = f >= READY_AFTER_FRAMES
        if f < self.POSE_LOCKED_FRAME:
            # connection down, or up but controller pose still stale (identity)
            return FakeTeleData(np.eye(4), np.eye(4), 10.0, 10.0, ready)

        t_live = (f - self.POSE_LOCKED_FRAME) * self.dt

        left_pose = np.eye(4)
        left_pose[:3, 3] = self.LEFT_BASE  # left stays put for the whole run

        right_pose = np.eye(4)
        move_alpha = min(t_live / RAMP_TIME, 1.0)
        right_pose[:3, 3] = self.RIGHT_BASE + np.array([move_alpha * MOVE_DISTANCE, 0.0, 0.0])

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

        return FakeTeleData(left_pose, right_pose, 10.0, right_trigger, True)


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
    ctrl = TeleopController(m, scale=1.0)  # unity here -- this test checks the
                                           # delta MAPPING; --teleop-scale is
                                           # exercised in scale_compresses_motion()

    right_site_start = None
    calibrated_at_step = None
    min_grip_seen = 1.0
    max_grip_seen = 0.0
    steps = int(3 * TRIGGER_RAMP_TIME / dt) + int(RAMP_TIME / dt) + 500

    for i in range(steps):
        data = source.get_tele_data()
        if not ctrl.calibrated:
            assert not np.any(np.isnan(d.qpos)), "NaN before calibration even happened"
            if ctrl.try_calibrate(m, d, data):
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

    assert calibrated_at_step is not None, "never calibrated -- gating too strict or logic broken"
    # the gate must NOT capture during the stale window (ready, but poses still
    # identity) -- that's the whole point of try_calibrate
    assert calibrated_at_step >= source.POSE_LOCKED_FRAME, (
        f"calibrated at step {calibrated_at_step}, before poses locked at "
        f"{source.POSE_LOCKED_FRAME} -- gate let a stale/identity pose through")
    print(f"gate held off calibration through the stale window "
          f"(poses locked at frame {source.POSE_LOCKED_FRAME})")

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

    calibration_gate_and_clamp()


def calibration_gate_and_clamp():
    """Focused checks on the calibration-race fix: the gate rejects not-ready
    and identity poses, request_recalibration() drops the reference, and an
    out-of-reach IK target (a bad reference would produce these every frame)
    is clamped instead of parking the arm at a limit."""
    from teleop_control import MAX_TARGET_DELTA

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    ctrl = TeleopController(m)
    I = np.eye(4)
    good = np.eye(4); good[:3, 3] = [0.35, -0.25, 1.05]

    # not ready -> never calibrates, however many frames
    for _ in range(50):
        assert not ctrl.try_calibrate(m, d, FakeTeleData(good, good, 10.0, 10.0, False))
    assert not ctrl.calibrated
    # ready but identity poses -> still never calibrates
    for _ in range(50):
        assert not ctrl.try_calibrate(m, d, FakeTeleData(I, I, 10.0, 10.0, True))
    assert not ctrl.calibrated
    # ready + valid + held still -> calibrates within ~CALIB_SETTLE_TIME
    frames = 0
    while not ctrl.try_calibrate(m, d, FakeTeleData(good, good, 10.0, 10.0, True)):
        frames += 1
        assert frames < 10 * ctrl._calib_need, "stable valid pose never calibrated"
    assert ctrl.calibrated
    print(f"gate: rejected not-ready + identity, calibrated after {frames} stable frames")

    # a wildly-out-of-reach controller (a bad reference produces these every
    # frame) -- held for a while so the rate-limited target fully ramps out;
    # the MAX_TARGET_DELTA clamp must still bound where it settles
    far = np.eye(4); far[:3, 3] = good[:3, 3] + np.array([5.0, 0.0, 0.0])  # 5 m away
    for _ in range(int(2.0 / m.opt.timestep)):
        ctrl.step(m, d, hold, FakeTeleData(far, far, 10.0, 10.0, True))
        mujoco.mj_step(m, d)
    for side in ("left", "right"):
        reached = np.linalg.norm(ctrl._target_pos_prev[side] - ctrl._robot_ref_pos[side])
        assert reached < MAX_TARGET_DELTA + 1e-6, (
            f"{side} IK target {reached:.2f} m from ref -- clamp didn't hold it in")
    assert not np.any(np.isnan(d.qpos)), "NaN after a clamped out-of-reach target"
    print(f"clamp: sustained 5 m controller offset held the IK target within "
          f"{MAX_TARGET_DELTA} m of the reference, no NaN")

    # request_recalibration drops the reference and re-gates
    ctrl.request_recalibration()
    assert not ctrl.calibrated
    assert not ctrl.try_calibrate(m, d, FakeTeleData(I, I, 10.0, 10.0, True)), \
        "re-calibration skipped the gate"
    print("request_recalibration: dropped calibration and re-gated")
    print("\ncalibration gate + clamp: PASS")

    jumpy_input_stays_stable()


def jumpy_input_stays_stable():
    """The real-hardware failure: Quest controller poses freeze then SNAP
    30-60cm on reacquire, and unfiltered that ran the arm's IK residual to
    ~700mm (permanent flailing). With _filter_vr + the target rate-limit the
    snaps are rejected, the arm stays near stance through the storm, and it
    tracks again once the input goes clean."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    rsite = m.site("right_gripper_site").id

    ctrl = TeleopController(m, scale=1.0)  # unity -- isolate recovery from scaling
    base_l = np.array([0.35, 0.25, 1.05])
    base_r = np.array([0.35, -0.25, 1.05])

    def pose(p):
        T = np.eye(4); T[:3, 3] = p
        return T

    steady = FakeTeleData(pose(base_l), pose(base_r), 10.0, 10.0, True)
    for _ in range(5 * ctrl._calib_need):
        if ctrl.try_calibrate(m, d, steady):
            break
    assert ctrl.calibrated, "calibration never completed on a steady pose"
    # let the arm ramp from stance to TELEOP_HOME under neutral input, THEN
    # measure -- the reference for "did it stay put" is HOME, not stance
    for _ in range(int(2.5 / dt)):
        ctrl.step(m, d, hold, steady)
        mujoco.mj_step(m, d)
    ref = d.site_xpos[rsite].copy()

    # regime 1: ~3s of freeze/snap alternation, each state well under the
    # re-seat hold time, so every snap is rejected and the arm should barely move
    rng = np.random.default_rng(0)
    max_excursion = 0.0
    for i in range(int(3.0 / dt)):
        r = base_r if (i // 40) % 2 == 0 else base_r + rng.uniform(-0.3, 0.3, 3)
        ctrl.step(m, d, hold, FakeTeleData(pose(base_l), pose(r), 10.0, 10.0, True))
        mujoco.mj_step(m, d)
        assert not np.any(np.isnan(d.qpos)) and not np.any(np.isnan(d.qvel)), f"NaN at step {i}"
        assert np.allclose(d.ctrl[LEG], hold[LEG]) and np.allclose(d.ctrl[WAIST], hold[WAIST]), \
            f"LEG/WAIST ctrl drifted at step {i}"
        max_excursion = max(max_excursion, float(np.linalg.norm(d.site_xpos[rsite] - ref)))
    print(f"jumpy input: right arm stayed within {max_excursion*100:.1f}cm of HOME "
          f"through 3s of freeze/snap (unfiltered this ran to ~70cm)")
    assert max_excursion < 0.12, "glitch rejection didn't hold the arm near HOME"

    # regime 2: input goes clean and ramps +12cm in x -- arm should recover and track it
    for i in range(int(3.0 / dt)):
        a = min(i * dt / 1.5, 1.0)
        r = base_r + np.array([0.12 * a, 0.0, 0.0])
        ctrl.step(m, d, hold, FakeTeleData(pose(base_l), pose(r), 10.0, 10.0, True))
        mujoco.mj_step(m, d)
    moved = d.site_xpos[rsite] - ref
    print(f"after the storm + a clean +12cm x ramp, right site moved {moved} m")
    assert not np.any(np.isnan(d.qpos))
    assert moved[0] > 0.05 and abs(moved[0]) > 2.0 * (abs(moved[1]) + abs(moved[2])), \
        "arm didn't recover clean tracking after the dropout storm"
    print("\njumpy input stability: PASS")

    scale_compresses_motion()


def scale_compresses_motion():
    """--teleop-scale: a controller move should map to `scale` x that move on
    the robot, so the operator's (larger) arm range fits the arm's envelope."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    rsite = m.site("right_gripper_site").id
    base_l = np.array([0.35, 0.25, 1.05])
    base_r = np.array([0.35, -0.25, 1.05])

    def pose(p):
        T = np.eye(4); T[:3, 3] = p
        return T

    moved_at = {}
    for s in (1.0, 0.5):
        mujoco.mj_resetDataKeyframe(m, d, key_id)
        d.ctrl[:] = hold
        mujoco.mj_forward(m, d)
        ctrl = TeleopController(m, scale=s)
        for _ in range(5 * ctrl._calib_need):
            if ctrl.try_calibrate(m, d, FakeTeleData(pose(base_l), pose(base_r), 10.0, 10.0, True)):
                break
        assert ctrl.calibrated
        steady = FakeTeleData(pose(base_l), pose(base_r), 10.0, 10.0, True)
        for _ in range(int(2.5 / dt)):        # settle at HOME first
            ctrl.step(m, d, hold, steady)
            mujoco.mj_step(m, d)
        ref = d.site_xpos[rsite].copy()
        for i in range(int(3.0 / dt)):
            a = min(i * dt / 1.5, 1.0)
            r = base_r + np.array([0.0, 0.0, 0.10 * a])   # +10cm controller move in z
            ctrl.step(m, d, hold, FakeTeleData(pose(base_l), pose(r), 10.0, 10.0, True))
            mujoco.mj_step(m, d)
        moved_at[s] = float(d.site_xpos[rsite][2] - ref[2])
    print(f"+10cm controller move -> robot site rose {moved_at[1.0]*100:.1f}cm at scale 1.0, "
          f"{moved_at[0.5]*100:.1f}cm at scale 0.5")
    assert moved_at[1.0] > 0.06, "scale 1.0 didn't track the 10cm move"
    assert 0.35 < moved_at[0.5] / moved_at[1.0] < 0.65, \
        f"scale 0.5 should roughly halve the motion, got ratio {moved_at[0.5]/moved_at[1.0]:.2f}"
    print("\nscale compresses motion: PASS")

    lopsided_calibration_warns()


def lopsided_calibration_warns():
    """A contorted (asymmetric) calibration pose maps normal hand positions to
    unreachable arm targets -- calibrate() should flag it (calibration_data_2.txt)."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    def pose(p):
        T = np.eye(4); T[:3, 3] = np.asarray(p, float)
        return T

    # neutral, symmetric -> no warning
    ctrl = TeleopController(m)
    for _ in range(5 * ctrl._calib_need):
        if ctrl.try_calibrate(m, d, FakeTeleData(pose([0.35, 0.25, 1.05]),
                                                 pose([0.35, -0.25, 1.05]), 10.0, 10.0, True)):
            break
    assert ctrl.calibrated and ctrl.calib_warning is None, \
        f"symmetric calibration should not warn (got {ctrl.calib_warning!r})"

    # the calibration_data_2.txt geometry: left 50cm more forward + 56cm more left
    ctrl2 = TeleopController(m)
    for _ in range(5 * ctrl2._calib_need):
        if ctrl2.try_calibrate(m, d, FakeTeleData(pose([0.897, 0.388, 0.198]),
                                                  pose([0.397, -0.170, 0.138]), 10.0, 10.0, True)):
            break
    assert ctrl2.calibrated and ctrl2.calib_warning is not None, \
        "a lopsided calibration pose should set calib_warning"
    print(f"lopsided calibration flagged: {ctrl2.calib_warning}")
    print("\nlopsided calibration warning: PASS")

    table_target_reachable()


def table_target_reachable():
    """From a neutral calibration, a controller move that maps to the brick1
    grasp pose must actually get the arm there -- not be blocked by the
    REACH_RADIUS / MAX_TARGET_DELTA clamps (calibration_data_3.txt: the user
    couldn't place the arm over the table)."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    rsite = m.site("right_gripper_site").id
    grasp_pos = d.xpos[m.body("brick1").id].copy() + np.array([0.0, 0.0, 0.04])  # GRASP_HEIGHT

    def pose(p):
        T = np.eye(4); T[:3, 3] = np.asarray(p, float)
        return T

    base_l, base_r = np.array([0.35, 0.25, 1.05]), np.array([0.35, -0.25, 1.05])
    scale = 0.5
    ctrl = TeleopController(m, scale=scale)
    for _ in range(5 * ctrl._calib_need):
        if ctrl.try_calibrate(m, d, FakeTeleData(pose(base_l), pose(base_r), 10.0, 10.0, True)):
            break
    assert ctrl.calibrated and ctrl.calib_warning is None
    steady = FakeTeleData(pose(base_l), pose(base_r), 10.0, 10.0, True)
    for _ in range(int(3.0 / dt)):        # ramp stance -> HOME
        ctrl.step(m, d, hold, steady)
        mujoco.mj_step(m, d)

    # controller pose that maps (via the delta, at this scale) to grasp_pos
    vr_target = ctrl._vr_ref_pos["right"] + (grasp_pos - ctrl._robot_ref_pos["right"]) / scale
    for i in range(int(7.0 / dt)):
        a = min(i * dt / 2.0, 1.0)                    # 2s ramp, then 5s hold to settle
        r = base_r + a * (vr_target - base_r)
        ctrl.step(m, d, hold, FakeTeleData(pose(base_l), pose(r), 10.0, 10.0, True))
        mujoco.mj_step(m, d)

    err = float(np.linalg.norm(d.site_xpos[rsite] - grasp_pos))
    print(f"teleop-commanded from HOME to the brick1 grasp pose {np.round(grasp_pos,3)}: "
          f"right gripper site landed {err*100:.1f}cm away")
    assert not np.any(np.isnan(d.qpos))
    assert err < 0.05, ("brick grasp pose not reachable via teleop -- a clamp is too "
                        "tight, or the arm_ik posture-bias parks it too far short")
    print("\ntable target reachable: PASS")

    orientation_mode_rotates_gripper()


def orientation_mode_rotates_gripper():
    """--teleop-orientation (EXPERIMENTAL): rotating the controller does rotate
    the gripper and doesn't NaN. It does NOT assert faithful tracking or tight
    position -- arm_ik's 6-DOF solve is erratic here (a 60deg command gives
    ~25-140deg depending on axis, position degrades 10-25cm); this just guards
    that the mode is wired and not catastrophically broken. Real
    grasp-alignment orientation needs a weighted IK replacing arm_ik."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)
    dt = m.opt.timestep
    rsite = m.site("right_gripper_site").id
    base_l, base_r = np.array([0.35, 0.25, 1.05]), np.array([0.35, -0.25, 1.05])

    def roty(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def pose(p, R=np.eye(3)):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(p, float)
        return T

    ctrl = TeleopController(m, scale=0.5, track_orientation=True)
    for _ in range(5 * ctrl._calib_need):
        if ctrl.try_calibrate(m, d, FakeTeleData(pose(base_l), pose(base_r), 10.0, 10.0, True)):
            break
    assert ctrl.calibrated
    steady = FakeTeleData(pose(base_l), pose(base_r), 10.0, 10.0, True)
    for _ in range(int(3.0 / dt)):                       # settle at HOME
        ctrl.step(m, d, hold, steady)
        mujoco.mj_step(m, d)
    R_home = d.site_xmat[rsite].reshape(3, 3).copy()

    theta = np.radians(40.0)                             # pitch the controller 40 deg about Y
    for i in range(int(5.0 / dt)):
        a = min(i * dt / 2.0, 1.0)
        R = roty(a * theta)
        ctrl.step(m, d, hold, FakeTeleData(pose(base_l), pose(base_r, R), 10.0, 10.0, True))
        mujoco.mj_step(m, d)

    R_now = d.site_xmat[rsite].reshape(3, 3)
    R_rel = R_now @ R_home.T
    ang = np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
    pos_err = float(np.linalg.norm(d.site_xpos[rsite] - ctrl._robot_ref_pos["right"]))
    print(f"controller pitched 40deg about Y -> gripper rotated {ang:.1f}deg; "
          f"position {pos_err*100:.1f}cm from HOME (erratic by design -- see docstring)")
    assert not np.any(np.isnan(d.qpos)) and not np.any(np.isnan(d.qvel))
    assert ang > 10.0, "orientation mode wired but the gripper didn't rotate at all"
    print("\norientation mode rotates gripper (experimental): PASS")

    try:
        import pinocchio  # noqa: F401
    except ImportError:
        print("\n[skip] --teleop-weighted-ik checks: pinocchio not installed "
              "(not a failure -- solver='lm' needs pinocchio, 'ipopt' also needs casadi)")
        return
    weighted_ik_orientation_follows()
    weighted_ik_position_in_regime()


# --------------------------------------------------------------------------- #
#  --teleop-weighted-ik (weighted_arm_ik.WeightedArmIK) -- the fix for        #
#  --teleop-orientation. Mirrors the sweep in                                 #
#  orientation_mode_rotates_gripper() that exposed the DLS solver's failure   #
#  and asserts the weighted IK does NOT reproduce it.                         #
# --------------------------------------------------------------------------- #
def _wik_ctrl(m, d, hold, dt, **kw):
    """Build a weighted-IK TeleopController, calibrate on a neutral symmetric
    pose, and settle at TELEOP_HOME. Returns (ctrl, steady_frame)."""
    base_l, base_r = np.array([0.35, 0.25, 1.05]), np.array([0.35, -0.25, 1.05])

    def pose(p, R=np.eye(3)):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(p, float)
        return T

    ctrl = TeleopController(m, scale=0.5, weighted_ik=True,
                            weighted_ik_mjcf=str(ROBOT_XML), **kw)
    steady = FakeTeleData(pose(base_l), pose(base_r), 10.0, 10.0, True)
    for _ in range(5 * ctrl._calib_need):
        if ctrl.try_calibrate(m, d, steady):
            break
    assert ctrl.calibrated, "weighted-IK controller never calibrated"
    for _ in range(int(3.0 / dt)):
        ctrl.step(m, d, hold, steady)
        mujoco.mj_step(m, d)
    return ctrl, steady, pose


def weighted_ik_orientation_follows():
    """The win: with --teleop-weighted-ik, a controller rotation produces a
    BOUNDED, monotonic gripper rotation (never the DLS solver's 130-160deg
    overshoot of a 45deg command), the moving arm HOLDS position, and the
    NON-MOVING arm stays put (DLS drifts it 13-17cm). Checked about all three
    site axes because the DLS failure was wildly axis-dependent."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    hold = m.key_ctrl[key_id].copy()
    dt = m.opt.timestep
    rsite = m.site("right_gripper_site").id
    lsite = m.site("left_gripper_site").id

    def rot(axis, a):
        c, s = np.cos(a), np.sin(a)
        if axis == "X":
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "Y":
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    CMD = 45.0
    for axis in ("X", "Y", "Z"):
        mujoco.mj_resetDataKeyframe(m, d, key_id)
        d.ctrl[:] = hold
        mujoco.mj_forward(m, d)
        ctrl, steady, pose = _wik_ctrl(m, d, hold, dt, track_orientation=True)
        R_home = d.site_xmat[rsite].reshape(3, 3).copy()
        L_home_p = d.site_xpos[lsite].copy()
        L_home_R = d.site_xmat[lsite].reshape(3, 3).copy()
        base_l, base_r = np.array([0.35, 0.25, 1.05]), np.array([0.35, -0.25, 1.05])

        theta = np.radians(CMD)
        for i in range(int(4.0 / dt)):
            a = min(i * dt / 2.0, 1.0)
            fr = FakeTeleData(pose(base_l), pose(base_r, rot(axis, a * theta)),
                              10.0, 10.0, True)
            ctrl.step(m, d, hold, fr)
            mujoco.mj_step(m, d)
            assert np.allclose(d.ctrl[LEG], hold[LEG]) and np.allclose(d.ctrl[WAIST], hold[WAIST]), \
                f"LEG/WAIST ctrl drifted ({axis})"
            assert not np.any(np.isnan(d.qpos)) and not np.any(np.isnan(d.qvel)), f"NaN ({axis})"

        R_now = d.site_xmat[rsite].reshape(3, 3)
        got = np.degrees(np.arccos(np.clip(
            (np.trace(R_now @ R_home.T) - 1.0) / 2.0, -1.0, 1.0)))
        pos_hold = float(np.linalg.norm(d.site_xpos[rsite] - ctrl._robot_ref_pos["right"]))
        l_drift = float(np.linalg.norm(d.site_xpos[lsite] - L_home_p))
        l_rot = np.degrees(np.arccos(np.clip(
            (np.trace(d.site_xmat[lsite].reshape(3, 3) @ L_home_R.T) - 1.0) / 2.0, -1.0, 1.0)))
        print(f"  weighted-IK {axis} 45deg cmd -> gripper {got:5.1f}deg | "
              f"moving-arm pos-hold {pos_hold*100:4.1f}cm | "
              f"non-moving arm drift {l_drift*100:4.1f}cm / {l_rot:4.1f}deg")

        # the gripper responded, but is NOT wildly overshooting the command the
        # way arm_ik's DLS 6-DOF solve did (it produced 127-159deg for 45 cmd) --
        # mink is faithful on all three axes (measured 43-46deg for a 45deg cmd)
        assert 5.0 < got < 1.6 * CMD, (
            f"{axis}: gripper rotated {got:.0f}deg for a {CMD:.0f}deg command -- "
            f"weighted IK should be bounded/monotonic, not the DLS overshoot")
        # position mostly holds (X/Z: <2cm) but the Y-axis rotation genuinely
        # drives the wrist assembly into a configuration where the G1's wrist
        # actuators (actuatorfrcrange="-5 5", +/-5 Nm -- see
        # weighted_arm_ik.py's docstring) can't fully track the commanded
        # joint angles -- measured ~6-7cm on Y, a real arm-hardware limit, not
        # a solver bug (still tighter than DLS's 11.4cm on the same axis).
        assert pos_hold < 0.08, f"{axis}: moving arm drifted {pos_hold*100:.1f}cm holding position"
        # the OTHER arm stays put (DLS let it wander 13-17cm) -- mink measured
        # ~0.01cm on every axis, essentially perfectly still
        assert l_drift < 0.03 and l_rot < 15.0, (
            f"{axis}: non-moving arm moved {l_drift*100:.1f}cm / {l_rot:.0f}deg")

    print("\nweighted-IK orientation follows (bounded, position held, other arm still): PASS")


def weighted_ik_position_in_regime():
    """Position parity check: in the real teleop operating regime (a pregrasp
    pose ~0.35m from the shoulder, above the table), --teleop-weighted-ik gets
    the gripper there about as well as the DLS solver does -- i.e. swapping the
    solver did not regress position tracking where it matters. (Both solvers
    degrade at fully-extended low reaches where the wrist actuators
    torque-saturate; that regime is a known arm-hardware limit, not tested
    here.)"""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    hold = m.key_ctrl[key_id].copy()
    dt = m.opt.timestep
    rsite = m.site("right_gripper_site").id
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)
    pregrasp = d.xpos[m.body("brick1").id].copy() + np.array([0.0, 0.0, 0.12])

    ctrl, steady, pose = _wik_ctrl(m, d, hold, dt)
    base_l, base_r = np.array([0.35, 0.25, 1.05]), np.array([0.35, -0.25, 1.05])
    scale = 0.5
    vr_target = ctrl._vr_ref_pos["right"] + (pregrasp - ctrl._robot_ref_pos["right"]) / scale
    for i in range(int(7.0 / dt)):
        a = min(i * dt / 2.5, 1.0)
        r = base_r + a * (vr_target - base_r)
        ctrl.step(m, d, hold, FakeTeleData(pose(base_l), pose(r), 10.0, 10.0, True))
        mujoco.mj_step(m, d)
        assert np.allclose(d.ctrl[LEG], hold[LEG]) and np.allclose(d.ctrl[WAIST], hold[WAIST])

    err = float(np.linalg.norm(d.site_xpos[rsite] - pregrasp))
    print(f"  weighted-IK reach to brick1 pregrasp {np.round(pregrasp, 3)}: "
          f"gripper landed {err*100:.1f}cm away")
    assert not np.any(np.isnan(d.qpos))
    assert err < 0.03, ("weighted IK regressed in-regime position tracking "
                        f"({err*100:.1f}cm, DLS gets ~0.3cm here)")
    print("\nweighted-IK position in operating regime: PASS")


if __name__ == "__main__":
    main()
