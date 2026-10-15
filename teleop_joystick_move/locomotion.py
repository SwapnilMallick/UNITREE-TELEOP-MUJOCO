"""
Joystick-driven base locomotion for --base-mode planar (Stage 1) and the
shared command-reading/smoothing layer Stage 2 (--base-mode policy) reuses.

Both VR controller thumbsticks were previously unused -- every other button
(aButton/bButton/squeeze/trigger) already drives something in
stand_next_to_table.py (reset, recalibrate, episode-record toggle, brick
spawn, grip) and is NOT touched here. Mapping (confirmed against televuer's
actual source, src/televuer/tv_wrapper.py, not assumed from memory):
  left_ctrl_thumbstickValue  = (x, y), front=(0,-1) back=(0,1) left=(-1,0) right=(1,0)
  right_ctrl_thumbstickValue = same shape, right controller

  left stick Y  -> forward velocity vx   (negated: front is -1, not +1)
  left stick X  -> lateral velocity vy   (+x = robot's right, see ROBOT FRAME below)
  right stick X -> yaw rate (+ccw, MuJoCo/world convention: +z rotation)

ROBOT FRAME: vx/vy here are in the ROBOT's own (torso/pelvis-local) frame --
vx is always "forward from the robot's own current facing", vy is always
"the robot's own current right/left", regardless of which way the robot is
currently turned. PlanarBaseController rotates this into world frame by the
base's live yaw before integrating position, every step -- so turning the
robot changes which way "forward" drives it, the way an actual joystick-
controlled character would behave.
"""
import numpy as np
import mujoco

# --- joystick -> raw command -------------------------------------------------
DEADZONE = 0.12          # thumbstick magnitude below this reads as exactly zero
                          # (cheap gimbal-mounted sticks rarely report a true
                          # (0,0) at rest; without this the base creeps)
MAX_VX = 0.6              # m/s forward/back -- a brisk, controllable walk pace
                          # for a legs-locked kinematic slide (not a real gait
                          # speed; this is Stage 1's "slide around", not
                          # walking -- see --base-mode policy for real gait)
MAX_VY = 0.4              # m/s lateral -- slower than forward, matching how
                          # most people find strafing less comfortable to aim
MAX_YAW_RATE = 1.2        # rad/s (~69 deg/s) -- fast enough to turn to face a
                          # brick in under a second, slow enough not to be
                          # disorienting in the headset
SMOOTH_TAU = 0.15         # s; low-pass time constant on the command (EMA) --
                          # smooths a jittery/noisy raw stick into something
                          # that doesn't jerk the base frame to frame; bigger
                          # = smoother but more input lag

# All three configurable per-run (stand_next_to_table.py --base-max-vx etc.)
# via BaseVelocityCommand's constructor, not just these module defaults.


def _deadzone(v, dz):
    """Rescale so |v|<dz reads as 0 and the deadzone edge is continuous (no
    jump at the boundary) rather than a hard clip -- avoids a visible snap
    to nonzero speed the instant the stick leaves the dead zone."""
    if abs(v) <= dz:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (abs(v) - dz) / (1.0 - dz)


def read_joystick_raw(tele_data, deadzone=DEADZONE):
    """Pull both thumbsticks off `tele_data` (real televuer.TeleData or any
    duck-typed stand-in -- see verify_locomotion.py's FakeJoystickSource) and
    return a raw (vx, vy, yaw_rate) command in [-1, 1]^3, ROBOT-frame, deadzone-
    applied but NOT yet scaled to m/s or smoothed (see BaseVelocityCommand).
    Returns exactly (0.0, 0.0, 0.0) if the fields are missing/None/non-finite
    (controller disconnected or a stale frame) -- never raises, never leaves
    a stale nonzero command in place, matching the "zero command when no
    input arrives" design requirement."""
    left = getattr(tele_data, "left_ctrl_thumbstickValue", None)
    right = getattr(tele_data, "right_ctrl_thumbstickValue", None)
    if left is None or right is None:
        return 0.0, 0.0, 0.0
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.shape[0] < 2 or right.shape[0] < 2 or not np.all(np.isfinite(left[:2])) \
            or not np.all(np.isfinite(right[:2])):
        return 0.0, 0.0, 0.0
    # front=(0,-1) -> forward is NEGATIVE y, hence the minus sign
    vx_raw = _deadzone(-float(left[1]), deadzone)
    vy_raw = _deadzone(float(left[0]), deadzone)
    yaw_raw = _deadzone(float(right[0]), deadzone)
    return vx_raw, vy_raw, yaw_raw


class BaseVelocityCommand:
    """Deadzone + per-axis max-speed scale + EMA low-pass on the joystick
    command. Call update() once per control step; .vx/.vy/.yaw_rate (m/s,
    m/s, rad/s, all ROBOT-frame) are the smoothed command PlanarBaseController
    (or Stage 2's policy observation) should actually use."""

    def __init__(self, max_vx=MAX_VX, max_vy=MAX_VY, max_yaw_rate=MAX_YAW_RATE,
                 deadzone=DEADZONE, smooth_tau=SMOOTH_TAU):
        self.max_vx, self.max_vy, self.max_yaw_rate = max_vx, max_vy, max_yaw_rate
        self.deadzone = deadzone
        self.smooth_tau = smooth_tau
        self.vx = self.vy = self.yaw_rate = 0.0

    def update(self, tele_data, dt):
        """tele_data=None (or missing/stale thumbstick fields) decays the
        command toward zero at the same smoothing rate, rather than freezing
        it -- a controller disconnect should coast to a stop, not keep
        driving the last command forever."""
        if tele_data is None:
            vx_raw = vy_raw = yaw_raw = 0.0
        else:
            vx_raw, vy_raw, yaw_raw = read_joystick_raw(tele_data, self.deadzone)
        target_vx = vx_raw * self.max_vx
        target_vy = vy_raw * self.max_vy
        target_yaw = yaw_raw * self.max_yaw_rate
        # exponential smoothing toward the target, dt-correct (not a fixed
        # per-frame alpha) so behavior doesn't change if the loop rate does
        a = 1.0 - np.exp(-dt / max(self.smooth_tau, 1e-6))
        self.vx += a * (target_vx - self.vx)
        self.vy += a * (target_vy - self.vy)
        self.yaw_rate += a * (target_yaw - self.yaw_rate)
        return self.vx, self.vy, self.yaw_rate


# --- base joints (make_planar_base.py) --------------------------------------
BASE_JOINT_NAMES = ("base_slide_x", "base_slide_y", "base_yaw")


class PlanarBaseController:
    """Drives the 3 planar base joints (make_planar_base.py) by directly
    setting d.qvel each step and letting MuJoCo's own integrator advance
    d.qpos -- no actuators involved (the joints have zero damping/stiffness/
    frictionloss, so nothing opposes the injected velocity). This is the
    "set velocities directly" option stand_next_to_table.py's base_mode=
    planar uses (vs. a velocity actuator) -- simpler (no new <actuator>
    entries, so hold_ctrl / nu=43 / every existing keyframe ctrl= stays
    completely untouched) and avoids a position-actuator PD fighting a
    directly-driven kinematic joint.

    No-ops gracefully (pose stays (0,0,0), step() does nothing) if the
    loaded model doesn't have these joints -- lets stand_next_to_table.py
    construct one unconditionally and just skip calling .step() based on
    --base-mode instead of needing two code paths."""

    def __init__(self, m):
        self.present = all(
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0
            for n in BASE_JOINT_NAMES)
        if not self.present:
            self._qadr = self._dadr = None
            return
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in BASE_JOINT_NAMES]
        self._qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self._dadr = np.array([m.jnt_dofadr[j] for j in jids])
        assert np.array_equal(self._qadr, [0, 1, 2]) and np.array_equal(self._dadr, [0, 1, 2]), (
            f"base joints not at the expected leading qpos/dof addresses "
            f"(qadr={self._qadr}, dadr={self._dadr}) -- make_planar_base.py "
            f"places them first on the pelvis (the root body) specifically "
            f"so this holds; a different model layout needs re-checking "
            f"actuator_groups_planar.py's slices too, not just this assert.")

    def pose_from(self, d):
        """Current (x, y, yaw) in world frame, read from `d` -- (0.0, 0.0,
        0.0) if the base joints aren't present (base_mode=fixed), so callers
        can pass this straight into TeleopController.step(base_pose=...)
        unconditionally regardless of base_mode."""
        if not self.present:
            return (0.0, 0.0, 0.0)
        return (float(d.qpos[self._qadr[0]]), float(d.qpos[self._qadr[1]]),
                float(d.qpos[self._qadr[2]]))

    def step(self, m, d, vx_robot, vy_robot, yaw_rate):
        """Rotate the robot-frame (vx, vy) command into world frame by the
        base's CURRENT yaw, then write world-frame linear velocity + yaw rate
        into d.qvel[base dofs]. Does NOT call mj_step -- the caller's normal
        per-frame mj_step() integrates qpos from this qvel exactly like any
        other unforced joint. No-op if the base joints aren't present."""
        if not self.present:
            return
        yaw = float(d.qpos[self._qadr[2]])
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        vx_world = cos_yaw * vx_robot - sin_yaw * vy_robot
        vy_world = sin_yaw * vx_robot + cos_yaw * vy_robot
        d.qvel[self._dadr[0]] = vx_world
        d.qvel[self._dadr[1]] = vy_world
        d.qvel[self._dadr[2]] = yaw_rate
