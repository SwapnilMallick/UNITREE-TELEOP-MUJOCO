"""
TeleopController: continuous, per-frame VR-driven control for both arms --
the live-input counterpart to pick_sequence.py's scripted PickSequence.

UNVERIFIED AGAINST REAL HARDWARE. Everything in this file is built from
televuer's documented interface (https://github.com/unitreerobotics/televuer)
and this repo's own already-verified ArmIK/hand_ctrl, but no VR headset is
reachable from this environment, so the coordinate-mapping math below has
never been run against a real controller. verify_teleop_control.py exercises
it headlessly against a synthetic input source (the "headless-testable input
stub" from the Phase 2 plan) to confirm the CONTROL LOOP wiring is correct
-- pinning, no-NaN, target tracking, grip response -- but that cannot
substitute for an actual on-hardware check. See the module docstring's
"First real-hardware checks" section before trusting this against a headset.

Why not reuse PickSequence's phase machine: real teleop has no phases. A
human continuously repositions the target and squeezes a trigger; there's no
SETTLE/CLOSE_TIME timer that makes sense here. TeleopController instead does
exactly one thing every frame: read wherever the operator's hands currently
are, solve IK toward it, and set the grip from the trigger -- matching
ArmIK.solve()/PickSequence.step()'s own advance-once-per-frame contract so
it composes with stand_next_to_table.py's live loop the same way.

Decoupled from televuer on purpose (matches the locked-in teleop design
decision: "don't couple the IK/grasp control loop tightly to VR-specific
APIs"). TeleopController.step() takes a `tele_data`-shaped object as a
plain argument -- any object with left_wrist_pose/right_wrist_pose (4x4
SE(3) numpy arrays: rotation in the top-left 3x3, translation in the top
3 of the last column) and left_ctrl_triggerValue/right_ctrl_triggerValue
(televuer's convention: 10.0 = released, 0.0 = fully pressed) works,
whether it's a real televuer.TeleVuerWrapper().get_tele_data() result or
the synthetic FakeTeleopSource used for headless testing.

Coordinate mapping is DELTA-based, not absolute: at calibrate() time, this
captures both the operator's current controller pose AND the robot's own
current gripper-site pose as a paired reference. Every subsequent frame
computes how far the controller has moved/rotated *since calibration* and
applies that same delta on top of the robot's reference pose -- not a raw
copy of VR world coordinates into robot world coordinates, which would
assume the two coordinate origins coincide (they don't; VR tracking space
and this MuJoCo scene have completely different origins). This only
requires the two spaces' AXIS CONVENTIONS to agree (both right-handed,
z-up), not their origins -- and per televuer's own documented convention
("Robot Convention: z up, y left, x front"), that's a reasonable match to
this scene's own layout (bricks sit in front of the robot at +x, with the
right arm's bricks at -y and the left arm's brick at +y -- consistent with
"y left"). Reasonable, not verified -- axis handedness bugs are exactly the
kind of thing that only shows up against a real headset.

Calibration is GATED (TeleopController.try_calibrate): the reference pose is
captured only once both controllers report a valid (finite, non-identity)
pose that has held still for CALIB_SETTLE_TIME. On calibrate() the robot side
is anchored to TELEOP_HOME (a fixed mid-workspace pose), NOT the arm's current
stance pose -- so the arm visibly RAMPS UP from stance to a "ready" pose out
in front when "teleop: calibrated" prints, and the operator's neutral hand
pose then maps to HOME with range in every direction. Still hold a RELAXED,
SYMMETRIC controller pose at calibration (elbows ~90 deg, hands in front of
your chest ~30cm apart, level): a lopsided one maps normal hand positions to
cramped/unreachable targets (calibration_data_2.txt: left controller was 50cm
more forward + 56cm more left than the right -> left arm never followed).
calibrate() prints "[calib] WARNING: controllers asymmetric" and sets
self.calib_warning when it detects this. If the arm ends up parked and
unresponsive, press 'c' in the viewer to drop the calibration and re-capture,
and/or re-run with --teleop-debug to see the reference and per-frame IK
target/residual.

First real-hardware checks, before trusting this for anything real:
1. With the robot at stance and the controller held still, calibrate, then
   move the controller straight up a few cm. The arm should rise, not
   drift sideways -- confirms the axis mapping isn't rotated/flipped.
2. Squeeze the trigger fully with the hand empty; confirm the fingers
   actually reach CLOSED (grip=1.0), not stuck partway, given the
   10.0->0.0 raw value's inversion below.
3. SCALE (below, default 0.5) compresses the operator's arm range into this
   arm's smaller ~0.49m envelope -- at 1.0, normal reaching/folding pushed the
   IK target out of the workspace and the divergence guard froze the arm
   (confirmed on hardware, calibration_data.txt). Tune via --teleop-scale:
   lower if the arm still hits its limits, raise toward 1.0 if motion feels
   sluggish.
4. The IK target is clamped to REACH_RADIUS (below) around the shoulder, so a
   controller reaching past the arm's envelope pins at the edge instead of
   diverging. With the TELEOP_HOME anchor, HOME sits ~0.35m from the shoulder
   and the brick grasp poses ~0.40m -- so from HOME the table is only a small
   forward+down move (a few cm of gripper travel, ~10cm of controller travel at
   scale 0.5). Move gradually; if "[teleop] ... reaching past the arm's
   envelope" prints you've pushed past the edge -- come back toward centre.

FpvStreamer (below) is the separate, previously-out-of-scope piece: pushing
stand_next_to_table.py's fpv_teleop camera view back to the headset, so the
operator sees the robot's own view instead of pass-through real-world video.
Per televuer's actual source (not guessed -- read directly): the client side
needs NO custom JS at all. TeleVuerWrapper.render_to_xr(image) just writes a
numpy array into a shared buffer; televuer's own already-built browser app
(bundled with the `vuer` package) picks it up and displays it in the XR
session over zmq or webrtc. So this is pure Python: offscreen-render a
camera, hand the array to render_to_xr() each frame. FpvStreamer is
duck-typed against `sink.render_to_xr(image)` the same way TeleopController
is duck-typed against tele_data -- decoupled from a hard televuer import,
testable with any stub that has that one method.

HandRetargeter (below) is the hand-tracking counterpart to trigger_to_grip --
only relevant if a caller switches TeleVuerWrapper to
use_hand_tracking=True, which supplies real per-finger keypoints
(left_hand_pos/right_hand_pos, (25,3) each) instead of a controller trigger.
Uses the dex_retargeting package (dexsuite/dex-retargeting, upstream --
deliberately NOT unitreerobotics/xr_teleoperate's forked pin; that fork's
nlopt<2.8.0 constraint has no prebuilt arm64 macOS wheel and fails to build
from source even with cmake/swig installed -- confirmed empirically, not
assumed. Upstream's nlopt>=2.8.0 installs cleanly with zero source builds).
The config in dex3_retargeting/unitree_dex3.yml is adapted from Unitree's
own asset (unitreerobotics/xr_teleoperate's assets/unitree_hand/) -- the
ONLY real change is renaming three now-separate fields
(target_link_human_indices_position/_vector/_dexpilot in the fork) into
upstream's single consolidated target_link_human_indices field, confirmed
by diffing the fork's and upstream's retargeting_config.py directly. Same
joint order, same URDF paths, same DexPilot indices, same low_pass_alpha --
nothing semantic changed. The URDF's mesh files are deliberately NOT
vendored -- confirmed empirically that dex_retargeting's optimizer only
needs the URDF's kinematic chain (joint types/axes/link transforms) to
build; it logs "Unable to resolve filename" for the missing meshes but
builds and retargets successfully regardless.
UNVERIFIED against real hand-tracking data or real hardware -- confirmed
only that the pipeline runs end-to-end (config loads, retarget() returns a
plausible 7-vector) against synthetic random keypoints, not that it
produces a good, natural grasp shape from an actual human hand.
"""
import pathlib
import numpy as np
import mujoco

from arm_ik import ArmIK
from grasp_primitive import hand_ctrl
from actuator_groups import LEFT_ARM, LEFT_HAND, RIGHT_ARM, RIGHT_HAND

SCALE = 0.5  # controller-motion -> robot-motion scale factor. A human's arm reach
             # (~0.7m + torso rotation) is bigger than this arm's ~0.49m envelope,
             # so at 1.0 a normal reach/fold pushes the IK target outside the
             # workspace and the divergence guard freezes the arm (confirmed on
             # hardware -- see calibration_data.txt). 0.5 compresses the operator's
             # range into the robot's; raise toward 1.0 if motion feels sluggish.
             # Per-run override: stand_next_to_table.py --teleop-scale.

# The delta mapping is anchored at calibration to TELEOP_HOME, NOT the current
# (stance) gripper pose. At stance the arm hangs nearly fully extended (~0.489m
# of ~0.499m reach), so anchoring there left almost no outward range -- any
# outward hand motion immediately clipped the REACH_RADIUS sphere and the arm
# looked frozen (calibration_data_4.txt: left arm tracked to sub-mm resid but was
# pinned at the edge every frame). TELEOP_HOME is a mid-workspace pose (~0.35m
# from the shoulder, out front at ~table height) so the operator has ~0.14m of
# range in every direction, and "reach to a brick" is a small forward+down move.
# calibrate() also ramps the arm from stance to here. Verified reachable to ~1cm
# under the per-frame re-solve drive, collision-free.
TELEOP_HOME = {"left": (0.24, 0.14, 0.82), "right": (0.24, -0.14, 0.82)}  # world frame
# The orientation (site quat, wxyz) the arm actually SETTLES at when driven to
# TELEOP_HOME by the per-frame re-solve -- measured, not a bare IK solve (which
# gives a different wrist config). --teleop-orientation uses this as the
# reference so a zero controller-rotation delta holds HOME cleanly instead of
# fighting between the position and a wrong orientation target.
TELEOP_HOME_QUAT = {"left":  (0.507,  -0.5578, 0.5105,  0.4138),
                    "right": (0.5061,  0.5585, 0.5113, -0.4129)}

# --- calibration gating (see TeleopController.try_calibrate) ---
# calibrate() captures the paired reference the ENTIRE delta mapping is built
# on. If it fires on a frame where the headset is still reporting a stale/default
# wrist pose (identity or zeros -- connection up, but controller tracking not
# locked on yet), every later step() computes a huge constant pos_delta and the
# IK target sits permanently out of reach, so the arm parks at a joint limit and
# barely responds. That race is the cause of the intermittent "the arm won't
# follow the controller, but it worked last run" failure. The gate below only
# captures once both controllers report a finite, non-identity pose that has
# held still for CALIB_SETTLE_TIME.
CALIB_SETTLE_TIME = 0.25     # s the operator must hold the controllers ~still,
                              # with valid tracking, before the reference is taken
CALIB_STABILITY_TOL = 0.03    # m; max controller travel between consecutive frames
                              # to still count as "settled"
REACH_RADIUS = 0.49          # m; the IK target is clamped to this sphere around the
                              # shoulder -- a big controller excursion then pins at the
                              # nearest REACHABLE point instead of diverging. Measured max
                              # reach ~0.499m, stance is ~0.489m shoulder->gripper (arm
                              # hangs nearly straight), so 0.49 keeps stance inside (no
                              # startup twitch) while clamping a raised-arm lunge. The
                              # brick grasp targets sit ~0.40m from the shoulder, inside.
MAX_TARGET_DELTA = 0.50       # m; secondary guard -- an IK target this far from the
                              # calibration reference means the reference is probably bad
                              # (stale controller at calibration); clamped and logged

# --- input filtering (see TeleopController._filter_vr / step) ---
# Real Quest controller tracking drops out constantly when the controllers leave
# the headset cameras' view (which they do in immersive mode -- the operator is
# looking at the FPV stream, not their hands): the pose freezes at its last value
# for a stretch, then SNAPS 30-60cm when re-acquired. Fed raw into IK + the
# kp=500 position actuators, those snaps slam the arm and the sim diverges (IK
# residual blows past 0.5m and never recovers). So: reject single-frame glitches,
# EMA-smooth what's left, and rate-limit how fast the IK target may translate.
# (xr_teleoperate ships weighted_moving_filter.py for the same reason.)
IK_ITERS = 8                 # per-frame IK iterations (was 4) -- track a moving target better
TELEOP_ORI_WEIGHT = 0.4      # --teleop-orientation (EXPERIMENTAL) only. arm_ik's DLS solver
                              # behaves erratically under a 6-DOF task at these tabletop poses
                              # -- measured: a 60deg controller rotation produces anywhere from
                              # ~25 to ~140deg of gripper rotation depending on axis, and
                              # position degrades 10-25cm. No weight tested fixes it. Real
                              # grasp-alignment orientation needs a weighted (Pinocchio/CasADi)
                              # IK, not this. This weight is just "least bad".
VR_FILTER_ALPHA = 0.2        # EMA weight on the new (accepted) controller pose each frame
VR_GLITCH_TOL = 0.10         # m; a single-frame controller jump larger than this is a
                              # tracking dropout/reacquire, not a hand -- rejected, hold last
VR_GLITCH_HOLD_TIME = 0.4    # s; if the rejected pose persists this long it's a real move
                              # (or the controller was set down and repositioned) -- re-seat
TARGET_MAX_SPEED = 2.0       # m/s; cap on IK-target translation speed. A burst of accepted
                              # motion (or a filter re-seat) then ramps in over ~0.1-0.2s
                              # instead of hitting the actuators as a step
DIVERGENCE_RESID = 0.15      # m; post-solve site-to-target error above this, sustained,
                              # means the arm has destabilised -- warn once, suggest 'c'

# Vendored, adapted Dex3 retargeting config -- see HandRetargeter's docstring
DEX3_RETARGETING_DIR = pathlib.Path(__file__).resolve().parent / "dex3_retargeting"

_SIDES = (
    ("left", "left_gripper_site", LEFT_ARM, LEFT_HAND),
    ("right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND),
)


def _se3_to_pos_quat(mat4x4):
    """Split a 4x4 SE(3) matrix into (pos(3,), quat(4,) wxyz). Assumes the
    standard [R t; 0 1] layout (rotation in the top-left 3x3, translation in
    the top 3 of the last column) -- the usual convention, but televuer's
    exact layout hasn't been cross-checked against real output."""
    mat4x4 = np.asarray(mat4x4)
    pos = mat4x4[:3, 3].copy()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat4x4[:3, :3].flatten())
    return pos, quat


def _site_quat(ik):
    """Current (scratch) world orientation of an ArmIK's gripper site, as a
    quaternion -- mirrors ArmIK.site_pos(), which only returns position.
    Same ik.scratch.site_xmat access pattern grasp_target_quat() already
    uses in grasp_primitive.py."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, ik.scratch.site_xmat[ik.site_id])
    return quat


def _relative_quat(q_ref, q_now):
    """World-frame rotation delta: the rotation that takes q_ref to q_now."""
    q_ref_inv = np.zeros(4)
    mujoco.mju_negQuat(q_ref_inv, q_ref)
    q_rel = np.zeros(4)
    mujoco.mju_mulQuat(q_rel, q_now, q_ref_inv)
    return q_rel


def _apply_relative_quat(q_rel, q_base):
    """Apply a world-frame rotation delta on top of a base orientation."""
    q_out = np.zeros(4)
    mujoco.mju_mulQuat(q_out, q_rel, q_base)
    return q_out


def trigger_to_grip(trigger_value):
    """televuer's raw convention is 10.0 (released) -> 0.0 (fully pressed);
    hand_ctrl's is 0 (open) -> 1 (closed). Clipped for safety against
    out-of-range values."""
    return float(np.clip(1.0 - trigger_value / 10.0, 0.0, 1.0))


class TeleopController:
    """Continuous two-arm VR follow controller.

    Usage:
        ctrl = TeleopController(m)
        ...
        while running:
            data = source.get_tele_data()          # real televuer or a stub
            if not ctrl.calibrated:
                ctrl.try_calibrate(m, d, data)     # gated -- see try_calibrate
            else:
                ctrl.step(m, d, hold_ctrl, data)
            mujoco.mj_step(m, d)

    step() writes only into the arm/hand slices for both sides (after first
    laying down hold_ctrl for everything else) -- never broadcasts across
    the full 43-actuator range, matching every other module here. LEG/WAIST
    stay pinned by hold_ctrl exactly like PickSequence.step() does.

    Real Quest controller poses are noisy and drop out constantly (the pose
    freezes then SNAPS 30-60cm when the controller leaves the headset
    cameras' view). step() runs them through _filter_vr (single-frame glitch
    rejection + EMA smoothing) and then rate-limits how fast the IK target
    may move (TARGET_MAX_SPEED), so a snap can't slam the position actuators
    and destabilise the arm; a sustained un-converged residual makes step()
    hold the arm in place until the input settles. IK is position-only by
    default (track_orientation) -- see __init__.
    """

    def __init__(self, m, hand_retargeter=None, debug=False, track_orientation=False,
                 scale=SCALE):
        """hand_retargeter, if given (a HandRetargeter instance), switches
        hand control from the controller-trigger path (hand_ctrl/
        trigger_to_grip) to real per-finger retargeting from tele_data's
        {side}_hand_pos keypoints -- only meaningful when the tele_data
        source actually supplies hand-tracking data (TeleVuerWrapper's own
        use_hand_tracking=True), not controller data. Omitting it (the
        default) preserves the original controller-trigger behavior exactly
        -- every existing call site is unaffected.

        debug=True prints the calibration reference poses once captured, and
        a throttled per-second line of vr_pos / |pos_delta| / IK target /
        residual from step() -- for diagnosing the arm not tracking.

        track_orientation=False (default) runs POSITION-ONLY IK. True adds a
        6-DOF solve following the controller's rotation-since-calibration --
        **EXPERIMENTAL and known to be poor**: arm_ik's DLS solver is erratic
        under a 6-DOF task at these tabletop poses (a controller rotation maps
        to a wildly axis-dependent gripper rotation, and position degrades
        10-25cm). It gives the operator *some* orientation influence but not
        faithful control. Real grasp-alignment orientation needs a weighted
        (Pinocchio/CasADi) IK replacing arm_ik -- see TELEOP_ORI_WEIGHT.

        scale (default SCALE) is the controller-motion -> robot-motion factor;
        step()'s own scale= arg still overrides per call."""
        self._iks = {side: ArmIK(m, site_name, arm_slice)
                     for side, site_name, arm_slice, _ in _SIDES}
        # body that carries each arm's first joint -- the shoulder anchor the
        # REACH_RADIUS workspace clamp is centred on (fixed: base welded, torso
        # pinned, so captured once in calibrate() from the live d.xpos)
        self._shoulder_bid = {side: int(m.jnt_bodyid[self._iks[side]._joint_ids[0]])
                              for side, _, _, _ in _SIDES}
        self._shoulder_pos = {}
        self._hand_slices = {side: hand_slice for side, _, _, hand_slice in _SIDES}
        self._hand_retargeter = hand_retargeter
        self._debug = debug
        self._track_orientation = track_orientation
        self._scale = float(scale)
        self._dt = m.opt.timestep
        self._steps_per_sec = max(int(round(1.0 / m.opt.timestep)), 1)
        self._calib_need = max(int(round(CALIB_SETTLE_TIME * self._steps_per_sec)), 1)
        self._diverge_need = max(int(round(0.3 / self._dt)), 1)  # ~0.3s sustained
        self._calib_streak = 0
        self._calib_prev_vr_pos = {}
        self._step_i = 0
        self._robot_ref_pos = {}
        self._robot_ref_quat = {}
        self._vr_ref_pos = {}
        self._vr_ref_quat = {}
        # input-filter state, (re)initialised in calibrate()
        self._vr_pos_prev = {}     # last RAW controller pos, for glitch detection
        self._vr_pos_filt = {}     # EMA-smoothed controller pos actually used
        self._vr_quat_filt = {}    # EMA-smoothed controller quat actually used
        self._glitch_frames = {}   # consecutive rejected-as-glitch frames, per side
        self._target_pos_prev = {} # last commanded IK target, for the rate limit
        self._diverge_frames = {}  # consecutive frames with a large post-solve residual
        self._diverge_warned = {}  # one-shot flag for the "holding position" message
        self.calibrated = False
        self.calib_warning = None   # set by calibrate() if the captured pose looks lopsided
        self.last_grip = {"left": 0.0, "right": 0.0}

    def try_calibrate(self, m, d, tele_data):
        """Gated calibration. Returns True only on the frame the reference is
        actually captured; until then returns False and the caller should
        keep the arms held where they are.

        Requires CALIB_SETTLE_TIME of consecutive frames that are ALL:
        motion_data_ready, a finite non-identity/non-zero 4x4 wrist pose for
        both hands, and within CALIB_STABILITY_TOL of the previous frame (the
        operator holding still). This is what stops calibrate() from firing
        on a stale first frame -- see the module-level calibration-gating
        comment for why that produces the intermittent 'arm won't follow'
        bug."""
        if self.calibrated:
            return False

        if not getattr(tele_data, "motion_data_ready", False):
            self._calib_reset("motion_data_ready is False")
            return False

        vr_pos = {}
        for side, _, _, _ in _SIDES:
            pose = np.asarray(getattr(tele_data, f"{side}_wrist_pose"), dtype=float)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                self._calib_reset(f"{side}_wrist_pose is not a finite 4x4")
                return False
            if np.linalg.norm(pose[:3, 3]) < 1e-6 or np.allclose(pose, np.eye(4), atol=1e-6):
                self._calib_reset(f"{side}_wrist_pose still identity/zero (tracking not locked)")
                return False
            vr_pos[side] = pose[:3, 3].copy()

        if self._calib_prev_vr_pos:
            jump = max(np.linalg.norm(vr_pos[s] - self._calib_prev_vr_pos[s]) for s in vr_pos)
            if jump > CALIB_STABILITY_TOL:
                if self._debug and self._calib_streak:
                    print(f"[calib] controllers still moving ({jump*100:.1f} cm/frame) -- streak reset")
                self._calib_streak = 0
                self._calib_prev_vr_pos = vr_pos
                return False
            self._calib_streak += 1
        else:
            self._calib_streak = 1
        self._calib_prev_vr_pos = vr_pos

        if self._calib_streak < self._calib_need:
            return False
        self.calibrate(m, d, tele_data)
        return True

    def _calib_reset(self, why):
        if self._debug and self._calib_streak:
            print(f"[calib] {why} -- streak reset")
        self._calib_streak = 0
        self._calib_prev_vr_pos = {}

    def request_recalibration(self):
        """Drop the current calibration; the next settled frames re-capture a
        fresh reference via try_calibrate(). Lets the operator re-center live
        -- after a bad capture, or after drifting to the edge of a
        comfortable arm range -- without restarting."""
        self.calibrated = False
        self._calib_streak = 0
        self._calib_prev_vr_pos = {}

    def calibrate(self, m, d, tele_data):
        """Capture the reference: the operator's CURRENT controller pose paired
        with TELEOP_HOME (a mid-workspace robot pose, NOT the current stance
        gripper pose -- see the TELEOP_HOME comment). The arm then ramps from
        wherever it is to HOME via the step() rate limiter. Prefer
        try_calibrate() -- calling this directly skips the stale-frame gate.
        Re-callable to re-center."""
        for side, _, _, _ in _SIDES:
            ik = self._iks[side]
            ik.sync(d.qpos)
            here = ik.site_pos()                       # the arm's ACTUAL pose now
            self._robot_ref_pos[side] = np.array(TELEOP_HOME[side], dtype=float)
            # orientation reference = the orientation the arm SETTLES at at HOME
            # (measured constant -- see TELEOP_HOME_QUAT), not the current stance
            # orientation and not a bare IK solve (both differ, and the mismatch
            # makes the 6-DOF solve fight position vs. a wrong orientation).
            self._robot_ref_quat[side] = np.array(TELEOP_HOME_QUAT[side], dtype=float)
            self._shoulder_pos[side] = d.xpos[self._shoulder_bid[side]].copy()
            vr_pos, vr_quat = _se3_to_pos_quat(getattr(tele_data, f"{side}_wrist_pose"))
            self._vr_ref_pos[side] = vr_pos
            self._vr_ref_quat[side] = vr_quat
            # seed the input filter at the reference so the first steps are no-ops
            self._vr_pos_prev[side] = vr_pos.copy()
            self._vr_pos_filt[side] = vr_pos.copy()
            self._vr_quat_filt[side] = vr_quat.copy()
            self._glitch_frames[side] = 0
            self._target_pos_prev[side] = here.copy()  # rate-limit ramps here -> HOME
            self._diverge_frames[side] = 0
            self._diverge_warned[side] = False
            if self._debug:
                print(f"[calib] {side}: vr_ref_pos={np.array2string(vr_pos, precision=3)}  "
                      f"HOME={np.array2string(self._robot_ref_pos[side], precision=3)}  "
                      f"(arm at {np.array2string(here, precision=3)}, will ramp to HOME)")

        # A neutral two-handed calibration pose has the controllers at similar x/z
        # and separated mostly in y. If they were far apart or lopsided, the
        # operator wasn't holding a relaxed symmetric pose -- and then a NORMAL
        # hand position maps, via the delta, to a cramped/unreachable arm target
        # and the arm looks frozen (this is exactly what calibration_data_2.txt
        # shows for the left arm: left controller was 50cm more forward + 56cm
        # more left than the right at calibration).
        lref, rref = self._vr_ref_pos["left"], self._vr_ref_pos["right"]
        dx, dz = abs(lref[0] - rref[0]), abs(lref[2] - rref[2])
        sep = float(np.linalg.norm(lref - rref))
        self.calib_warning = None
        if dx > 0.30 or dz > 0.30 or sep > 0.70:
            self.calib_warning = (f"controllers asymmetric at calibration "
                                  f"(dx={dx*100:.0f}cm dz={dz*100:.0f}cm sep={sep*100:.0f}cm)")
            print(f"[calib] WARNING: {self.calib_warning} -- you likely weren't holding a "
                  f"relaxed, symmetric pose (elbows ~90 deg, hands in front of your chest, "
                  f"~30cm apart). Normal hand positions will map to unreachable arm targets. "
                  f"Press 'c' to re-calibrate.")

        self.calibrated = True
        self._calib_streak = 0
        self._calib_prev_vr_pos = {}

    def _filter_vr(self, side, pos_raw, quat_raw):
        """Reject single-frame tracking glitches (a Quest controller that
        leaves the headset cameras' view freezes at its last pose, then SNAPS
        30-60cm when re-acquired), then EMA-smooth what's left. Returns
        (pos, quat, dropped) -- `dropped` True when this frame's raw pose was
        rejected and the held value returned instead."""
        jump = float(np.linalg.norm(pos_raw - self._vr_pos_prev[side]))
        self._vr_pos_prev[side] = pos_raw.copy()

        if jump > VR_GLITCH_TOL:
            self._glitch_frames[side] += 1
            if self._glitch_frames[side] * self._dt > VR_GLITCH_HOLD_TIME:
                # persisted too long to be a glitch -- the operator really moved
                # (or set a controller down and picked it up elsewhere); re-seat
                self._vr_pos_filt[side] = pos_raw.copy()
                self._vr_quat_filt[side] = quat_raw.copy()
                self._glitch_frames[side] = 0
                return self._vr_pos_filt[side].copy(), self._vr_quat_filt[side].copy(), False
            return self._vr_pos_filt[side].copy(), self._vr_quat_filt[side].copy(), True

        self._glitch_frames[side] = 0
        a = VR_FILTER_ALPHA
        self._vr_pos_filt[side] = (1.0 - a) * self._vr_pos_filt[side] + a * pos_raw
        qf = self._vr_quat_filt[side]
        if float(np.dot(qf, quat_raw)) < 0.0:   # keep both quats in the same hemisphere
            quat_raw = -quat_raw
        qf = (1.0 - a) * qf + a * quat_raw
        n = float(np.linalg.norm(qf))
        if n > 1e-9:
            self._vr_quat_filt[side] = qf / n
        return self._vr_pos_filt[side].copy(), self._vr_quat_filt[side].copy(), False

    def step(self, m, d, hold_ctrl, tele_data, scale=None):
        """Advance one control step for both arms. Must not be called before
        calibrate(). scale defaults to the instance's (see __init__)."""
        assert self.calibrated, "TeleopController.step() called before calibrate()"
        if scale is None:
            scale = self._scale
        d.ctrl[:] = hold_ctrl
        self._step_i += 1
        verbose = self._debug and (self._step_i % self._steps_per_sec == 0)
        max_target_step = TARGET_MAX_SPEED * self._dt
        for side, _, arm_slice, _ in _SIDES:
            ik = self._iks[side]
            ik.sync(d.qpos)  # re-seed from the live sim every frame -- the arm
                              # actually moves under real physics between calls,
                              # unlike a scripted sequence's own warm-started scratch

            vr_pos_raw, vr_quat_raw = _se3_to_pos_quat(getattr(tele_data, f"{side}_wrist_pose"))
            vr_pos, vr_quat, dropped = self._filter_vr(side, vr_pos_raw, vr_quat_raw)

            pos_delta = scale * (vr_pos - self._vr_ref_pos[side])
            delta_norm = float(np.linalg.norm(pos_delta))
            clamped = delta_norm > MAX_TARGET_DELTA
            if clamped:
                pos_delta = pos_delta * (MAX_TARGET_DELTA / delta_norm)
            raw_target = self._robot_ref_pos[side] + pos_delta

            # workspace clamp: pull the target onto the REACH_RADIUS sphere around
            # the shoulder, so a big controller excursion pins at the nearest
            # REACHABLE point instead of diverging (the table/brick targets sit
            # ~0.40m from the shoulder, inside; a raised-arm lunge sits outside)
            reach_vec = raw_target - self._shoulder_pos[side]
            reach = float(np.linalg.norm(reach_vec))
            reached_limit = reach > REACH_RADIUS
            if reached_limit:
                raw_target = self._shoulder_pos[side] + reach_vec * (REACH_RADIUS / reach)

            # rate-limit how fast the IK target may translate: a burst of accepted
            # motion, or a post-dropout filter re-seat, then ramps in over
            # ~0.1-0.2s instead of hitting the kp=500 position actuators as a step
            # (unfiltered snaps destabilised the arm -- ik_resid ran to 700mm)
            step_vec = raw_target - self._target_pos_prev[side]
            step_len = float(np.linalg.norm(step_vec))
            if step_len > max_target_step:
                target_pos = self._target_pos_prev[side] + step_vec * (max_target_step / step_len)
            else:
                target_pos = raw_target
            self._target_pos_prev[side] = target_pos

            if self._track_orientation:
                # apply the controller's rotation-since-calibration onto the arm's
                # natural HOME orientation
                quat_delta = _relative_quat(self._vr_ref_quat[side], vr_quat)
                target_quat = _apply_relative_quat(quat_delta, self._robot_ref_quat[side])
                q = ik.solve(target_pos, target_quat=target_quat, iters=IK_ITERS,
                             ori_weight=TELEOP_ORI_WEIGHT)
            else:
                target_quat = None  # position-only IK -- see __init__ docstring
                q = ik.solve(target_pos, iters=IK_ITERS)
            resid = float(np.linalg.norm(ik.site_pos() - target_pos))

            if resid > DIVERGENCE_RESID:
                self._diverge_frames[side] += 1
            else:
                self._diverge_frames[side] = 0
                self._diverge_warned[side] = False
            diverged = self._diverge_frames[side] > self._diverge_need
            if diverged:
                # arm can't reach the target and isn't recovering -- hold it
                # where it is rather than let it flail; resumes on its own once
                # resid drops back under DIVERGENCE_RESID
                d.ctrl[arm_slice] = d.qpos[arm_slice]
                if not self._diverge_warned[side]:
                    self._diverge_warned[side] = True
                    print(f"[teleop] {side} arm not converging (site {resid*100:.0f}cm off "
                          f"target) -- holding position until the input settles; press 'c' "
                          f"to re-calibrate if it stays stuck")
            else:
                d.ctrl[arm_slice] = q

            if self._step_i % self._steps_per_sec == 0:  # ~1/s, on regardless of debug
                if clamped:
                    print(f"[teleop] {side} IK target {delta_norm*100:.0f}cm past the "
                          f"calibration reference, clamped -- reference is likely bad "
                          f"(press 'c' to re-calibrate)")
                elif reached_limit:
                    print(f"[teleop] {side} controller reaching past the arm's "
                          f"{REACH_RADIUS*100:.0f}cm envelope -- target pinned at the edge; "
                          f"move back toward centre, or the table is just out of reach here")
            if verbose:
                ori_str = ""
                if self._track_orientation:
                    cur_q = np.zeros(4)
                    mujoco.mju_mat2Quat(cur_q, ik.scratch.site_xmat[ik.site_id])
                    oe = np.zeros(3)
                    mujoco.mju_subQuat(oe, np.asarray(target_quat), cur_q)
                    ori_str = f" ori_err={np.degrees(np.linalg.norm(oe)):4.1f}deg"
                print(f"[teleop {self._step_i:6d}] {side}: "
                      f"vr_pos={np.array2string(vr_pos, precision=3)} "
                      f"|delta|={delta_norm*100:5.1f}cm "
                      f"target={np.array2string(target_pos, precision=3)} "
                      f"ik_resid={resid*1000:5.1f}mm{ori_str}"
                      f"{'  DROPOUT' if dropped else ''}"
                      f"{'  REACH' if reached_limit else ''}"
                      f"{'  DIVERGED' if diverged else ''}")

            if self._hand_retargeter is not None:
                # real per-finger control -- bypasses hand_ctrl()'s 3-pose
                # interpolation entirely, see HandRetargeter's docstring
                hand_pos = getattr(tele_data, f"{side}_hand_pos")
                q_hand = self._hand_retargeter.retarget(side, hand_pos)
                self.last_grip[side] = None  # not a single scalar in this mode
                d.ctrl[self._hand_slices[side]] = q_hand
            else:
                trigger_value = getattr(tele_data, f"{side}_ctrl_triggerValue")
                grip = trigger_to_grip(trigger_value)
                self.last_grip[side] = grip
                d.ctrl[self._hand_slices[side]] = hand_ctrl(grip, side)


class FpvStreamer:
    """Offscreen-renders one camera and pushes frames to a televuer-like sink
    every step() call, rate-limited to roughly `fps` real-time rather than
    every physics step -- offscreen rendering is comparatively expensive
    (unlike the pure-ctrl-array work everything else here does), and the
    headset doesn't need physics-rate frames. See module docstring for why
    no client-side code is needed for this at all.

    Usage:
        streamer = FpvStreamer(m, "fpv_teleop")
        ...
        while running:
            streamer.step(d, tele_source)   # once per control step; internally
                                             # rate-limited, most calls no-op
            mujoco.mj_step(m, d)

    `sink` (tele_source above) only needs a `.render_to_xr(image)` method --
    a real televuer.TeleVuerWrapper works, and so does any stub with that
    one method, matching TeleopController's tele_data duck-typing.
    """

    def __init__(self, m, camera_name, height=480, width=640, fps=30):
        self.renderer = mujoco.Renderer(m, height=height, width=width)
        self.camera_name = camera_name
        # mirrors verify_stack_sequence.py --record's own frame_every pattern:
        # sample every Nth physics step to approximate fps in real time
        self.frame_every = max(int(round((1.0 / fps) / m.opt.timestep)), 1)
        self._step_count = 0

    def step(self, d, sink):
        """Call once per control step. Returns True on steps a frame was
        actually rendered and sent (most calls no-op, by design)."""
        self._step_count += 1
        if self._step_count % self.frame_every != 0:
            return False
        self.renderer.update_scene(d, camera=self.camera_name)
        frame = self.renderer.render()
        # televuer's own render loop applies cv2.cvtColor(BGR2RGB) to whatever
        # we hand it (it assumes an OpenCV-style BGR source) -- MuJoCo's
        # Renderer.render() is already RGB, so pre-swap to BGR here so their
        # conversion undoes it and the final displayed color is correct.
        # Confirmed on real hardware: without this, a wood-brown table showed
        # up blue in the headset (R/B channels swapped).
        sink.render_to_xr(frame[..., ::-1])
        return True


class HandRetargeter:
    """Retargets televuer hand-tracking keypoints onto Dex3-1 joint angles,
    via dex_retargeting -- the hand-tracking counterpart to trigger_to_grip.
    See module docstring for the fork-vs-upstream story and what's verified.

    Lazy-imports dex_retargeting/yaml (not installed in most dev environments
    -- only needed if a caller actually uses hand-tracking mode), matching
    the pattern already used for televuer in stand_next_to_table.py.

    Usage:
        retargeter = HandRetargeter()
        ...
        q = retargeter.retarget("right", tele_data.right_hand_pos)  # (7,)
        d.ctrl[RIGHT_HAND] = q
    """

    def __init__(self, config_dir=None):
        try:
            import yaml
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as e:
            raise ImportError(
                "HandRetargeter needs the upstream dex_retargeting package: "
                "pip install dex_retargeting (plus CPU torch: pip install torch "
                "--index-url https://download.pytorch.org/whl/cpu). NOT "
                "unitreerobotics/xr_teleoperate's forked pin -- see this module's "
                "docstring for why.") from e

        config_dir = pathlib.Path(config_dir) if config_dir else DEX3_RETARGETING_DIR
        RetargetingConfig.set_default_urdf_dir(str(config_dir))
        with open(config_dir / "unitree_dex3.yml") as f:
            cfg = yaml.safe_load(f)

        self._retargeting = {}
        self._hardware_reindex = {}
        for side in ("left", "right"):
            retargeting = RetargetingConfig.from_dict(cfg[side]).build()
            self._retargeting[side] = retargeting
            # retargeting.joint_names is dex_retargeting's own internal order
            # (alphabetical-ish); reindex to OUR hardware/ctrl order (the
            # YAML's own target_joint_names, verified to already match
            # LEFT_HAND/RIGHT_HAND's documented convention -- left:
            # thumb,middle,index; right: thumb,index,middle)
            hardware_names = cfg[side]["target_joint_names"]
            self._hardware_reindex[side] = [retargeting.joint_names.index(n)
                                             for n in hardware_names]

    def retarget(self, side, hand_pos):
        """hand_pos: (25,3) keypoints (televuer's {side}_hand_pos). Returns a
        (7,) array of joint angles in OUR hardware order, ready to write
        directly into d.ctrl[LEFT_HAND or RIGHT_HAND] -- no hand_ctrl(), no
        grip scalar, this is real per-finger control."""
        hand_pos = np.asarray(hand_pos)
        retargeting = self._retargeting[side]
        indices = retargeting.optimizer.target_link_human_indices
        ref_value = hand_pos[indices[1, :]] - hand_pos[indices[0, :]]
        q = retargeting.retarget(ref_value)
        return q[self._hardware_reindex[side]]
