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

First real-hardware checks, before trusting this for anything real:
1. With the robot at stance and the controller held still, calibrate, then
   move the controller straight up a few cm. The arm should rise, not
   drift sideways -- confirms the axis mapping isn't rotated/flipped.
2. Squeeze the trigger fully with the hand empty; confirm the fingers
   actually reach CLOSED (grip=1.0), not stuck partway, given the
   10.0->0.0 raw value's inversion below.
3. SCALE=1.0 (below) assumes roughly human-arm-scale controller motion
   maps 1:1 onto this robot's own (also roughly human-scale) reach --
   untested; if the mapped motion feels too twitchy or too sluggish,
   retune SCALE first before suspecting anything else.

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

SCALE = 1.0  # controller-motion -> robot-motion scale factor; untested, see above

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
            if not ctrl.calibrated and data.motion_data_ready:
                ctrl.calibrate(m, d, data)
            elif ctrl.calibrated:
                ctrl.step(m, d, hold_ctrl, data)
            mujoco.mj_step(m, d)

    step() writes only into the arm/hand slices for both sides (after first
    laying down hold_ctrl for everything else) -- never broadcasts across
    the full 43-actuator range, matching every other module here. LEG/WAIST
    stay pinned by hold_ctrl exactly like PickSequence.step() does.
    """

    def __init__(self, m, hand_retargeter=None):
        """hand_retargeter, if given (a HandRetargeter instance), switches
        hand control from the controller-trigger path (hand_ctrl/
        trigger_to_grip) to real per-finger retargeting from tele_data's
        {side}_hand_pos keypoints -- only meaningful when the tele_data
        source actually supplies hand-tracking data (TeleVuerWrapper's own
        use_hand_tracking=True), not controller data. Omitting it (the
        default) preserves the original controller-trigger behavior exactly
        -- every existing call site is unaffected."""
        self._iks = {side: ArmIK(m, site_name, arm_slice)
                     for side, site_name, arm_slice, _ in _SIDES}
        self._hand_slices = {side: hand_slice for side, _, _, hand_slice in _SIDES}
        self._hand_retargeter = hand_retargeter
        self._robot_ref_pos = {}
        self._robot_ref_quat = {}
        self._vr_ref_pos = {}
        self._vr_ref_quat = {}
        self.calibrated = False
        self.last_grip = {"left": 0.0, "right": 0.0}

    def calibrate(self, m, d, tele_data):
        """Call once, as soon as tele_data.motion_data_ready is True (the
        first real frame -- calibrating against stale/default data before
        that would capture a meaningless reference). Pairs the operator's
        CURRENT controller pose with the robot's CURRENT gripper-site pose
        as the reference for the delta mapping described in the module
        docstring. Re-callable if the operator wants to re-center (e.g.
        after reaching the edge of a comfortable arm range)."""
        for side, _, _, _ in _SIDES:
            ik = self._iks[side]
            ik.sync(d.qpos)
            self._robot_ref_pos[side] = ik.site_pos()
            self._robot_ref_quat[side] = _site_quat(ik)
            vr_pos, vr_quat = _se3_to_pos_quat(getattr(tele_data, f"{side}_wrist_pose"))
            self._vr_ref_pos[side] = vr_pos
            self._vr_ref_quat[side] = vr_quat
        self.calibrated = True

    def step(self, m, d, hold_ctrl, tele_data, scale=SCALE):
        """Advance one control step for both arms. Must not be called before
        calibrate()."""
        assert self.calibrated, "TeleopController.step() called before calibrate()"
        d.ctrl[:] = hold_ctrl
        for side, _, arm_slice, _ in _SIDES:
            ik = self._iks[side]
            ik.sync(d.qpos)  # re-seed from the live sim every frame -- the arm
                              # actually moves under real physics between calls,
                              # unlike a scripted sequence's own warm-started scratch

            vr_pos, vr_quat = _se3_to_pos_quat(getattr(tele_data, f"{side}_wrist_pose"))
            pos_delta = scale * (vr_pos - self._vr_ref_pos[side])
            target_pos = self._robot_ref_pos[side] + pos_delta

            quat_delta = _relative_quat(self._vr_ref_quat[side], vr_quat)
            target_quat = _apply_relative_quat(quat_delta, self._robot_ref_quat[side])

            q = ik.solve(target_pos, target_quat=target_quat, iters=4)
            d.ctrl[arm_slice] = q

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
