"""
Damped least-squares position IK for one G1 arm, solved against a gripper site
(left_gripper_site / right_gripper_site, injected by make_fixed_base.py).

Solves against a private scratch MjData, never the live simulation's qpos --
the arm is position-actuated (kp=500, dampratio=1), so the correct pattern is
to solve for a target joint angle and write it into d.ctrl, letting the PD
actuators converge to it physically. Writing an IK result straight into the
live d.qpos would teleport the arm, bypassing physics entirely.

Only LEG/WAIST need to stay pinned (see CLAUDE.md); this module only ever
touches one arm's 7 dof (LEFT_ARM or RIGHT_ARM from stand_next_to_table.py),
so it composes directly with that pinning -- it has no opinion about the rest
of the robot.

Includes a nullspace posture bias toward the stance pose. Necessary, not
cosmetic: solving a 3-DOF position target with a 7-DOF arm leaves 4 DOF of
redundancy, and plain DLS has no preference among the solutions that satisfy
the primary task -- verified empirically (verify_arm_ik.py) that without a
bias, the solver can find a valid-in-position but physically awkward
"elbow-down, wrist-low" configuration that grazes the table even when the
target itself sits comfortably inside the arm's reach envelope. Biasing
toward the stance pose (arm raised, per g1_fixed_upper.xml's "stand"
keyframe) keeps the redundant DOF near a known-clear posture, deviating only
as far as the primary task requires.

Tradeoff: the posture bias trades a small amount of steady-state task
accuracy for a well-behaved configuration. dq_task shrinks toward zero as
position error shrinks, but dq_posture doesn't (it's proportional to distance
from the preferred posture in the nullspace, not to task error), so the two
settle at a nonzero equilibrium instead of driving error exactly to zero.
posture_gain=0.12 was picked empirically (verify_arm_ik.py) as the largest
gain that still converges within a 2cm tolerance for the two tested targets
while reliably avoiding the table-grazing configuration at gain=0; retune if
either target changes materially.
"""
import numpy as np
import mujoco


class ArmIK:
    """IK for one arm's gripper site. `joint_slice` must be a qpos/qvel/ctrl
    slice for this model's arm dof (e.g. LEFT_ARM = slice(15, 22)) -- valid
    because this model has 43 single-dof hinge joints and no free joints, so
    qpos index == dof index == joint declaration order for the robot."""

    def __init__(self, model, site_name, joint_slice, damping=0.15, step_scale=0.6,
                 posture_gain=0.12):
        self.model = model
        self.site_id = model.site(site_name).id
        self.joint_slice = joint_slice
        self.damping = damping
        self.step_scale = step_scale
        self.posture_gain = posture_gain  # 0 disables the nullspace posture bias

        self._joint_ids = [j for j in range(model.njnt)
                            if joint_slice.start <= model.jnt_qposadr[j] < joint_slice.stop]
        assert len(self._joint_ids) == joint_slice.stop - joint_slice.start, (
            f"joint_slice {joint_slice} doesn't line up with whole joints in this model")
        self._qlo = model.jnt_range[self._joint_ids, 0].copy()
        self._qhi = model.jnt_range[self._joint_ids, 1].copy()
        self._n = joint_slice.stop - joint_slice.start

        self.scratch = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._q_pref = None
        self._synced = False

    def sync(self, qpos_full, pref=None):
        """(Re)seed the scratch state from the robot's actual current qpos --
        call once before the first solve(), and again whenever the pinned
        LEG/WAIST pose changes (it doesn't, under the current stance-holding
        design) or if the IK guess needs to be snapped back to reality.
        `pref` sets the nullspace posture-bias target (default: this arm's
        slice of qpos_full itself, i.e. whatever pose it's called with --
        typically the stance pose)."""
        self.scratch.qpos[:] = qpos_full
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        self._q_pref = np.asarray(pref if pref is not None else qpos_full[self.joint_slice]).copy()
        self._synced = True

    def solve(self, target_pos, target_quat=None, iters=6, tol=1e-4, ori_tol=1e-2,
              ori_weight=0.5):
        """Returns the solved qpos for this arm's joint_slice (7,). Warm-starts
        from the scratch state's current arm angles (whatever the last solve()
        or sync() left them at), so repeated calls converge smoothly toward a
        moving target across frames instead of resolving from scratch.

        target_quat is optional (default None = position-only, the original
        3-DOF behavior). When given, adds a 3-DOF orientation task using the
        standard MuJoCo IK pattern (mju_subQuat against the rotational
        Jacobian, stacked under the position rows) -- necessary for grasping:
        verified empirically that plain position-only IK reaching for a
        tabletop target naturally settles with the reach axis near-horizontal
        (dot product with world -Z of only ~0.03), not pointing down at the
        object, because nothing in a 3-DOF task constrains orientation at all.
        ori_weight scales the orientation rows relative to position (position
        error is in metres, orientation error in radians -- comparable
        magnitudes here, but the knob exists to retune if targets change)."""
        assert self._synced, "call sync() once before solve()"
        use_ori = target_quat is not None
        for _ in range(iters):
            site_pos = self.scratch.site_xpos[self.site_id]
            pos_err = target_pos - site_pos
            mujoco.mj_jacSite(self.model, self.scratch, self._jacp, self._jacr, self.site_id)
            if use_ori:
                cur_quat = np.zeros(4)
                mujoco.mju_mat2Quat(cur_quat, self.scratch.site_xmat[self.site_id])
                ori_err = np.zeros(3)
                mujoco.mju_subQuat(ori_err, np.asarray(target_quat), cur_quat)
                err = np.concatenate([pos_err, ori_weight * ori_err])
                J = np.vstack([self._jacp[:, self.joint_slice],
                                ori_weight * self._jacr[:, self.joint_slice]])  # 6x7
                converged = (np.linalg.norm(pos_err) < tol) and (np.linalg.norm(ori_err) < ori_tol)
            else:
                err = pos_err
                J = self._jacp[:, self.joint_slice]                      # 3x7
                converged = np.linalg.norm(pos_err) < tol
            lam2 = self.damping ** 2
            Jsharp = J.T @ np.linalg.inv(J @ J.T + lam2 * np.eye(J.shape[0]))  # 7xN, damped pseudoinverse
            dq_task = Jsharp @ err                                       # 7,
            if converged and self.posture_gain == 0:
                break
            # nullspace posture bias: pulls toward q_pref using only the DOF
            # combinations that don't move the site (N = I - J#J), so it never
            # fights the primary position/orientation task.
            q_now = self.scratch.qpos[self.joint_slice]
            N = np.eye(self._n) - Jsharp @ J
            dq_posture = self.posture_gain * (N @ (self._q_pref - q_now))
            q = q_now + self.step_scale * (dq_task + dq_posture)
            self.scratch.qpos[self.joint_slice] = np.clip(q, self._qlo, self._qhi)
            mujoco.mj_kinematics(self.model, self.scratch)
            mujoco.mj_comPos(self.model, self.scratch)
            if converged:
                break
        return self.scratch.qpos[self.joint_slice].copy()

    def site_pos(self):
        """Current (scratch) world position of this arm's gripper site."""
        return self.scratch.site_xpos[self.site_id].copy()
