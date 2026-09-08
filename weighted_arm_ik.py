"""
WeightedArmIK: a weighted whole-arm IK for one G1 arm, packaged as a DROP-IN
PARALLEL to arm_ik.py's ArmIK -- built for live VR teleop, never touching the
scripted pick/stack pipeline.

WHY THIS EXISTS -- and why arm_ik.py is NOT touched
--------------------------------------------------
arm_ik.py's damped-least-squares solver is the IK for the entire verified
scripted pipeline (PickSequence / ApproachPath / every verify_*). It works
there and must not change. But for live VR teleop it has two failure modes
(documented in CLAUDE.md "Teleop Input (Phase 2)" and the memory notes):

  (a) it parks ~2-5 cm short on extended reaches -- its nullspace posture
      bias settles at a NONZERO task-error equilibrium.
  (b) it is ERRATIC under a 6-DOF position+orientation task at tabletop
      poses -- a 45-60 deg controller rotation produced anywhere from ~25 to
      ~160 deg of gripper rotation depending on axis, with 10-25 cm of
      position degradation AND the non-moving arm drifting 13-17 cm. No
      ori_weight fixed it. That is why --teleop-orientation shipped
      EXPERIMENTAL.

A first attempt at fixing this (a hand-rolled weighted Levenberg-Marquardt on
a Pinocchio kinematic model, still here as solver="lm") measurably improved
orientation but was only faithful on one of three axes and needed real
tuning to avoid its own failure modes (see "solver='lm'/'ipopt'" below). On a
real headset run it was NOT enough -- orientation still wasn't faithful and
some joints weren't folding the way they should. The actual fix is
**solver="mink" (the default)**: kevinzakka/mink, a maintained differential
IK library built directly on MuJoCo (Apache-2.0). It solves a proper
QP-based task (position + orientation + posture, with real joint-limit
constraints) using MuJoCo's OWN model and Jacobians -- no separate
Pinocchio model, no coordinate-frame offset to get wrong, and a much
better-conditioned redundant-DOF resolution than the hand-rolled version.

Measured head-to-head vs DLS AND the hand-rolled "lm" solver, through the
REAL TeleopController pipeline (calibration, input filter, rate limiter,
divergence guard -- not a simplified standalone harness), 45 deg controller
rotation about each site axis from TELEOP_HOME:

  |                        | DLS (ArmIK)     | "lm" (Pinocchio)  | "mink" (default)  |
  |------------------------|-----------------|-------------------|-------------------|
  | gripper rotation X/Y/Z | 127/159/43 deg  | 27/41/29 deg      | 45.5/43.2/45.3 deg|
  | non-moving arm drift   | 13-17 cm        | ~0.1 cm           | ~0.01 cm          |
  | moving-arm pos hold    | 3-11 cm         | 1-2 cm            | 0.4/6.8/2.0 cm    |
  | position-only reach*   | 0.35/2.1/6.1 cm | 1.1/6.8/7.4 cm    | 0.36/1.95/0.41 cm |
  | solve time             | --              | ~1 ms             | ~0.7 ms           |

  * pregrasp(+12cm)/grasp(+4cm)/HOME+8cm-forward, same three targets each column.

mink is faithful on ALL THREE axes (not just one -- X/Z within ~1deg of
commanded, Y within ~2deg), holds the non-moving arm essentially perfectly
still, and tracks position at least as well as DLS across every target
tested -- a clear win on both (a) and (b), not just (b).

Two real, solver-specific things had to be fixed to get the numbers above,
both found by tracing per-joint commanded-vs-actual qpos (the same technique
that found the DLS park-short and the wrist-torque issue below) -- worth
knowing before retuning further:

1. **`TELEOP_HOME_QUAT` (teleop_control.py) is solver-specific.** It was
   measured for arm_ik.py's DLS solver settling at TELEOP_HOME; mink settles
   at a genuinely different natural orientation there (~57deg apart,
   measured). Reusing DLS's constant for mink recreated the exact
   "position fights a wrong orientation reference" problem that constant was
   invented to avoid in the first place, just for a different solver
   (measured: held HOME position to only ~6-7cm instead of <1cm).
   `teleop_control.py` now has a separate `MINK_HOME_QUAT`, and
   `TeleopController.calibrate()` picks the one matching the active solver.
2. **`mink_posture_cost_arm` needs to be high enough to prevent the
   redundant-DOF solution from wandering frame-to-frame.** Mink fully
   converges its QP every call (unlike DLS's small warm-started steps), so
   with too little posture weight (tried 0.01-0.05) a STATIC target's
   commanded joint config kept changing slightly frame to frame even though
   the TASK residual stayed near zero each time -- the physical arm chased a
   moving target in joint space and never settled (measured: 6cm+ static
   position error that wasn't improving). `mink_posture_cost_arm=0.1`
   (the default) stabilizes it (<1cm) without reintroducing park-short.

One thing NO solver fixes: the G1 arm's WRIST actuators are torque-limited
(`actuatorfrcrange="-5 5"`, i.e. +/-5 Nm) -- the Dex3 hand on a forward lever
easily exceeds that. This shows up two ways: (i) at fully-extended
low-forward POSITION reaches, wrist_pitch sags to a gravity/torque balance
past its own joint limit regardless of what any IK commands; (ii) a Y-axis
ORIENTATION command that reconfigures the wrist substantially (measured
above: 6.8cm position hold vs X/Z's <2cm, even though the Y rotation itself
is just as faithful) runs into the same ceiling -- the commanded joint
angles are kinematically correct but the actuator can't fully develop the
torque to track them. Both are an arm-hardware ceiling, not an IK bug;
mink's better posture/limit handling narrows it (see the table above, still
tighter than DLS on every one of these) but cannot eliminate it.

Kept behind stand_next_to_table.py's --teleop-weighted-ik flag (default
OFF): if this path misbehaves on the real headset, drop the flag and you're
straight back on the confirmed-working DLS position teleop with zero risk to
any scripted grasp/stack work.

CALL SURFACE -- identical to arm_ik.ArmIK, so TeleopController swaps solvers
in one line and its internals (_shoulder_bid via ik._joint_ids[0],
_site_quat(ik) reading ik.scratch.site_xmat[ik.site_id], the --teleop-debug
block) keep working unchanged, regardless of which solver is selected:

    ik = WeightedArmIK(m, "right_gripper_site", RIGHT_ARM, mjcf_path=...)
    ik.sync(d.qpos)                       # reseed from the live sim each frame
    q = ik.solve(target_pos, target_quat) # (7,) joint angles for d.ctrl[slice]
    ik.site_pos()                         # (3,) world pos of the gripper site
    ik.site_id, ik.scratch, ik._joint_ids # mirrored attributes

SOLVERS
-------
solver="mink" (DEFAULT) -- a mink.Configuration built directly from `model`
             (the SAME MjModel the caller already has -- no separate
             kinematic model, no coordinate offset). Per instance: one
             FrameTask on this arm's gripper site (position_cost /
             orientation_cost), one PostureTask over the WHOLE model
             weighted heavily off this arm's own 7 DOF (so the QP
             concentrates all its motion on this arm, matching the
             "only this arm's slice moves" invariant everywhere else in this
             repo) and lightly on it (to resolve the arm's own redundancy),
             plus a ConfigurationLimit (real joint-limit QP constraints, not
             a penalty term). Solved via `qpsolvers`' "quadprog" backend
             (already installed in this env; MIT-licensed alternatives like
             daqp/osqp are drop-in via `mink_qp_solver` if ever needed).
             `mink.solve_ik`'s dt is fixed to the model's OWN physics
             timestep -- using a larger value was tried and measurably
             caused real drift (the QP's velocity gets integrated too far
             per call). Needs `pip install mink` (pulls in `qpsolvers`).
solver="lm"  -- the original hand-rolled weighted Levenberg-Marquardt on a
             Pinocchio kinematic model (still correct, still available, but
             measurably worse than mink on orientation faithfulness -- see
             the table above). Kept as a fallback in case mink ever has a
             dependency or licensing problem on a given machine. Needs
             `pinocchio`.
solver="ipopt" -- the same weighted cost as solver="lm" but solved as a
             CasADi Opti/IPOPT NLP (the "same approach as xr_teleoperate's
             robot_arm_ik.py" this repo was originally asked for). This
             cmeel Pinocchio build has no `pinocchio.casadi`, so the task
             term is a `casadi.Callback` and IPOPT runs with an L-BFGS
             Hessian -- it converges but 60-120 ms/solve, decimate it
             (`ik_every`) to stay real-time. Needs `casadi`. Kept for
             completeness, not recommended for live use.
"""
import numpy as np
import mujoco

_ARM_JOINTS = {
    "left":  ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
              "left_shoulder_yaw_joint", "left_elbow_joint",
              "left_wrist_roll_joint", "left_wrist_pitch_joint",
              "left_wrist_yaw_joint"],
    "right": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
              "right_shoulder_yaw_joint", "right_elbow_joint",
              "right_wrist_roll_joint", "right_wrist_pitch_joint",
              "right_wrist_yaw_joint"],
}

# Pinocchio-root -> MuJoCo-world is a pure translation of the welded pelvis
# height (solver="lm"/"ipopt" only -- solver="mink" needs no such offset,
# it operates directly on the caller's own MuJoCo model). Computed in
# __init__ (not trusted from this constant), but this is the expected value
# and a sanity bound.
_EXPECTED_BASE_OFFSET_Z = 0.793


class WeightedArmIK:
    """Weighted IK for one arm's gripper site. `joint_slice` is that arm's
    qpos/ctrl slice (LEFT_ARM = slice(15,22) / RIGHT_ARM = slice(29,36));
    valid because this model has 43 single-DOF hinge joints and no free
    joints, so qpos index == dof index == joint declaration order (holds for
    both nq- and nv-space indexing, which is why `joint_slice` alone is
    enough to index the mink PostureTask's per-DOF cost vector too).

    `mjcf_path` is only needed for solver="lm"/"ipopt" (it must point at the
    robot MJCF `model` was built from -- `g1_fixed_upper.xml` inside
    MODEL_DIR -- parsed by Pinocchio to build the arm kinematics, with its
    joint order and limits asserted against `model` at construction so the
    two cannot drift). solver="mink" (the default) ignores it -- mink builds
    directly from `model`, so pass it or not, it costs nothing either way.
    """

    def __init__(self, model, site_name, joint_slice, *, mjcf_path=None,
                 solver="mink", ik_every=1,
                 # solver="mink" params
                 mink_pos_cost=1.0, mink_ori_cost=0.5,
                 mink_posture_cost_arm=0.1, mink_posture_cost_rest=10.0,
                 mink_lm_damping=1e-2, mink_qp_damping=1e-6,
                 mink_qp_solver="quadprog",
                 # solver="lm"/"ipopt" params (Pinocchio-based fallback)
                 pos_weight=3000.0, ori_weight=1.0, posture_weight=5e-3,
                 smooth_weight=0.1, limit_weight=2.0, limit_margin_frac=0.12,
                 solve_blend=1.0, max_iters=25, dq_max=0.3,
                 ipopt_max_iter=60, debug=False):
        self.model = model
        self.site_name = site_name
        self.site_id = model.site(site_name).id
        self.joint_slice = joint_slice
        self._solver = solver
        self._ik_every = max(int(ik_every), 1)
        self._debug = debug

        side = "left" if "left" in site_name else "right"
        self._side = side
        arm_joint_names = _ARM_JOINTS[side]
        self._n = joint_slice.stop - joint_slice.start
        assert self._n == 7, "expected a 7-DOF arm slice"

        # --- MuJoCo joint ids for this arm slice (mirrors ArmIK._joint_ids) ---
        self._joint_ids = [j for j in range(model.njnt)
                           if joint_slice.start <= model.jnt_qposadr[j] < joint_slice.stop]
        assert len(self._joint_ids) == self._n, (
            f"joint_slice {joint_slice} doesn't line up with whole joints")
        self._qlo = model.jnt_range[self._joint_ids, 0].copy()
        self._qhi = model.jnt_range[self._joint_ids, 1].copy()

        # --- scratch MjData for FK / site reporting (mirrors ArmIK.scratch) ---
        self.scratch = mujoco.MjData(model)
        q_stance_full = self._stance_qpos(model, model.nq)
        self.scratch.qpos[:] = q_stance_full
        mujoco.mj_kinematics(model, self.scratch)
        mujoco.mj_comPos(model, self.scratch)
        q_arm_ref = np.asarray(self.scratch.qpos[joint_slice]).copy()

        # posture target / warm-start state, (re)seeded by sync()
        self._q_rest = q_arm_ref.copy()
        self._q_seed = q_arm_ref.copy()
        self._q_prev = q_arm_ref.copy()
        self._q_last = q_arm_ref.copy()          # last solve() output (for decimation)
        self._synced = False
        self._call_i = 0
        # used by solve() regardless of solver
        self._solve_blend = float(solve_blend)

        if solver == "mink":
            self._init_mink(model, site_name, joint_slice, q_stance_full,
                            mink_pos_cost, mink_ori_cost, mink_posture_cost_arm,
                            mink_posture_cost_rest, mink_lm_damping,
                            mink_qp_damping, mink_qp_solver)
        else:
            self._limit_margin_frac = float(limit_margin_frac)
            self._lim_margin = self._limit_margin_frac * (self._qhi - self._qlo)
            self._pos_w = float(pos_weight)
            self._ori_w = float(ori_weight)
            self._posture_w = float(posture_weight)
            self._smooth_w = float(smooth_weight)
            self._limit_w = float(limit_weight)
            self._max_iters = int(max_iters)
            self._dq_max = float(dq_max)
            self._ipopt_max_iter = int(ipopt_max_iter)
            self._init_pinocchio(model, site_name, joint_slice, arm_joint_names,
                                 mjcf_path, q_stance_full)
            if solver == "ipopt":
                self._build_ipopt()

    # ------------------------------------------------------------------ helpers
    def _stance_qpos(self, model, nq):
        for name in ("stand_at_table", "stand"):
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
            if kid >= 0:
                return model.key_qpos[kid][:nq].copy()
        if model.nkey > 0:
            return model.key_qpos[0][:nq].copy()
        return np.zeros(nq)

    # ================================================================= mink
    def _init_mink(self, model, site_name, joint_slice, q_stance_full,
                   pos_cost, ori_cost, posture_cost_arm, posture_cost_rest,
                   lm_damping, qp_damping, qp_solver):
        try:
            import mink
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "WeightedArmIK(solver='mink') needs mink: `pip install mink` "
                "(pulls in qpsolvers). Or use solver='lm', which needs only "
                "pinocchio.") from e
        self._mink = mink
        # mink.solve_ik() calls configuration.check_limits(safety_break=False)
        # every call, which logs a root-logger WARNING for any joint outside
        # its declared range -- and the stance keyframe's CLOSED Dex3 thumb
        # joints sit ~0.003 rad past their own limit (a pre-existing rounding
        # artifact in the keyframe's authored literals, confirmed harmless and
        # unrelated to the ARM this class controls). Uncaught, that's one
        # warning per joint per solve() call -- i.e. per physics step. Raise
        # the root logger level once so real teleop use isn't log-flooded;
        # does not affect solving, only this log message.
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        self._mink_ori_cost_default = float(ori_cost)
        self._mink_config = mink.Configuration(model, q=q_stance_full.copy())
        self._mink_task = mink.FrameTask(
            frame_name=site_name, frame_type="site",
            position_cost=pos_cost, orientation_cost=ori_cost,
            lm_damping=lm_damping)
        # posture cost: high everywhere (keeps legs/waist/other-arm/hands from
        # moving in mink's own internal solution -- irrelevant to the physical
        # robot since only this arm's slice is ever read out, but keeps the QP
        # from wasting its budget on DOF this instance has no business moving),
        # low on THIS arm's own 7 DOF (redundancy resolution only, not a task
        # fight -- same rationale as the "lm" solver's posture_weight, but
        # per-DOF and much better conditioned here)
        cost = np.full(model.nv, posture_cost_rest)
        cost[joint_slice] = posture_cost_arm
        self._mink_posture = mink.PostureTask(model, cost=cost, lm_damping=lm_damping)
        self._mink_limits = [mink.ConfigurationLimit(model)]
        self._mink_qp_damping = float(qp_damping)
        self._mink_qp_solver = qp_solver
        # mink.solve_ik's dt controls how far one QP solution is integrated;
        # it MUST match the physics timestep this solve() is called at (once
        # per mj_step, like every other solver here) -- using a larger value
        # was tried and measurably caused real position/orientation drift
        # (the "one unit of work per call" contract every module here follows
        # depends on this).
        self._mink_dt = model.opt.timestep

    def _solve_mink(self, target_pos, target_quat, use_ori, w_ori):
        mink = self._mink
        if use_ori:
            self._mink_task.set_orientation_cost(w_ori)
            rot = mink.SO3(wxyz=np.asarray(target_quat, float))
        else:
            # true position-only: zero orientation cost, so the arm is free
            # to settle wherever the task+posture nullspace puts it -- the
            # placeholder rotation below is never actually weighted
            self._mink_task.set_orientation_cost(0.0)
            rot = mink.SO3.identity()
        se3 = mink.SE3.from_rotation_and_translation(rot, np.asarray(target_pos, float))
        self._mink_task.set_target(se3)
        vel = mink.solve_ik(self._mink_config, [self._mink_task, self._mink_posture],
                            self._mink_dt, self._mink_qp_solver,
                            damping=self._mink_qp_damping, limits=self._mink_limits)
        self._mink_config.integrate_inplace(vel, self._mink_dt)
        return self._mink_config.q[self.joint_slice].copy()

    # ============================================================= pinocchio
    def _init_pinocchio(self, model, site_name, joint_slice, arm_joint_names,
                        mjcf_path, q_stance_full):
        try:
            import pinocchio as pin
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "WeightedArmIK(solver='lm'/'ipopt') needs pinocchio (already "
                "a dex_retargeting dependency in this repo's env). "
                "`python -m pip install pin`. Or use the default "
                "solver='mink', which needs `pip install mink` instead."
            ) from e
        self._pin = pin
        if mjcf_path is None:
            raise ValueError("solver='lm'/'ipopt' needs mjcf_path "
                             "(path to g1_fixed_upper.xml in MODEL_DIR)")

        # --- Pinocchio reduced model: just this arm's 7 joints ---
        full = pin.buildModelFromMJCF(str(mjcf_path))
        # full is parsed from the ROBOT-only MJCF (nq == 43, no bricks); the
        # scene model's stance qpos has the same robot joints first (an
        # already-established invariant -- see actuator_groups.py), so a
        # straight truncation gets the matching robot-only stance vector
        # regardless of whether `model` itself is the bare robot or the
        # brick-laden scene.
        q_lock_ref = q_stance_full[:full.nq]
        keep_ids = [full.getJointId(n) for n in arm_joint_names]
        lock_ids = [j for j in range(1, full.njoints) if j not in keep_ids]
        self._rmodel = pin.buildReducedModel(full, lock_ids, q_lock_ref)
        self._rdata = self._rmodel.createData()
        self._fid = self._rmodel.getFrameId(site_name)

        # order + limits must match MuJoCo's arm slice exactly
        reduced_joint_names = list(self._rmodel.names)[1:]  # drop 'universe'
        assert reduced_joint_names == arm_joint_names, (
            f"reduced Pinocchio joint order {reduced_joint_names} != "
            f"{arm_joint_names} -- MJCF/actuator_groups drift")
        assert np.allclose(self._rmodel.lowerPositionLimit, self._qlo, atol=1e-6) and \
               np.allclose(self._rmodel.upperPositionLimit, self._qhi, atol=1e-6), \
            "Pinocchio joint limits != MuJoCo jnt_range for this arm"

        # --- constant Pinocchio-root -> MuJoCo-world translation offset ---
        q_arm_ref = np.asarray(self.scratch.qpos[joint_slice]).copy()
        pin.forwardKinematics(self._rmodel, self._rdata, q_arm_ref)
        pin.updateFramePlacements(self._rmodel, self._rdata)
        self._base_offset = (self.scratch.site_xpos[self.site_id]
                             - self._rdata.oMf[self._fid].translation).copy()
        assert abs(self._base_offset[2] - _EXPECTED_BASE_OFFSET_Z) < 0.05 and \
               np.linalg.norm(self._base_offset[:2]) < 1e-3, (
            f"unexpected Pinocchio<->MuJoCo base offset {self._base_offset}")

        self._lim_margin = self._limit_margin_frac * (self._qhi - self._qlo)

    def sync(self, qpos_full, pref=None):
        """Reseed from the robot's live qpos -- call once per frame before
        solve(), exactly like ArmIK.sync(). Warm-starts the next solve() from
        where the arm actually is (it physically moved between frames).
        `pref` overrides the posture target (default: the stance arm pose
        captured at construction). For solver="mink" the posture task always
        targets the CURRENT live pose (matches the arm's own natural
        continuation, empirically the best-behaved choice -- see the module
        docstring's measured table); `pref`, if given, overrides only this
        arm's own 7-DOF slice of that target, same as the "lm"/"ipopt" path's
        (much lower-weight) posture bias."""
        self.scratch.qpos[:] = qpos_full
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        self._q_seed = np.asarray(qpos_full[self.joint_slice]).copy()
        self._q_prev = self._q_seed.copy()
        if pref is not None:
            self._q_rest = np.asarray(pref).copy()

        if self._solver == "mink":
            self._mink_config.update(qpos_full.copy())
            posture_target = qpos_full.copy()
            if pref is not None:
                posture_target[self.joint_slice] = pref
            self._mink_posture.set_target(posture_target)

        self._synced = True

    # ------------------------------------------------------------- pin residual
    def _res_jac(self, q, p_tgt_pin, R_tgt, use_ori):
        """Task residual [pos_err(3); ori_err(3)] and its 6x7 (or 3x7)
        analytic Jacobian, in the Pinocchio root frame. Validated against
        finite differences to ~3e-10. (solver="lm"/"ipopt" only.)"""
        pin = self._pin
        pin.forwardKinematics(self._rmodel, self._rdata, q)
        pin.updateFramePlacements(self._rmodel, self._rdata)
        oMf = self._rdata.oMf[self._fid]
        R_wf = oMf.rotation
        p_err = oMf.translation - p_tgt_pin
        Jl = pin.computeFrameJacobian(self._rmodel, self._rdata, q, self._fid, pin.LOCAL)
        Jpos = R_wf @ Jl[:3, :]                       # world-frame linear vel
        if not use_ori:
            return p_err, Jpos
        R_err = R_tgt.T @ R_wf
        o_err = pin.log3(R_err)
        Jori = pin.Jlog3(R_err) @ Jl[3:, :]
        return np.concatenate([p_err, o_err]), np.vstack([Jpos, Jori])

    def _limit_resid(self, q):
        """One-sided residual, nonzero only within lim_margin of a joint limit
        -- added as extra rows of the weighted least-squares system so the
        solver is pushed off the limits. A fully-converged weighted solve
        otherwise likes to park the redundant DOF against a limit, and the
        physical position-controlled arm then can't hold that config (it sags
        past the limit under the hand's weight and never settles). mink's
        ConfigurationLimit does this natively as a real QP constraint, which
        is why solver="mink" doesn't need this at all."""
        hi = np.minimum(self._qhi - self._lim_margin - q, 0.0)   # <0 near hi
        lo = np.maximum(self._qlo + self._lim_margin - q, 0.0)   # >0 near lo
        return hi + lo                                           # want -> 0

    # ------------------------------------------------------------------- solvers
    def _solve_lm(self, p_tgt_pin, R_tgt, use_ori, w_ori):
        """Weighted Levenberg-Marquardt on a stacked least-squares system:
        position (high weight) + orientation (low weight, so an infeasible
        orientation request never trades position away) + a light posture and
        continuity regulariser + a joint-limit repulsion. Unlike arm_ik.py's
        damped-pinv step (which settles at a nonzero task-error equilibrium
        and parks 2-5 cm short), this drives the position residual to ~0 in
        JOINT space -- the physical arm can still fall short where its wrist
        actuators torque-saturate (module docstring)."""
        q = np.clip(self._q_seed.copy(), self._qlo, self._qhi)
        q_rest, q_prev = self._q_rest, self._q_prev
        wp, wpost, wsm, wlim = (self._pos_w, self._posture_w,
                                self._smooth_w, self._limit_w)
        W = (np.concatenate([np.full(3, wp), np.full(3, w_ori)])
             if use_ori else np.full(3, wp))
        n = self._n
        Ipost = (wpost + wsm) * np.eye(n)

        def cost(qq, r=None):
            if r is None:
                r, _ = self._res_jac(qq, p_tgt_pin, R_tgt, use_ori)
            return float(r @ (W * r) + wpost * np.sum((qq - q_rest) ** 2)
                         + wsm * np.sum((qq - q_prev) ** 2)
                         + wlim * np.sum(self._limit_resid(qq) ** 2))

        lam = 1e-3
        it = 0
        for it in range(self._max_iters):
            r, J = self._res_jac(q, p_tgt_pin, R_tgt, use_ori)
            lg = self._limit_resid(q)
            lim_active = (lg != 0.0)
            JtW = J.T * W
            # normal equations: task + posture/continuity + limit repulsion
            H = JtW @ J + Ipost + wlim * np.diag(lim_active.astype(float))
            g = (JtW @ r + wpost * (q - q_rest) + wsm * (q - q_prev)
                 - wlim * lg)                      # d/dq (lg^2)/2 = -lg where active
            c0 = cost(q, r)
            improved = False
            for _ in range(10):                    # LM trust adaptation
                dq = np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), -g)
                nrm = np.linalg.norm(dq)
                if nrm > self._dq_max:
                    dq *= self._dq_max / nrm       # trust-region cap
                qn = np.clip(q + dq, self._qlo, self._qhi)
                if cost(qn) < c0:
                    q, lam, improved = qn, max(lam * 0.5, 1e-9), True
                    break
                lam *= 4.0
            if not improved or np.linalg.norm(dq) < 1e-6:
                break
        self._last_iters = it + 1
        return q

    def _build_ipopt(self):
        try:
            import casadi as ca
        except ImportError as e:                       # pragma: no cover
            raise ImportError(
                "WeightedArmIK(solver='ipopt') needs casadi: "
                "`python -m pip install casadi`. Or use the default "
                "solver='mink', which needs `pip install mink` instead.") from e
        self._ca = ca
        outer = self

        class _TaskCB(ca.Callback):
            def __init__(self):
                ca.Callback.__init__(self)
                self.p_tgt = np.zeros(3)
                self.R_tgt = np.eye(3)
                self.use_ori = True
                self.construct("wik_task", {})

            def get_n_in(self): return 1
            def get_n_out(self): return 1
            def get_sparsity_in(self, i): return ca.Sparsity.dense(7, 1)
            def get_sparsity_out(self, i): return ca.Sparsity.dense(6, 1)

            def eval(self, arg):
                q = np.asarray(arg[0]).ravel()
                r, _ = outer._res_jac(q, self.p_tgt, self.R_tgt, True)
                return [ca.DM(r)]

            def has_jacobian(self): return True

            def get_jacobian(self, name, inames, onames, opts):
                class _JacCB(ca.Callback):
                    def __init__(s):
                        ca.Callback.__init__(s)
                        s.construct(name, {})
                    def get_n_in(s): return 2
                    def get_n_out(s): return 1
                    def get_sparsity_in(s, i):
                        return ca.Sparsity.dense(7, 1) if i == 0 else ca.Sparsity.dense(6, 1)
                    def get_sparsity_out(s, i): return ca.Sparsity.dense(6, 7)
                    def eval(s, arg):
                        q = np.asarray(arg[0]).ravel()
                        _, Jr = outer._res_jac(q, self.p_tgt, self.R_tgt, True)
                        return [ca.DM(Jr)]
                self._jac_keepalive = _JacCB()
                return self._jac_keepalive

        self._task_cb = _TaskCB()
        opti = ca.Opti()
        q = opti.variable(7)
        q_prev = opti.parameter(7)
        w_ori = opti.parameter(1)
        r = self._task_cb(q)
        cost = (self._pos_w * ca.sumsqr(r[:3]) + w_ori * ca.sumsqr(r[3:])
                + self._posture_w * ca.sumsqr(q - ca.DM(self._q_rest))
                + self._smooth_w * ca.sumsqr(q - q_prev))
        opti.minimize(cost)
        opti.subject_to(opti.bounded(ca.DM(self._qlo), q, ca.DM(self._qhi)))
        opti.solver("ipopt", {"print_time": 0},
                    {"print_level": 0, "sb": "yes", "max_iter": self._ipopt_max_iter,
                     "tol": 1e-4, "acceptable_tol": 1e-3, "acceptable_iter": 5,
                     "warm_start_init_point": "yes", "mu_strategy": "adaptive",
                     "nlp_scaling_method": "gradient-based",
                     "hessian_approximation": "limited-memory",
                     "limited_memory_max_history": 10})
        self._opti, self._opti_q, self._opti_qprev, self._opti_wori = opti, q, q_prev, w_ori

    def _solve_ipopt(self, p_tgt_pin, R_tgt, use_ori, w_ori):
        ca = self._ca
        self._task_cb.p_tgt = np.asarray(p_tgt_pin, float)
        self._task_cb.R_tgt = np.asarray(R_tgt, float) if use_ori else np.eye(3)
        # position-only == orientation weight 0 (residual rows still there, zeroed)
        self._opti.set_value(self._opti_wori, w_ori if use_ori else 0.0)
        self._opti.set_value(self._opti_qprev, self._q_prev)
        self._opti.set_initial(self._opti_q, np.clip(self._q_seed, self._qlo, self._qhi))
        try:
            sol = self._opti.solve()
            q = np.asarray(sol.value(self._opti_q)).ravel()
        except Exception:
            q = np.asarray(self._opti.debug.value(self._opti_q)).ravel()
        self._last_iters = self._ipopt_max_iter
        return np.clip(q, self._qlo, self._qhi)

    # -------------------------------------------------------------------- solve
    def solve(self, target_pos, target_quat=None, iters=None, tol=1e-4,
              ori_tol=1e-2, ori_weight=None):
        """Returns the solved qpos for this arm's joint_slice (7,). Same
        signature shape as ArmIK.solve(); `iters`/`tol`/`ori_tol` are accepted
        for drop-in compatibility (every solver here uses its own convergence
        control instead). `ori_weight`, if given, overrides the instance's
        orientation weight for this call (TeleopController passes
        TELEOP_WEIGHTED_ORI_WEIGHT)."""
        assert self._synced, "call sync() once before solve()"
        self._call_i += 1
        use_ori = target_quat is not None

        # decimation: only run the (possibly expensive) solve every ik_every-th
        # call; between solves return the last result. Mirrors FpvStreamer's
        # "every Nth step" pattern. Default ik_every=1 (every call) -- mink and
        # "lm" are both fast enough to run every physics step; only "ipopt"
        # really needs decimation.
        if self._ik_every > 1 and (self._call_i % self._ik_every) != 1:
            return self._q_last.copy()

        if self._solver == "mink":
            w_ori = (self._mink_ori_cost_default if ori_weight is None
                     else float(ori_weight))
            q_full = self._solve_mink(target_pos, target_quat, use_ori, w_ori)
        else:
            w_ori = self._ori_w if ori_weight is None else float(ori_weight)
            p_tgt_pin = np.asarray(target_pos, float) - self._base_offset
            if use_ori:
                R_tgt = np.zeros(9)
                mujoco.mju_quat2Mat(R_tgt, np.asarray(target_quat, float))
                R_tgt = R_tgt.reshape(3, 3)
            else:
                R_tgt = np.eye(3)
            if self._solver == "ipopt":
                q_full = self._solve_ipopt(p_tgt_pin, R_tgt, use_ori, w_ori)
            else:
                q_full = self._solve_lm(p_tgt_pin, R_tgt, use_ori, w_ori)

        # solve_blend<1 commands only a fraction of the way from the live pose
        # to the fully-converged solution (like arm_ik.py's step_scale),
        # converging over frames instead of in one jump -- a tuning knob for
        # if a fully-converged command ever looks twitchy on real hardware.
        # Default 1.0 (no blend): both solvers are already continuity- and
        # limit-regularised, and blending measurably HURT the well-behaved
        # near-target cases in sim, so it is off unless a headset run needs it.
        q_sol = np.clip(self._q_seed + self._solve_blend * (q_full - self._q_seed),
                        self._qlo, self._qhi) if self._solve_blend < 1.0 else q_full

        # reflect the solution in the scratch MjData so site_pos() /
        # scratch.site_xmat report it (identical contract to ArmIK.solve)
        self.scratch.qpos[self.joint_slice] = q_sol
        mujoco.mj_kinematics(self.model, self.scratch)
        mujoco.mj_comPos(self.model, self.scratch)
        self._q_last = q_sol.copy()
        return q_sol.copy()

    def site_pos(self):
        """Current (scratch) world position of this arm's gripper site."""
        return self.scratch.site_xpos[self.site_id].copy()
