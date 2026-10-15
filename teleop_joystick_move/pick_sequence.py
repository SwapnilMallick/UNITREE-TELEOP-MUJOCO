"""
PickSequence: assembles ArmIK + ApproachPath + grasp_primitive + grasp_state
into ONE incremental, per-frame-advanceable state machine -- the live-loop
counterpart to verify_grasp_hold.py's blocking reference sequence (approach
-> orient -> descend -> close -> lift -> hold), which this mirrors exactly
(same GRASP_HEIGHT/LIFT_HEIGHT/timing constants) so the behavior verified
there carries over here rather than drifting into a second, untested
implementation of the same recipe.

Advance-once-per-frame design, matching ArmIK.solve() and ApproachPath's own
pattern: .step() does one control step's worth of work and returns, so it
composes with a real viewer loop (stand_next_to_table.py) instead of
blocking it the way the verification scripts' `while` loops do.

Phase 1 scripted-target driving: the brick to pick is fixed at .start()
time, read from the live simulation (never a hardcoded position -- see
CLAUDE.md's verification pattern). This is exactly the interface Phase 2's
real teleop input will eventually drive instead -- .step()'s per-frame
contract doesn't change, only what supplies the target does.

Known scope boundary carried over from verify_grasp_hold.py: only tested
reliable for brick1/brick2 via the right arm. brick3 (left arm) is a
documented, unresolved grasp-approach-geometry problem (see CLAUDE.md's
"Fingertip/Brick Contact Tuning") -- PickSequence will still run it if
asked, it just isn't expected to succeed.

Optional PLACE extension (pass place_pos to start()): after HOLD confirms a
stable grasp, transits to place_pos, releases, and retreats (TRANSIT_OVER ->
TRANSIT_DOWN -> SETTLE_PLACE -> RELEASE -> RELEASE_SETTLE -> RETREAT ->
DONE). Omitting place_pos (every existing call site) leaves HOLD as the
terminal, indefinitely-repeating phase exactly as before -- the place
phases are strictly additive, not a rewrite of the already-verified
pick/hold behavior.

Two real problems surfaced building this, both found empirically (via
verify_stack_sequence.py, not guessed) and both necessary to fix together:

1. **Orientation must stay constant through the WHOLE transit, not just
   descend.** The first attempt reused ApproachPath (like APPROACH does),
   which only constrains orientation on its descend leg by design --
   correct for the empty-handed pick approach, where nothing is being
   carried yet and constraining orientation early only fights the position
   task (see approach_path.py). But transit IS carrying a firmly-closed
   grasp, and letting the hand's orientation drift freely through the
   lift/move-over legs let the carried brick's real position diverge
   several cm from the commanded site position. Fixed by not reusing
   ApproachPath for transit: target_quat is held constant for the entire
   move instead (see point 2 for the two legs this now happens over).

2. **The idealized site-to-brick offset (site = brick_center +
   GRASP_HEIGHT, used for the pick target above) does not hold once an
   object is actually gripped.** Measured directly: the real offset after a
   genuine grasp was found to have the OPPOSITE sign of the idealized
   assumption in one case (the brick sitting *above* the site by ~3cm,
   not the site sitting above the brick by GRASP_HEIGHT) -- real
   friction/contact settling under load isn't the same as the zero-load
   kinematic convergence point the site was originally calibrated against
   (see CLAUDE.md's "Corrected gripper site"). Fixed by measuring the real
   offset once, right after the grasp closes (CLOSE_SETTLE->LIFT
   transition), and using THAT instead of the idealized formula to compute
   where the site needs to go for the object to end up at place_pos.

Even after both fixes, a single diagonal straight-line transit (move
sideways and descend simultaneously) still risked sweeping the carried
brick past the target's side rather than only ever approaching from
directly above -- the same reason the ORIGINAL pick approach uses a
lift/move-over/descend structure instead of a straight Cartesian line (see
CLAUDE.md's "Arm IK"). TRANSIT_OVER (horizontal, at the current lifted
height) then TRANSIT_DOWN (straight down onto place_target) applies that
same lesson to placing, with target_quat held constant across both legs
(point 1) using the corrected place_target (point 2).

**Release-from-hover (found the session after the two fixes above -- a
third, deeper problem, not solved by either of them).** Even with an
accurate place_target and a clean two-leg approach, placing brick2 onto
brick1 still violently knocked brick1 away. Measured directly via
d.contact: the collision wasn't primarily the carried brick missing its
target -- it was the HAND ITSELF (thumb_1/thumb_2/middle_0/middle_1 links)
making contact with brick1, because CLOSED wraps the fingers most of the
way around the held object, and those finger links sit several cm below
the gripper site (measured ~4.5-5.4cm below the site through a real
grasp's CLOSE_SETTLE/LIFT/HOLD phases -- this is `_hand_clearance` below,
measured once per grasp, not a hardcoded guess).

Two follow-up ideas were tried and both ruled out empirically before
landing on the real fix:
- Reducing hold_grip's magnitude (uniformly scaling the SAME curl path
  toward CLOSED) -- tested down to 0.55; finger-vs-brick1 contact persisted
  at every level, and penetration depth got slightly WORSE, not better, as
  grip decreased (a looser grip holds the carried brick less precisely).
- Redistributing curl between the base knuckle and tip joints (a "keep the
  base open, curl only the tip" pinch) -- a pure forward-kinematics sweep
  (no arm, no brick) found the base knuckle is what does almost all the
  work of sweeping the fingertip toward the thumb on this hand; backing it
  off from 1.0 to 0.85 alone nearly doubled the thumb-to-fingertip gap
  (1.52cm -> 3.02cm), and it only gets worse from there. The curl needed
  for a real opposing grip on an object this size IS the curl that wraps
  under it -- there's no alternative shape available on this hand that
  keeps both.

The real finding: computed out, the fingers would need to sit ~6.6cm BELOW
brick1's own top surface even at the theoretically correct final placement
position -- meaning continuing to hold brick2 all the way down to
brick-contact height is geometrically impossible without the hand
clipping brick1, regardless of targeting precision or grip tuning. So
TRANSIT_DOWN/SETTLE_PLACE/RELEASE no longer target place_target directly
-- they target a HOVER point instead: the site height where the measured
`_hand_clearance` (plus HAND_CLEARANCE_MARGIN) keeps the lowest hand link
above the landing surface (place_surface_z, passed to start() by a caller
that knows the object's geometry -- e.g. stack_sequence.py knows brick
dimensions; PickSequence itself stays object-shape-agnostic). RELEASE
opens the grip while still hovering there, and the brick free-falls the
remaining distance under gravity -- RELEASE_SETTLE_TIME was lengthened
accordingly (0.5s -> 1.0s) to give a real several-cm fall time to settle,
not the few-mm gap a fixed-height release would have needed.
"""
import numpy as np
import mujoco

from arm_ik import ArmIK
from approach_path import ApproachPath
from grasp_primitive import hand_ctrl, grasp_target_quat
from grasp_state import grasp_confidence

# Mirrors verify_grasp_hold.py's constants exactly -- see that module's
# docstring for how GRASP_HEIGHT and LIFT_HEIGHT were derived/verified.
GRASP_HEIGHT = 0.04       # m above brick center
LIFT_HEIGHT = 0.20        # m
SETTLE_TIME = 0.3         # s, after reaching grasp pose, before closing
CLOSE_TIME = 1.2          # s, grip ramp 0 -> 1
CLOSE_SETTLE_TIME = 0.5   # s, after closing, before lifting
LIFT_TIME = 2.0           # s

# Place extension -- mirrors the pick timing constants above rather than
# inventing new untested numbers. NOTE: the place target is NOT computed as
# place_pos + GRASP_HEIGHT (the idealized site-above-brick-center relationship
# used for the pick target above) -- verified empirically (see module
# docstring and verify_stack_sequence.py) that once an object is actually
# gripped, its real position relative to the site diverges from that
# idealized formula, sometimes by more than the target brick's own footprint
# margin. The real offset is measured once, right after the grasp closes
# (see the CLOSE_SETTLE->LIFT transition), and used instead.
HOLD_BEFORE_PLACE_TIME = 1.0   # s, confirm the grasp reads stable before transiting to place
TRANSIT_TIME = 2.0             # s total for the two-leg transit below, split proportionally
                                # to each leg's distance -- mirrors LIFT_TIME's pacing for a
                                # similarly-sized motion carrying the same grasp
HAND_CLEARANCE_MARGIN = 0.02   # m, safety margin added on top of the MEASURED hand
                                # under-hang (see CLOSE_SETTLE->LIFT) when computing the
                                # hover height to release from -- see module docstring's
                                # "Release-from-hover" section for why the descent stops
                                # here instead of continuing down to brick-contact height
SETTLE_PLACE_TIME = 1.0        # s -- longer than SETTLE_TIME's 0.3s (a separate constant,
                                # not a shared reuse, so the already-verified pick-side SETTLE
                                # is untouched). TRIED as a fix for the residual XY landing
                                # error and found NOT to help -- measured the real site
                                # position sitting at a flat ~1.7cm error the whole dwell, not
                                # slowly converging. That error is a steady-state equilibrium
                                # from ArmIK's own nullspace posture bias (see arm_ik.py's
                                # docstring: dq_posture doesn't shrink with task error the way
                                # dq_task does), not a convergence-speed problem -- more time
                                # here can't close it. Kept at 1.0s/8 iters anyway (harmless,
                                # doesn't hurt); see CLAUDE.md's "Stacking Sequencing" for the
                                # full negative result and what's actually still worth trying.
SETTLE_PLACE_ITERS = 8         # vs. 2 elsewhere -- see above, doesn't meaningfully help but
                                # doesn't hurt either
RELEASE_TIME = 1.0             # s, grip ramp 1 -> 0 (CLOSE_TIME's ramp, reversed)
RELEASE_SETTLE_TIME = 1.0      # s -- longer than CLOSE_SETTLE_TIME's 0.5s on purpose: the
                                # brick now free-falls several cm from the hover height (see
                                # "Release-from-hover"), not the few-mm gap a fixed-height
                                # release would have, so it needs real time to fall and settle
RETREAT_HEIGHT = 0.15          # m, straight-up clearance after release -- less than
                                # LIFT_HEIGHT since this only needs to clear the brick just
                                # placed, not transit the whole table
RETREAT_TIME = 1.0             # s

PHASES = ("APPROACH", "SETTLE", "CLOSE", "CLOSE_SETTLE", "LIFT", "HOLD",
          "TRANSIT_OVER", "TRANSIT_DOWN", "SETTLE_PLACE", "RELEASE",
          "RELEASE_SETTLE", "RETREAT", "DONE")


class PickSequence:
    """One pick -> lift -> hold cycle for one arm, one brick.

    Usage:
        seq = PickSequence(m, "right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick1")
        seq.start(m, d)
        ...
        while running:
            seq.step(m, d, hold_ctrl)   # once per control step
            mujoco.mj_step(m, d)

    step() writes only into d.ctrl[arm_slice] and d.ctrl[hand_slice] (after
    first laying down hold_ctrl for everything else) -- never broadcasts
    across the full 43-actuator range, matching every other module here.
    """

    def __init__(self, m, side, site_name, arm_slice, hand_slice, brick_name, hold_grip=1.0):
        """hold_grip caps the grip value used from CLOSE onward (default 1.0,
        i.e. every existing call site is unaffected). Tried as a candidate
        fix for the finger-vs-neighboring-object collision described in the
        module docstring's "Release-from-hover" section -- ruled out
        empirically (tested down to 0.55; contact persisted at every level,
        penetration got slightly worse, not better). Kept as infrastructure
        for future experiments, not because lowering it is known to help;
        the actual fix is release-from-hover, not a smaller grip."""
        self.side = side
        self.arm_slice = arm_slice
        self.hand_slice = hand_slice
        self.brick_name = brick_name
        self.hold_grip = hold_grip
        self.ik = ArmIK(m, site_name, arm_slice)
        # cached once: every geom belonging to this side's hand, used to measure
        # _hand_clearance (see the CLOSE_SETTLE->LIFT transition and the module
        # docstring's "Release-from-hover" section)
        self._hand_geom_ids = [gid for gid in range(m.ngeom)
                                if f"{side}_hand" in (mujoco.mj_id2name(
                                    m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[gid]) or "")]
        self.phase = None
        self.confidence = None  # last grasp_confidence() reading, updated every HOLD step
        self._approach = None
        self._target_pos = None
        self._target_quat = None
        self._brick_short_axis = None
        self._lift_target = None
        self._lift_start = None
        self._place_pos = None
        self._place_surface_z = None
        self._place_target = None
        self._hover_target = None
        self._hand_clearance = None
        self._transit_start = None
        self._transit_mid = None
        self._transit_over_time = None
        self._grasp_offset = None
        self._retreat_target = None
        self._hold_elapsed = 0.0
        self._t = 0.0

    def start(self, m, d, place_pos=None, place_surface_z=None):
        """Call once, right after d.qpos reflects where the arm actually is
        (e.g. just after a stance reset) -- reads the brick's CURRENT
        position fresh, same as every verification script's pattern.

        place_pos, if given, is the WORLD position the brick's center should
        end up at (e.g. another brick's body xpos, for stacking on top of
        it) -- read fresh by the caller, same rule as the pick target.

        place_surface_z, if given, is the world Z of the surface place_pos
        rests ON (e.g. another brick's own top surface) -- used to compute
        the release-from-hover height (see module docstring). PickSequence
        deliberately doesn't know the held object's own dimensions (it's
        object-shape-agnostic), so a caller that DOES know them (e.g.
        stack_sequence.py, which knows brick heights) should pass this
        explicitly. Omitting it while place_pos is given falls back to
        treating place_pos's own height as the surface -- a more
        conservative (hovers a bit higher than strictly necessary) but
        workable default for a caller that doesn't track surface height
        separately.
        Omitting it (the default) preserves the original pick-hold-forever
        behavior exactly."""
        self.ik.sync(d.qpos)
        self._target_pos = d.xpos[m.body(self.brick_name).id].copy()
        self._target_pos[2] += GRASP_HEIGHT
        self._brick_short_axis = d.xmat[m.body(self.brick_name).id].reshape(3, 3)[:, 1]
        self._target_quat = None
        self._approach = ApproachPath(self.ik.site_pos(), self._target_pos)
        self._lift_target = self._target_pos.copy()
        self._lift_target[2] += LIFT_HEIGHT
        self._place_pos = None if place_pos is None else np.asarray(place_pos, dtype=float).copy()
        self._place_surface_z = (place_surface_z if place_surface_z is not None
                                  else (None if self._place_pos is None else self._place_pos[2]))
        self._place_target = None   # computed once the real grasp offset is known -- see CLOSE_SETTLE
        self._hover_target = None   # computed alongside _place_target -- see HOLD's transit trigger
        self._hand_clearance = None  # measured at CLOSE_SETTLE->LIFT
        self._grasp_offset = None
        self._retreat_target = None
        self._hold_elapsed = 0.0
        self.phase = "APPROACH"
        self._t = 0.0

    @property
    def done(self):
        return self.phase == ("DONE" if self._place_pos is not None else "HOLD")

    def step(self, m, d, hold_ctrl):
        """Advance one control step. Returns the current phase name (check
        against `done`/`self.phase == "HOLD"` to know when the cycle has
        reached its terminal, indefinitely-repeating hold)."""
        d.ctrl[:] = hold_ctrl
        dt = m.opt.timestep

        if self.phase == "APPROACH":
            # compute the grasp orientation once, right as descend begins --
            # not upfront, since it's derived from the arm's own natural
            # pose near the target (see grasp_target_quat), which isn't
            # known yet during the lift/move-over legs.
            if self._approach.on_descend_leg and self._target_quat is None:
                self._target_quat = grasp_target_quat(self.ik, self._target_pos, self._brick_short_axis)
                self._approach.set_target_quat(self._target_quat)
            q = self._approach.advance(self.ik, self.arm_slice, dt)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(0.0, self.side)
            if self._approach.finished:
                self.phase = "SETTLE"
                self._t = 0.0

        elif self.phase == "SETTLE":
            q = self.ik.solve(self._target_pos, target_quat=self._target_quat, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(0.0, self.side)
            self._t += dt
            if self._t >= SETTLE_TIME:
                self.phase = "CLOSE"
                self._t = 0.0

        elif self.phase == "CLOSE":
            grip = min(self._t / CLOSE_TIME, 1.0) * self.hold_grip
            q = self.ik.solve(self._target_pos, target_quat=self._target_quat, iters=1)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(grip, self.side)
            self._t += dt
            if self._t >= CLOSE_TIME:
                self.phase = "CLOSE_SETTLE"
                self._t = 0.0

        elif self.phase == "CLOSE_SETTLE":
            q = self.ik.solve(self._target_pos, target_quat=self._target_quat, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(self.hold_grip, self.side)
            self._t += dt
            if self._t >= CLOSE_SETTLE_TIME:
                self.phase = "LIFT"
                self._t = 0.0
                self._lift_start = self.ik.site_pos()
                # real grasp offset (brick's actual center minus the site),
                # measured now rather than assumed -- see module docstring:
                # the idealized site - GRASP_HEIGHT relationship doesn't
                # hold once the object is actually gripped (real friction/
                # contact settling, not a perfect kinematic pinch point).
                # Only meaningful for placement; harmless to compute even
                # when place_pos is None.
                self._grasp_offset = d.xpos[m.body(self.brick_name).id].copy() - self._lift_start
                # real hand under-hang (site height minus the lowest point of any
                # hand geom), measured now for the same reason as _grasp_offset --
                # see module docstring's "Release-from-hover" section. geom_rbound
                # (bounding-sphere radius) is a conservative lower-bound proxy for
                # "lowest point", not exact mesh geometry, but consistent with how
                # this was first diagnosed.
                site_z = self._lift_start[2]
                lowest_z = min(d.geom_xpos[gid][2] - m.geom_rbound[gid]
                                for gid in self._hand_geom_ids)
                self._hand_clearance = site_z - lowest_z

        elif self.phase == "LIFT":
            alpha = min(self._t / LIFT_TIME, 1.0)
            wp = self._lift_start + alpha * (self._lift_target - self._lift_start)
            q = self.ik.solve(wp, target_quat=self._target_quat, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(self.hold_grip, self.side)
            self._t += dt
            if self._t >= LIFT_TIME:
                self.phase = "HOLD"

        elif self.phase == "HOLD":
            q = self.ik.solve(self._lift_target, target_quat=self._target_quat, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(self.hold_grip, self.side)
            body_id = m.body(self.brick_name).id
            self.confidence = grasp_confidence(d, self.hand_slice, self.side, self.ik.site_id, body_id)
            if self._place_pos is not None:
                self._hold_elapsed += dt
                if self._hold_elapsed >= HOLD_BEFORE_PLACE_TIME:
                    # site target = desired brick center minus the REAL,
                    # measured grasp offset (not the idealized GRASP_HEIGHT
                    # formula) -- see the CLOSE_SETTLE->LIFT transition
                    self._place_target = self._place_pos - self._grasp_offset
                    # hover target: same XY, but the Z is only ever the HIGHER
                    # of (a) place_target's own Z or (b) the height where the
                    # measured hand under-hang clears the landing surface --
                    # see module docstring's "Release-from-hover". TRANSIT_DOWN/
                    # SETTLE_PLACE/RELEASE target this, never place_target
                    # directly -- the hand release the grip HERE and lets the
                    # object fall the rest of the way, rather than continuing
                    # to descend the (still-wrapped) hand down to contact height.
                    min_hover_z = self._place_surface_z + self._hand_clearance + HAND_CLEARANCE_MARGIN
                    self._hover_target = np.array([self._place_target[0],
                                                    self._place_target[1],
                                                    max(self._place_target[2], min_hover_z)])
                    self._transit_start = self.ik.site_pos()
                    # move-over waypoint: same XY as the hover target, same Z
                    # as right now -- so the carried brick only ever
                    # approaches from directly above, instead of clipping the
                    # target's side on a diagonal
                    self._transit_mid = np.array([self._hover_target[0],
                                                   self._hover_target[1],
                                                   self._transit_start[2]])
                    dist_over = np.linalg.norm(self._transit_mid - self._transit_start)
                    dist_down = np.linalg.norm(self._hover_target - self._transit_mid)
                    total_dist = dist_over + dist_down
                    frac = 0.5 if total_dist < 1e-6 else dist_over / total_dist
                    self._transit_over_time = max(TRANSIT_TIME * frac, 0.3)
                    self.phase = "TRANSIT_OVER"
                    self._t = 0.0

        elif self.phase == "TRANSIT_OVER":
            # target_quat held constant here too (unlike APPROACH's
            # ApproachPath-based lift/move-over legs) -- see module docstring
            alpha = min(self._t / self._transit_over_time, 1.0)
            wp = self._transit_start + alpha * (self._transit_mid - self._transit_start)
            q = self.ik.solve(wp, target_quat=self._target_quat, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(self.hold_grip, self.side)  # still holding
            self._t += dt
            if self._t >= self._transit_over_time:
                self.phase = "TRANSIT_DOWN"
                self._t = 0.0

        elif self.phase == "TRANSIT_DOWN":
            down_time = max(TRANSIT_TIME - self._transit_over_time, 0.3)
            alpha = min(self._t / down_time, 1.0)
            wp = self._transit_mid + alpha * (self._hover_target - self._transit_mid)
            q = self.ik.solve(wp, target_quat=self._target_quat, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(self.hold_grip, self.side)  # still holding
            self._t += dt
            if self._t >= down_time:
                self.phase = "SETTLE_PLACE"
                self._t = 0.0

        elif self.phase == "SETTLE_PLACE":
            # target is the HOVER point, not place_target -- see module
            # docstring's "Release-from-hover"; the object is released from
            # here, not lowered all the way to brick-contact height.
            # SETTLE_PLACE_TIME/SETTLE_PLACE_ITERS (longer dwell, more solver
            # iterations than the pick-side SETTLE): tried as a fix for the
            # residual XY landing error, found NOT to help -- see the
            # constants' own comments and CLAUDE.md's "Stacking Sequencing"
            # for the full negative result (it's a steady-state posture-bias
            # equilibrium, not a convergence-speed problem).
            q = self.ik.solve(self._hover_target, target_quat=self._target_quat,
                               iters=SETTLE_PLACE_ITERS)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(self.hold_grip, self.side)
            self._t += dt
            if self._t >= SETTLE_PLACE_TIME:
                self.phase = "RELEASE"
                self._t = 0.0

        elif self.phase == "RELEASE":
            grip = max(self.hold_grip * (1.0 - self._t / RELEASE_TIME), 0.0)
            q = self.ik.solve(self._hover_target, target_quat=self._target_quat, iters=1)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(grip, self.side)
            self._t += dt
            if self._t >= RELEASE_TIME:
                self.phase = "RELEASE_SETTLE"
                self._t = 0.0

        elif self.phase == "RELEASE_SETTLE":
            # hand stays put at the hover point (already clear of the landing
            # surface) while the released object free-falls and settles --
            # see module docstring's "Release-from-hover"
            q = self.ik.solve(self._hover_target, target_quat=self._target_quat, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(0.0, self.side)
            self._t += dt
            if self._t >= RELEASE_SETTLE_TIME:
                self.phase = "RETREAT"
                self._t = 0.0
                self._retreat_target = self._hover_target.copy()
                self._retreat_target[2] += RETREAT_HEIGHT

        elif self.phase == "RETREAT":
            alpha = min(self._t / RETREAT_TIME, 1.0)
            wp = self._hover_target + alpha * (self._retreat_target - self._hover_target)
            q = self.ik.solve(wp, target_quat=self._target_quat, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(0.0, self.side)
            self._t += dt
            if self._t >= RETREAT_TIME:
                self.phase = "DONE"

        elif self.phase == "DONE":
            q = self.ik.solve(self._retreat_target, iters=2)
            d.ctrl[self.arm_slice] = q
            d.ctrl[self.hand_slice] = hand_ctrl(0.0, self.side)

        else:
            raise RuntimeError(f"PickSequence.step() called before start() (phase={self.phase})")

        return self.phase
