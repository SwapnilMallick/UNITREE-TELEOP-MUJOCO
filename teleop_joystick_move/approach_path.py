"""
Re-targetable up/over/down approach path for one arm: lift straight up to a
safe height clear of the table, move horizontally to above the target, then
descend onto it -- re-solving IK fresh each step (ArmIK's warm-start design
is built for exactly this).

Promoted out of verify_arm_ik.py/verify_grasp_primitive.py, where this logic
existed twice, copy-pasted per test case and hardcoded to one static target
each. A real control loop needs to call this per pick/place with whatever
start/goal it has that frame, so this is a re-targetable component (any
start -> any goal), not a script.

Why up/over/down at all (see CLAUDE.md's "Arm IK" section for the full
story): an instant joint-space jump and a straight-line Cartesian
interpolation were both tried first and both collided the arm with the
table -- an IK solver only promises the destination is valid, not the path
to it.

Advance-once-per-frame design, not a blocking loop: the real control loop
needs to interleave this with other per-frame work (hand control, reading
teleop input, etc.), so ApproachPath.advance() computes one step's joint
target and returns it -- the caller owns writing d.ctrl and calling
mj_step(), exactly like ArmIK.solve() already does.

Obstacle-aware safe_z (see verify_approach_obstacles.py): the original
up/over/down path only ever proved clear of the TABLE -- safe_z was picked
from start/target height alone. That's not enough once a brick is actually
moving: the move-over leg travels in a straight line at a fixed height, and
if another brick (or a partially-built stack) sits near that line and is
taller than the original safe_z, the path clips it. `obstacles` lets a
caller list what else is on the table (or already stacked) so safe_z gets
raised whenever the straight-line move-over path would pass close enough to
one, in the target's z units. This is a "go over everything relevant"
fix, not a real path planner -- appropriate for a handful of small, known,
static-during-the-move obstacles on a tabletop; it would not scale to
dense clutter or moving obstacles.
"""
import numpy as np


def _point_segment_distance(p, a, b):
    """Shortest distance from point p to the line segment a->b (all 2D)."""
    ab = b - a
    denom = np.dot(ab, ab)
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0) if denom > 1e-12 else 0.0
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def brick_obstacle(pos, radius=0.044):
    """Convenience constructor for one obstacle entry from a brick's world
    position (its body xpos, e.g. d.xpos[body_id]) -- pos[2] is its body
    origin, not top surface, so this adds the brick's own half-height
    (1.725cm, see scene_fixed_table.xml's brick collision geom) to get the
    top. `radius` defaults to the brick's own bounding-circle radius in the
    XY plane (7.875 x 3.75cm footprint, worst case over any yaw:
    sqrt(3.9375^2 + 1.875^2) ~= 4.4cm) -- ApproachPath's own
    `obstacle_margin` adds clearance on top of this for whatever's flying
    near it (the hand, not just the wrist point)."""
    pos = np.asarray(pos, dtype=float)
    return dict(xy=pos[:2].copy(), top_z=pos[2] + 0.01725, radius=radius)


class ApproachPath:
    """One lift -> move-over -> descend path from start_pos to target_pos.

    target_quat, if given, is applied only on the descend leg -- orientation
    doesn't matter while clearing the table, and constraining it early only
    fights the position task (see grasp_primitive.py's grasp_target_quat for
    why descend-only also isn't fully aligned, just blended toward it).

    safe_z defaults to clearance above whichever of start/target is higher,
    not a fixed absolute height, so this generalizes to place targets on top
    of a growing stack, not just pick targets down at table height.

    `obstacles` (optional): a list of dicts with keys `xy`, `top_z`,
    `radius` (see brick_obstacle() for the usual way to build one from a
    brick's world position) -- anything else on the table the move-over leg
    needs to clear, beyond just the start/target heights. Only raises
    safe_z for obstacles whose footprint actually comes within
    `obstacle_margin` of the straight-line move-over path; an obstacle off
    to the side doesn't force extra height for no reason. Explicitly passed
    by the caller, not queried from a live simulation here, so this stays
    usable with any object-tracking source (sim ground truth now, real
    perception later) and testable without a live MjData.

    obstacle_clearance (not `clearance`) sets how far above an obstacle's
    top the raised path clears it, and defaults higher (0.20m vs 0.12m) for
    a real, measured reason: with no orientation constraint during lift/
    move-over, the arm's natural pose can leave the WRIST trailing well
    below the SITE -- measured 13.8cm in one case in
    verify_approach_obstacles.py, more than the site's own ~11.2cm
    pinch-point offset. Raising safe_z by only `clearance` (correct for the
    table, which the baseline up/over/down path was originally proven
    against -- a large flat surface with no "wrist dips below a nearby tall
    object" failure mode) was verified insufficient: the site cleared a
    pinned obstacle by 10+cm while the wrist still clipped it.

    KNOWN, UNRESOLVED LIMITATION (do not treat obstacle_clearance as a
    guarantee): the site-to-wrist vertical gap is NOT a fixed worst case --
    it varies through the transit as the nullspace-biased IK's natural
    orientation shifts, and was observed exceeding even a 0.35m clearance
    in further testing (verify_approach_obstacles.py's Part 2 reports this
    rather than hiding it). Root cause: nothing constrains orientation
    during the lift/move-over legs (target_quat only applies on descend --
    see above), so there's no bound on how far the physical hand can swing
    away from the site's commanded position. The height-only fix here
    correctly handles obstacles when the natural transit orientation stays
    reasonably level (verified: the pure-geometry safe_z calculation is
    exactly correct in check_geometry()), but is not a collision-avoidance
    guarantee for every possible obstacle placement/transit combination.
    Properly fixing this needs an orientation constraint through lift/
    move-over near obstacles (e.g. holding a level/neutral pose, not the
    free nullspace-biased one) or real path planning -- neither implemented
    here. Flagged as an open item, same as brick3's grasp-approach geometry
    (see CLAUDE.md's "Grasp Primitive" section) -- not silently worked
    around.
    """

    def __init__(self, start_pos, target_pos, target_quat=None,
                 safe_z=None, clearance=0.12, approach_speed=0.15,
                 min_leg_duration=0.3, obstacles=None, obstacle_margin=0.06,
                 obstacle_clearance=0.20):
        start_pos = np.asarray(start_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        if safe_z is None:
            safe_z = max(start_pos[2], target_pos[2]) + clearance
            for obs in (obstacles or []):
                gap = _point_segment_distance(obs["xy"], start_pos[:2], target_pos[:2])
                if gap < obs["radius"] + obstacle_margin:
                    safe_z = max(safe_z, obs["top_z"] + obstacle_clearance)

        lift_end = np.array([start_pos[0], start_pos[1], safe_z])
        move_end = np.array([target_pos[0], target_pos[1], safe_z])
        self._waypoints = [start_pos, lift_end, move_end, target_pos]
        # quat only on the descend leg (index 2 -> 3, i.e. leg index 2)
        self._leg_quats = [None, None, target_quat]

        self._leg_durations = []
        for a, b in zip(self._waypoints[:-1], self._waypoints[1:]):
            self._leg_durations.append(max(np.linalg.norm(b - a) / approach_speed,
                                            min_leg_duration))
        self._total_duration = sum(self._leg_durations)
        self._elapsed = 0.0

    @property
    def finished(self):
        return self._elapsed >= self._total_duration

    @property
    def target_pos(self):
        return self._waypoints[-1]

    @property
    def on_descend_leg(self):
        """True once advance() has reached (or passed) the descend leg --
        the point at which set_target_quat() actually starts taking effect.
        Useful for callers like grasp_target_quat that need to compute an
        orientation target only once, right as descend begins."""
        return self._leg_index(min(self._elapsed, self._total_duration)) == 2

    def _leg_index(self, t):
        t_in_leg = t
        for i, dur in enumerate(self._leg_durations):
            if t_in_leg <= dur or i == len(self._leg_durations) - 1:
                return i
            t_in_leg -= dur
        return len(self._leg_durations) - 1

    def set_target_quat(self, target_quat):
        """Set (or replace) the descend leg's orientation target after
        construction -- for grasp_target_quat-style callers that need to
        discover the target orientation from the arm's own natural pose near
        the target, which isn't known yet when the path starts out at the
        lift leg. Takes effect the next time advance() reaches the descend
        leg (or immediately, if already there)."""
        self._leg_quats[2] = target_quat

    def advance(self, ik, arm_slice, dt, iters=2):
        """Call once per control step. Returns the solved qpos (7,) for
        arm_slice -- write it into d.ctrl[arm_slice] yourself, same as a
        direct ArmIK.solve() call. Once finished, keeps re-solving toward
        the final target_pos/target_quat (acts as an implicit settle phase
        under continued calls, same as the explicit settle loops the two
        verification scripts used before this was promoted out of them)."""
        t = min(self._elapsed, self._total_duration)
        leg = self._leg_index(t)
        t_in_leg = t - sum(self._leg_durations[:leg])
        dur = self._leg_durations[leg]
        alpha = min(t_in_leg / dur, 1.0) if dur > 0 else 1.0
        a, b = self._waypoints[leg], self._waypoints[leg + 1]
        waypoint = a + alpha * (b - a)
        quat = self._leg_quats[leg]

        self._elapsed += dt
        if quat is not None:
            return ik.solve(waypoint, target_quat=quat, iters=iters)
        return ik.solve(waypoint, iters=iters)
