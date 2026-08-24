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
"""
import numpy as np


class ApproachPath:
    """One lift -> move-over -> descend path from start_pos to target_pos.

    target_quat, if given, is applied only on the descend leg -- orientation
    doesn't matter while clearing the table, and constraining it early only
    fights the position task (see grasp_primitive.py's grasp_target_quat for
    why descend-only also isn't fully aligned, just blended toward it).

    safe_z defaults to clearance above whichever of start/target is higher,
    not a fixed absolute height, so this generalizes to place targets on top
    of a growing stack, not just pick targets down at table height.
    """

    def __init__(self, start_pos, target_pos, target_quat=None,
                 safe_z=None, clearance=0.12, approach_speed=0.15,
                 min_leg_duration=0.3):
        start_pos = np.asarray(start_pos, dtype=float)
        target_pos = np.asarray(target_pos, dtype=float)
        if safe_z is None:
            safe_z = max(start_pos[2], target_pos[2]) + clearance

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
