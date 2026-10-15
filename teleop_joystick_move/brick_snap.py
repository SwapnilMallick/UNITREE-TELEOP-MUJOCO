"""
BrickSnapper: stud/groove snap-fit for the yellow PLAEX bricks, done as
"check alignment, then seat + lock" rather than physical stud geometry.

Why this shape (see CLAUDE.md "Brick Snap-Fit"): each brick's collision is one
plain box (the studded mesh is display-only), so two stacked bricks simply rest
on each other 10.5 mm too high (on the stud tips) whatever their alignment.
LEGO_PLANNING_LEARNING_MUJOCO -- the user's simulator that already solves this --
does not seat studs physically either: its runner checks alignment against a
tolerance and, if it passes, sets the exact seated pose and locks it
("SNAP_LOCKED"); a failed check is "released to physics" and rests unseated.
This ports that rule to runtime, for a robot placing free bodies live.

Each step, for every brick U resting on brick L:
  * U touches L and NO robot geom (a held brick is left alone -- snap happens on
    release, like the reference simulator's "snap at release"),
  * both are flat (tilt <= TILT_TOL_DEG),
  * long axes agree mod 180 deg (the brick is symmetric under a half turn) to
    within yaw_tol_deg,
  * centres agree horizontally to within horiz_tol,
  * U's height above L is in the resting/seating window,
then U is moved to the exact seated pose (SEAT_HEIGHT above L, L's orientation)
and the pre-declared weld constraint for that pair is activated with that
relative pose. The seated pair's collision bits are also cleared (the seated
pose puts the two boxes 10.5 mm inside each other -- a plain box has no
grooves -- and contact would otherwise fight the weld); they're restored only
needed again. The weld is broken if its tension (pull-apart) exceeds pull_force or its shear exceeds
shear_force -- the operator pulled or shoved it; compression (just carrying
weight) never breaks it, for BREAK_DEBOUNCE consecutive steps (ignores one-step
spikes). Breaking UNSEATS the brick: it is moved up REST_HEIGHT-SEAT_HEIGHT (as
if the studs popped out of the grooves, so it rests on the stud tips) and the
pair's collisions are restored at once -- leaving them off until the boxes
separate would let a brick whose hand let go mid-pull fall through the one
below. Re-snap is then blocked for RESNAP_COOLDOWN so it doesn't instantly
re-seat.

Tolerances default to the reference simulator's (5 deg yaw; 0.05 plan units
horizontal, plan unit = 0.035 m here since its mesh scale is 0.01 and ours
0.035 -- 1.75 mm). The physical stud/groove clearance is tighter (~1 mm, ~1.3
deg) -- these are the looser, teleop-oriented numbers, all overridable.

SIDE (dovetail) joint -- PHYSICAL, no snapper involvement. The yellow long mesh has a trapezoidal
dovetail tab on its -X end and a matching slot (through the full height, narrow at the face, wide
inside) on its +X end. Each brick's collision is therefore a compound of boxes (see the scene
XML): main body + stepped tab + stepped jaws forming the slot. A neighbour's tab can only be
lowered into the slot from above and the head/neck geometry then blocks sliding apart, so no weld
and no teleport is needed: it seats under gravity and holds by shape alone.

Inactive welds cost nothing, and a scene/verify script that never builds a
BrickSnapper is untouched, so this is opt-in per run.
"""
import numpy as np
import mujoco

# --- geometry, measured 2026-09-19 from the yellowPLAEXLong mesh at scale 0.035 ---
SEAT_HEIGHT = 0.070     # m, centre-to-centre with studs fully in grooves
REST_HEIGHT = 0.0805    # m, centre-to-centre resting on stud tips (= box height)
PLAN_UNIT = 0.035       # m per plan unit at this mesh scale

# --- tolerances: the reference simulator's Phase 2 values (see module docstring) ---
DEFAULT_HORIZ_TOL = 0.05 * PLAN_UNIT   # 1.75 mm
DEFAULT_YAW_TOL_DEG = 5.0
TILT_TOL_DEG = 5.0
HEIGHT_WINDOW = (SEAT_HEIGHT - 0.004, REST_HEIGHT + 0.006)

# Weld tension = pull F on the seated brick minus its own weight, so the pull limit has to be
# BELOW one brick's weight (7.6 N) or the brick underneath leaves the table before the snap ever
# breaks (measured: a 25 N limit lifted the whole pair 8 m). Shear (sliding) is limited higher so
# an incidental finger bump doesn't unseat a brick.
DEFAULT_PULL_FORCE = 5.0     # N of weld tension (pulling apart) before the snap breaks
DEFAULT_SHEAR_FORCE = 20.0   # N of weld shear before the snap breaks
RESNAP_COOLDOWN = 0.5        # s
BREAK_DEBOUNCE = 5           # consecutive control steps a break condition must hold (10 ms)


def _yaw(R):
    return np.arctan2(R[1, 0], R[0, 0])


def _wrap(a, period):
    return (a + period / 2.0) % period - period / 2.0


class BrickSnapper:
    def __init__(self, m, brick_names, horiz_tol=DEFAULT_HORIZ_TOL,
                 yaw_tol_deg=DEFAULT_YAW_TOL_DEG, pull_force=DEFAULT_PULL_FORCE,
                 shear_force=DEFAULT_SHEAR_FORCE):
        self.m = m
        self.names = tuple(brick_names)
        self.horiz_tol = float(horiz_tol)
        self.yaw_tol = np.radians(yaw_tol_deg)
        self.pull_force = float(pull_force)
        self.shear_force = float(shear_force)
        self.events = []                       # human-readable snap/break log

        self.body = {n: m.body(n).id for n in self.names}
        self.geom = {n: m.geom(f"{n}_collision").id for n in self.names}   # main box: pose/height reference
        self.geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.body[n]
                          and m.geom_contype[g]] for n in self.names}      # every collision geom (dovetail pieces too)
        self.qadr = {n: m.jnt_qposadr[m.body(n).jntadr[0]] for n in self.names}
        self.dadr = {n: m.jnt_dofadr[m.body(n).jntadr[0]] for n in self.names}
        self._geom_to_brick = {g: n for n, gs in self.geoms.items() for g in gs}
        contypes = [int(m.geom_contype[self.geom[n]]) for n in self.names]
        assert all(len({int(m.geom_contype[g]) for g in gs}) == 1 for gs in self.geoms.values()), \
            "all of a brick's collision geoms must share one contype bit"
        assert len(set(contypes)) == len(self.names), \
            "each brick collision geom needs its own contype bit (see scene_fixed_table.xml)"
        # Baseline = collide with the environment (bit 1) and with every brick. Computed from the
        # bricks' own bits, NOT read back from the model: this class mutates geom_conaffinity, so
        # reading it would pick up another BrickSnapper's (or a crashed run's) cleared bits.
        self._full_affinity = 1
        for c in contypes:
            self._full_affinity |= c
        self._all_geoms = [g for gs in self.geoms.values() for g in gs]
        m.geom_conaffinity[self._all_geoms] = self._full_affinity
        assert np.allclose(m.geom_quat[list(self.geom.values())], [1, 0, 0, 0]), \
            "snap pose math assumes the collision geom has identity orientation in its body"

        self.eq = {}                            # frozenset({a, b}) -> eq id
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                for nm in (f"snap_{a}_{b}", f"snap_{b}_{a}"):
                    e = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, nm)
                    if e >= 0:
                        self.eq[frozenset((a, b))] = e
        missing = [(a, b) for i, a in enumerate(self.names) for b in self.names[i + 1:]
                   if frozenset((a, b)) not in self.eq]
        assert not missing, f"scene is missing snap welds for pairs {missing}"

        # bodies that count as "the environment" rather than "a robot holding it"
        env = {0, m.body("table").id} | set(self.body.values())
        self._robot_geoms = np.array([g for g in range(m.ngeom)
                                      if m.geom_bodyid[g] not in env and m.geom_contype[g]], int)
        self._robot_geom_mask = np.zeros(m.ngeom, bool)
        self._robot_geom_mask[self._robot_geoms] = True

        self._weld = {}                         # upper name -> (lower name, eq id)
        self._blocked_until = {}                # upper name -> sim time
        self._over = {}                         # upper name -> consecutive steps over a break limit

    # ------------------------------------------------------------------ public
    @property
    def welded(self):
        """{upper: lower} for every active snap."""
        return {u: lo for u, (lo, _) in self._weld.items()}

    def reset(self, d):
        """Deactivate every snap and restore all brick-brick collisions (call
        alongside any brick reset)."""
        for _, e in self._weld.values():
            d.eq_active[e] = 0
        self._weld.clear()
        self._blocked_until.clear()
        self._over.clear()
        self.m.geom_conaffinity[self._all_geoms] = self._full_affinity

    def step(self, d):
        """Call once per control step, after mj_step()."""
        self._check_breaks(d)
        self._try_snaps(d)

    # ---------------------------------------------------------------- internals
    def _weld_force_vec(self, d, e):
        n = d.nefc
        rows = np.flatnonzero((d.efc_type[:n] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                              & (d.efc_id[:n] == e))
        return d.efc_force[rows[:3]].copy() if len(rows) >= 3 else np.zeros(3)

    def _set_collide(self, a, b, on):
        ca, cb = int(self.m.geom_contype[self.geom[a]]), int(self.m.geom_contype[self.geom[b]])
        for g in self.geoms[a]:
            if on:
                self.m.geom_conaffinity[g] |= cb
            else:
                self.m.geom_conaffinity[g] &= ~cb
        for g in self.geoms[b]:
            if on:
                self.m.geom_conaffinity[g] |= ca
            else:
                self.m.geom_conaffinity[g] &= ~ca

    def _tension_shear(self, d, e, n1, n2):
        """(tension, shear) of weld e between bricks n1 (eq obj1) and n2 (obj2). The constraint
        force is in world axes; positive tension = pulling the pair apart. Sign is taken relative
        to the obj1->obj2 line (measured: carrying weight reads negative for BOTH body orders --
        a fixed 'z is up' sign broke stacks whose upper brick was listed first in the weld)."""
        f = self._weld_force_vec(d, e)
        u = d.geom_xpos[self.geom[n2]] - d.geom_xpos[self.geom[n1]]
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            return 0.0, float(np.linalg.norm(f))
        u = u / nu
        tension = float(f @ u)
        return tension, float(np.linalg.norm(f - tension * u))

    def _break_reason(self, tension, shear):
        if tension > self.pull_force:
            return f"pulled off ({tension:.1f} N tension)"
        if shear > self.shear_force:
            return f"knocked sideways ({shear:.1f} N shear)"
        return None

    def _check_breaks(self, d):
        for upper in list(self._weld):
            lower, e = self._weld[upper]
            n1, n2 = self._eq_names(e)
            why = self._break_reason(*self._tension_shear(d, e, n1, n2))
            if why is None:
                self._over[upper] = 0
                continue
            self._over[upper] = self._over.get(upper, 0) + 1
            if self._over[upper] >= BREAK_DEBOUNCE:
                self._unseat(d, upper, lower, e)
                self.events.append(f"snap broken: {upper} {why} from {lower}")

    def _eq_names(self, e):
        b1, b2 = int(self.m.eq_obj1id[e]), int(self.m.eq_obj2id[e])
        rev = {v: k for k, v in self.body.items()}
        return rev[b1], rev[b2]

    def _unseat(self, d, upper, lower, e):
        """Break the snap: weld off, brick popped up onto the stud tips, collisions back on."""
        d.eq_active[e] = 0
        del self._weld[upper]
        self._over.pop(upper, None)
        self._blocked_until[upper] = d.time + RESNAP_COOLDOWN
        rise = (d.geom_xpos[self.geom[lower]][2] + REST_HEIGHT + 0.0005) - d.geom_xpos[self.geom[upper]][2]
        if rise > 0:
            d.qpos[self.qadr[upper] + 2] += rise
            mujoco.mj_kinematics(self.m, d)
        self._set_collide(upper, lower, True)

    def _contacts(self, d):
        """({(upper_candidate, lower_candidate)} brick-brick touching pairs, {bricks touched by a robot geom})"""
        n = d.ncon
        if n == 0:
            return set(), set()
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        pairs, held = set(), set()
        for a, b in zip(g1, g2):
            ba, bb = self._geom_to_brick.get(a), self._geom_to_brick.get(b)
            if ba and bb:
                pairs.add((ba, bb)); pairs.add((bb, ba))
            elif ba and self._robot_geom_mask[b]:
                held.add(ba)
            elif bb and self._robot_geom_mask[a]:
                held.add(bb)
        return pairs, held

    def _try_snaps(self, d):
        pairs, held = self._contacts(d)
        if not pairs:
            return
        supporting = {lo for lo, _ in self._weld.values()}
        for upper, lower in pairs:
            if upper in self._weld or upper in held or upper in supporting:
                continue        # already seated / being held / something is seated on it
            if d.time < self._blocked_until.get(upper, 0.0):
                continue
            if self._aligned(d, upper, lower):
                self._snap(d, upper, lower)

    def _aligned(self, d, upper, lower):
        gu, gl = self.geom[upper], self.geom[lower]
        Ru, Rl = d.geom_xmat[gu].reshape(3, 3), d.geom_xmat[gl].reshape(3, 3)
        cu, cl = d.geom_xpos[gu], d.geom_xpos[gl]
        dz = cu[2] - cl[2]
        if not (HEIGHT_WINDOW[0] <= dz <= HEIGHT_WINDOW[1]):
            return False                       # not resting on top (or it's the lower one)
        cos_tilt = np.cos(np.radians(TILT_TOL_DEG))
        if Ru[2, 2] < cos_tilt or Rl[2, 2] < cos_tilt:
            return False
        if abs(_wrap(_yaw(Ru) - _yaw(Rl), np.pi)) > self.yaw_tol:
            return False
        # studs are symmetric about the BODY origin (mesh origin), not the collision box centre
        # (which is offset along x): a half-turned brick would otherwise be misjudged by ~9 mm
        bu, bl = self.body[upper], self.body[lower]
        return float(np.linalg.norm(d.xpos[bu][:2] - d.xpos[bl][:2])) <= self.horiz_tol

    def _write_weld(self, d, e):
        """Freeze the current relative pose of eq e's two bodies into the weld and switch it on."""
        m = self.m
        b1, b2 = m.eq_obj1id[e], m.eq_obj2id[e]
        R1, p1 = d.xmat[b1].reshape(3, 3), d.xpos[b1]
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, _neg(d.xquat[b1]), d.xquat[b2])
        m.eq_data[e, 0:3] = 0.0                      # anchor at body2 origin
        m.eq_data[e, 3:6] = R1.T @ (d.xpos[b2] - p1)
        m.eq_data[e, 6:10] = rel_q
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1

    def _snap(self, d, upper, lower):
        m = self.m
        gu, gl = self.geom[upper], self.geom[lower]
        Rl = d.geom_xmat[gl].reshape(3, 3)
        Ru = d.geom_xmat[gu].reshape(3, 3)
        flip = abs(_wrap(_yaw(Ru) - _yaw(Rl), 2 * np.pi)) > np.pi / 2
        Rt = Rl @ np.diag([-1.0, -1.0, 1.0]) if flip else Rl.copy()   # keep whichever half-turn is closer
        body_pos = d.xpos[self.body[lower]] + SEAT_HEIGHT * Rl[:, 2]      # stud grid centred on the lower's
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, Rt.flatten())

        q, v = self.qadr[upper], self.dadr[upper]
        d.qpos[q:q + 3], d.qpos[q + 3:q + 7] = body_pos, quat
        d.qvel[v:v + 6] = d.qvel[self.dadr[lower]:self.dadr[lower] + 6]
        mujoco.mj_kinematics(m, d)

        e = self.eq[frozenset((upper, lower))]
        self._write_weld(d, e)
        self._set_collide(upper, lower, False)
        self._weld[upper] = (lower, e)
        self.events.append(f"snapped: {upper} seated on {lower}")


def _neg(q):
    out = np.zeros(4)
    mujoco.mju_negQuat(out, q)
    return out
