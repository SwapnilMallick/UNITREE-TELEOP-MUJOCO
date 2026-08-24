"""
Headless verification for arm_ik.py, following this repo's established
smoke-test pattern (see CLAUDE.md's "Verification Pattern"): no GUI, just
assertions.

For each arm, solves IK toward a pre-grasp point above the corresponding
brick and checks two things:
  1. The IK solve itself converges (scratch site position -> target).
  2. Driving the real simulation there along an up/over/down waypoint path
     (re-solving IK fresh each step, per ArmIK's warm-start design) gets the
     ACTUAL simulated gripper site close to the target too -- i.e. the target
     is genuinely reachable and approachable, not just mathematically
     solvable in isolation.

This test earned its complexity: an instant joint-space jump, an
independently-rate-limited joint ramp, and a straight-line Cartesian
interpolation were all tried first and all failed for different reasons
(see approach_path.py's docstring) before landing on the up/over/down path
approach_path.ApproachPath now implements, which is also what an
unconstrained position-only IK needs in practice -- see arm_ik.py's
nullspace posture bias for the companion fix on the IK side.

Run: python verify_arm_ik.py
"""
import pathlib, os
import numpy as np
import mujoco

from actuator_groups import LEG, WAIST, LEFT_ARM, LEFT_HAND, RIGHT_ARM, RIGHT_HAND, UPPER_BODY
from arm_ik import ArmIK
from approach_path import ApproachPath

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
SCENE = MODEL_DIR / "scene_fixed_table.xml"

# right arm naturally reaches -Y (right_shoulder mounts at y=-0.10021), left
# arm reaches +Y -- pair each arm with the brick on its own side.
#
# brick2 originally sat at y=0 (dead center), equidistant from both
# shoulders. Testing both arms against it found BOTH failed (~4.4-4.8cm vs a
# 2cm tolerance): reaching dead center requires the shoulder to rotate across
# the body's own centerline, producing a real torso/shoulder self-collision
# for either arm (confirmed via d.contact) that a taller pregrasp height did
# not fix (self-collision persisted up to 20cm above the brick -- it's a
# lateral geometry problem, not a vertical clearance one). Moved brick2 to
# y=-0.07 in scene_fixed_table.xml (confirmed no brick1<->brick2 collision-geom
# overlap at rest); with that fix the right arm reaches it cleanly (1.6cm,
# matching brick1/brick3's margin) while the left arm now clearly can't
# (10.5cm) -- so the assignment is: right arm takes both brick1 and brick2,
# left arm takes brick3. That's why only three cases are tested below instead
# of the four originally used to make this call.
CASES = [
    ("right", "right_gripper_site", RIGHT_ARM, "brick1"),  # brick1 y=-0.15
    ("right", "right_gripper_site", RIGHT_ARM, "brick2"),  # brick2 y=-0.07 (moved off-center)
    ("left",  "left_gripper_site",  LEFT_ARM,  "brick3"),  # brick3 y=+0.18
]

# A pre-grasp point above the brick, not its exact resting center -- the brick
# center sits essentially on the table surface, so commanding the gripper site
# there drives the hand straight into the tabletop. This is what a real
# approach trajectory would do too -- descend onto the brick from above, not
# drive straight at its center.
PREGRASP_HEIGHT = 0.08  # m above brick center
POS_TOL = 0.02  # 2 cm -- generous for a position-only, no-orientation first pass


def reset_stance(m, d, key_id):
    """Reset to stand_at_table with hands forced open. Returns the (modified)
    ctrl array to hold LEG/WAIST/hands at. Called once per case so each test
    starts clean from stance instead of wherever the previous case's arm
    ended up -- matters now that brick2 is tested with both arms in the same
    run."""
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    # The stance keyframe's hand pose has thumb_1 curled to +-1.05 rad. Hold
    # the hands open (all dof = 0) instead for this test -- grasp pose is a
    # later step's concern, not arm-IK's, and open is the physically correct
    # thing to do during pure reaching anyway.
    hold[LEFT_HAND] = 0.0
    hold[RIGHT_HAND] = 0.0
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)
    return hold


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")

    hold = reset_stance(m, d, key_id)
    # NOTE: left_hand_thumb_1_link and right_hand_thumb_1_link show a tiny
    # (0.27mm) static overlap with their own wrist_yaw_link at ANY thumb
    # angle, including 0 -- a pre-existing artifact of the stock Menagerie
    # collision meshes, not something introduced here or affected by pose.
    # Confirmed (separately) not to be the cause of any tracking failure in
    # this test; harmless at this magnitude. Flagging for the future
    # contact-tuning step rather than fixing here, out of scope for arm IK.
    if d.ncon:
        for c in d.contact[:d.ncon]:
            b1, b2 = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
            if "hand" in b1 or "hand" in b2:
                print(f"  (known, harmless: {b1} <-> {b2} dist={c.dist:.5f})")

    results = {}
    all_ok = True
    for side, site_name, arm_slice, brick_name in CASES:
        hold = reset_stance(m, d, key_id)
        pelvis_start = d.xpos[m.body("pelvis").id].copy()

        target = d.xpos[m.body(brick_name).id].copy()
        target[2] += PREGRASP_HEIGHT
        print(f"\n=== {side} arm -> {brick_name} pre-grasp point at {target} ===")

        ik = ArmIK(m, site_name, arm_slice)
        ik.sync(d.qpos)
        solved = ik.solve(target, iters=30)
        ik_err = np.linalg.norm(ik.site_pos() - target)
        print(f"  IK solve converged: scratch site error = {ik_err:.4f} m "
              f"(joint angles: {np.round(solved, 3)})")

        # That diagnostic solve() advanced ik's internal scratch state all the
        # way to the target -- re-sync it back to the robot's actual current
        # pose before driving the real simulation, or the "gradual path" below
        # would start from an already-solved scratch state instead of from
        # where the arm really is, collapsing it back into an instant jump.
        ik.sync(d.qpos)

        # Drive the real simulation along the up/over/down approach path --
        # see approach_path.py's docstring for why (an instant joint-space
        # jump, an independent per-joint rate ramp, and a straight-line
        # Cartesian interpolation were all tried first and all failed).
        path = ApproachPath(ik.site_pos(), target)
        while not path.finished:
            q = path.advance(ik, arm_slice, m.opt.timestep)
            d.ctrl[:] = hold
            d.ctrl[arm_slice] = q
            mujoco.mj_step(m, d)
        settle_steps = int(1.0 / m.opt.timestep)  # extra time to settle after arrival
        for _ in range(settle_steps):
            q = path.advance(ik, arm_slice, m.opt.timestep)
            d.ctrl[:] = hold
            d.ctrl[arm_slice] = q
            mujoco.mj_step(m, d)

        actual_site_pos = d.site(site_name).xpos.copy()
        actual_err = np.linalg.norm(actual_site_pos - target)
        pelvis_drift = np.linalg.norm(d.xpos[m.body("pelvis").id] - pelvis_start)
        has_nan = bool(np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)))

        print(f"  physically simulated site error = {actual_err:.4f} m "
              f"(pelvis drift = {pelvis_drift:.4f} m, NaN = {has_nan})")

        ok = (ik_err < POS_TOL) and (actual_err < POS_TOL) and not has_nan and (pelvis_drift < 1e-3)
        print(f"  {'PASS' if ok else 'FAIL'}")
        all_ok = all_ok and ok
        results[(side, brick_name)] = dict(ok=ok, ik_err=ik_err, actual_err=actual_err)

    # Confirms the settled pick assignment: right arm takes brick1 and
    # brick2, left arm takes brick3 (see the CASES comment above for how
    # that was decided -- both arms were tested against brick2 before it
    # was moved off dead-center, and again after, before landing here).
    if ("right", "brick2") in results:
        print(f"\nbrick2 assignment: right arm, actual_err="
              f"{results[('right', 'brick2')]['actual_err']:.4f} m")

    print("\n" + ("ALL CASES PASSED" if all_ok else "SOME CASES FAILED"))
    assert all_ok


if __name__ == "__main__":
    main()
