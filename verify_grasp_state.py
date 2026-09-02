"""
Headless verification + calibration for grasp_state.py.

Measures aperture_stall and gripper_object_distance across four reference
scenarios so the thresholds in grasp_state.py are picked from real numbers,
not guessed:
  1. GRASPED: full approach + close + lift 20cm + hold (verify_grasp_hold.py's
     passing case, brick1) -- confidence should read True throughout.
  2. EMPTY: approach + close on empty space, far from any brick (no object
     within reach) -- confidence should read False.
  3. DROPPED: same as GRASPED, but with contact/solref left at MuJoCo's
     default (soft) values so the grip is known to fail under load (see
     CLAUDE.md's "Fingertip/Brick Contact Tuning") -- confidence should
     read False once the brick has slipped back down.
  4. LONG_HOLD (brick2, 15s): aperture_stall is NOT a stable long-term
     signal -- it decays asymptotically even while genuinely still grasped
     (confirmed via gripper_object_distance staying flat), from ~0.011 right
     after closing toward an asymptote around ~0.0019. A threshold picked
     from a 3s hold (0.002) looked fine but would have failed by ~t=10s on
     this exact case -- caught only by testing a hold long enough to reach
     the asymptote, not by assuming a short window generalizes.

Run: python verify_grasp_state.py
"""
import pathlib, os
import numpy as np
import mujoco

from actuator_groups import RIGHT_ARM, RIGHT_HAND, LEFT_HAND
from arm_ik import ArmIK
from approach_path import ApproachPath
from grasp_primitive import hand_ctrl, grasp_target_quat
from grasp_state import aperture_stall, gripper_object_distance, grasp_confidence

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
SCENE = MODEL_DIR / "scene_fixed_table.xml"
GRASP_HEIGHT = 0.04


def reset_stance(m, d, key_id):
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    hold[LEFT_HAND] = 0.0
    hold[RIGHT_HAND] = 0.0
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)
    return hold


def approach_and_close(m, d, key_id, target_pos, brick_name, soften_contact=False):
    hold = reset_stance(m, d, key_id)

    if soften_contact:
        # deliberately revert to MuJoCo's default (softer) solref on this
        # brick's collision geom, so the grip is known to fail under load --
        # the DROPPED reference scenario.
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{brick_name}_collision")
        m.geom_solref[gid] = [0.02, 1]

    brick_short_axis = d.xmat[m.body(brick_name).id].reshape(3, 3)[:, 1]
    ik = ArmIK(m, "right_gripper_site", RIGHT_ARM)
    ik.sync(d.qpos)
    approach = ApproachPath(ik.site_pos(), target_pos)
    while not approach.on_descend_leg:
        q = approach.advance(ik, RIGHT_ARM, m.opt.timestep)
        d.ctrl[:] = hold; d.ctrl[RIGHT_ARM] = q; mujoco.mj_step(m, d)
    target_quat = grasp_target_quat(ik, target_pos, brick_short_axis)
    approach.set_target_quat(target_quat)
    while not approach.finished:
        q = approach.advance(ik, RIGHT_ARM, m.opt.timestep)
        d.ctrl[:] = hold; d.ctrl[RIGHT_ARM] = q; mujoco.mj_step(m, d)
    for _ in range(int(0.3 / m.opt.timestep)):
        q = ik.solve(target_pos, target_quat=target_quat, iters=2)
        d.ctrl[:] = hold; d.ctrl[RIGHT_ARM] = q; mujoco.mj_step(m, d)

    close_steps = int(1.2 / m.opt.timestep)
    for step in range(close_steps):
        grip = step / close_steps
        q = ik.solve(target_pos, target_quat=target_quat, iters=1)
        d.ctrl[:] = hold; d.ctrl[RIGHT_ARM] = q; d.ctrl[RIGHT_HAND] = hand_ctrl(grip, "right")
        mujoco.mj_step(m, d)
    for _ in range(int(0.5 / m.opt.timestep)):
        d.ctrl[RIGHT_HAND] = hand_ctrl(1.0, "right"); mujoco.mj_step(m, d)

    return hold, ik, target_quat


def lift_and_hold(m, d, ik, target_pos, target_quat, hold, site_id, body_id,
                   lift_height=0.20, hold_time=3.0):
    lift_target = target_pos.copy()
    lift_target[2] += lift_height
    lift_steps = int(2.0 / m.opt.timestep)
    for step in range(lift_steps):
        alpha = min(step / lift_steps, 1.0)
        wp = target_pos + alpha * (lift_target - target_pos)
        q = ik.solve(wp, target_quat=target_quat, iters=2)
        d.ctrl[:] = hold; d.ctrl[RIGHT_ARM] = q; d.ctrl[RIGHT_HAND] = hand_ctrl(1.0, "right")
        mujoco.mj_step(m, d)

    readings = []
    for step in range(int(hold_time / m.opt.timestep)):
        q = ik.solve(lift_target, target_quat=target_quat, iters=2)
        d.ctrl[:] = hold; d.ctrl[RIGHT_ARM] = q; d.ctrl[RIGHT_HAND] = hand_ctrl(1.0, "right")
        mujoco.mj_step(m, d)
        if step % int(0.25 / m.opt.timestep) == 0:
            stall = aperture_stall(d, RIGHT_HAND, "right")
            dist = gripper_object_distance(d, site_id, body_id)
            readings.append((step * m.opt.timestep, stall, dist))
    return readings


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_gripper_site")
    body_id = m.body("brick1").id

    print("=== Scenario 1: GRASPED (brick1) ===")
    reset_stance(m, d, key_id)  # populate d.xpos via forward kinematics before reading it
    target_pos = d.xpos[body_id].copy()
    target_pos[2] += GRASP_HEIGHT
    hold, ik, target_quat = approach_and_close(m, d, key_id, target_pos, "brick1")
    readings_grasped = lift_and_hold(m, d, ik, target_pos, target_quat, hold, site_id, body_id)
    for t, stall, dist in readings_grasped:
        print(f"  t={t:.2f}  aperture_stall={stall:.4f}  gripper_obj_dist={dist:.4f}")
    r = grasp_confidence(d, RIGHT_HAND, "right", site_id, body_id)
    print(f"  final: {r}")
    assert r["grasped"], "GRASPED scenario must report grasped=True"

    print("\n=== Scenario 2: EMPTY (close on empty space, no brick nearby) ===")
    hold = reset_stance(m, d, key_id)
    empty_target = d.xpos[body_id].copy()
    empty_target[1] -= 0.30  # well clear of any brick
    empty_target[2] += GRASP_HEIGHT
    hold, ik, target_quat = approach_and_close(m, d, key_id, empty_target, "brick1")
    r = grasp_confidence(d, RIGHT_HAND, "right", site_id, body_id)
    print(f"  {r}")
    assert not r["grasped"], "EMPTY scenario must report grasped=False"

    print("\n=== Scenario 3: DROPPED (default solref -- known to fail under load) ===")
    m2 = mujoco.MjModel.from_xml_path(str(SCENE))
    d2 = mujoco.MjData(m2)
    hold2, ik2, target_quat2 = approach_and_close(m2, d2, key_id, target_pos, "brick1", soften_contact=True)
    readings_dropped = lift_and_hold(m2, d2, ik2, target_pos, target_quat2, hold2, site_id, body_id)
    for t, stall, dist in readings_dropped:
        print(f"  t={t:.2f}  aperture_stall={stall:.4f}  gripper_obj_dist={dist:.4f}")
    r = grasp_confidence(d2, RIGHT_HAND, "right", site_id, body_id)
    print(f"  final: {r}")
    assert not r["grasped"], "DROPPED scenario must report grasped=False once it has slipped"

    print("\n=== Scenario 4: LONG_HOLD (brick2, 15s -- aperture_stall's decay asymptote) ===")
    body2_id = m.body("brick2").id
    hold = reset_stance(m, d, key_id)
    target2 = d.xpos[body2_id].copy()
    target2[2] += GRASP_HEIGHT
    hold, ik, target_quat2 = approach_and_close(m, d, key_id, target2, "brick2")
    readings_long = lift_and_hold(m, d, ik, target2, target_quat2, hold, site_id, body2_id,
                                   hold_time=15.0)
    for t, stall, dist in readings_long:
        if t % 2.0 < 0.25:
            print(f"  t={t:.1f}  stall={stall:.5f}  dist={dist:.4f}")
    r = grasp_confidence(d, RIGHT_HAND, "right", site_id, body2_id)
    print(f"  final (t=15s): {r}")
    assert r["grasped"], "LONG_HOLD scenario must still report grasped=True at t=15s"

    print("\nALL CALIBRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
