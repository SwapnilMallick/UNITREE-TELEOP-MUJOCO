"""
Headless verification for locomotion.py + --base-mode planar, against a
SYNTHETIC joystick+controller source -- there is no VR hardware reachable
from this environment (same caveat as verify_teleop_control.py). Confirms
the CONTROL LOOP wiring: command smoothing/deadzone, base velocity tracking,
legs/waist staying pinned, no NaN, and -- the actual point of this feature --
that the arm IK targets really do move with the body (torso-frame tracking)
instead of staying anchored to a fixed point in the room. It canNOT confirm
the joystick axis mapping/signs against a real headset's actual output.

Run: python verify_locomotion.py [--record]
--record saves a short offscreen MP4 per scenario under videos/ (off by
default, no effect on pass/fail) -- same opt-in pattern
verify_stack_sequence.py's own --record uses.
"""
import argparse, pathlib, os
import numpy as np
import mujoco

from actuator_groups import LEG, WAIST
import actuator_groups_planar as agp
from teleop_control import TeleopController
from locomotion import BaseVelocityCommand, PlanarBaseController

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
SCENE = MODEL_DIR / "scene_planar_table.xml"

RIGHT_BASE = np.array([0.35, -0.25, 1.05])
LEFT_BASE = np.array([0.35, 0.25, 1.05])


class FakeTeleData:
    def __init__(self, left_wrist_pose, right_wrist_pose, left_trigger, right_trigger,
                 left_stick, right_stick, motion_data_ready=True):
        self.left_wrist_pose = left_wrist_pose
        self.right_wrist_pose = right_wrist_pose
        self.left_ctrl_triggerValue = left_trigger
        self.right_ctrl_triggerValue = right_trigger
        self.left_ctrl_thumbstickValue = np.asarray(left_stick, dtype=float)
        self.right_ctrl_thumbstickValue = np.asarray(right_stick, dtype=float)
        self.motion_data_ready = motion_data_ready
        # inert defaults for fields stand_next_to_table.py also reads
        self.left_ctrl_aButton = self.right_ctrl_aButton = False
        self.right_ctrl_bButton = False
        self.left_ctrl_squeeze = self.right_ctrl_squeeze = False
        self.left_ctrl_squeezeValue = self.right_ctrl_squeezeValue = 0.0


class FakeJoystickSource:
    """Scriptable fake tele_data source: holds controllers at a realistic
    resting pose (wrist_offset added for scenario 4's scripted reach), and
    lets the test driver set the commanded stick values each frame via
    .left_stick / .right_stick (mutated directly -- no method call needed
    mid-scenario, matching how a real headset's state just IS whatever it
    currently is)."""

    def __init__(self):
        self.left_stick = np.zeros(2)   # (x, y)
        self.right_stick = np.zeros(2)  # (x, y) -- only x (yaw) is read
        self.wrist_offset = {"left": np.zeros(3), "right": np.zeros(3)}
        self.triggers = {"left": 10.0, "right": 10.0}  # released

    def get_tele_data(self):
        left_pose = np.eye(4)
        left_pose[:3, 3] = LEFT_BASE + self.wrist_offset["left"]
        right_pose = np.eye(4)
        right_pose[:3, 3] = RIGHT_BASE + self.wrist_offset["right"]
        return FakeTeleData(left_pose, right_pose, self.triggers["left"], self.triggers["right"],
                             self.left_stick, self.right_stick)


def build_sim():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    reset_to_stance(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    return m, d, hold, key_id


def reset_to_stance(m, d, key_id):
    """Full reset: robot AND base back to the stand_at_table keyframe (base
    qpos=0 there -- make_planar_scene.py extends it that way). Used between
    scenarios instead of zeroing ONLY the base's qpos/qvel -- doing that
    alone leaves the ARM's joint angles wherever the previous scenario left
    them, under a now-SUDDENLY-DIFFERENT base orientation (base yaw snapped
    to 0 while the arm's actual joint config stayed put) -- forward
    kinematics then places the gripper site somewhere wildly different from
    where it physically was an instant before, which calibrate() then (not
    wrongly, just fed a bad premise) treats as the real "current" position
    to ramp away from. A real --teleop session's base never teleports (only
    continuous velocity integration), so this whole failure mode is a
    test-harness artifact of this script's own "start each scenario from a
    clean pose" design, not a --base-mode planar bug -- confirmed by tracing
    it to exactly this mechanism before writing this fix, not assumed."""
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    d.ctrl[:] = m.key_ctrl[key_id]
    mujoco.mj_forward(m, d)


def torso_frame_site_pos(d, base_ctrl, site_world):
    """World-frame site position, expressed in the base's CURRENT local
    frame -- the direct test of "hands move with the body": if the arm's VR
    controller target hasn't moved, this value should stay constant as the
    base walks/turns around underneath it."""
    bx, by, byaw = base_ctrl.pose_from(d)
    c, s = np.cos(-byaw), np.sin(-byaw)
    dx, dy, dz = site_world[0] - bx, site_world[1] - by, site_world[2]
    return np.array([c * dx - s * dy, s * dx + c * dy, dz])


def run_scenario(name, m, d, hold, ctrl, base_ctrl, base_cmd, src, duration, key_id,
                  left_stick=(0.0, 0.0), right_stick=(0.0, 0.0), reach=None,
                  qpos_slices=None, recorder=None, reset_base=True):
    """Drives `duration` seconds with a constant (left_stick, right_stick)
    command (reach, if given, is a callable t -> {"left": xyz, "right": xyz}
    scripted wrist-offset motion, for scenario 4). Returns a metrics dict.

    reset_base=True (default): zero the base's qpos/qvel before this
    scenario runs, so each scenario tests ITS OWN command in isolation from
    a clean (0,0,0) pose, rather than accumulating displacement across the
    whole test suite's sequential scenarios (which, chained at these speeds/
    durations, travels several metres and can run the base into its own
    generous-backstop joint range -- a test-harness artifact, not a real
    --teleop session behavior, where an operator naturally walks continuously
    and IS expected to accumulate position)."""
    if reset_base and base_ctrl.present:
        reset_to_stance(m, d, key_id)
        # An instant base teleport (this reset) is NOT something a real
        # --teleop session ever does (the base only ever moves continuously,
        # by velocity integration) -- it's purely this test's "start each
        # scenario from a clean pose" harness design. But TeleopController's
        # rate limiter (_target_pos_prev, by design -- see TARGET_MAX_SPEED)
        # assumes continuous motion between frames; fed a multi-metre
        # discontinuity it would (correctly, for a REAL glitch) ramp back in
        # slowly over many seconds, which would look exactly like a locomotion
        # bug in this test's numbers without actually being one. Re-calibrate
        # right after the reset -- the test-harness equivalent of an operator
        # pressing 'c' -- so each scenario's rate-limiter state starts clean,
        # same as it would after any real calibration.
        src.wrist_offset["left"][:] = 0.0
        src.wrist_offset["right"][:] = 0.0
        ctrl.calibrate(m, d, src.get_tele_data())
    src.left_stick = np.asarray(left_stick, dtype=float)
    src.right_stick = np.asarray(right_stick, dtype=float)
    n_steps = int(round(duration / m.opt.timestep))
    base_speed_err = []
    torso_site_hist = {"left": [], "right": []}
    pelvis_z_hist = []
    any_nan = False
    cmd_vx_hist, cmd_vy_hist, cmd_yaw_hist = [], [], []

    for i in range(n_steps):
        t = i * m.opt.timestep
        if reach is not None:
            offsets = reach(t)
            src.wrist_offset["left"] = offsets.get("left", np.zeros(3))
            src.wrist_offset["right"] = offsets.get("right", np.zeros(3))
        data = src.get_tele_data()

        vx, vy, yaw_rate = base_cmd.update(data, m.opt.timestep)
        base_ctrl.step(m, d, vx, vy, yaw_rate)
        cmd_vx_hist.append(vx); cmd_vy_hist.append(vy); cmd_yaw_hist.append(yaw_rate)

        if not ctrl.calibrated:
            if not ctrl.try_calibrate(m, d, data):
                d.ctrl[LEG] = hold[LEG]
                d.ctrl[WAIST] = hold[WAIST]
        else:
            ctrl.step(m, d, hold, data, base_pose=base_ctrl.pose_from(d))

        mujoco.mj_step(m, d)

        if np.isnan(d.qpos).any() or np.isnan(d.qvel).any():
            any_nan = True
            break

        # realised base velocity (world-frame qvel) vs. the ROBOT-frame command,
        # rotated the same way base_ctrl.step() rotates it
        if base_ctrl.present:
            bx, by, byaw = base_ctrl.pose_from(d)
            c, s = np.cos(byaw), np.sin(byaw)
            want_world = np.array([c * vx - s * vy, s * vx + c * vy])
            got_world = d.qvel[[0, 1]].copy()
            base_speed_err.append(float(np.linalg.norm(got_world - want_world)))
            pelvis_z_hist.append(float(d.xpos[m.body("pelvis").id][2]))

        if ctrl.calibrated:
            for side in ("left", "right"):
                site_world = ctrl._iks[side].site_pos()
                torso_site_hist[side].append(torso_frame_site_pos(d, base_ctrl, site_world))

        if recorder is not None:
            recorder(d)

    out = {
        "name": name, "nan": any_nan, "steps_run": i + 1 if any_nan else n_steps,
        "base_speed_err_mean": float(np.mean(base_speed_err)) if base_speed_err else None,
        "base_speed_err_max": float(np.max(base_speed_err)) if base_speed_err else None,
        "pelvis_z_min": float(np.min(pelvis_z_hist)) if pelvis_z_hist else None,
        "pelvis_z_max": float(np.max(pelvis_z_hist)) if pelvis_z_hist else None,
    }
    # baseline AFTER a short settle window, not frame 0 -- right after
    # calibrate() the arm is still RAMPING from stance to TELEOP_HOME (rate-
    # limited by TARGET_MAX_SPEED, by design -- see teleop_control.py), which
    # is real but unrelated arm motion that would otherwise swamp the "does
    # the hand stay put in torso frame" measurement this is actually checking
    settle_steps = min(int(round(1.0 / m.opt.timestep)), max(len(torso_site_hist["left"]) - 1, 0))
    for side in ("left", "right"):
        h = torso_site_hist[side]
        if len(h) > settle_steps + 1:
            h = np.array(h[settle_steps:])
            drift = np.linalg.norm(h - h[0], axis=1)
            out[f"{side}_torso_drift_mean_cm"] = float(np.mean(drift)) * 100
            out[f"{side}_torso_drift_max_cm"] = float(np.max(drift)) * 100
        else:
            out[f"{side}_torso_drift_mean_cm"] = out[f"{side}_torso_drift_max_cm"] = None
    return out


def print_report(r):
    print(f"\n=== {r['name']} ===")
    print(f"  steps run: {r['steps_run']}   NaN: {r['nan']}")
    if r["base_speed_err_mean"] is not None:
        print(f"  base velocity tracking error: mean={r['base_speed_err_mean']:.4f} m/s "
              f"max={r['base_speed_err_max']:.4f} m/s")
    if r["pelvis_z_min"] is not None:
        print(f"  pelvis height range: {r['pelvis_z_min']:.4f} .. {r['pelvis_z_max']:.4f} m "
              f"(stance ~0.793m -- large deviation would mean legs didn't stay locked)")
    for side in ("left", "right"):
        mean_, max_ = r[f"{side}_torso_drift_mean_cm"], r[f"{side}_torso_drift_max_cm"]
        if mean_ is not None:
            print(f"  {side} hand drift in TORSO frame (VR controller held still): "
                  f"mean={mean_:.2f}cm max={max_:.2f}cm "
                  f"(near 0 means the hand correctly moved WITH the body)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", nargs="?", const="videos", default=None, metavar="DIR")
    p.add_argument("--weighted-ik", action="store_true",
                    help="also run every scenario with --teleop-weighted-ik (mink) "
                         "instead of the default DLS solver")
    args = p.parse_args()

    if not SCENE.exists():
        raise SystemExit(f"{SCENE} not found -- run make_planar_base.py + "
                          f"make_planar_scene.py and copy the output into MODEL_DIR")

    solvers = [False]
    if args.weighted_ik:
        solvers.append(True)

    all_results = []
    for weighted in solvers:
        tag = "mink" if weighted else "DLS"
        m, d, hold, key_id = build_sim()
        base_ctrl = PlanarBaseController(m)
        assert base_ctrl.present, "scene_planar_table.xml has no base joints"
        kw = dict(arm_qpos_slices={"left": agp.LEFT_ARM_QPOS, "right": agp.RIGHT_ARM_QPOS})
        if weighted:
            kw.update(weighted_ik=True, weighted_ik_solver="mink")
        ctrl = TeleopController(m, **kw)
        src = FakeJoystickSource()
        # calibrate immediately -- this test isn't exercising the calibration
        # gate (verify_teleop_control.py already covers that), just locomotion
        ctrl.calibrate(m, d, src.get_tele_data())
        base_cmd = BaseVelocityCommand()

        recorder = None
        frames = []
        if args.record:
            renderer = mujoco.Renderer(m, height=360, width=480)
            def recorder(dd, renderer=renderer, frames=frames):
                if len(frames) % 5 == 0:   # thin to ~100fps/5=20fps-ish sampling
                    renderer.update_scene(dd, camera="fpv_teleop")
                    frames.append(renderer.render().copy())
                else:
                    frames.append(None)

        print(f"\n########## solver={tag} ##########")

        r1 = run_scenario(f"[{tag}] 1. standing still, zero command, 10s", m, d, hold, ctrl,
                           base_ctrl, base_cmd, src, duration=10.0, key_id=key_id, recorder=recorder)
        print_report(r1)

        r2a = run_scenario(f"[{tag}] 2a. walk forward", m, d, hold, ctrl, base_ctrl, base_cmd,
                            src, duration=4.0, key_id=key_id, left_stick=(0.0, -1.0), recorder=recorder)
        print_report(r2a)
        r2b = run_scenario(f"[{tag}] 2b. strafe lateral", m, d, hold, ctrl, base_ctrl, base_cmd,
                            src, duration=4.0, key_id=key_id, left_stick=(1.0, 0.0), recorder=recorder)
        print_report(r2b)
        r2c = run_scenario(f"[{tag}] 2c. turn in place", m, d, hold, ctrl, base_ctrl, base_cmd,
                            src, duration=4.0, key_id=key_id, right_stick=(1.0, 0.0), recorder=recorder)
        print_report(r2c)

        r3 = run_scenario(f"[{tag}] 3. combined walk+turn", m, d, hold, ctrl, base_ctrl, base_cmd,
                           src, duration=5.0, key_id=key_id, left_stick=(0.3, -0.7), right_stick=(0.5, 0.0),
                           recorder=recorder)
        print_report(r3)

        def reach_fn(t):
            # small scripted reach, both arms, sinusoidal -- big enough to
            # exercise the IK, small enough to stay reachable
            a = 0.06 * np.sin(2 * np.pi * t / 2.0)
            return {"left": np.array([a, 0.0, 0.0]), "right": np.array([a, 0.0, 0.0])}

        r4a = run_scenario(f"[{tag}] 4a. reach while standing still", m, d, hold, ctrl, base_ctrl,
                            base_cmd, src, duration=4.0, key_id=key_id, reach=reach_fn, recorder=recorder)
        print_report(r4a)
        r4b = run_scenario(f"[{tag}] 4b. reach while walking forward", m, d, hold, ctrl, base_ctrl,
                            base_cmd, src, duration=4.0, key_id=key_id, left_stick=(0.0, -1.0), reach=reach_fn,
                            recorder=recorder)
        print_report(r4b)

        all_results += [r1, r2a, r2b, r2c, r3, r4a, r4b]

        if args.record:
            import imageio
            out_dir = pathlib.Path(args.record)
            out_dir.mkdir(parents=True, exist_ok=True)
            real_frames = [f for f in frames if f is not None]
            path = out_dir / f"locomotion_{tag}.mp4"
            imageio.mimsave(str(path), real_frames, fps=20)
            print(f"  saved {path} ({len(real_frames)} frames)")

    n_nan = sum(r["nan"] for r in all_results)
    print(f"\n{'='*60}\n{len(all_results)} scenarios run, {n_nan} hit NaN.")
    if n_nan:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
