"""
Headless verification for stack_sequence.py: drives StackSequence exactly
the way stand_next_to_table.py's live viewer loop will -- one .step() call
per mj_step() -- through the full STACK_ORDER (brick2 then brick3, per
stack_sequence.py's own docstring for why that order/arm assignment), and
checks the REAL, SETTLED resting position of each brick afterward, not just
whether the state machine reports DONE.

This matters because DONE only means "the phases ran to completion", not
"the brick actually ended up in the stack" -- PickSequence's phases don't
themselves check contact/grasp success at any transition (see
grasp_state.py: that's a separate, non-privileged signal, not something the
timing-based phase transitions themselves gate on). So this script checks
ground truth: does brick2 actually end up resting on brick1's position
after a real settle period, and does brick1 (the untouched base) stay where
it started?

brick3 is included and NOT required to pass -- it's expected to fail for
two independent, already-documented reasons (see stack_sequence.py's
docstring): the existing brick3 grasp-approach problem, and a newly-found
left-arm-to-this-stack-location reach problem. Reported honestly, not
hidden or asserted.

Optional video recording (--record [OUTPUT.mp4]): saves an offscreen render
of the run, sampled at ~30fps of simulated time -- the same pattern
CLAUDE.md's "Verification Pattern" already recommends for visual evidence in
this environment (MUJOCO_GL=glfw on macOS; mujoco.Renderer, not the
interactive viewer, which needs a real display). Off by default -- doesn't
change the pass/fail behavior or add a hard dependency (imageio is only
imported if --record is actually used) for the normal regression-check run.

Run: python verify_stack_sequence.py
       python verify_stack_sequence.py --record                    # -> stack_sequence.mp4
       python verify_stack_sequence.py --record out/attempt3.mp4
"""
import argparse, pathlib, os
import numpy as np
import mujoco

from stack_sequence import StackSequence, BASE_BRICK, BRICK_HALF_HEIGHT, LAYER_GAP

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
SCENE = MODEL_DIR / "scene_fixed_table.xml"

MAX_TIME = 40.0        # s -- generous upper bound for two full pick+place cycles
SETTLE_AFTER = 2.0     # s of physics settle after StackSequence reports done, before
                        # reading final resting positions
XY_TOL = 0.03           # m, lenient lateral tolerance for "landed roughly in the stack"
Z_TOL = 0.015           # m, vertical tolerance (about half a brick's height)

# Recording -- framed to show the table, both bricks, and the arm's transit/descend
# clearly (same view used to diagnose the placement collision this session).
RECORD_FPS = 30
RECORD_SIZE = (640, 480)     # width, height
RECORD_LOOKAT = np.array([0.35, -0.12, 0.85])
RECORD_DISTANCE = 0.9
RECORD_AZIMUTH = -60
RECORD_ELEVATION = -20


def _make_recorder():
    """Sets up an offscreen renderer + fixed camera for --record. Imports
    imageio lazily so it's only a hard dependency when recording is actually
    requested."""
    try:
        import imageio
    except ImportError as e:
        raise SystemExit("--record needs imageio: pip install imageio") from e
    os.environ.setdefault("MUJOCO_GL", "glfw")  # macOS: egl/osmesa aren't valid here
    cam = mujoco.MjvCamera()
    cam.lookat = RECORD_LOOKAT
    cam.distance = RECORD_DISTANCE
    cam.azimuth = RECORD_AZIMUTH
    cam.elevation = RECORD_ELEVATION
    return imageio, cam


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", nargs="?", const="stack_sequence.mp4", default=None,
                         metavar="OUTPUT.mp4",
                         help="save an MP4 recording of the run (path optional, "
                              "defaults to stack_sequence.mp4 in the cwd)")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    brick1_start = d.xpos[m.body(BASE_BRICK).id].copy()
    expected_layer2 = brick1_start.copy()
    expected_layer2[2] += 2 * BRICK_HALF_HEIGHT + LAYER_GAP
    print(f"brick1 (base) starts at {brick1_start}")
    print(f"expected brick2 stacked position ~ {expected_layer2}")

    imageio = renderer = cam = None
    frames = []
    dt = m.opt.timestep
    frame_every = max(int(round((1.0 / RECORD_FPS) / dt)), 1)
    if args.record:
        imageio, cam = _make_recorder()
        renderer = mujoco.Renderer(m, height=RECORD_SIZE[1], width=RECORD_SIZE[0])

    def maybe_capture(step_idx):
        if renderer is not None and step_idx % frame_every == 0:
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())

    seq = StackSequence(m)
    seq.start(m, d)

    max_steps = int(MAX_TIME / dt)
    phases_seen = []
    last_key = None
    steps = 0
    while steps < max_steps and not seq.done:
        brick_name, phase = seq.step(m, d, hold)
        mujoco.mj_step(m, d)
        steps += 1
        maybe_capture(steps)
        key = (brick_name, phase)
        if key != last_key:
            phases_seen.append(key)
            last_key = key

    reached_done = seq.done
    print(f"\nStackSequence.done={reached_done} after {steps} steps ({steps*dt:.1f}s)")
    print("phase transitions:")
    for brick_name, phase in phases_seen:
        print(f"  {brick_name}: {phase}")

    for _ in range(int(SETTLE_AFTER / dt)):
        d.ctrl[:] = hold
        mujoco.mj_step(m, d)
        steps += 1
        maybe_capture(steps)

    if args.record:
        imageio.mimsave(args.record, frames, fps=RECORD_FPS)
        print(f"\nsaved {len(frames)} frames ({len(frames)/RECORD_FPS:.1f}s) to {args.record}")

    has_nan = bool(np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)))
    brick1_final = d.xpos[m.body(BASE_BRICK).id].copy()
    brick2_final = d.xpos[m.body("brick2").id].copy()
    brick3_final = d.xpos[m.body("brick3").id].copy()

    base_drift = np.linalg.norm(brick1_final - brick1_start)
    brick2_err = np.linalg.norm(brick2_final - expected_layer2)
    brick2_xy_ok = np.linalg.norm(brick2_final[:2] - expected_layer2[:2]) < XY_TOL
    brick2_z_ok = abs(brick2_final[2] - expected_layer2[2]) < Z_TOL

    print(f"\nNaN={has_nan}")
    print(f"brick1 (base) drift from start: {base_drift*100:.2f}cm  "
          f"({'OK, undisturbed' if base_drift < 0.02 else 'MOVED -- placement disturbed the base'})")
    print(f"brick2 final position: {brick2_final}  "
          f"(target {expected_layer2}, err={brick2_err*100:.2f}cm, "
          f"xy_ok={brick2_xy_ok} z_ok={brick2_z_ok})")
    print(f"brick3 final position: {brick3_final}  "
          f"(started elsewhere -- not required to have moved; see stack_sequence.py's docstring)")

    brick2_stacked = brick2_xy_ok and brick2_z_ok
    ok = (not has_nan) and (base_drift < 0.02) and brick2_stacked
    print(f"\n{'PASS' if ok else 'FAIL'}: brick2 stacked on brick1, base undisturbed, no NaN "
          f"(required)")
    print("brick3: known open issue (grasp-approach geometry + cross-body place-reach), "
          "not required to pass -- see stack_sequence.py's docstring for the two independent reasons.")

    assert ok


if __name__ == "__main__":
    main()
