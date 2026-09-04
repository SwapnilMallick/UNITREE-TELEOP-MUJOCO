"""
G1 standing fixed next to a table (no balance control needed), running
either a scripted pick-lift-hold sequence or live VR teleoperation.

The base is welded (freejoint removed) and the legs/waist are held at the
'stand' keyframe by their position actuators. Everything above the waist
(arms + Dex3-1 3-finger hands, indices 15..42) is available -- PickSequence
(pick_sequence.py) drives it through approach -> orient -> close -> lift ->
hold with a SCRIPTED target (--pick chooses which brick, read fresh from
the live sim -- never hardcoded). --teleop instead drives it with LIVE
input from a Quest 3S (or other televuer-supported headset) via
TeleopController (teleop_control.py) -- same per-frame .step() contract,
just fed by real controller data instead of a scripted timer. See
teleop_control.py's module docstring for what's verified (the control-loop
wiring, headlessly) vs. not (the coordinate mapping against real hardware,
since no headset is reachable from this dev environment).

    python stand_next_to_table.py                       # third-person view, picks brick1
    python stand_next_to_table.py --view egocentric      # robot head POV, for teleoperation
    python stand_next_to_table.py --pick brick2          # pick brick2 instead (also right arm)
    python stand_next_to_table.py --pick none            # old behavior: just hold stance
    python stand_next_to_table.py --teleop --cert-file cert.pem --key-file key.pem
                                                          # live VR teleop, both arms (needs
                                                          # `pip install televuer` and a headset)
    python stand_next_to_table.py --teleop --cert-file cert.pem --key-file key.pem --hand-tracking
                                                          # same, but real per-finger control
                                                          # via dex_retargeting instead of
                                                          # controller triggers (see
                                                          # teleop_control.py's HandRetargeter)
    python stand_next_to_table.py --view third_person /path/to/scene.xml
    python stand_next_to_table.py --pick brick1 --record-episodes recordings/pick_brick1
                                                          # also records frames+joint states/
                                                          # actions per step via EpisodeRecorder
                                                          # (episode_recording.py) -- works with
                                                          # any driver (--pick, --teleop, ...),
                                                          # runs in parallel, not wired into any
                                                          # of them
"""
import argparse, time, pathlib, os
import numpy as np
import mujoco, mujoco.viewer

from actuator_groups import LEG, WAIST, LEFT_ARM, LEFT_HAND, RIGHT_ARM, RIGHT_HAND, UPPER_BODY
from pick_sequence import PickSequence
from teleop_control import TeleopController, FpvStreamer, HandRetargeter
from episode_recording import EpisodeRecorder

# Only brick1/brick2 (right arm) are verified reliable -- see
# verify_grasp_hold.py and CLAUDE.md's "Fingertip/Brick Contact Tuning".
# brick3 (left arm) is a documented, unresolved grasp-approach-geometry
# problem; included here so --pick brick3 is possible for testing, not
# because it's expected to succeed.
PICK_TARGETS = {
    "brick1": ("right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick1"),
    "brick2": ("right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick2"),
    "brick3": ("left",  "left_gripper_site",  LEFT_ARM,  LEFT_HAND,  "brick3"),
}

# Resolved relative to this file (not the caller's cwd) so the script runs
# from anywhere. The Menagerie clone is expected as a sibling of this repo:
#   <parent>/mujoco_menagerie/unitree_g1
# Override with the MODEL_DIR env var, or pass an explicit scene path as argv[1].
MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
DEFAULT_SCENE = MODEL_DIR / "scene_fixed_table.xml"

# the fixed camera baked into g1_fixed_upper.xml by make_fixed_base.py
EGOCENTRIC_CAMERA = "fpv_teleop"

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", nargs="?", type=pathlib.Path, default=DEFAULT_SCENE,
                         help=f"path to scene xml (default: {DEFAULT_SCENE})")
    parser.add_argument("--view", choices=["third_person", "egocentric"], default="third_person",
                         help="camera view: 'third_person' (default, free orbit camera) or "
                              "'egocentric' (robot head POV, for teleoperation)")
    parser.add_argument("--pick", choices=[*PICK_TARGETS, "none"], default="brick1",
                         help="which brick to run the scripted pick-lift-hold sequence on "
                              "(default: brick1). 'none' reverts to just holding stance, the "
                              "pre-PickSequence behavior. brick3 is a known, documented open "
                              "issue (see CLAUDE.md) -- included for testing, not expected to succeed. "
                              "Ignored if --teleop is given.")
    parser.add_argument("--teleop", action="store_true",
                         help="drive both arms live from a VR headset via televuer, instead of "
                              "running the scripted --pick sequence. Requires `pip install "
                              "televuer` (not installed in this dev environment) and a connected "
                              "headset -- see teleop_control.py for what's verified vs. not.")
    parser.add_argument("--cert-file", type=pathlib.Path, default=None,
                         help="SSL cert file for televuer's HTTPS/WebRTC headset connection "
                              "(see televuer's README, 'Generate Certificate Files'). Required "
                              "with --teleop for most headsets.")
    parser.add_argument("--key-file", type=pathlib.Path, default=None,
                         help="SSL key file, paired with --cert-file.")
    parser.add_argument("--display-mode", choices=["pass-through", "ego", "immersive"],
                         default="pass-through",
                         help="what the headset shows with --teleop (default: pass-through, "
                              "the real world through the headset's own cameras -- no "
                              "streaming). 'ego' streams the fpv_teleop camera into a small "
                              "window with the real world around it; 'immersive' replaces the "
                              "view entirely with fpv_teleop. Both stream over televuer's zmq "
                              "transport, enabled automatically when set.")
    parser.add_argument("--hand-tracking", action="store_true",
                         help="with --teleop, use real per-finger hand-tracking (via "
                              "televuer's use_hand_tracking=True + dex_retargeting) instead "
                              "of controller triggers. Requires `pip install dex_retargeting` "
                              "(the upstream package, NOT unitreerobotics/xr_teleoperate's "
                              "forked pin -- see teleop_control.py's HandRetargeter docstring) "
                              "plus CPU torch. Default is controller mode (a trigger's grip "
                              "scalar), matching the locked-in teleop design decision.")
    parser.add_argument("--record-episodes", type=pathlib.Path, default=None, metavar="DIR",
                         help="also record frames + joint states/actions per step via "
                              "EpisodeRecorder (episode_recording.py), into DIR in "
                              "xr_teleoperate's own per-episode data.json format. Works "
                              "alongside any driver (--pick, --teleop, or --pick none) -- "
                              "runs in parallel, not wired into any of them. Off by default.")
    parser.add_argument("--teleop-debug", action="store_true",
                         help="with --teleop, print calibration + per-frame IK diagnostics "
                              "(captured reference poses, |pos_delta|, IK target, residual) "
                              "-- for diagnosing the arm not following the controller. "
                              "Off by default.")
    parser.add_argument("--teleop-orientation", action="store_true",
                         help="with --teleop, add 6-DOF IK following controller ORIENTATION. "
                              "EXPERIMENTAL / poor: arm_ik's solver maps a controller "
                              "rotation to a wildly axis-dependent gripper rotation and "
                              "degrades position 10-25cm -- gives some orientation influence "
                              "but not faithful control (needs a weighted IK). Default is "
                              "position-only.")
    parser.add_argument("--teleop-scale", type=float, default=None, metavar="S",
                         help="with --teleop, controller-motion -> robot-motion scale "
                              "(default: teleop_control.SCALE = 0.5). The operator's arm "
                              "range is bigger than this arm's ~0.45m envelope, so <1.0 "
                              "keeps targets reachable. Lower if the arm still hits limits; "
                              "raise toward 1.0 if motion feels sluggish.")
    return parser.parse_args()

def main():
    args = parse_args()
    scene = args.scene
    if not scene.exists():
        raise SystemExit(f"scene file not found: {scene}\n"
                          f"set MODEL_DIR or pass the scene path as an argument")
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)

    # Prefer "stand_at_table" (full-scene keyframe incl. brick poses, defined in
    # scene_fixed_table.xml) over index 0: the robot-only "stand" keyframe from
    # g1_fixed_upper.xml gets zero/identity-padded by the compiler for any bodies
    # (e.g. bricks) it doesn't know about, which resets them to the world origin.
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    if key_id < 0:
        key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        key_id = 0

    # start in the stance pose and hold it
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()  # legs/waist -> 0, arms -> raised stance
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    seq = None
    tele_source = None
    teleop_ctrl = None
    fpv_streamer = None
    if args.teleop:
        # Lazy import: televuer isn't installed in most dev environments
        # (this one included) -- only needed when --teleop is actually used,
        # matching the pattern already used for imageio in verify_stack_sequence.py.
        try:
            from televuer import TeleVuerWrapper
        except ImportError as e:
            raise SystemExit(
                "--teleop needs the televuer package: pip install televuer\n"
                "(see https://github.com/unitreerobotics/televuer)") from e
        # display_mode="pass-through" (default) shows the real world through the
        # headset's own cameras, no streaming. --display-mode ego/immersive streams
        # the fpv_teleop camera instead -- see FpvStreamer in teleop_control.py;
        # televuer requires zmq (or webrtc) enabled for either of those modes.
        tele_source = TeleVuerWrapper(
            use_hand_tracking=args.hand_tracking,  # controllers by default, matching
                                                     # the locked-in teleop design decision
            binocular=False,  # FpvStreamer renders one mono frame per camera, not a
                               # side-by-side stereo pair
            img_shape=(480, 640),  # (height, width) -- must match FpvStreamer's actual
                                    # rendered frame size exactly; televuer's img2display
                                    # buffer is sized from THIS, independent of binocular
                                    # (binocular=False alone did NOT fix the shape mismatch --
                                    # confirmed empirically, this is the real knob)
            display_mode=args.display_mode,
            zmq=(args.display_mode != "pass-through"),
            cert_file=str(args.cert_file) if args.cert_file else None,
            key_file=str(args.key_file) if args.key_file else None,
        )
        hand_retargeter = None
        if args.hand_tracking:
            try:
                hand_retargeter = HandRetargeter()
            except ImportError as e:
                raise SystemExit(
                    "--hand-tracking needs the upstream dex_retargeting package "
                    "(NOT unitreerobotics/xr_teleoperate's forked pin -- see "
                    "teleop_control.py's HandRetargeter docstring for why):\n"
                    "  pip install dex_retargeting\n"
                    "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
                ) from e
        teleop_kw = dict(hand_retargeter=hand_retargeter, debug=args.teleop_debug,
                         track_orientation=args.teleop_orientation)
        if args.teleop_scale is not None:
            teleop_kw["scale"] = args.teleop_scale   # else TeleopController's own default
        teleop_ctrl = TeleopController(m, **teleop_kw)
        fpv_streamer = (FpvStreamer(m, EGOCENTRIC_CAMERA)
                         if args.display_mode != "pass-through" else None)
        print(f"teleop: waiting for headset connection + first "
              f"{'hand-tracking' if args.hand_tracking else 'controller'} data...")
    elif args.pick != "none":
        side, site_name, arm_slice, hand_slice, brick_name = PICK_TARGETS[args.pick]
        seq = PickSequence(m, side, site_name, arm_slice, hand_slice, brick_name)
        seq.start(m, d)
        print(f"picking {brick_name} with the {side} arm...")

    # EpisodeRecorder runs in PARALLEL to whatever drives the arms above
    # (--teleop / --pick / --pick none) -- it only READS the live MjData each
    # step and never touches d.ctrl, so it records identically regardless of
    # the driver. See episode_recording.py.
    recorder = None
    episode_n = 1
    record_state = {"want_next": False, "awaiting_next": False}
    calib_state = {"waiting_printed": False}  # for the 'hold still to calibrate' hint
    if args.record_episodes is not None:
        if args.teleop:
            episode_goal = f"teleop ({'hand-tracking' if args.hand_tracking else 'controllers'})"
        elif args.pick != "none":
            episode_goal = f"pick {args.pick} with the {PICK_TARGETS[args.pick][0]} arm"
        else:
            episode_goal = "hold stance"
        recorder = EpisodeRecorder(m, task_dir=str(args.record_episodes),
                                    goal=episode_goal,
                                    extra_bodies=list(PICK_TARGETS))  # brick1/2/3 ground truth
        if not recorder.start_episode():
            raise SystemExit(f"could not start recording in {args.record_episodes}/ "
                              f"(EpisodeWriter reported busy at startup -- unexpected)")
        print(f"recording episode {episode_n} into {args.record_episodes}/ -- press 'n' in "
              f"the viewer window to save it and begin the next take")

    def key_callback(keycode):
        # GLFW reports letter keys as their uppercase ASCII code.
        # 'n' -> finalize the current episode and start a fresh one, so one
        #        session can capture many takes without restarting the sim.
        # 'c' -> drop teleop calibration and re-capture a fresh reference,
        #        so a bad initial capture is recoverable without a restart.
        if recorder is not None and keycode == ord("N"):
            record_state["want_next"] = True
        if teleop_ctrl is not None and keycode == ord("C"):
            teleop_ctrl.request_recalibration()
            calib_state["waiting_printed"] = False
            print("\nteleop: re-calibrating -- hold both controllers still for a moment")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        if args.view == "egocentric":
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = m.camera(EGOCENTRIC_CAMERA).id

        last_phase = None
        # try/finally so the in-progress episode is flushed to disk (and the
        # writer's non-daemon worker thread joined) on a viewer close, an
        # exception, or Ctrl-C -- without close() the process would hang on
        # exit waiting for that thread.
        try:
            while viewer.is_running():
                t0 = time.time()
                if teleop_ctrl is not None:
                    data = tele_source.get_tele_data()
                    if not teleop_ctrl.calibrated:
                        # Gated: try_calibrate captures the reference only once the
                        # controllers report a stable, non-stale pose for a moment
                        # (see TeleopController.try_calibrate -- calibrating on a
                        # stale first frame is the cause of the intermittent "arm
                        # won't follow" failure). Until then, freeze the arms where
                        # they are; legs/waist stay pinned to stance.
                        if teleop_ctrl.try_calibrate(m, d, data):
                            print("teleop: calibrated -- arms ramping to the ready pose, "
                                  "then following the controllers")
                            if teleop_ctrl.calib_warning:
                                print(f"teleop: !! {teleop_ctrl.calib_warning} -- hold a "
                                      f"relaxed, symmetric pose (hands in front of your chest, "
                                      f"~30cm apart) and press 'c' to re-calibrate")
                            calib_state["waiting_printed"] = False
                        else:
                            d.ctrl[LEG]   = hold[LEG]
                            d.ctrl[WAIST] = hold[WAIST]
                            if not calib_state["waiting_printed"]:
                                print("teleop: hold both controllers still to calibrate...")
                                calib_state["waiting_printed"] = True
                    else:
                        # TeleopController.step() lays down hold_ctrl for the WHOLE
                        # array first, same contract as PickSequence.step() -- see
                        # teleop_control.py
                        teleop_ctrl.step(m, d, hold, data)
                        if args.hand_tracking:
                            # last_grip is None per side in hand-tracking mode --
                            # there's no single scalar, real per-finger joints instead
                            print("  hand-tracking: retargeting live", end="\r")
                        else:
                            print(f"  grip L={teleop_ctrl.last_grip['left']:.2f} "
                                  f"R={teleop_ctrl.last_grip['right']:.2f}", end="\r")
                    if fpv_streamer is not None:
                        # streams regardless of calibration state -- the operator
                        # should be able to see the robot's view even before their
                        # controllers are tracked; internally rate-limited, most
                        # calls are a no-op (see FpvStreamer)
                        fpv_streamer.step(d, tele_source)
                elif seq is None:
                    # --pick none: old behavior, just hold the stance every step
                    d.ctrl[LEG]   = hold[LEG]
                    d.ctrl[WAIST] = hold[WAIST]
                    d.ctrl[UPPER_BODY] = hold[UPPER_BODY]
                else:
                    # PickSequence.step() lays down hold_ctrl for the WHOLE array
                    # first, then overwrites only its own arm_slice/hand_slice --
                    # so LEG/WAIST and the other arm/hand stay pinned to stance
                    # without a separate write here.
                    phase = seq.step(m, d, hold)
                    if phase != last_phase:
                        print(f"  phase: {phase}")
                        last_phase = phase
                    if phase == "HOLD" and seq.confidence is not None:
                        print(f"    grasped={seq.confidence['grasped']}  "
                              f"confidence={seq.confidence['confidence']:.2f}", end="\r")
                mujoco.mj_step(m, d)

                if recorder is not None:
                    # 'n' key -> save this take and roll to the next. create_episode()
                    # is async (returns False while the previous save drains), so retry
                    # each frame until it takes.
                    if record_state["want_next"]:
                        recorder.end_episode()
                        record_state["want_next"] = False
                        record_state["awaiting_next"] = True
                    if record_state["awaiting_next"] and recorder.start_episode():
                        record_state["awaiting_next"] = False
                        episode_n += 1
                        print(f"\nrecording episode {episode_n} into {args.record_episodes}/")
                    # one sample per control step; internally rate-limited to the
                    # writer's fps, so most calls are a no-op (see EpisodeRecorder)
                    recorder.step(m, d)

                viewer.sync()
                dt = m.opt.timestep - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        finally:
            if recorder is not None:
                recorder.end_episode()
                recorder.close()
                print(f"\nepisode recording(s) saved under {args.record_episodes}/")

if __name__ == "__main__":
    # televuer spawns its own server as a child process. Python's multiprocessing
    # DEFAULT start method on macOS is "spawn" (full pickling of everything handed
    # to the child) -- Linux defaults to "fork" (copies process memory directly, no
    # pickling needed), which is almost certainly what televuer was only ever
    # tested against. On macOS this crashes with "cannot pickle '_thread.lock'
    # object" the moment --teleop constructs TeleVuerWrapper. Force "fork" here,
    # before any Process gets created, to match what televuer actually needs --
    # untested beyond fixing this specific crash; if --teleop still misbehaves
    # after this, that's a new, separate issue, not this one.
    import multiprocessing
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass  # already set (e.g. re-imported) -- fine, don't fail on it
    main()
