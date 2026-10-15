"""
Headless verification for episode_recording.py / episode_writer.py: drives
PickSequence (already-verified, deterministic -- see verify_pick_sequence.py)
one .step() per mj_step() while EpisodeRecorder samples alongside it, exactly
matching the "runs in parallel" usage pattern this was built for. Confirms
the written dataset is well-formed: data.json parses, has the right number
of per-step records with the right shapes, images referenced actually exist
and load back with the right size, and sim_state's ground-truth brick
position matches the live simulation at save time.

Does NOT verify anything about actual downstream use of a recorded dataset
(training a policy, etc.) -- out of scope for this repo.

Run: python verify_episode_recording.py
"""
import json, pathlib, os, shutil, tempfile
import numpy as np
import mujoco
import cv2

from actuator_groups import RIGHT_ARM, RIGHT_HAND
from pick_sequence import PickSequence
from episode_recording import EpisodeRecorder

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR",
    pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
SCENE = MODEL_DIR / "scene_fixed_table.xml"


def main():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    task_dir = tempfile.mkdtemp(prefix="verify_episode_recording_")
    try:
        rec = EpisodeRecorder(m, task_dir=task_dir, fps=10,  # low fps -- keeps the test fast
                               goal="verify recording",
                               extra_bodies=["brick1", "brick2", "brick3"])
        ok = rec.start_episode()
        assert ok, "start_episode() failed"

        seq = PickSequence(m, "right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick1")
        seq.start(m, d)

        dt = m.opt.timestep
        recorded_steps = 0
        for i in range(int(9.0 / dt)):
            phase = seq.step(m, d, hold)
            mujoco.mj_step(m, d)
            if rec.step(m, d):
                recorded_steps += 1
            if phase == "HOLD":
                break

        expected_brick1 = d.xpos[m.body("brick1").id].copy()

        rec.end_episode()
        rec.close()  # blocks until the async save actually finishes

        print(f"recorded_steps (samples taken)={recorded_steps}")
        assert recorded_steps > 5, "too few samples -- rate limiting or wiring is broken"

        # NOTE: episode numbering starts at 0001, not 0000, here -- EpisodeWriter
        # (matching upstream xr_teleoperate exactly, not a bug introduced here)
        # only starts numbering at 0 if task_dir did NOT already exist at
        # construction time; tempfile.mkdtemp() always pre-creates it, so glob
        # for the actual episode dir instead of assuming its number.
        episode_dirs = sorted(pathlib.Path(task_dir).glob("episode_*"))
        assert len(episode_dirs) == 1, f"expected exactly one episode dir, found {episode_dirs}"
        episode_dir = episode_dirs[0]
        json_path = episode_dir / "data.json"
        assert json_path.exists(), f"data.json missing at {json_path}"

        with open(json_path) as f:
            episode = json.load(f)

        assert "info" in episode and "text" in episode and "data" in episode
        assert episode["text"]["goal"] == "verify recording", \
            f"task goal not written through: {episode['text']['goal']!r}"
        assert len(episode["data"]) == recorded_steps, \
            f"data.json has {len(episode['data'])} records, expected {recorded_steps}"

        first = episode["data"][0]
        last = episode["data"][-1]
        for rec_item in (first, last):
            assert set(rec_item["states"].keys()) == {"left_arm", "left_ee", "right_arm", "right_ee"}
            assert len(rec_item["states"]["right_arm"]) == 7, "right_arm state should be 7 joints"
            assert len(rec_item["actions"]["right_arm"]) == 7, "right_arm action should be 7 joints"
            assert set(rec_item["sim_state"].keys()) == {"brick1", "brick2", "brick3"}
            assert len(rec_item["sim_state"]["brick1"]) == 3, "brick1 sim_state should be xyz"

        # confirm sim_state's LAST recorded brick1 position is close to where
        # brick1 actually was at save time (ground truth, not stale/wrong)
        last_brick1 = np.array(last["sim_state"]["brick1"])
        drift = np.linalg.norm(last_brick1 - expected_brick1)
        print(f"last recorded brick1 sim_state vs actual at save time: {drift*100:.3f}cm apart")
        assert drift < 0.15, "sim_state brick1 position doesn't match live sim -- logging bug"

        # confirm the actual image file exists and loads with the right size
        img_rel_path = first["colors"]["fpv_teleop"]
        img_path = episode_dir / img_rel_path
        assert img_path.exists(), f"recorded image missing: {img_path}"
        img = cv2.imread(str(img_path))
        assert img is not None, "recorded image failed to load"
        assert img.shape == (480, 640, 3), f"unexpected image shape: {img.shape}"
        print(f"first recorded image: {img_path.name}, shape={img.shape}, "
              f"mean_pixel={img.mean():.1f} (0=black, confirms non-blank)")
        assert img.mean() > 5, "recorded image looks blank/black"

        print("\nsingle-episode: PASS")
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)

    toggle_segmentation()


def toggle_segmentation():
    """Exercises the 's'-toggle episode path in stand_next_to_table.py's loop:
    nothing records until the first 's'; 's' starts an episode, 's' again
    finalizes+saves it (retrying start_episode() each frame while the async
    previous-save drains), repeat. Ends up with two independent, well-formed
    episode dirs from one sim run, and NO samples written while stopped."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    task_dir = tempfile.mkdtemp(prefix="verify_episode_recording_toggle_")
    try:
        rec = EpisodeRecorder(m, task_dir=task_dir, fps=10, goal="toggle take",
                               extra_bodies=["brick1", "brick2", "brick3"])
        # NOT started here -- the first 's' does it, matching the new wiring.
        assert not rec.is_recording

        seq = PickSequence(m, "right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick1")
        seq.start(m, d)
        dt = m.opt.timestep
        state = {"toggle": False, "awaiting_start": False}
        episode_n = 0
        # 's' presses scheduled (frame -> registered, handled next frame, like
        # the async key_callback): start, stop, start, stop.
        presses = {int(0.3 / dt), int(2.5 / dt), int(3.0 / dt), int(5.4 / dt)}
        samples_while_stopped = 0
        for i in range(int(6.0 / dt)):
            seq.step(m, d, hold)
            mujoco.mj_step(m, d)
            # --- mirror stand_next_to_table.py's loop exactly ---
            if state["toggle"]:
                state["toggle"] = False
                if rec.is_recording:
                    rec.end_episode()
                elif state["awaiting_start"]:
                    state["awaiting_start"] = False
                else:
                    state["awaiting_start"] = True
            if state["awaiting_start"] and rec.start_episode():
                state["awaiting_start"] = False
                episode_n += 1
            sampled = rec.step(m, d)
            # ---------------------------------------------------
            if not rec.is_recording and sampled:
                samples_while_stopped += 1
            if i in presses:
                state["toggle"] = True
        rec.end_episode()
        rec.close()

        assert samples_while_stopped == 0, "recorded frames while stopped between/before takes"
        assert episode_n == 2, f"expected 2 episodes started via toggle, got {episode_n}"

        episode_dirs = sorted(pathlib.Path(task_dir).glob("episode_*"))
        assert len(episode_dirs) == 2, f"expected 2 episode dirs, found {episode_dirs}"
        for ep in episode_dirs:
            with open(ep / "data.json") as f:
                episode = json.load(f)          # parses => header was closed correctly
            assert len(episode["data"]) > 3, f"{ep.name} has too few samples"
            img_rel = episode["data"][0]["colors"]["fpv_teleop"]
            assert (ep / img_rel).exists(), f"{ep.name}: first image missing"
        print(f"toggle-segmentation: 2 dirs, "
              f"{[len(json.load(open(ep / 'data.json'))['data']) for ep in episode_dirs]} samples, "
              f"0 samples while stopped")
        print("toggle-segmentation: PASS")
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
