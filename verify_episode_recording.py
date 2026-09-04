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

    multi_episode()


def multi_episode():
    """Exercises the 'n'-key episode-segmentation path added to
    stand_next_to_table.py's loop: end_episode() then retry start_episode()
    each frame until the async previous-save drains, keep recording, and end
    up with two independent, well-formed episode dirs from one sim run."""
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    hold = m.key_ctrl[key_id].copy()
    d.ctrl[:] = hold
    mujoco.mj_forward(m, d)

    task_dir = tempfile.mkdtemp(prefix="verify_episode_recording_multi_")
    try:
        rec = EpisodeRecorder(m, task_dir=task_dir, fps=10, goal="multi take",
                               extra_bodies=["brick1", "brick2", "brick3"])
        assert rec.start_episode()

        seq = PickSequence(m, "right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick1")
        seq.start(m, d)
        dt = m.opt.timestep
        counts = []
        n = 0
        want_next, awaiting_next = False, False
        for i in range(int(6.0 / dt)):
            seq.step(m, d, hold)
            mujoco.mj_step(m, d)
            # mirror stand_next_to_table.py's loop exactly
            if want_next:
                rec.end_episode()
                want_next, awaiting_next = False, True
            if awaiting_next and rec.start_episode():
                awaiting_next = False
                n += 1
            rec.step(m, d)
            if i == int(3.0 / dt):   # request the cut halfway through
                want_next = True
        rec.end_episode()
        rec.close()

        episode_dirs = sorted(pathlib.Path(task_dir).glob("episode_*"))
        assert len(episode_dirs) == 2, f"expected 2 episode dirs, found {episode_dirs}"
        for ep in episode_dirs:
            with open(ep / "data.json") as f:
                episode = json.load(f)          # parses => header was closed correctly
            assert len(episode["data"]) > 3, f"{ep.name} has too few samples"
            img_rel = episode["data"][0]["colors"]["fpv_teleop"]
            assert (ep / img_rel).exists(), f"{ep.name}: first image missing"
        print(f"multi-episode: 2 dirs, "
              f"{[len(json.load(open(ep / 'data.json'))['data']) for ep in episode_dirs]} samples")
        print("multi-episode: PASS")
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
