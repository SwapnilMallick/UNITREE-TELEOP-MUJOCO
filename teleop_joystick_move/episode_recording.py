"""
EpisodeRecorder: samples the fpv_teleop camera + arm/hand joint states and
actions together at a controlled rate, and writes them via the vendored
EpisodeWriter (episode_writer.py, adapted from xr_teleoperate's own data
collection module) into its per-episode data.json + colors/ format.

Deliberately decoupled from TeleopController -- runs in PARALLEL to whatever
is actually driving the arms, not wired into it. It only reads `d` (the live
MjData) each step, so it works identically whether the arm is being driven
by PickSequence's scripted phases, StackSequence, or TeleopController's live
VR input -- matches the same duck-typing/decoupling philosophy already used
for FpvStreamer and HandRetargeter (this repo consistently keeps "what
drives the arm" separate from "what observes/records it").

Rate-limited like FpvStreamer (not every physics step -- offscreen rendering
is comparatively expensive, and a recorded dataset doesn't need physics-rate
samples; EpisodeWriter's own `frequency` param, default 30fps, is what this
matches against).

"states" vs "actions": states are the arm/hand's actual CURRENT qpos
(d.qpos); actions are the CURRENT COMMANDED ctrl target (d.ctrl) -- these
differ on a position-controlled robot (the actuator hasn't necessarily
reached the commanded target yet), and recording both is the standard
convention for imitation-learning-style datasets. sim_state (optional) logs
real ground-truth body positions (e.g. brick1/brick2/brick3) this repo
already has privileged access to in simulation -- useful for later
training/debugging, even though a real robot wouldn't have this signal.

UNVERIFIED beyond "writes a well-formed dataset headlessly" -- confirmed
that a recorded episode's data.json parses, has the right per-step shape,
and its images load back correctly (verify_episode_recording.py). NOT
verified against any actual downstream use (training a policy, etc.) --
that's out of scope for this repo.
"""
import numpy as np
import mujoco

from episode_writer import EpisodeWriter
from actuator_groups import LEFT_ARM, LEFT_HAND, RIGHT_ARM, RIGHT_HAND


class EpisodeRecorder:
    """
    Usage:
        rec = EpisodeRecorder(m, task_dir="recordings/pick_brick1",
                               goal="Pick up brick1",
                               extra_bodies=["brick1", "brick2", "brick3"])
        rec.start_episode()
        while running:
            <drive the arm however -- PickSequence.step(), TeleopController.step(), ...>
            mujoco.mj_step(m, d)
            rec.step(m, d)        # once per control step; internally rate-limited
        rec.end_episode()
        rec.close()               # call once, at real shutdown -- flushes the writer
    """

    def __init__(self, m, task_dir, camera_name="fpv_teleop", fps=30,
                 height=480, width=640, extra_bodies=None,
                 goal=None, desc=None, steps=None):
        # goal/desc/steps go into every episode's data.json "text" block. They're
        # set once here, not per-episode: upstream xr_teleoperate's create_episode()
        # doesn't take them either (it's construction-time there too), and in
        # practice one recording session captures many takes of the SAME task.
        self.writer = EpisodeWriter(task_dir=task_dir, frequency=fps,
                                     image_size=[width, height], rerun_log=False,
                                     task_goal=goal, task_desc=desc, task_steps=steps)
        self.renderer = mujoco.Renderer(m, height=height, width=width)
        self.camera_name = camera_name
        self.frame_every = max(int(round((1.0 / fps) / m.opt.timestep)), 1)
        self.extra_bodies = extra_bodies or []
        self._step_count = 0
        self._recording = False

    def start_episode(self):
        """Begin a new episode directory. The task text (goal/desc/steps) is
        fixed at construction time, not per-episode -- see __init__. Returns
        False if the writer is still busy saving a previous episode (matches
        EpisodeWriter.create_episode's contract), True otherwise."""
        ok = self.writer.create_episode()
        self._recording = ok
        self._step_count = 0
        return ok

    @property
    def is_recording(self):
        """True between a successful start_episode() and its end_episode() --
        lets a caller drive a start/stop toggle without poking internals."""
        return self._recording

    def step(self, m, d, base_cmd=None, base_qpos_slice=None, base_dof_slice=None):
        """Call once per control step. Internally rate-limited to ~fps; most
        calls are a no-op by design. Returns True on steps a sample was
        actually recorded. No-ops entirely (returns False) if start_episode()
        hasn't been called or the previous episode hasn't ended yet.

        base_cmd (optional): the smoothed (vx, vy, yaw_rate) locomotion
        command (locomotion.BaseVelocityCommand), recorded as actions["base"]
        -- --base-mode planar/policy only; omitted (default None) for
        --base-mode fixed, so old recordings' actions dict shape is
        unaffected (no "base" key at all) and loaders that don't know about
        it are unaffected (EpisodeWriter stores these as opaque dicts, no
        schema). base_qpos_slice/base_dof_slice (optional, qpos/qvel-space
        slices for the 3 base joints): when given, records states["base"] =
        [x, y, yaw] and states["base_vel"] = [vx, vy, yaw_rate] (live sim
        ground truth, not the commanded values actions["base"] already has)."""
        if not self._recording:
            return False
        self._step_count += 1
        if self._step_count % self.frame_every != 0:
            return False

        self.renderer.update_scene(d, camera=self.camera_name)
        frame = self.renderer.render()
        # cv2.imwrite (used inside EpisodeWriter) expects BGR; MuJoCo renders
        # RGB -- same color-channel lesson already confirmed via FpvStreamer.
        frame_bgr = np.ascontiguousarray(frame[..., ::-1])

        states = {
            "left_arm": d.qpos[LEFT_ARM].tolist(),
            "left_ee": d.qpos[LEFT_HAND].tolist(),
            "right_arm": d.qpos[RIGHT_ARM].tolist(),
            "right_ee": d.qpos[RIGHT_HAND].tolist(),
        }
        actions = {
            "left_arm": d.ctrl[LEFT_ARM].tolist(),
            "left_ee": d.ctrl[LEFT_HAND].tolist(),
            "right_arm": d.ctrl[RIGHT_ARM].tolist(),
            "right_ee": d.ctrl[RIGHT_HAND].tolist(),
        }
        if base_cmd is not None:
            actions["base"] = [float(base_cmd[0]), float(base_cmd[1]), float(base_cmd[2])]
        if base_qpos_slice is not None:
            states["base"] = d.qpos[base_qpos_slice].tolist()
            if base_dof_slice is not None:
                states["base_vel"] = d.qvel[base_dof_slice].tolist()
        sim_state = None
        if self.extra_bodies:
            sim_state = {name: d.xpos[m.body(name).id].tolist() for name in self.extra_bodies}

        self.writer.add_item(colors={self.camera_name: frame_bgr},
                              states=states, actions=actions, sim_state=sim_state)
        return True

    def end_episode(self):
        """Triggers the (async, background-thread) save of the current
        episode. Safe to call start_episode() again right after -- it'll
        return False until the previous save actually finishes."""
        if self._recording:
            self.writer.save_episode()
        self._recording = False

    def close(self):
        """Call once, at real shutdown. Blocks until any pending save
        completes, then stops the writer's background thread."""
        self.writer.close()
