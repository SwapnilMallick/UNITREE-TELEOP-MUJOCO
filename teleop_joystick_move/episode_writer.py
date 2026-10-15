"""
EpisodeWriter: vendored, adapted from unitreerobotics/xr_teleoperate's
teleop/utils/episode_writer.py. Writes per-episode teleop recordings to disk:
one directory per episode (colors/, depths/, audios/, data.json), with
data.json holding an "info" header, a "text" (task goal/description) block,
and a "data" array of per-timestep records (idx, colors, depths, states,
actions, tactiles, audios, sim_state).

The ONLY real change from the original: the `rerun_log` option (live 3D
debug visualization via the `rerun` SDK) now lazy-imports `rerun_visualizer`
inside __init__ instead of unconditionally at module level. The original
imports it at the top of the file regardless of whether rerun_log is used,
meaning `rerun` (a heavy, separate Rust-backed visualization package) would
be a hard import-time dependency for anyone using this module at all, even
with rerun_log=False. Confirmed by reading rerun_visualizer.py directly:
its only real dependency is `import rerun as rr`. Matches this repo's
existing pattern of not forcing a heavy optional dependency (televuer,
dex_retargeting) at plain import time. Default here is rerun_log=False --
nothing in this repo's own integration (episode_recording.py) uses it.
Everything else (the on-disk format, add_item's fields, the background
writer thread) is unchanged from upstream.
"""
import os
import cv2
import json
import datetime
import numpy as np
import time
from queue import Queue, Empty
from threading import Thread
import logging
logger_mp = logging.getLogger(__name__)


class EpisodeWriter():
    def __init__(self, task_dir, task_goal=None, task_desc=None, task_steps=None,
                 frequency=30, image_size=[640, 480], rerun_log=False):
        """
        image_size: [width, height]
        """
        logger_mp.info("==> EpisodeWriter initializing...\n")
        self.task_dir = task_dir
        self.text = {
            "goal": "Pick up the red cup on the table.",
            "desc": "task description",
            "steps": "step1: do this; step2: do that; ...",
        }
        if task_goal is not None:
            self.text['goal'] = task_goal
        if task_desc is not None:
            self.text['desc'] = task_desc
        if task_steps is not None:
            self.text['steps'] = task_steps

        self.frequency = frequency
        self.image_size = image_size

        self.rerun_log = rerun_log
        if self.rerun_log:
            # lazy import -- see module docstring for why this isn't at the top
            from rerun_visualizer import RerunLogger
            logger_mp.info("==> RerunLogger initializing...\n")
            self.rerun_logger = RerunLogger(prefix="online/", IdxRangeBoundary=60, memory_limit="300MB")
            logger_mp.info("==> RerunLogger initializing ok.\n")

        self.item_id = -1
        self.episode_id = -1
        if os.path.exists(self.task_dir):
            episode_dirs = [episode_dir for episode_dir in os.listdir(self.task_dir)
                             if 'episode_' in episode_dir and not episode_dir.endswith('.zip')]
            episode_last = sorted(episode_dirs)[-1] if len(episode_dirs) > 0 else None
            self.episode_id = 0 if episode_last is None else int(episode_last.split('_')[-1])
            logger_mp.info(f"==> task_dir directory already exist, now self.episode_id is:{self.episode_id}\n")
        else:
            os.makedirs(self.task_dir)
            logger_mp.info(f"==> episode directory does not exist, now create one.\n")
        self.data_info()

        self.is_available = True
        self.item_data_queue = Queue(-1)
        self.stop_worker = False
        self.need_save = False
        self.worker_thread = Thread(target=self.process_queue)
        self.worker_thread.start()

        logger_mp.info("==> EpisodeWriter initialized successfully.\n")

    def is_ready(self):
        return self.is_available

    def data_info(self, version='1.0.0', date=None, author=None):
        self.info = {
            "version": "1.0.0" if version is None else version,
            "date": datetime.date.today().strftime('%Y-%m-%d') if date is None else date,
            "author": "unitree" if author is None else author,
            "image": {"width": self.image_size[0], "height": self.image_size[1], "fps": self.frequency},
            "depth": {"width": self.image_size[0], "height": self.image_size[1], "fps": self.frequency},
            "audio": {"sample_rate": 16000, "channels": 1, "format": "PCM", "bits": 16},
            "joint_names": {
                "left_arm": [],
                "left_ee": [],
                "right_arm": [],
                "right_ee": [],
                "body": [],
            },
            "tactile_names": {
                "left_ee": [],
                "right_ee": [],
            },
            "sim_state": ""
        }

    def create_episode(self):
        """Create a new episode. Returns True on success, False if the writer
        is still busy saving a previous episode."""
        if not self.is_available:
            logger_mp.info("==> The class is currently unavailable for new operations. Please wait until ongoing tasks are completed.")
            return False

        self.item_id = -1
        self.episode_id = self.episode_id + 1

        self.episode_dir = os.path.join(self.task_dir, f"episode_{str(self.episode_id).zfill(4)}")
        self.color_dir = os.path.join(self.episode_dir, 'colors')
        self.depth_dir = os.path.join(self.episode_dir, 'depths')
        self.audio_dir = os.path.join(self.episode_dir, 'audios')
        self.json_path = os.path.join(self.episode_dir, 'data.json')
        os.makedirs(self.episode_dir, exist_ok=True)
        os.makedirs(self.color_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        with open(self.json_path, "w", encoding="utf-8") as f:
            f.write('{\n')
            f.write('"info": ' + json.dumps(self.info, ensure_ascii=False, indent=4) + ',\n')
            f.write('"text": ' + json.dumps(self.text, ensure_ascii=False, indent=4) + ',\n')
            f.write('"data": [\n')
        self.first_item = True

        if self.rerun_log:
            from rerun_visualizer import RerunLogger
            self.online_logger = RerunLogger(prefix="online/", IdxRangeBoundary=60, memory_limit="300MB")

        self.is_available = False
        logger_mp.info(f"==> New episode created: {self.episode_dir}")
        return True

    def add_item(self, colors, depths=None, states=None, actions=None, tactiles=None, audios=None, sim_state=None):
        self.item_id += 1
        item_data = {
            'idx': self.item_id,
            'colors': colors,
            'depths': depths,
            'states': states,
            'actions': actions,
            'tactiles': tactiles,
            'audios': audios,
            'sim_state': sim_state,
        }
        self.item_data_queue.put(item_data)

    def process_queue(self):
        while not self.stop_worker or not self.item_data_queue.empty():
            try:
                item_data = self.item_data_queue.get(timeout=1)
                try:
                    self._process_item_data(item_data)
                except Exception as e:
                    logger_mp.info(f"Error processing item_data (idx={item_data['idx']}): {e}")
                self.item_data_queue.task_done()
            except Empty:
                pass

            if self.need_save and self.item_data_queue.empty():
                self._save_episode()

    def _process_item_data(self, item_data):
        idx = item_data['idx']
        colors = item_data.get('colors', {})
        depths = item_data.get('depths', {})
        audios = item_data.get('audios', {})

        if colors:
            for idx_color, (color_key, color) in enumerate(colors.items()):
                color_name = f'{str(idx).zfill(6)}_{color_key}.jpg'
                if not cv2.imwrite(os.path.join(self.color_dir, color_name), color):
                    logger_mp.info(f"Failed to save color image.")
                item_data['colors'][color_key] = os.path.join('colors', color_name)

        if depths:
            for idx_depth, (depth_key, depth) in enumerate(depths.items()):
                depth_name = f'{str(idx).zfill(6)}_{depth_key}.jpg'
                if not cv2.imwrite(os.path.join(self.depth_dir, depth_name), depth):
                    logger_mp.info(f"Failed to save depth image.")
                item_data['depths'][depth_key] = os.path.join('depths', depth_name)

        if audios:
            for mic, audio in audios.items():
                audio_name = f'audio_{str(idx).zfill(6)}_{mic}.npy'
                np.save(os.path.join(self.audio_dir, audio_name), audio.astype(np.int16))
                item_data['audios'][mic] = os.path.join('audios', audio_name)

        with open(self.json_path, "a", encoding="utf-8") as f:
            if not self.first_item:
                f.write(",\n")
            f.write(json.dumps(item_data, ensure_ascii=False, indent=4))
            self.first_item = False

        if self.rerun_log:
            curent_record_time = time.time()
            logger_mp.info(f"==> episode_id:{self.episode_id}  item_id:{idx}  current_time:{curent_record_time}")
            self.rerun_logger.log_item_data(item_data)

    def save_episode(self):
        """Trigger the save. The background worker thread handles it once the
        queue drains -- this returns immediately, doesn't block."""
        self.need_save = True
        logger_mp.info(f"==> Episode saved start...")

    def _save_episode(self):
        with open(self.json_path, "a", encoding="utf-8") as f:
            f.write("\n]\n}")
        self.need_save = False
        self.is_available = True
        logger_mp.info(f"==> Episode saved successfully to {self.json_path}.")

    def close(self):
        """Stop the worker thread. Blocks until any pending save completes."""
        self.item_data_queue.join()
        if not self.is_available:
            self.save_episode()
        while not self.is_available:
            time.sleep(0.01)
        self.stop_worker = True
        self.worker_thread.join()
