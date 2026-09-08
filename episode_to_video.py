"""
Stitch a recorded episode's color frames into an .mp4.

EpisodeRecorder (episode_recording.py) writes each take as
    <task_dir>/episode_XXXX/colors/NNNNNN_fpv_teleop.jpg   (+ data.json, depths/, audios/)
at data.json's info.image.fps (30 by default). This turns one such colors/ dir back
into a video, written to  <task_dir>/video/episode_XXXX.mp4 .

Usage:
    python episode_to_video.py recordings/teleop_demo1/episode_0001/colors
    python episode_to_video.py recordings/teleop_demo1/episode_0001        # episode dir also OK
    python episode_to_video.py recordings/teleop_demo1                     # task dir -> every episode
    python episode_to_video.py .../episode_0001/colors --fps 20 --output /tmp/ep1.mp4

fps is taken from the episode's data.json (info.image.fps) unless --fps overrides it;
falls back to 30 if neither is available. The on-disk jpgs are already correct-colour
(EpisodeWriter did the RGB->BGR swap before cv2.imwrite), so no channel juggling here.
"""
import argparse
import json
import pathlib
import sys

import imageio.v2 as imageio

DEFAULT_FPS = 30
FRAME_GLOB = "*.jpg"


def _frame_index(p: pathlib.Path) -> int:
    """Sort key: leading zero-padded integer of 'NNNNNN_fpv_teleop.jpg'."""
    stem = p.name.split("_", 1)[0]
    return int(stem) if stem.isdigit() else -1


def _resolve_targets(path: pathlib.Path):
    """Yield (colors_dir, episode_dir) pairs for whatever `path` points at:
    a colors/ dir, an episode dir, or a task dir (every episode under it)."""
    if not path.exists():
        sys.exit(f"error: path does not exist: {path}")

    # a colors/ dir directly
    if any(path.glob(FRAME_GLOB)):
        yield path, path.parent
        return

    # an episode dir (has colors/)
    if (path / "colors").is_dir():
        yield path / "colors", path
        return

    # a task dir (has episode_*/colors/)
    episodes = sorted(p for p in path.glob("episode_*") if (p / "colors").is_dir())
    if episodes:
        for ep in episodes:
            yield ep / "colors", ep
        return

    sys.exit(f"error: found no frames, no colors/ dir, and no episode_*/ under: {path}")


def _fps_for(episode_dir: pathlib.Path, override: int | None) -> int:
    if override is not None:
        return override
    data_json = episode_dir / "data.json"
    try:
        info = json.loads(data_json.read_text())["info"]["image"]
        fps = int(info.get("fps") or 0)
        if fps > 0:
            return fps
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return DEFAULT_FPS


def make_video(colors_dir: pathlib.Path, episode_dir: pathlib.Path,
               fps_override: int | None, out_override: pathlib.Path | None) -> pathlib.Path:
    frames = sorted(colors_dir.glob(FRAME_GLOB), key=_frame_index)
    if not frames:
        sys.exit(f"error: no {FRAME_GLOB} frames in {colors_dir}")

    fps = _fps_for(episode_dir, fps_override)

    if out_override is not None:
        out_path = out_override
    else:
        # <task_dir>/video/<episode_name>.mp4   (task_dir = episode_dir's parent)
        out_path = episode_dir.parent / "video" / f"{episode_dir.name}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(out_path, fps=fps, macro_block_size=None) as writer:
        for f in frames:
            writer.append_data(imageio.imread(f))

    dur = len(frames) / fps
    print(f"{colors_dir}  ->  {out_path}")
    print(f"  {len(frames)} frames @ {fps} fps  ({dur:.1f}s)")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Stitch a recorded episode's colors/ frames into an .mp4 under "
                    "<task_dir>/video/.")
    ap.add_argument("path", type=pathlib.Path,
                    help="an episode's colors/ dir, an episode dir, or a whole task dir "
                         "(recordings/teleop_demoX) to convert every episode under it")
    ap.add_argument("--fps", type=int, default=None,
                    help="frames per second (default: read from the episode's data.json "
                         f"info.image.fps, else {DEFAULT_FPS})")
    ap.add_argument("-o", "--output", type=pathlib.Path, default=None,
                    help="explicit output .mp4 path (single episode only; overrides the "
                         "default <task_dir>/video/<episode>.mp4)")
    args = ap.parse_args()

    targets = list(_resolve_targets(args.path.resolve()))
    if args.output is not None and len(targets) > 1:
        sys.exit("error: --output can't be used when converting multiple episodes")

    for colors_dir, episode_dir in targets:
        make_video(colors_dir, episode_dir, args.fps, args.output)


if __name__ == "__main__":
    main()
