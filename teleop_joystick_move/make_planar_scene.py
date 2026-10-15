"""Generates scene_planar_table.xml from scene_fixed_table.xml: swaps the
g1_fixed_upper.xml include for g1_planar_base.xml (make_planar_base.py) and
extends the 'stand_at_table' keyframe's qpos to match the new nq (134 -> 137
-- the table scene's existing 13-brick pool qpos is untouched, only the
robot's leading block grows by the 3 new base dof, same "0 0 0" prepended by
make_planar_base.py for the robot-only 'stand' keyframe). ctrl is untouched
(nu stays 43 -- no base actuators).

Everything else (floor, table, bricks, snap welds, brick pool) is copied
verbatim -- this is the same "reuse the existing scene's asset/body XML,
swap only what the base-mode change actually touches" pattern
inspire_hand/make_scene_inspire.py already uses in this repo.

Usage: python make_planar_scene.py [scene_fixed_table.xml]   (run from
teleop_joystick_move/; writes scene_planar_table.xml next to it. Copy both
this output AND g1_planar_base.xml into MODEL_DIR for --base-mode planar to
actually load, same two-location-split convention as every other generated
scene file in this repo.)
"""
import re
import sys
import pathlib

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "scene_fixed_table.xml")
DST = pathlib.Path("scene_planar_table.xml")
src = SRC.read_text()

INC_RE = re.compile(r'<include file="g1_fixed_upper\.xml"/>')
assert INC_RE.search(src), "g1_fixed_upper.xml include not found"
src = INC_RE.sub('<include file="g1_planar_base.xml"/>', src, count=1)

KEY_RE = re.compile(r'(<key\s+name="stand_at_table"\s+qpos=")([^"]+)("\s+ctrl=)')
km = KEY_RE.search(src)
assert km, "'stand_at_table' keyframe not found"
src = src[:km.start()] + km.group(1) + "0 0 0 " + km.group(2) + km.group(3) + src[km.end():]

DST.write_text(src)
print(f"wrote {DST}  (include -> g1_planar_base.xml, stand_at_table qpos "
      f"extended by 3 leading base-dof zeros)")
