"""Generates g1_planar_base.xml from g1_fixed_upper.xml: adds a 3-DOF planar
base (slide x, slide y, hinge yaw) directly on the pelvis body, replacing the
fully-welded base used everywhere else in this repo. For base_mode="planar"
only (locomotion.py) -- base_mode="policy" restores a real freejoint instead
(separate generator, not this one) and base_mode="fixed" keeps using the
untouched g1_fixed_upper.xml.

Why 3 explicit joints and not a freejoint: a freejoint gives 6 DOF (it can
also pitch/roll/fall over) and costs 7 qpos / 6 dof per the usual quaternion
mismatch. This scene never needs out-of-plane motion for a legs-locked
"slide around" base (the real balancing problem is Stage 2's, via the
pretrained RL policy) -- 3 single-DOF hinge/slide joints are simpler (1
qpos == 1 dof each, no quaternion), physically prevent the robot from ever
tipping over, and are driven directly via d.qvel injection in locomotion.py
(no actuators added -- ctrl stays all 43 original entries untouched, so
hold_ctrl / TeleopController's `d.ctrl[:] = hold_ctrl` pattern needs no
changes at all for this mode).

Index shift (unavoidable): the pelvis is the root body, so any joint placed
on it occupies the FIRST qpos/dof slots, ahead of every existing joint --
confirmed by reading how MuJoCo assigns qpos by body-tree traversal order,
not XML element order. qpos layout becomes:
    [0:3]  base_slide_x, base_slide_y, base_yaw   (new)
    [3:46] the original 43, each index shifted +3
See actuator_groups_planar.py for the shifted slice constants (same pattern
already used by inspire_hand/actuator_groups_inspire.py for a similar
qpos-layout change in this codebase). nu (actuator count) is UNCHANGED at 43
-- no actuators were added for the base.

Usage: python make_planar_base.py [g1_fixed_upper.xml]   (run from
teleop_joystick_move/; writes g1_planar_base.xml next to it)
"""
import re
import sys
import pathlib

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "g1_fixed_upper.xml")
DST = pathlib.Path("g1_planar_base.xml")
src = SRC.read_text()

# Insert the 3 base joints right after the pelvis's own imu site, before its
# first child body opens -- anchored on the imu_in_pelvis site tag (unique,
# present in every known source variant) rather than a line number.
SITE_RE = re.compile(r'(<site name="imu_in_pelvis"[^>]*/>\n)')
m = SITE_RE.search(src)
assert m, "imu_in_pelvis site not found (g1_fixed_upper.xml format changed?)"

# No damping/stiffness/frictionloss (all default to 0): locomotion.py drives
# these purely via qvel injection every frame, and anything that opposes
# qvel would fight that command instead of just integrating it. range="-6 6"
# on x/y is a generous safety backstop (the table/brick area is small; this
# just keeps a runaway command or bug from sliding the robot into empty
# space indefinitely) -- NOT the real travel limit, which belongs in
# locomotion.py's command clamping where it can be tuned without
# recompiling the model. yaw is unlimited (repeated spins are physically
# fine for a hinge; only sin/cos of it is ever used downstream).
BASE_JOINTS = (
    '      <joint name="base_slide_x" type="slide" axis="1 0 0" '
    'range="-6 6" limited="true"/>\n'
    '      <joint name="base_slide_y" type="slide" axis="0 1 0" '
    'range="-6 6" limited="true"/>\n'
    '      <joint name="base_yaw" type="hinge" axis="0 0 1" limited="false"/>\n'
)
src = src[:m.end()] + BASE_JOINTS + src[m.end():]

# Extend the 'stand' keyframe qpos: 3 new leading zeros (base at the origin,
# facing +x) ahead of the original 43 values -- mirrors make_fixed_base.py's
# own keyframe-trim logic in reverse (there it drops 7 leading free-joint
# values; here it adds 3). ctrl is untouched (never included base dof).
KEY_RE = re.compile(r'(<key\s+name="stand"\s+qpos=")([^"]+)("\s+ctrl="[^"]+"\s*/>)')
km = KEY_RE.search(src)
assert km, "'stand' keyframe not found"
vals = km.group(2).split()
assert len(vals) == 43, f"expected 43 qpos values (g1_fixed_upper.xml, no free joint), got {len(vals)}"
new_qpos = " ".join(["0", "0", "0"] + vals)
src = src[:km.start()] + km.group(1) + new_qpos + km.group(3) + src[km.end():]

DST.write_text(src)
print(f"wrote {DST}  (base_slide_x/base_slide_y/base_yaw added to pelvis, "
      f"nq 43 -> 46, nu unchanged at 43, 'stand' keyframe qpos extended)")
