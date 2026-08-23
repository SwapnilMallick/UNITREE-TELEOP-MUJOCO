"""
Generate a fixed-base upper-body G1 from a stock Menagerie G1 MJCF -- either the
bare g1.xml (no fingers) or g1_with_hands.xml (Dex3-1 3-finger hands: thumb/
index/middle, 7 DoF/hand). Works on either source; the joint/actuator count
(29 vs 43) is read from the file itself, nothing is hardcoded per-variant.

Three edits, all required and verified:
  1. Remove the pelvis <freejoint>        -> welds the base to the world (zero drift).
  2. Trim the 'stand' keyframe qpos by 7  -> drop the free-joint (x y z qw qx qy qz)
                                              values so len(qpos) == nq once the
                                              freejoint is gone, or the model will
                                              NOT compile. ctrl is untouched (it
                                              never included free-joint dof).
  3. Inject the forward-facing FPV camera into torso_link, tilted down so a robot
     standing at the table (per scene_fixed_table.xml's layout) has the tabletop
     in frame.

Regex-based (not literal string match): g1.xml and g1_with_hands.xml format the
imu_in_torso site with different attribute order (pos-then-size vs
size-then-pos), which a literal match silently fails on. Matching by structure
instead of exact formatting is what "version-safe" means here.

Run this from inside your Menagerie 'unitree_g1/' directory (next to assets/):
    python make_fixed_base.py g1.xml             # bare hand (rubber_hand, no grasp)
    python make_fixed_base.py g1_with_hands.xml   # Dex3-1 dexterous hand
"""
import re
import sys
import pathlib

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "g1.xml")
DST = pathlib.Path("g1_fixed_upper.xml")
src = SRC.read_text()

# 1) remove freejoint
FREEJOINT_RE = re.compile(r'[ \t]*<freejoint\s+name="floating_base_joint"\s*/>\n')
assert FREEJOINT_RE.search(src), "freejoint line not found (Menagerie version mismatch?)"
src = FREEJOINT_RE.sub("", src, count=1)

# 2) trim keyframe qpos: drop the first 7 free-joint values (x y z qw qx qy qz).
# Length is read from the file, not hardcoded, so this works for both the 29-DoF
# bare model (36 -> 29) and the 43-DoF hands model (50 -> 43).
KEY_RE = re.compile(
    r'(<key\s+name="stand"\s+qpos=")([^"]+)("\s+ctrl="[^"]+"\s*/>)')
m = KEY_RE.search(src)
assert m, "'stand' keyframe not found (Menagerie version mismatch?)"
vals = m.group(2).split()
assert len(vals) % 7 == 1, f"unexpected keyframe qpos length {len(vals)} (not free-joint(7) + n*1)"
trimmed = " ".join(vals[7:])
src = src[:m.start()] + m.group(1) + trimmed + m.group(3) + src[m.end():]
print(f"keyframe qpos trimmed {len(vals)} -> {len(vals) - 7} (dropped free-joint dof)")

# 3) inject verified FPV camera into torso_link. xyaxes = right axis, up axis;
# view dir = cross(right, up), camera looks -z. Right axis (0 -1 0) is
# unchanged; the up axis is rotated by theta=49deg about it:
#   up = (sin(theta), 0, cos(theta))  ->  forward = (cos(theta), 0, -sin(theta))
# theta was solved from camera_xpos -> table-top-center at the stand keyframe
# (see scene_fixed_table.xml); re-derive it if the stance or table position
# changes. Matches the imu_in_torso site regardless of attribute order.
SITE_RE = re.compile(r'<site name="imu_in_torso"[^>]*/>')
site_m = SITE_RE.search(src)
assert site_m, "imu_in_torso site not found (Menagerie version mismatch?)"
camera_tag = ('\n            <camera name="fpv_teleop" mode="fixed" '
              'pos="0.08 0 0.45" fovy="70" xyaxes="0 -1 0  0.7558 0 0.6549"/>')
src = src[:site_m.end()] + camera_tag + src[site_m.end():]

DST.write_text(src)
print(f"wrote {DST}  (source={SRC.name}, base welded, keyframe trimmed, fpv_teleop added)")
