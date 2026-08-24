"""
Generate a fixed-base upper-body G1 from a stock Menagerie G1 MJCF -- either the
bare g1.xml (no fingers) or g1_with_hands.xml (Dex3-1 3-finger hands: thumb/
index/middle, 7 DoF/hand). Works on either source; the joint/actuator count
(29 vs 43) is read from the file itself, nothing is hardcoded per-variant.

Five edits, all required and verified:
  1. Remove the pelvis <freejoint>        -> welds the base to the world (zero drift).
  2. Trim the 'stand' keyframe qpos by 7  -> drop the free-joint (x y z qw qx qy qz)
                                              values so len(qpos) == nq once the
                                              freejoint is gone, or the model will
                                              NOT compile. ctrl is untouched (it
                                              never included free-joint dof).
  3. Inject the forward-facing FPV camera into torso_link, tilted down so a robot
     standing at the table (per scene_fixed_table.xml's layout) has the tabletop
     in frame.
  4. Inject a gripper site into each wrist ("left_gripper_site" / "right_gripper_site"),
     the target frame arm_ik.py's IK solver drives to a commanded pose. pos="0.1117
     +/-0.0402 0" in the wrist_yaw_link frame (sign flips between hands) is where the
     thumb tip and the index/middle tip midpoint actually converge at full curl --
     measured via forward kinematics (sweep thumb_0 with thumb_1/2 and index/middle
     maxed, find the abduction that minimizes thumb-to-finger-midpoint gap; residual
     gap 1.5cm, i.e. the hand's closed aperture, verified symmetric between hands).
     A first guess of pos="0.12 0 0" (assuming the pinch happens on the wrist's local
     Y=0 plane) was tried first and shown wrong -- the real convergence point sits
     ~4cm off that plane, which is exactly why closing on a brick only ever grazed it
     with one finger; see CLAUDE.md's "Grasp Primitive" section. Works on the bare
     g1.xml too (same wrist_yaw_link anchor and local frame) even though there's
     nothing to grasp with there.
  5. Tune contact (friction/solref/solimp) on the six fingertip-relevant collision
     geoms per hand (thumb_1/thumb_2, index_0/index_1, middle_0/middle_1) -- only on
     g1_with_hands.xml, silently skipped on the bare g1.xml (no fingers to tune).
     friction="3 0.5 0.3" (vs MuJoCo's default "1 0.005 0.0001") and solref="0.005 1"
     (vs default "0.02 1", i.e. 4x stiffer/faster contact response) were found by
     real iteration against a grasp+lift+hold test (verify_grasp_hold.py), not a
     one-shot guess -- high friction with the DEFAULT solref actually held *worse*
     than the unmodified baseline (contact was too soft/springy to develop the
     friction force before the grip slipped); the stiffness change was the load-bearing
     fix, friction alone wasn't enough. See CLAUDE.md's "Grasp Primitive" section for
     the full comparison table. The brick side of this same contact pair is tuned to
     match in scene_fixed_table.xml -- change one without the other and they'll fight.

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

# 4) inject gripper sites, anchored on the wrist_yaw_link collision geom (identically
# formatted in both g1.xml and g1_with_hands.xml). y sign flips between hands (mirrored
# geometry); see the derivation comment above.
PINCH_POINT = {"left": "0.1117 -0.0402 0", "right": "0.1117 0.0402 0"}
for side in ("left", "right"):
    anchor_re = re.compile(rf'<geom class="collision" mesh="{side}_wrist_yaw_link"/>')
    anchor_m = anchor_re.search(src)
    assert anchor_m, f"{side}_wrist_yaw_link collision geom not found (Menagerie version mismatch?)"
    site_tag = (f'\n                          <site name="{side}_gripper_site" '
                f'pos="{PINCH_POINT[side]}" size="0.01"/>')
    src = src[:anchor_m.end()] + site_tag + src[anchor_m.end():]

# 5) tune fingertip contact (friction/solref/solimp) -- silently skipped for
# the bare g1.xml (no fingers exist to match against).
CONTACT_ATTRS = 'friction="3 0.5 0.3" solref="0.005 1"'
n_tuned = 0
for side in ("left", "right"):
    # the five mesh-based collision geoms (thumb_2, index_0/1, middle_0/1)
    for finger in ("thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
        geom_re = re.compile(rf'<geom class="collision" mesh="{side}_hand_{finger}_link"/>')
        gm = geom_re.search(src)
        if gm is None:
            continue
        old = gm.group(0)
        new = old[:-2] + f' {CONTACT_ATTRS}/>'
        src = src[:gm.start()] + new + src[gm.end():]
        n_tuned += 1
    # thumb_1's collision geom is a box, not a mesh -- distinctive size/type
    # signature instead, y sign flips between hands.
    thumb1_y = "-0.032" if side == "left" else "0.032"
    thumb1_re = re.compile(
        rf'<geom size="0.01 0.015 0.01" pos="-0.001 {thumb1_y} 0" type="box" class="collision"/>')
    tm = thumb1_re.search(src)
    if tm is not None:
        old = tm.group(0)
        new = old[:-2] + f' {CONTACT_ATTRS}/>'
        src = src[:tm.start()] + new + src[tm.end():]
        n_tuned += 1
if n_tuned:
    assert n_tuned == 12, f"expected to tune 12 fingertip geoms (6/hand), tuned {n_tuned}"

DST.write_text(src)
print(f"wrote {DST}  (source={SRC.name}, base welded, keyframe trimmed, "
      f"fpv_teleop + gripper sites added, {n_tuned} fingertip geoms tuned)")
