"""
qpos/qvel dof groups for g1_planar_base.xml (make_planar_base.py): the same
43-actuator G1+Dex3-1 robot as actuator_groups.py, but with 3 new joints
(base_slide_x, base_slide_y, base_yaw) added directly on the pelvis body for
--base-mode planar/policy locomotion (locomotion.py).

Because the pelvis is the root body, MuJoCo assigns qpos/dof addresses by
body-tree traversal order -- any joint placed on it occupies the FIRST
slots, ahead of every joint already declared inside it. So qpos/dof index
== actuator_groups.py's index + 3 for every pre-existing joint; BASE is new,
at the front. Mirrors the exact same kind of index-shift pattern this repo
already has for inspire_hand/actuator_groups_inspire.py (a different model
change, same underlying MuJoCo behavior).

nu (actuator count) is UNCHANGED at 43 -- no actuators were added for the
base (locomotion.py drives it via direct d.qvel injection, not position
actuators), so CTRL-space slices below are identical to actuator_groups.py.
Only QPOS/QVEL-space slices shift. Anything that indexes d.ctrl can keep
using actuator_groups.py's constants unchanged even in planar/policy mode;
anything that indexes d.qpos/d.qvel for the robot (ArmIK.joint_slice,
TeleopController's shoulder-body lookup, etc.) must use the *_QPOS names
below instead when the loaded model has the 3 base joints.

Quirk inherited unchanged from actuator_groups.py: the two hands are NOT
internally symmetric -- left is thumb,middle,index, right is
thumb,index,middle.
"""

# --- qpos/qvel (dof) space: shifted +3 for the 3 new base joints -----------
BASE_QPOS      = slice(0, 3)     # base_slide_x, base_slide_y, base_yaw (qpos == dof here: all 1-dof joints)
LEG_QPOS       = slice(3, 15)
WAIST_QPOS     = slice(15, 18)
LEFT_ARM_QPOS  = slice(18, 25)
LEFT_HAND_QPOS = slice(25, 32)
RIGHT_ARM_QPOS = slice(32, 39)
RIGHT_HAND_QPOS = slice(39, 46)
UPPER_BODY_QPOS = slice(18, 46)

# --- ctrl (actuator) space: UNCHANGED from actuator_groups.py --------------
# (re-exported here so a caller that's already switched to this module for
# qpos doesn't also need to keep importing the other one just for ctrl)
LEG        = slice(0, 12)
WAIST      = slice(12, 15)
LEFT_ARM   = slice(15, 22)
LEFT_HAND  = slice(22, 29)
RIGHT_ARM  = slice(29, 36)
RIGHT_HAND = slice(36, 43)
UPPER_BODY = slice(15, 43)
