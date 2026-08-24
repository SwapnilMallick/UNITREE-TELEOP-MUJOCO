"""
Actuator/qpos/qvel dof groups for g1_fixed_upper.xml (generated from
g1_with_hands.xml: 43 actuators -- 29 body + Dex3-1 hands, 7 DoF/hand: thumb x3,
index x2, middle x2). Order is declaration order in the source MJCF's <actuator>
block, which for the arm+hand subtrees also matches qpos/qvel order (fingers are
nested under wrist_yaw_link with no sibling interleaving, and the robot has no
free joints, so qpos index == dof index == joint declaration order throughout).

Shared by stand_next_to_table.py and arm_ik.py callers so the two never drift
apart.

Quirk: the two hands are NOT internally symmetric -- left is thumb,middle,index
(22:25, 25:27, 27:29) but right is thumb,index,middle (36:39, 39:41, 41:43).
Doesn't affect the slices below, but matters if you later index a specific
finger.
"""

LEG        = slice(0, 12)    # hips / knees / ankles
WAIST      = slice(12, 15)   # yaw / roll / pitch
LEFT_ARM   = slice(15, 22)   # shoulder x3 / elbow / wrist x3
LEFT_HAND  = slice(22, 29)   # Dex3-1: thumb x3, middle x2, index x2
RIGHT_ARM  = slice(29, 36)   # shoulder x3 / elbow / wrist x3
RIGHT_HAND = slice(36, 43)   # Dex3-1: thumb x3, index x2, middle x2
UPPER_BODY = slice(15, 43)   # LEFT_ARM+LEFT_HAND+RIGHT_ARM+RIGHT_HAND (contiguous)
