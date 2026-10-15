"""
Dex3-1 grasp primitive: hand shape (a single 0->1 grip scalar piecewise-
interpolating through open -> pre-grasp -> closed, not full per-finger IK)
plus the approach orientation for the arm to use while closing on a brick.

CLOSED drives every joint to (near) its declared <joint range=...> limit.
Verified by forward-kinematics sweep (not the original guess -- see
make_fixed_base.py's gripper-site derivation comment for the full story) that
thumb_0=0 with thumb_1/thumb_2 and index/middle all at their curled limit is
where the thumb tip and the index/middle tip midpoint actually converge
(1.5cm residual gap = the hand's closed aperture); driving CLOSED to the
joint limit rather than a moderate ~70% guess is what makes the fingers
actually reach that convergence point instead of stopping short of the
object. It's fine for a position actuator to target beyond where a real
object will stop it -- that's what generates grip force against the object,
not a bug.

Per-finger joint order within each 7-vector matches actuator_groups.py's
LEFT_HAND/RIGHT_HAND slices: thumb_0, thumb_1, thumb_2, then, per hand's own
(non-symmetric) order, middle_0, middle_1, index_0, index_1 (left) or
index_0, index_1, middle_0, middle_1 (right).
"""
import numpy as np
import mujoco

# --- Left hand: thumb_0,1,2, middle_0,1, index_0,1 -------------------------
# Ranges (radians): thumb_0 [-1.0472,1.0472] thumb_1 [-0.7243,1.0472]
# thumb_2 [0,1.7453] middle_0 [-1.5708,0] middle_1 [-1.7453,0]
# index_0 [-1.5708,0] index_1 [-1.7453,0]. For middle/index, 0 = straight
# (open), the negative bound = fully curled.
LEFT_HAND_OPEN     = np.array([0.0,  0.0,     0.0,      0.0,     0.0,      0.0,     0.0])
LEFT_HAND_PREGRASP = np.array([0.0,  0.4712,  0.7854,  -0.7071, -0.7854,  -0.7071, -0.7854])
LEFT_HAND_CLOSED   = np.array([0.0,  1.0472,  1.74533, -1.5708, -1.74533, -1.5708, -1.74533])

# --- Right hand: thumb_0,1,2, index_0,1, middle_0,1 (mirrored ranges) ------
RIGHT_HAND_OPEN     = np.array([0.0,  0.0,      0.0,      0.0,     0.0,     0.0,    0.0])
RIGHT_HAND_PREGRASP = np.array([0.0, -0.4712,  -0.7854,   0.7071,  0.7854,  0.7071, 0.7854])
RIGHT_HAND_CLOSED   = np.array([0.0, -1.0472,  -1.74533,  1.5708,  1.74533, 1.5708, 1.74533])


def hand_ctrl(grip, side):
    """grip in [0,1]: 0=open, 0.5=pre-grasp, 1=closed, piecewise linear
    through the three waypoints. side is 'left' or 'right'."""
    grip = float(np.clip(grip, 0.0, 1.0))
    if side == "left":
        a, b, c = LEFT_HAND_OPEN, LEFT_HAND_PREGRASP, LEFT_HAND_CLOSED
    elif side == "right":
        a, b, c = RIGHT_HAND_OPEN, RIGHT_HAND_PREGRASP, RIGHT_HAND_CLOSED
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if grip <= 0.5:
        t = grip / 0.5
        return a + t * (b - a)
    else:
        t = (grip - 0.5) / 0.5
        return b + t * (c - b)


def grasp_target_quat(ik, target_pos, brick_short_axis_world, blend=0.6, natural_iters=100):
    """Target orientation for the gripper site to approach target_pos with,
    for grasping a brick whose short axis (in world frame -- its body xmat's
    Y column) is brick_short_axis_world.

    Position-only IK has no notion of orientation at all: verified that its
    natural solution for a tabletop target has the finger-spread axis nearly
    perpendicular to the brick's short axis (dot product ~0.13), not aligned
    with it, purely because nothing in a 3-DOF task constrains it.

    This blends `blend` of the way from that natural orientation toward an
    "ideal" one (finger-spread axis, local Z, exactly aligned with the
    brick's short axis; reach axis, local X, left unchanged from natural).
    blend=1.0 (full alignment) was tried first and found to require the
    elbow to sit at its exact upper joint limit (2.0944 rad) -- verified via
    per-iteration tracing that it doesn't settle there, it oscillates in and
    out of the limit indefinitely instead of converging. blend=0.6 was picked
    empirically as the largest value that converges reliably (position error
    <2cm, orientation error small) for all three assigned pick targets.
    Alignment quality varies by target -- it's a soft blend, not a guarantee:
    brick1 0.13->0.83, brick2 (tighter reach geometry, see CLAUDE.md) 0.06->0.27,
    brick3 -0.13->0.64.

    Must be called right after ik.sync() and before the caller's own solve()
    -- this function runs its own internal position-only solve() to discover
    the natural orientation, which usefully leaves the scratch state warm-started
    close to the answer for the caller's subsequent orientation-aware solve()."""
    ik.solve(target_pos, iters=natural_iters)
    natural_quat = np.zeros(4)
    mujoco.mju_mat2Quat(natural_quat, ik.scratch.site_xmat[ik.site_id])
    natural_R = ik.scratch.site_xmat[ik.site_id].reshape(3, 3)
    x = natural_R[:, 0]

    z = brick_short_axis_world - np.dot(brick_short_axis_world, x) * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R_ideal = np.column_stack([x, y, z])
    ideal_quat = np.zeros(4)
    mujoco.mju_mat2Quat(ideal_quat, R_ideal.flatten())

    dq = np.zeros(3)
    mujoco.mju_subQuat(dq, ideal_quat, natural_quat)
    blended = natural_quat.copy()
    mujoco.mju_quatIntegrate(blended, dq * blend, 1.0)
    return blended
