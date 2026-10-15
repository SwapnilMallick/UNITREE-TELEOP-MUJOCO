"""
Grasp-state detection: contact-only holding (see CLAUDE.md's "Grasp
Primitive" section -- no kinematic weld) has no weld-on-detect flag, so "is
the object actually grasped" has to be inferred from simulation state, not
just assumed true once the hand finishes closing.

Two signals, chosen to be realistic/deployable rather than reads of
simulator-only ground truth (no d.contact introspection here -- that's
useful for verification/debugging, as verify_grasp_hold.py already does, but
a real Dex3-1 hand's control loop wouldn't have privileged access to exact
contact geometry):

  1. Aperture stall: compare the *commanded* fully-closed joint targets
     (grasp_primitive.hand_ctrl(1.0, side)) against the *actual* current
     finger joint positions (d.qpos[hand_slice] -- valid directly, no index
     translation needed, since qpos index == ctrl index for this model's
     hinge-only, no-freejoint robot dof, same fact arm_ik.py relies on).
     Closing on empty space, fingers reach their commanded target almost
     exactly (confirmed in verify_grasp_primitive.py). Closing on a real
     object, they get physically blocked and stall short of it -- a
     persistent position error, the same signal a real position-controlled
     gripper would expose via motor current or commanded-vs-actual error.
  2. Gripper-to-object proximity: how close the gripper site is to the
     target object's position -- requires knowing where the object is (from
     perception in a real system; ground-truth pose is fine in sim).

Both combine into a single confidence score via grasp_confidence(), not a
one-shot binary flag -- meant to be evaluated every control step (gates the
pick->lift transition, and keeps working afterward to catch a dropped
object mid-carry, not just at the moment the hand finishes closing).
"""
import numpy as np

from grasp_primitive import hand_ctrl

# Calibrated empirically against verify_grasp_state.py's three reference
# scenarios (grasped-and-held, closed-on-empty-space, dropped-mid-hold) --
# not a one-shot guess, and revised three times as more real data came in.
# aperture_stall is the noisier, less reliable of the two signals: closed-
# on-empty-space/dropped sits flat at ~0.0007 rad, but grasped values keep
# decaying over a long hold rather than settling -- brick2 goes
# 0.011 (right after closing) -> 0.0067 (lift complete) -> 0.0021 (7s into
# the hold), still slowly falling, extrapolating toward an asymptote around
# ~0.0019 (geometric-decay fit on the per-second deltas). Confirmed
# genuinely still grasped throughout via gripper_object_distance, which
# stays essentially flat (~0.043-0.045m) the entire time -- proximity is
# the physically stable signal for sustained holds; aperture_stall is most
# discriminating right at the moment of closing and gets noisier/weaker the
# longer the hold continues, likely because a settled grip can relax its
# actuator position error over time without the grip itself loosening.
# A first threshold of 0.25 was ~19x too high (misclassified every grasp as
# ungrasped); 0.005 was still too tight (misclassified brick2 shortly after
# lift); 0.002 looked fine on a 3s hold but would have failed by ~t=10s on
# the same case. 0.0012 keeps margin under the ~0.0019 long-hold asymptote
# while staying above the 0.0007 empty/dropped baseline -- tighter margin
# than ideal on the low side, acceptable given proximity is the
# authoritative signal for anything beyond the initial grasp confirmation.
APERTURE_STALL_THRESHOLD = 0.0012  # rad, mean |commanded - actual| across the 7 hand joints
PROXIMITY_THRESHOLD = 0.07         # m, gripper site to object-body distance -- ~0.043-0.051m
                                    # grasped (either brick) vs. ~0.21-0.26m dropped/empty


def aperture_stall(d, hand_slice, side):
    """Mean absolute joint-angle gap between the commanded fully-closed pose
    and the actual current finger joint positions. Near 0 when closing on
    empty space (fingers reach their target); larger when something blocks
    closure."""
    commanded = hand_ctrl(1.0, side)
    actual = d.qpos[hand_slice]
    return float(np.mean(np.abs(commanded - actual)))


def gripper_object_distance(d, site_id, body_id):
    """Straight-line distance from the gripper site to the object body's
    position (its own body-frame origin, not the nearest surface point --
    close enough given the object is brick-sized)."""
    return float(np.linalg.norm(d.site_xpos[site_id] - d.xpos[body_id]))


def grasp_confidence(d, hand_slice, side, site_id, body_id,
                      aperture_threshold=APERTURE_STALL_THRESHOLD,
                      proximity_threshold=PROXIMITY_THRESHOLD):
    """Returns a dict with both raw signals, a `grasped` boolean (both
    signals past their threshold), and a 0-1 `confidence` score (how far
    past each threshold, capped at 2x margin, zero if either signal fails).
    Evaluate every control step, not just once -- confidence dropping back
    to 0 mid-carry means the object was lost, not just "not yet picked up"."""
    stall = aperture_stall(d, hand_slice, side)
    dist = gripper_object_distance(d, site_id, body_id)

    grasped = (stall >= aperture_threshold) and (dist <= proximity_threshold)
    if grasped:
        stall_margin = min(stall / aperture_threshold, 2.0) / 2.0
        prox_margin = min(proximity_threshold / max(dist, 1e-6), 2.0) / 2.0
        confidence = min(stall_margin, prox_margin)
    else:
        confidence = 0.0
    return dict(aperture_stall=stall, gripper_object_distance=dist,
                confidence=confidence, grasped=grasped)
