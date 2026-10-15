"""
StackSequence: orchestrates multiple PickSequence pick-and-place cycles, one
brick at a time, to build a physical stack -- the concrete deliverable for
Phase 1 step 8 ("stacking sequencing").

Stack location, pick order, and arm assignment (decided empirically, not
guessed -- see the checks that produced these numbers):

- **Stack location = brick1's own resting position** (x=0.28, y=-0.15), not
  a new, untested spot. brick1 stays exactly where it is and becomes the
  stack's base -- no pick/place cycle needed for it. Reusing this position
  means the only genuinely NEW physics being exercised is the place/release
  mechanism itself (PickSequence's TRANSIT/RELEASE/RETREAT phases), not a
  brand new pick-side reach target, keeping this step's real risk contained
  to what it's actually testing.
- **Pick order: brick2 then brick3.** brick2 (right arm) is picked from its
  own resting spot and placed directly on top of brick1 -- both bricks
  already right-arm/right-side (y<0), both already individually verified
  reachable (verify_grasp_hold.py), so this is the FIRST concrete win here.
- **brick3 (left arm) is included for the same reason every other module in
  this repo keeps its known-broken case rather than dropping it** (see
  verify_grasp_hold.py, verify_pick_sequence.py): running it and reporting
  the failure honestly is more useful than hiding it. But it is not expected
  to succeed, and for TWO independent, compounding reasons, not one:
    1. The already-documented brick3 grasp-approach problem (the left wrist
       collides with brick3 at its own resting spot regardless of height/
       offset/orientation blend -- see CLAUDE.md's "Fingertip/Brick Contact
       Tuning") means brick3 is never actually picked up in the first place.
    2. Even setting that aside, a real simulated drive (not just an IK
       solve) of the LEFT arm to THIS stack's location (on the right side,
       at brick1/brick2's x=0.28, y=-0.15) measured a 15.6cm settled
       position error -- essentially the same self-collision problem
       already found and documented for brick2's dead-center (y=0) case
       (which was ~4.4-4.8cm), except *worse*, because reaching a
       right-side stack from the left arm crosses further past the body's
       centerline than merely reaching y=0 does. Pure IK-only convergence
       looked fine here too (~1cm) -- exactly the same trap CLAUDE.md
       already warns about ("IK correctness alone does not mean a target is
       safely reachable"): position-only IK has no notion of the physical
       self-collision that only shows up once the solved joint targets are
       actually driven through real physics via an approach path.
  So brick3 is doubly blocked: it can't be picked up, and even if it could,
  it can't reach this particular stack location. A left-arm-reachable stack
  location would need its own dedicated spot on the left side -- out of
  scope here (this is one stack, one column, one shared target location by
  definition), and moot until issue #1 is resolved anyway.

Run() drives each brick's PickSequence to completion (or to its documented
failure point) one at a time, matching the per-frame .step() contract every
other module here uses -- see verify_stack_sequence.py for a headless
regression check and stand_next_to_table.py for the live integration.
"""
import numpy as np
import mujoco

from actuator_groups import LEFT_ARM, LEFT_HAND, RIGHT_ARM, RIGHT_HAND
from pick_sequence import PickSequence

BRICK_HALF_HEIGHT = 0.01725   # m, half of a brick's 3.45cm height (scene_fixed_table.xml)
LAYER_GAP = 0.0008            # m, observed rest gap above the surface below (matches a
                               # brick's own gap when resting flat on the table)

# (side, site_name, arm_slice, hand_slice, brick_name) -- same shape as
# stand_next_to_table.py's PICK_TARGETS, in the pick order decided above.
STACK_ORDER = [
    ("right", "right_gripper_site", RIGHT_ARM, RIGHT_HAND, "brick2"),
    ("left",  "left_gripper_site",  LEFT_ARM,  LEFT_HAND,  "brick3"),
]
BASE_BRICK = "brick1"   # stays put -- the stack's base, never picked


class StackSequence:
    """Drives STACK_ORDER's bricks through pick-and-place, one at a time,
    onto a column above BASE_BRICK's own resting position.

    Usage:
        seq = StackSequence(m)
        seq.start(m, d)
        while running:
            seq.step(m, d, hold_ctrl)   # once per control step
            mujoco.mj_step(m, d)
    """

    def __init__(self, m):
        self._specs = STACK_ORDER
        self._active = None       # PickSequence for the current brick
        self._index = -1
        self._layer = None        # world xy,z of the NEXT open stack slot
        self.done = False

    def start(self, m, d):
        base_pos = d.xpos[m.body(BASE_BRICK).id].copy()
        self._layer = base_pos.copy()
        self._layer[2] += 2 * BRICK_HALF_HEIGHT + LAYER_GAP
        self._index = -1
        self.done = False
        self._advance(m, d)

    def _advance(self, m, d):
        self._index += 1
        if self._index >= len(self._specs):
            self._active = None
            self.done = True
            return
        side, site_name, arm_slice, hand_slice, brick_name = self._specs[self._index]
        self._active = PickSequence(m, side, site_name, arm_slice, hand_slice, brick_name)
        # place_surface_z = the top of whatever this brick is landing ON (the
        # previous layer, or brick1's own top for the first placement) --
        # StackSequence knows brick dimensions, so it computes this explicitly
        # rather than relying on PickSequence's more conservative fallback
        # (treating place_pos's own height as the surface). See
        # pick_sequence.py's "Release-from-hover" for why this matters.
        surface_z = self._layer[2] - BRICK_HALF_HEIGHT
        self._active.start(m, d, place_pos=self._layer.copy(), place_surface_z=surface_z)

    @property
    def current_brick(self):
        return None if self._active is None else self._active.brick_name

    @property
    def phase(self):
        return None if self._active is None else self._active.phase

    def step(self, m, d, hold_ctrl):
        """Advance one control step. Returns (brick_name, phase) for the
        currently active pick, or (None, None) once every brick in
        STACK_ORDER has reached DONE (or is stuck -- see stuck detection in
        verify_stack_sequence.py; StackSequence itself doesn't time out a
        stuck brick, matching PickSequence's own no-timeout HOLD design)."""
        if self._active is None:
            d.ctrl[:] = hold_ctrl
            return None, None
        phase = self._active.step(m, d, hold_ctrl)
        if phase == "DONE":
            # this brick's place succeeded (or ran its full cycle); the next
            # brick stacks one layer higher regardless of whether THIS one
            # actually landed there -- verify_stack_sequence.py checks real
            # resting height separately, this only advances the sequence
            self._layer = self._layer.copy()
            self._layer[2] += 2 * BRICK_HALF_HEIGHT + LAYER_GAP
            self._advance(m, d)
        return self.current_brick, phase
