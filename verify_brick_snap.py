"""Headless regression check for brick_snap.py (stud/groove snap-fit).

Drops brick2 onto brick1 at various offsets and checks: aligned seats at
~70 mm and holds; misaligned rests on the stud tips (~80 mm) unsnapped; a
pull breaks the snap with the bottom brick staying; a 3-stack holds; reset
clears the welds. Sim-only -- never exercised with a gripper holding a brick.
"""
import os, pathlib, sys
import numpy as np
import mujoco
from brick_snap import BrickSnapper

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR", pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
# brick1-3 only: brick4..13 are the parked spare pool (brick_spawner.py, verified separately)
NAMES = [f"brick{i}" for i in range(1, 4)]
m = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene_fixed_table.xml"))
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def fresh():
    S = BrickSnapper(m, NAMES)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[:] = m.key_ctrl[key]
    run(S, d, 800)
    return S, d


def run(S, d, n):
    for _ in range(n):
        mujoco.mj_step(m, d)
        S.step(d)


def drop(S, d, up, lo, dx=0.0, dy=0.0, yaw=0.0, gap=0.03):
    a = S.qadr[up]
    p = d.body(lo).xpos.copy()
    d.qpos[a:a + 3] = [p[0] + dx, p[1] + dy, p[2] + 0.0805 + gap]
    d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    d.qvel[S.dadr[up]:S.dadr[up] + 6] = 0


def dz(S, d, up, lo):
    return (d.geom_xpos[S.geom[up]][2] - d.geom_xpos[S.geom[lo]][2]) * 1000


# lower brick yaw is whatever the keyframe gives (brick1 yaw 0)
S, d = fresh(); drop(S, d, "brick2", "brick1"); run(S, d, 1500)
check("aligned drop seats", "brick2" in S.welded and abs(dz(S, d, "brick2", "brick1") - 70) < 0.5,
      f"dz={dz(S, d, 'brick2', 'brick1'):.2f}mm")
run(S, d, 3000)
check("aligned stays seated 6 s", "brick2" in S.welded and abs(dz(S, d, "brick2", "brick1") - 70) < 0.5)

for label, kw in [("3 mm offset", dict(dx=0.003)), ("10 mm offset", dict(dx=0.010)),
                  ("yaw 8 deg", dict(yaw=np.radians(8))), ("yaw 15 deg", dict(yaw=np.radians(15))),
                  ("cross 90 deg", dict(yaw=np.radians(90)))]:
    S, d = fresh(); drop(S, d, "brick2", "brick1", **kw); run(S, d, 2000)
    h = dz(S, d, "brick2", "brick1")
    check(f"misaligned ({label}) does not snap", "brick2" not in S.welded and h > 75, f"dz={h:.1f}mm")

S, d = fresh(); drop(S, d, "brick2", "brick1", dx=0.001); run(S, d, 1500)
check("within tolerance (1 mm) snaps", "brick2" in S.welded)

S, d = fresh(); drop(S, d, "brick2", "brick1"); run(S, d, 1500)
b = m.body("brick2").id; f = 0.0; z1 = d.body("brick1").xpos[2]
while "brick2" in S.welded and f < 60:
    f += 5.0 * m.opt.timestep; d.xfrc_applied[b, 2] = f; mujoco.mj_step(m, d); S.step(d)
broke = "brick2" not in S.welded
d.xfrc_applied[:] = 0; run(S, d, 1000)
check("pull-off breaks snap near weight+5 N", broke and 8 < f < 20, f"applied={f:.1f}N")
check("bottom brick stays put", abs(d.body("brick1").xpos[2] - z1) < 0.002)
check("released brick rests, no ejection", dz(S, d, "brick2", "brick1") < 90)

S, d = fresh(); drop(S, d, "brick2", "brick1"); run(S, d, 1500)
drop(S, d, "brick3", "brick2"); run(S, d, 2000)
check("3-stack holds", len(S.welded) == 2 and abs(dz(S, d, "brick3", "brick2") - 70) < 0.5,
      f"welded={sorted(S.welded)}")

# stack whose upper brick is listed FIRST in the weld (was: broke instantly, sign bug)
S, d = fresh(); drop(S, d, "brick1", "brick2"); a = S.qadr["brick1"]
d.qpos[a + 3:a + 7] = d.body("brick2").xquat; run(S, d, 3000)
check("reverse-order stack holds", S.welded == {"brick1": "brick2"} and len(S.events) == 1, str(S.events))

S.reset(d)
snap_eqs = [i for i in range(m.neq) if m.eq(i).name.startswith("snap_")]   # park_* welds stay on
check("reset clears welds", len(S.welded) == 0 and not d.eq_active[snap_eqs].any())

# ------------------------------------------- dovetail joint: PHYSICAL (compound collision), from above
# Tab on the -X end, slot on the +X end (scene XML / make_brick_dovetail.py). No weld, no teleport:
# a brick lowered with its tab over a neighbour's slot drops in under gravity and the shape locks it.
SP = 0.175   # mated body-to-body spacing along the long axis


def arena(anchor_name, x, y=-0.05):
    """Fresh scene, every brick but `anchor_name` stashed on the far side of the table, anchor re-laid
    with identity yaw at (x, y) so the mate slots are clear."""
    S, d = fresh()
    ys = iter([0.12, 0.22, 0.32])
    for n in S.names:
        a = S.qadr[n]
        d.qpos[a:a + 3] = [x, y, 0.79] if n == anchor_name else [0.5, next(ys), 0.79]
        d.qpos[a + 3:a + 7] = [1, 0, 0, 0]; d.qvel[S.dadr[n]:S.dadr[n] + 6] = 0
    run(S, d, 800)
    return S, d


def lower(S, d, up, lo, side=1, lat=0.0, dx=0.0, yaw=0.0, gap=0.04):
    """Release `up` `gap` m above the height where it would sit level with `lo`, its tab over lo's
    slot (side=+1, up on lo's +x side) or its slot over lo's tab (side=-1)."""
    Rl = d.xmat[S.body[lo]].reshape(3, 3)
    p = d.xpos[S.body[lo]] + side * (SP + dx) * Rl[:, 0] + lat * Rl[:, 1] + np.array([0, 0, gap])
    y = np.arctan2(Rl[1, 0], Rl[0, 0]) + yaw
    a = S.qadr[up]; d.qpos[a:a + 3] = p; d.qpos[a + 3:a + 7] = [np.cos(y / 2), 0, 0, np.sin(y / 2)]
    d.qvel[S.dadr[up]:S.dadr[up] + 6] = 0; mujoco.mj_kinematics(m, d)


def rel(S, d, a, b):
    return d.xmat[S.body[a]].reshape(3, 3).T @ (d.xpos[S.body[b]] - d.xpos[S.body[a]])


def is_in(S, d, a="brick1", b="brick2", side=1):
    r = rel(S, d, a, b)
    return abs(r[2]) < 0.002 and abs(r[0] - side * SP) < 0.004 and abs(r[1]) < 0.004


for gap in (0.02, 0.06):
    S, d = arena("brick1", 0.35); lower(S, d, "brick2", "brick1", gap=gap); run(S, d, 1200)
    r = rel(S, d, "brick1", "brick2")
    check(f"dovetail: tab dropped {gap * 100:.0f} cm above the slot seats by gravity alone",
          is_in(S, d) and abs(r[0] - SP) < 0.0005 and not S.welded, f"rel={np.round(r * 1000, 2)}mm")
run(S, d, 3000)
check("dovetail: stays seated 6 s", is_in(S, d))

S, d = arena("brick2", 0.50); lower(S, d, "brick1", "brick2", side=-1); run(S, d, 1500)
check("dovetail: slot lowered over a tab seats too", is_in(S, d, "brick2", "brick1", side=-1))

for label, kw, want in [("lateral 2 mm", dict(lat=0.002), True), ("lateral 3.5 mm", dict(lat=0.0035), False),
                        ("axial +1 mm", dict(dx=0.001), True), ("axial +3 mm", dict(dx=0.003), False),
                        ("axial -3 mm", dict(dx=-0.003), False),
                        ("yaw 1 deg", dict(yaw=np.radians(1)), True), ("yaw 4 deg", dict(yaw=np.radians(4)), False)]:
    S, d = arena("brick1", 0.35); lower(S, d, "brick2", "brick1", **kw); run(S, d, 1200)
    check(f"dovetail: {label} {'drops in' if want else 'does not drop in'}", is_in(S, d) == want)

# Pushing bricks together ALONG THE TABLE cannot mate them: the tab head is wider than the slot neck.
S, d = arena("brick1", 0.35)
a = S.qadr["brick2"]
d.qpos[a:a + 3] = d.xpos[S.body["brick1"]] + np.array([0.20, 0, 0.0]); d.qpos[a + 3:a + 7] = [1, 0, 0, 0]; d.qvel[:] = 0
run(S, d, 300)
for _ in range(1500):
    d.qpos[a] -= 0.0002; d.qvel[S.dadr["brick2"]:S.dadr["brick2"] + 6] = 0
    mujoco.mj_step(m, d); S.step(d)
    if rel(S, d, "brick1", "brick2")[0] < 0.176:
        break
run(S, d, 800)
check("dovetail: bricks slid together on the table do not mate", not is_in(S, d), f"rel={np.round(rel(S, d, 'brick1', 'brick2') * 1000, 1)}mm")

# The mated joint locks against sliding apart (25 N each way for 1 s); the same test on a tab-facing-tab
# (flipped) pair just separates.
b1, b2 = m.body("brick1").id, m.body("brick2").id
S, d = arena("brick1", 0.35); lower(S, d, "brick2", "brick1", gap=0.02); run(S, d, 900)
d.xfrc_applied[b2, 0], d.xfrc_applied[b1, 0] = 25.0, -25.0
run(S, d, 500); d.xfrc_applied[:] = 0
check("dovetail: 25 N each way pulling along the axis does not separate it", rel(S, d, "brick1", "brick2")[0] < 0.182,
      f"gap grew to {(rel(S, d, 'brick1', 'brick2')[0] - SP) * 1000:.1f} mm")
S, d = arena("brick1", 0.35); lower(S, d, "brick2", "brick1", yaw=np.pi, gap=0.02); run(S, d, 900)
d.xfrc_applied[b2, 0], d.xfrc_applied[b1, 0] = 25.0, -25.0
run(S, d, 500); d.xfrc_applied[:] = 0
check("dovetail: a tab-facing-tab pair (nothing interlocks) separates under the same pull",
      rel(S, d, "brick1", "brick2")[0] > 0.19)

# Lifting straight up slides it out (only vertical extraction works); the neighbour stays put.
S, d = arena("brick1", 0.35); lower(S, d, "brick2", "brick1", gap=0.02); run(S, d, 900)
z1 = d.xpos[S.body["brick1"]][2]
for i in range(400):
    d.xfrc_applied[b2, 2] = 14.0; mujoco.mj_step(m, d); S.step(d)
d.xfrc_applied[:] = 0
check("dovetail: lifting one brick slides it out and leaves the other", rel(S, d, "brick1", "brick2")[2] > 0.02
      and abs(d.xpos[S.body["brick1"]][2] - z1) < 0.002)

S, d = arena("brick1", 0.26); lower(S, d, "brick2", "brick1", gap=0.02); run(S, d, 800)
lower(S, d, "brick3", "brick2", gap=0.02); run(S, d, 2000)
check("dovetail: 3-brick chain seats", is_in(S, d, "brick1", "brick2") and is_in(S, d, "brick2", "brick3"))

S, d = arena("brick1", 0.30); lower(S, d, "brick2", "brick1", gap=0.02); run(S, d, 800)
a = S.qadr["brick3"]; p = d.xpos[S.body["brick1"]]
d.qpos[a:a + 3] = [p[0], p[1], p[2] + 0.0805 + 0.03]; d.qpos[a + 3:a + 7] = d.body("brick1").xquat; d.qvel[S.dadr["brick3"]:S.dadr["brick3"] + 6] = 0
run(S, d, 3000)
check("dovetail: stud-stacking onto a dovetail-linked brick works and keeps the joint",
      S.welded == {"brick3": "brick1"} and is_in(S, d), str(S.events))

S, d = arena("brick1", 0.35); lower(S, d, "brick2", "brick1", gap=0.02); run(S, d, 800)
S.reset(d)
check("reset does not disturb a dovetail joint", is_in(S, d))

print("ALL PASS" if all(results) else "SOME FAILED")
sys.exit(0 if all(results) else 1)
