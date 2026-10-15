"""Headless check of the brick pool + BrickSpawner (scene_fixed_table.xml, make_brick_pool.py).

Run with the env that has mujoco (e.g. `conda run -n robosuite python verify_brick_spawner.py`).
"""
import os, pathlib, sys
import numpy as np
import mujoco

from brick_spawner import BrickSpawner, _obb_overlap, ENV_HALF
from brick_snap import BrickSnapper

MODEL_DIR = pathlib.Path(os.environ.get(
    "MODEL_DIR", pathlib.Path(__file__).resolve().parent.parent / "mujoco_menagerie" / "unitree_g1"))
NAMES = [f"brick{i}" for i in range(1, 14)]
fails = []


def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  [{info}]" if info else ""))
    if not ok:
        fails.append(name)


def bpos(m, d, n):
    return d.xpos[m.body(n).id].copy()


def run(m, d, key, snapper, secs):
    for _ in range(int(secs / m.opt.timestep)):
        d.ctrl[:] = m.key_ctrl[key]
        mujoco.mj_step(m, d)
        snapper.step(d)


def fresh():
    m = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene_fixed_table.xml"))
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_at_table")
    mujoco.mj_resetDataKeyframe(m, d, key)
    mujoco.mj_forward(m, d)
    return m, d, key


def main():
    # --- SAT sanity
    I = np.eye(3)
    c = np.zeros(3)
    check("sat: identical boxes overlap", _obb_overlap(c, I, ENV_HALF, c, I, ENV_HALF))
    check("sat: 0.5 m apart don't", not _obb_overlap(c, I, ENV_HALF, c + [0, 0.5, 0], I, ENV_HALF))
    yaw = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    check("sat: crossed boxes overlap", _obb_overlap(c, I, ENV_HALF, c + [0.0, 0.1, 0], yaw, ENV_HALF))
    check("sat: just clear along y", not _obb_overlap(c, I, ENV_HALF, c + [0, 2 * 0.04375 + 0.01, 0], I, ENV_HALF))

    m, d, key = fresh()
    check("nq == key length == 43 + 13*7", m.nq == len(m.key_qpos[key]) == 43 + 13 * 7, str(m.nq))
    sn = BrickSnapper(m, NAMES)
    sp = BrickSpawner(m, key)
    check("10 spares in inventory", sp.inventory == 10)
    park0 = {n: bpos(m, d, n) for n in sp.spares}

    run(m, d, key, sn, 3.0)
    check("no NaN after 3 s", not np.isnan(d.qpos).any())
    drift = max(np.linalg.norm(bpos(m, d, n) - park0[n]) for n in sp.spares)
    check("parked spares stay parked (3 s)", drift < 0.01, f"max drift {drift*1000:.2f} mm")
    zs = [bpos(m, d, f"brick{i}")[2] for i in (1, 2, 3)]
    check("brick1-3 rest on table", all(0.770 < z < 0.780 for z in zs), str(np.round(zs, 4)))

    # --- occupied slots refuse, inventory untouched
    for slot in (1, 3):
        ok, msg = sp.spawn(d, slot)
        check(f"slot {slot} occupied -> refused", not ok and sp.inventory == 10, msg)

    # --- clear slot 1 by moving brick1 away; spawn succeeds
    q1 = m.jnt_qposadr[m.body("brick1").jntadr[0]]
    d.qpos[q1:q1 + 3] = [0.45, -0.40, 0.78]
    mujoco.mj_forward(m, d)
    ok, msg = sp.spawn(d, 1)
    check("slot 1 vacant -> spawned", ok and sp.inventory == 9, msg)
    run(m, d, key, sn, 2.0)
    p = bpos(m, d, "brick4")
    check("brick4 rests in slot 1", abs(p[0] - 0.28) < 0.01 and abs(p[1] + 0.235) < 0.01 and 0.770 < p[2] < 0.780,
          str(np.round(p, 4)))
    ok, msg = sp.spawn(d, 1)
    check("slot 1 now occupied again -> refused, inventory 9", not ok and sp.inventory == 9, msg)
    others = [bpos(m, d, n) for n in sp.spares if n != "brick4"]
    check("other spares still parked", all(abs(o[0] + 30) < 0.01 and o[2] > 0.99 for o in others))

    # --- a robot geom over the slot blocks the spawn: drop the right hand onto slot 3's spot
    d.qpos[m.jnt_qposadr[m.body("brick3").jntadr[0]]:][:3] = [0.45, 0.40, 0.78]
    mujoco.mj_forward(m, d)
    check("slot 3 vacant after moving brick3", sp.vacant(d, 3))
    hb = m.body("left_wrist_yaw_link").id
    # fake a robot geom position at the slot (geom_xpos of one hand geom)
    g = int(sp._robot_geoms[np.flatnonzero(m.geom_bodyid[sp._robot_geoms] == hb)[0]])
    old = d.geom_xpos[g].copy()
    d.geom_xpos[g] = sp.slot_pose[3][:3]
    who = sp.blocker(d, 3)          # (spawn() re-runs mj_forward, which would undo the fake)
    d.geom_xpos[g] = old
    check("robot geom in slot -> blocked", who == "left_wrist_yaw_link", str(who))
    mujoco.mj_forward(m, d)
    ok, msg = sp.spawn(d, 3)
    check("slot 3 spawn ok", ok and sp.inventory == 8, msg)

    # --- exhaust inventory
    sp._parked.clear()
    ok, msg = sp.spawn(d, 1)
    check("empty inventory -> refused", not ok and "no spare" in msg, msg)

    # --- restock: X-button semantics
    sn.reset(d)
    for n in NAMES[:3]:
        a = m.jnt_qposadr[m.body(n).jntadr[0]]
        d.qpos[a:a + 7] = m.key_qpos[key][a:a + 7]
        d.qvel[m.jnt_dofadr[m.body(n).jntadr[0]]:][:6] = 0
    sp.restock(d)
    check("restock -> 10 spares", sp.inventory == 10)
    run(m, d, key, sn, 3.0)
    drift = max(np.linalg.norm(bpos(m, d, n) - park0[n]) for n in sp.spares)
    check("re-parked spares hold", drift < 0.01, f"{drift*1000:.2f} mm")
    zs = [bpos(m, d, f"brick{i}")[2] for i in (1, 2, 3)]
    check("brick1-3 back on table", all(0.770 < z < 0.780 for z in zs))
    ok, msg = sp.spawn(d, 1)
    check("slot 1 occupied again after reset -> refused", not ok, msg)
    check("no NaN at the end", not np.isnan(d.qpos).any())

    print("\nALL PASS" if not fails else f"\nFAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
