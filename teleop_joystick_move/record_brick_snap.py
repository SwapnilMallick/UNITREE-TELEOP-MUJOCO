"""Renders videos/brick_snap.mp4 (misaligned drop rests, aligned drop snaps) and
videos/brick_side_snap.mp4 (dovetail: a brick lowered onto the end of another drops its tab into the slot by physics).
Run: MUJOCO_GL=glfw python record_brick_snap.py"""
import sys, numpy as np, mujoco, imageio
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from brick_snap import BrickSnapper
m=mujoco.MjModel.from_xml_path('/Users/swapnilmallick/Desktop/mujoco_menagerie/unitree_g1/scene_fixed_table.xml')
key=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'stand_at_table')
S=BrickSnapper(m,[f'brick{i}' for i in range(1,4)]); d=mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m,d,key); d.ctrl[:]=m.key_ctrl[key]
r=mujoco.Renderer(m,480,640)
cam=mujoco.MjvCamera(); cam.type=mujoco.mjtCamera.mjCAMERA_FREE
opt=mujoco.MjvOption(); opt.geomgroup[:]=[1,1,1,0,0,0]
frames=[]
def step(n):
    for _ in range(n):
        mujoco.mj_step(m,d); S.step(d)
        if int(round(d.time/m.opt.timestep))%int(1/(60*m.opt.timestep))==0:
            r.update_scene(d,cam,opt); frames.append(r.render().copy())
for _ in range(800): mujoco.mj_step(m,d); S.step(d)
d.time=0
p=d.body('brick1').xpos
cam.lookat[:]=[p[0],p[1],p[2]+0.06]; cam.distance=0.55; cam.azimuth=60; cam.elevation=-15
def place(up,lo,dx,yaw,gap):
    a=S.qadr[up]; q=d.body(lo).xpos
    d.qpos[a:a+3]=[q[0]+dx,q[1],q[2]+0.0805+gap]; d.qpos[a+3:a+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    d.qvel[S.dadr[up]:S.dadr[up]+6]=0
step(60)
place('brick2','brick1',0.012,np.radians(10),0.10); step(500)   # misaligned: rests on top
place('brick2','brick1',0.0,0.0,0.10); step(700)                # aligned: seats
print(S.events, len(frames))
imageio.mimsave('videos/brick_snap.mp4',frames,fps=30)


# ------------------------------------------------ side (dovetail) snap: lowered from ABOVE
S = BrickSnapper(m, [f'brick{i}' for i in range(1, 4)]); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, key); d.ctrl[:] = m.key_ctrl[key]
frames = []
ys = iter([0.12, 0.22, 0.32, 0.42])          # clear the table: stash the bricks we don't use
for n in S.names:
    a = S.qadr[n]
    d.qpos[a:a + 3] = [0.35, -0.05, 0.79] if n == 'brick1' else [0.5, next(ys), 0.79]
    d.qpos[a + 3:a + 7] = [1, 0, 0, 0]; d.qvel[S.dadr[n]:S.dadr[n] + 6] = 0
for _ in range(800): mujoco.mj_step(m, d); S.step(d)
d.time = 0
SP = 0.175
def lower_in(name, anchor, lat=0.0, start_gap=0.06, speed=0.12):
    """Hand stand-in: hold `name` above `anchor`'s cavity end (tab over the slot) and lower it
    straight down at `speed` m/s until it touches; then let go (physics + snapper decide)."""
    Rl = d.geom_xmat[S.geom[anchor]].reshape(3, 3)
    c = d.geom_xpos[S.geom[anchor]] + SP * Rl[:, 0] + lat * Rl[:, 1]
    bp = c - Rl @ m.geom_pos[S.geom[name]]
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rl.flatten())
    a = S.qadr[name]; d.qpos[a + 3:a + 7] = q
    z_hi = bp[2] + 0.0805 + start_gap
    d.qpos[a:a + 3] = [bp[0], bp[1], z_hi]
    while d.qpos[a + 2] > bp[2] + 0.0805 - 0.001:
        d.qpos[a + 2] -= speed * m.opt.timestep; d.qvel[S.dadr[name]:S.dadr[name] + 6] = 0
        step(1)
p = d.body('brick1').xpos + np.array([SP / 2, 0, 0.0])
cam.lookat[:] = [p[0], p[1], p[2] + 0.04]; cam.distance = 0.6; cam.azimuth = 55; cam.elevation = -22
step(60)
lower_in('brick2', 'brick1', lat=0.012); step(500)   # tab 12 mm off the slot: lands on top and tips off
lower_in('brick2', 'brick1', lat=0.0); step(700)     # tab over the slot: drops in under gravity and locks by shape
print(S.events, len(frames))
imageio.mimsave('videos/brick_side_snap.mp4', frames, fps=30)
