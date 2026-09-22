"""Rewrites every brick's collision geoms in scene_fixed_table.xml as a physical dovetail
compound (idempotent: strips any previous brickN_collision / _jaw_* / _tab_* geoms first).

Per brick (body frame, u = 0.035 m = the mesh's plan unit; x = long axis):
  brickN_collision  main box, tab base .. slot back wall, studs included in its height (as the old box)
  brickN_tab_head/_tab_neck   stepped dovetail TAB on the -X end (head 0.5u half-width at the tip,
                              neck 0.25u, 0.25u deep, full brick height)
  brickN_jaw_in_l/r, _jaw_out_l/r   stepped JAWS on the +X end forming the matching SLOT (0.25u
                              neck at the face, 0.5u wide inside, through the full height)
The head is wider than the neck, so a neighbour's tab can only be lowered in from above, and
once in, the shape blocks sliding apart. Clearances are looser than the visual mesh (~1 mm) so a
human can line it up: LAT_CLEAR per side laterally, AX_CLEAR axially, FACE_GAP between the mated
main-body face and the jaws' outer face (a coplanar face + friction 3 made the tab creep in).
conaffinity is derived from the brick count (1 | every brick's contype bit), so it grows with the pool.
Also imported by make_brick_pool.py, which owns the brick bodies/welds/keyframe.
Usage: python make_brick_dovetail.py [scene.xml]   (then copy to <MODEL_DIR>)
"""
import re, sys

U = 0.035
W = 0.04375                 # half width incl. the +/-Y ribs (same envelope as the old single box)
LAT_CLEAR = 0.0025
AX_CLEAR = 0.002
FACE_GAP = 0.001
DENSITY = 640               # keeps each brick at ~0.776 kg (the old single box at density 600)


def f(x):
    return f"{x:.6f}".rstrip("0").rstrip(".")


def box(name, cx, cy, cz, hx, hy, hz, ct, aff):
    return (f'      <geom name="{name}" type="box" size="{f(hx)} {f(hy)} {f(hz)}" pos="{f(cx)} {f(cy)} {f(cz)}" '
            f'density="{DENSITY}" group="3" contype="{ct}" conaffinity="{aff}" friction="3 0.5 0.3" solref="0.005 1"/>')


def affinity(n_bricks):
    """1 (robot/table/floor) | one contype bit (2^k) per brick."""
    return 1 | sum(1 << k for k in range(1, n_bricks + 1))


def geoms(k, ct, aff=63):
    n = f"brick{k}"
    xb = 2.25 * U - AX_CLEAR            # slot back wall
    xi = 2.375 * U + AX_CLEAR           # step between the slot's wide inside and its narrow neck
    xo = 2.5 * U - FACE_GAP             # jaws' outer face (just short of the +X face)
    x0 = -2.5 * U                       # tab base
    yi, yo = 0.5 * U + LAT_CLEAR, 0.25 * U + LAT_CLEAR
    out = [box(f"{n}_collision", (x0 + xb) / 2, 0, 0.00525, (xb - x0) / 2, W, 0.04025, ct, aff)]
    for s, tag in ((1, "l"), (-1, "r")):
        out.append(box(f"{n}_jaw_in_{tag}", (xb + xi) / 2, s * (yi + W) / 2, 0, (xi - xb) / 2, (W - yi) / 2, 0.035, ct, aff))
        out.append(box(f"{n}_jaw_out_{tag}", (xi + xo) / 2, s * (yo + W) / 2, 0, (xo - xi) / 2, (W - yo) / 2, 0.035, ct, aff))
    out.append(box(f"{n}_tab_head", (-2.75 * U - 2.625 * U) / 2, 0, 0, 0.125 * U / 2, 0.5 * U, 0.035, ct, aff))
    out.append(box(f"{n}_tab_neck", (-2.625 * U - 2.5 * U) / 2, 0, 0, 0.125 * U / 2, 0.25 * U, 0.035, ct, aff))
    return "\n".join(out)


def main(path):
    s = open(path).read()
    ks = sorted({int(k) for k in re.findall(r'name="brick(\d+)_collision"', s)})
    assert ks, "no brick collision geoms found"
    aff = affinity(len(ks))
    for k in ks:
        pat = re.compile(rf'(?:      <geom name="brick{k}_(?:collision|jaw_\w+|tab_\w+)"[^>]*?/>\n?)+', re.S)
        mo = pat.search(s)
        ct = re.search(r'contype="(\d+)"', mo.group(0)).group(1)
        s = s[:mo.start()] + geoms(k, ct, aff) + "\n" + s[mo.end():]
    open(path, "w").write(s)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scene_fixed_table.xml")
