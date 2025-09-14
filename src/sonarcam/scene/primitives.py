import numpy as np

def make_ground(size=20.0, y_divs=1, x_divs=1, z=0.0):
    half = size * 0.5
    v0 = np.array([-half, -half, z], dtype=np.float32)
    v1 = np.array([ half, -half, z], dtype=np.float32)
    v2 = np.array([ half,  half, z], dtype=np.float32)
    v3 = np.array([-half,  half, z], dtype=np.float32)
    tris = np.stack([
        np.stack([v0, v1, v2], axis=0),
        np.stack([v0, v2, v3], axis=0),
    ], axis=0)
    colors = np.array([[120, 170, 120], [100, 150, 100]], dtype=np.uint8)
    return tris, colors

def make_box(center, size, color=(200, 80, 60)):
    cx, cy, cz = center
    sx, sy, sz = (size, size, size) if np.isscalar(size) else size
    x0, x1 = cx - sx/2, cx + sx/2
    y0, y1 = cy - sy/2, cy + sy/2
    z0, z1 = cz - sz/2, cz + sz/2
    V = np.array([
        [x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0],
        [x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1],
    ], dtype=np.float32)
    F = [
        (0,1,2), (0,2,3),
        (4,6,5), (4,7,6),
        (0,4,5), (0,5,1),
        (1,5,6), (1,6,2),
        (2,6,7), (2,7,3),
        (3,7,4), (3,4,0),
    ]
    tris = np.stack([V[list(f)] for f in F], axis=0)
    colors = np.tile(np.array(color, dtype=np.uint8), (len(F),1))
    return tris, colors
