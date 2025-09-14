import numpy as np
from ..scene.objects import make_tris, transform_pts

def _sample_surface_points(shape, size, position, rpy, n_per_face=49):
    """Dense-ish barycentric sampling on each triangle (≈7x7 per face)."""
    V, F = make_tris(shape, size)
    Vw = transform_pts(V, position, rpy)
    pts = []
    root = int(max(4, np.sqrt(n_per_face)))
    # stratified samples in triangle using (u,v,w) with u+v<=1, w=1-u-v
    for f in F:
        a, b, c = Vw[f]
        for i in range(root):
            for j in range(root):
                # jittered grid -> barycentric
                u = (i + 0.35) / root
                v = (j + 0.62) / root
                if u + v > 1.0:
                    u, v = 1.0 - u, 1.0 - v
                w = 1.0 - u - v
                p = a * u + b * v + c * w
                pts.append(p)
    return np.asarray(pts, dtype=np.float32)

def _project_points_cam(Xc, fx, fy, cx, cy, W, H):
    """Pinhole projection with camera axes: +Z forward, +X right, +Y down."""
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    valid = z > 1e-6
    u = fx * (x / (z + 1e-12)) + cx
    v = fy * (y / (z + 1e-12)) + cy  # +Y down maps to increasing v
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.where(valid & inside)[0]
    return idx, u[idx].astype(np.int32), v[idx].astype(np.int32), z[idx]

def render_camera(scene, T_world_cam, K_tuple, width, height):
    """
    Simple point-based renderer with z-buffer on a white background.
    T_world_cam: world->camera (row-vector convention)
    K_tuple: (fx, fy, cx, cy)
    Returns: (rgb_img, depth, xyz_cam_of_drawn_points)
    """
    fx, fy, cx, cy = K_tuple
    # white background
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)

    all_pts_c = []  # optional return of points that actually got drawn

    for obj in scene.objects:
        P = _sample_surface_points(obj.shape, obj.size, obj.position, obj.rpy, n_per_face=49)
        # world -> camera (row-vector convention: [x y z 1] @ T^T)
        Pc = (np.c_[P, np.ones(len(P), dtype=np.float32)] @ T_world_cam.T)[:, :3]
        idx, uu, vv, zz = _project_points_cam(Pc, fx, fy, cx, cy, width, height)
        if idx.size == 0:
            continue

        # z-buffer update
        # flatten 2D index
        lin = vv * width + uu
        # for speed: group by linear index and keep nearest z
        # sort by lin then z
        order = np.argsort(lin)
        lin_s = lin[order]; z_s = zz[order]
        # first occurrence for each pixel in sorted unique gives nearest due to later pass overriding,
        # but we want explicit min per pixel:
        unique_lin, first_idx = np.unique(lin_s, return_index=True)
        # for each unique pixel, find min z among its run
        # compute run lengths
        counts = np.diff(np.append(first_idx, len(lin_s)))
        ptr = 0
        for k, count in enumerate(counts):
            run_z = z_s[ptr:ptr+count]
            # nearest in this run
            j = np.argmin(run_z)
            pix = unique_lin[k]
            y = pix // width; x = pix % width
            zval = run_z[j]
            if zval < depth[y, x]:
                depth[y, x] = zval
                img[y, x, :] = np.array(obj.color, dtype=np.uint8)
            ptr += count

        all_pts_c.append(Pc[idx])

    pts_c = np.concatenate(all_pts_c, axis=0) if all_pts_c else np.zeros((0, 3), dtype=np.float32)
    return img, depth, pts_c
