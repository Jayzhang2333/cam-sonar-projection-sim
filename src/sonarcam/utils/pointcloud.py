import numpy as np

def depth_to_points(depth_m, fx, fy, cx, cy, downsample=2):
    """
    Back-project a depth map (meters) to 3D points in the CAMERA frame.
    Returns:
      pts_c: Nx3 float32
      uv:    Nx2 float32 (pixel coords in original resolution)
    """
    D = depth_m
    H, W = D.shape[:2]
    ds = max(1, int(downsample))

    ys = np.arange(0, H, ds)
    xs = np.arange(0, W, ds)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = D[grid_y, grid_x].astype(np.float32)
    mask = np.isfinite(z) & (z > 0.0)
    if not np.any(mask):
        return np.zeros((0,3), np.float32), np.zeros((0,2), np.float32)

    u = grid_x[mask].astype(np.float32)
    v = grid_y[mask].astype(np.float32)

    X = (u - cx) / fx * z[mask]
    Y = (v - cy) / fy * z[mask]   # +Y down in camera; that's fine in camera frame
    Z = z[mask]

    pts = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    uv  = np.stack([u, v], axis=-1).astype(np.float32)
    return pts, uv
