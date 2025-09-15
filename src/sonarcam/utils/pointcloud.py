import numpy as np

def depth_to_points(depth_m: np.ndarray, fx, fy, cx, cy, downsample=2, max_depth_m: float = 100.0):
    """
    Convert depth (meters) to camera-frame points (X right, Y down, Z forward),
    returning:
      pts_c : (N,3) float32
      uv    : (N,2) float32 pixel coords (u=x, v=y)

    Invalid depths (NaN/Inf), <= 0, or > max_depth_m are ignored.
    """
    D = np.asarray(depth_m, dtype=np.float32)
    H, W = D.shape
    ds = max(1, int(downsample))

    # Sample grid
    v = np.arange(0, H, ds, dtype=np.float32)
    u = np.arange(0, W, ds, dtype=np.float32)
    UU, VV = np.meshgrid(u, v)
    Z = D[::ds, ::ds]

    # Valid mask (≤ max depth enforced)
    M = np.isfinite(Z) & (Z > 0.0) & (Z <= float(max_depth_m))
    if not np.any(M):
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32)

    UU = UU[M]; VV = VV[M]; Z = Z[M]

    X = (UU - float(cx)) * Z / float(fx)
    Y = (VV - float(cy)) * Z / float(fy)
    pts = np.stack([X, Y, Z], axis=1).astype(np.float32)
    uv = np.stack([UU, VV], axis=1).astype(np.float32)
    return pts, uv
