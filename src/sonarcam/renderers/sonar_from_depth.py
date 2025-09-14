import numpy as np
from ..models.sonar import ScanOrientation

def render_sonar_from_depth(pts_c, uv, sonar, T_cam_to_sonar):
    """
    Render sonar polar image from a CAMERA-frame point cloud (pts_c).
    Sonar frame convention (as requested):
      X = acoustic axis (forward), Y = left, Z = up.
    The user-provided extrinsics are sonar pose in camera frame (sonar->camera).
    Here we receive camera->sonar transform, T_cam_to_sonar, to convert points.

    Returns: polar (num_bins x num_beams) uint8-like, used_mask (N,), angle_grid (beams,)
    """
    if pts_c.size == 0:
        return np.zeros((sonar.num_bins, sonar.num_beams), np.float32), np.zeros((0,), bool), np.linspace(0,1,sonar.num_beams)

    # Transform CAMERA points -> SONAR frame (row-vector convention)
    ones = np.ones((pts_c.shape[0], 1), dtype=np.float32)
    Ps = np.hstack([pts_c, ones]) @ T_cam_to_sonar.T
    Xs, Ys, Zs = Ps[:,0], Ps[:,1], Ps[:,2]

    # Range
    rng = np.sqrt(Xs*Xs + Ys*Ys + Zs*Zs)

    # Angles per sonar orientation
    # Horizontal scan: angle = azimuth around Z, from +X toward +Y (left), angle = atan2(Y, X)
    # Vertical scan: angle = elevation, angle = atan2(Z, sqrt(X^2+Y^2))
    if sonar.orientation == ScanOrientation.horizontal:
        ang = np.arctan2(Ys, Xs)
        gate = np.abs(np.arctan2(Zs, np.sqrt(Xs*Xs + Ys*Ys))) <= np.deg2rad(sonar.elevation_fov_deg) * 0.5
    else:  # vertical
        ang = np.arctan2(Zs, np.sqrt(Xs*Xs + Ys*Ys))
        gate = np.abs(np.arctan2(Ys, Xs)) <= np.deg2rad(sonar.elevation_fov_deg) * 0.5

    # Sector gating
    a0 = np.deg2rad(sonar.sector_start_deg)
    a1 = np.deg2rad(sonar.sector_end_deg)
    # unwrap into increasing interval
    span = (a1 - a0) % (2*np.pi)
    if span == 0: span = 2*np.pi
    if a1 < a0: a1 = a0 + span

    # normalize angle into [a0, a1]
    ang_norm = (ang - a0) % (2*np.pi)
    in_sector = ang_norm <= span

    # Range gating
    in_range = (rng >= sonar.range_min) & (rng <= sonar.range_max)

    valid = gate & in_sector & in_range & np.isfinite(rng)
    if not np.any(valid):
        angle_grid = a0 + (np.arange(sonar.num_beams)+0.5) * (span/sonar.num_beams)
        return np.zeros((sonar.num_bins, sonar.num_beams), np.float32), np.zeros_like(valid), angle_grid

    # Bin indices
    bins  = ((rng - sonar.range_min) / (sonar.range_max - sonar.range_min + 1e-12) * (sonar.num_bins-1)).astype(np.int32)
    beams = ((ang_norm / span) * sonar.num_beams).astype(np.int32) % sonar.num_beams

    # Keep only valid ones
    idx = np.where(valid)[0]
    bins, beams = bins[idx], beams[idx]

    # Polar accumulation (presence)
    polar = np.zeros((sonar.num_bins, sonar.num_beams), dtype=np.float32)
    np.maximum.at(polar, (bins, beams), 255.0)

    # Used points mask (for overlay)
    used = np.zeros(pts_c.shape[0], dtype=bool)
    used[idx] = True

    # Angle grid for fan visualization
    angle_grid = a0 + (np.arange(sonar.num_beams)+0.5) * (span/sonar.num_beams)
    return polar, used, angle_grid
