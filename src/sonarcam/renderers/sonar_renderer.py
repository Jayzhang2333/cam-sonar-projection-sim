import numpy as np
from ..scene.objects import make_tris, transform_pts

def _sample_surface_points(shape, size, position, rpy, n_per_face=25):
    """Jittered barycentric sampling on each triangle (≈5x5 per face)."""
    V, F = make_tris(shape, size)
    Vw = transform_pts(V, position, rpy)
    pts = []
    root = int(max(3, np.sqrt(n_per_face)))
    rng = np.random.RandomState(42)
    for f in F:
        a, b, c = Vw[f]
        for i in range(root):
            for j in range(root):
                u = (i + 0.3 + 0.4 * rng.rand()) / root
                v = (j + 0.6 + 0.3 * rng.rand()) / root
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                p = a * u + b * v + c * (1 - u - v)
                pts.append(p)
    return np.asarray(pts, dtype=np.float32)

def render_sonar(scene, T_world_sonar, sonar):
    start = np.deg2rad(sonar.sector_start_deg)
    end   = np.deg2rad(sonar.sector_end_deg)
    span = (end - start) % (2 * np.pi)
    if span == 0: span = 2 * np.pi
    if end < start: end = start + span
    azis = start + (np.arange(sonar.num_beams) + 0.5) * (span / sonar.num_beams)  # beam centers

    polar = np.zeros((sonar.num_bins, sonar.num_beams), dtype=np.float32)

    # nearest per beam
    nearest = np.full(sonar.num_beams, np.inf, dtype=np.float32)
    nearest_world = np.full((sonar.num_beams, 3), np.nan, dtype=np.float32)

    # accumulate samples from all objects
    for obj in scene.objects:
        P = _sample_surface_points(obj.shape, obj.size, obj.position, obj.rpy, n_per_face=25)  # faster
        Ps = (np.c_[P, np.ones(len(P))] @ T_world_sonar.T)[:, :3]  # world->sonar (row-vector convention)
        xs, ys, zs = Ps[:, 0], Ps[:, 1], Ps[:, 2]
        rng = np.sqrt(xs * xs + ys * ys + zs * zs)
        az  = np.arctan2(ys, xs)

        az_norm = (az - start) % (2 * np.pi)
        in_sector = az_norm < span
        in_range = (rng >= sonar.range_min) & (rng <= sonar.range_max)
        valid = in_sector & in_range
        if not np.any(valid):
            continue

        vv = np.where(valid)[0]
        bins  = ((rng[vv] - sonar.range_min) / (sonar.range_max - sonar.range_min + 1e-9) * (sonar.num_bins - 1)).astype(int)
        beams = ((az_norm[vv] / span) * sonar.num_beams).astype(int) % sonar.num_beams

        # Vectorized "any return" marking (255)
        np.maximum.at(polar, (bins, beams), 255.0)

        # Nearest per beam (loop over unique beams only)
        uniq_beams = np.unique(beams)
        for m in uniq_beams:
            sel = (beams == m)
            if not np.any(sel): 
                continue
            idx_local = vv[sel]
            # find local min range
            j = np.argmin(rng[idx_local])
            k = idx_local[j]
            if rng[k] < nearest[m]:
                nearest[m] = rng[k]
                nearest_world[m] = P[k]  # world point corresponding to this sample

    hit_pts_world = nearest_world[np.isfinite(nearest)]
    return polar, hit_pts_world, nearest, azis
