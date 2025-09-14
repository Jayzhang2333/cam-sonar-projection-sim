import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _unwrap_span(azis_rad):
    start = float(azis_rad[0]); end = float(azis_rad[-1])
    if end < start: end += 2*np.pi
    span = end - start
    if span <= 0: span = 2*np.pi; end = start + span
    center = 0.5 * (start + end)
    return start, end, span, center

def polar_to_fan_image(
    polar, rmin, rmax, azis_rad, *,
    out_size=360, grid_rings=6, angle_step_deg=15, show_labels=True,
    top_pad_px=2, side_pad_px=2, bottom_pad_px=4
):
    """
    Centered fan:
      - origin bottom-center; wedge symmetric about vertical centerline
      - outside wedge: WHITE; inside: BLACK; echoes: WHITE; grid+labels: RED
      - auto-fit radius; crop top & sides to wedge bbox with small paddings
    """
    H, W = polar.shape
    img = np.full((out_size, out_size, 3), 255, dtype=np.uint8)

    # Geometry
    margin = 6
    cx = out_size // 2
    cy = out_size - margin

    start, end, span, center = _unwrap_span(azis_rad)
    half = 0.5 * span

    max_r_vert = out_size - 2 * margin
    s = max(np.sin(half), 1e-6)
    max_r_horiz = (out_size / 2 - margin) / s
    R = int(np.floor(max(1.0, min(max_r_vert, max_r_horiz))))

    # Inside-wedge mask
    Y, X = np.indices((out_size, out_size))
    dx = X - cx
    dy = cy - Y
    phi_rel = np.arctan2(dx, dy)  # 0 along vertical up
    inside_angle = (phi_rel >= -half) & (phi_rel <= +half)
    rad = np.sqrt(dx * dx + dy * dy)
    inside_radius = rad <= R
    mask = inside_angle & inside_radius

    # Fill inside wedge black
    img[mask] = 0

    # Vectorized echo plotting
    bb, mm = np.nonzero(polar > 0)
    if bb.size:
        rs = rmin + (rmax - rmin) * (bb / max(1, H - 1))
        az = azis_rad[mm]
        phi = (az - center + np.pi) % (2 * np.pi) - np.pi
        good = (phi >= -half) & (phi <= +half)
        if np.any(good):
            rs = rs[good]; phi = phi[good]
            rr = (rs - rmin) / (rmax - rmin + 1e-12) * R
            xs = (cx + rr * np.sin(phi)).round().astype(int)
            ys = (cy - rr * np.cos(phi)).round().astype(int)
            keep = (xs >= 0) & (xs < out_size) & (ys >= 0) & (ys < out_size)
            xs = xs[keep]; ys = ys[keep]
            img[ys, xs, :] = 255

    red = np.array([255, 0, 0], dtype=np.uint8)

    # Range rings
    for k in range(1, grid_rings + 1):
        radk = int(round(R * (k / grid_rings)))
        steps = max(120, int(span * radk / 2))
        t = np.linspace(-half, +half, steps + 1)
        xs = (cx + radk * np.sin(t)).round().astype(int)
        ys = (cy - radk * np.cos(t)).round().astype(int)
        keep = (xs >= 0) & (xs < out_size) & (ys >= 0) & (ys < out_size) & mask[ys, xs]
        img[ys[keep], xs[keep], :] = red

    # Angle spokes
    span_deg = np.degrees(span)
    degs = np.arange(-np.floor(span_deg/2), np.floor(span_deg/2) + 0.5, angle_step_deg, dtype=float)
    for deg in degs:
        phi = np.radians(deg)
        if phi < -half or phi > half: continue
        rr = np.arange(0, R + 1)
        xs = (cx + rr * np.sin(phi)).round().astype(int)
        ys = (cy - rr * np.cos(phi)).round().astype(int)
        keep = (xs >= 0) & (xs < out_size) & (ys >= 0) & (ys < out_size) & mask[ys, xs]
        img[ys[keep], xs[keep], :] = red

    # Labels
    if show_labels:
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font_small = ImageFont.load_default()
        except Exception:
            font_small = None

        phi_label = min(half - 0.25, max(0.15, 0.35 * half))
        for k in range(1, grid_rings + 1):
            radk = R * (k / grid_rings)
            r_m = rmin + (k / grid_rings) * (rmax - rmin)
            txt = f"{int(round(r_m))} m" if (rmax - rmin) >= 10 else f"{r_m:.1f} m"
            x = int(round(cx + radk * np.sin(phi_label)))
            y = int(round(cy - radk * np.cos(phi_label)))
            draw.text((x + 6, y - 6), txt, fill=(255, 0, 0), font=font_small)

        rlab = R - 12
        for deg in degs:
            phi = np.radians(deg)
            if phi < -half or phi > half: continue
            txt = f"{int(round(deg))}°"
            x = int(round(cx + rlab * np.sin(phi)))
            y = int(round(cy - rlab * np.cos(phi)))
            t_sign = -1 if deg < 0 else (1 if deg > 0 else 0)
            draw.text((x + 4 * t_sign, y - 6), txt, fill=(255, 0, 0), font=font_small)

        img = np.array(pil, dtype=np.uint8)

    # ---- Crop top & sides tightly around wedge, keep a small bottom pad ----
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        top_idx = int(np.argmax(rows))                      # first row with wedge
        bottom_idx = int(np.where(rows)[0][-1])             # last row with wedge
        left_idx = int(np.argmax(cols))
        right_idx = int(np.where(cols)[0][-1])

        top = max(0, top_idx - int(top_pad_px))
        bottom = min(out_size - 1, bottom_idx + int(bottom_pad_px))
        left = max(0, left_idx - int(side_pad_px))
        right = min(out_size - 1, right_idx + int(side_pad_px))

        img = img[top:bottom+1, left:right+1, :]

    return img
