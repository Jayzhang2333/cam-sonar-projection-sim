from PyQt5 import QtWidgets
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ProjectionView(QtWidgets.QWidget):
    """
    Matplotlib-based projection plane:
      - Axis in pixels (u: 0..W, v: 0..H), v increasing downward
      - Draws image boundary and projected elevation-ambiguity arcs
      - Annotates per-arc pixel lengths (label placed ABOVE the arc)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        self.fig = Figure(figsize=(5, 4), facecolor="white")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.06, right=0.995, bottom=0.08, top=0.995)
        self.ax.set_aspect('equal', adjustable='box')

        v.addWidget(self.canvas)

        self._W = 640
        self._H = 480
        self.last_lengths = []  # list[float] per sample

    def _setup_axes(self, W, H):
        self.ax.clear()
        self.ax.set_xlim(0, W)
        self.ax.set_ylim(H, 0)  # invert so +v goes downward
        self.ax.set_xlabel("u (px)")
        self.ax.set_ylabel("v (px)")
        # image boundary
        self.ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], color='k', linewidth=1.2)

    @staticmethod
    def _project_point(Pc, fx, fy, cx, cy):
        Z = Pc[2]
        if not np.isfinite(Z) or Z <= 0:
            return np.nan, np.nan, False
        u = fx * (Pc[0] / Z) + cx
        v = fy * (Pc[1] / Z) + cy
        return u, v, True

    @staticmethod
    def _polyline_length(uv):
        uv = np.asarray(uv, dtype=float)
        length = 0.0
        for i in range(1, uv.shape[0]):
            p0 = uv[i - 1]; p1 = uv[i]
            if np.all(np.isfinite(p0)) and np.all(np.isfinite(p1)):
                length += float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        return length

    @staticmethod
    def _topmost_point(uv):
        """
        Return the finite point with minimum v (topmost in image coords).
        """
        uv = np.asarray(uv, dtype=float)
        ok = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        if not np.any(ok):
            return None
        v_vals = uv[ok, 1]
        idx_ok = np.where(ok)[0]
        k = idx_ok[np.argmin(v_vals)]
        return uv[k]

    def _plot_polyline_clip(self, uv, color_rgba, W, H, annotate_len=True, label_offset_px=12.0):
        """
        Plot polyline with clipping to view; place length label ABOVE the arc
        (at the topmost point, offset upward by label_offset_px).
        Returns the measured pixel length.
        """
        uv = np.asarray(uv, dtype=float)
        ok = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        ok &= (uv[:, 0] >= -1) & (uv[:, 0] <= W + 1) & (uv[:, 1] >= -1) & (uv[:, 1] <= H + 1)

        # length from finite points (not the clipped mask)
        finite = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        length = self._polyline_length(uv[finite])

        # contiguous segments for plotting
        idx_bad = np.where(~ok)[0]
        start = 0
        stops = list(idx_bad) + [len(uv)]
        for stop in stops:
            if stop - start >= 2:
                seg = uv[start:stop]
                self.ax.plot(seg[:, 0], seg[:, 1], color=color_rgba, linewidth=2.0)
            start = stop + 1

        # label ABOVE the arc: take the topmost finite point and move up (negative v)
        if annotate_len:
            top = self._topmost_point(uv)
            if top is not None and np.all(np.isfinite(top)):
                label_u = float(np.clip(top[0], 0, W))
                label_v = float(np.clip(top[1] - label_offset_px, 0, H))
                self.ax.text(label_u, label_v, f"{length:.1f}px",
                             color=color_rgba, fontsize=8, ha='center', va='bottom',
                             bbox=dict(facecolor='white', edgecolor='none', alpha=0.70, pad=0.6))

        return length

    def update_projection(self, W, H, K, T_cam_from_sonar, sonar, samples):
        """
        Draw elevation-ambiguity curves for each sonar sample and compute their pixel lengths.

        samples: list of dicts:
           {'az': deg, 'r': meters, 'color': (r,g,b)}
        """
        self._W, self._H = int(W), int(H)
        fx, fy, cx, cy = K

        self._setup_axes(W, H)
        self.last_lengths = []

        elev_half = 0.5 * np.deg2rad(float(sonar.elevation_fov_deg))
        n_el = 64  # smooth arcs

        for s in samples:
            az = float(s.get('az', 0.0))
            r  = float(s.get('r', 1.0))
            col = s.get('color', (0, 200, 255))
            rgba = (col[0]/255.0, col[1]/255.0, col[2]/255.0, 1.0)

            az_rad = np.deg2rad(az)
            el_list = np.linspace(-elev_half, +elev_half, n_el, dtype=float)

            # sonar frame directions: X fwd, Y right, Z up
            dirs_s = np.stack([
                np.cos(el_list) * np.cos(az_rad),
                np.cos(el_list) * np.sin(az_rad),
                np.sin(el_list)
            ], axis=1)  # (n_el, 3)

            # points at distance r -> camera frame
            Ps = r * dirs_s
            Ps_h = np.concatenate([Ps, np.ones((Ps.shape[0], 1), dtype=float)], axis=1)
            Pc = (T_cam_from_sonar @ Ps_h.T).T[:, :3]

            # project
            uv = np.empty((Pc.shape[0], 2), dtype=float); uv[:] = np.nan
            for i in range(Pc.shape[0]):
                u, v, ok = self._project_point(Pc[i], fx, fy, cx, cy)
                if ok: uv[i] = (u, v)

            # draw & label (label placed ABOVE arc)
            L = self._plot_polyline_clip(uv, rgba, W, H, annotate_len=True, label_offset_px=12.0)
            self.last_lengths.append(L)

        self.canvas.draw_idle()
