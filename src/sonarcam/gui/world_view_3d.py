from PyQt5 import QtWidgets
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..utils.transforms import inv_T
from ..models.sonar import ScanOrientation


class WorldView3D(QtWidgets.QWidget):
    """
    Matplotlib 3D viewer with zoom/pan; trimmed whitespace and larger axes.

    Camera-aligned initial view:
      - Viewer looks roughly along +X_cam
      - Screen vertical = +Y_cam (but inverted so +Y goes DOWN on screen)
      - Screen horizontal = Z_cam
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Figure with zero margins
        self.fig = Figure(figsize=(4, 4), facecolor="white")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_position([0, 0, 1, 1])

        # Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("QToolBar { padding: 0px; spacing: 0px; }")
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Visual tuning
        self.axis_len_frac = 0.28
        self.axis_len_min = 0.3
        self.axis_linewidth = 3.0
        self.axis_fontsize = 11

        # Tight ticks to cut padding
        self.ax.tick_params(pad=1, labelsize=7)

        # Zoom handlers
        self._zoom_factor = 0.9
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('key_press_event', self._on_key)

    # ---- Public API ----
    def update_scene(self, pts_c, T_cam_to_sonar, sonar):
        self.ax.clear()
        self.ax.tick_params(pad=1, labelsize=7)

        # Equal aspect & minimal margins
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        # Downsample for speed
        if pts_c.shape[0] > 60000:
            step = int(np.ceil(pts_c.shape[0] / 60000))
            P = pts_c[::step]
        else:
            P = pts_c

        span = self._scene_span(P, fallback=float(sonar.range_max))
        axis_len = max(self.axis_len_min, self.axis_len_frac * span)

        if P.size:
            self.ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.4, c='k', alpha=0.4, depthshade=False)

        # Camera axes
        self._draw_axes(np.eye(4), length=axis_len, colors=('r', 'g', 'b'),
                        labels=('X', 'Y', 'Z'),
                        linewidth=self.axis_linewidth, fontsize=self.axis_fontsize)

        # Sonar axes
        T_sonar_in_cam = inv_T(T_cam_to_sonar)
        self._draw_axes(T_sonar_in_cam, length=axis_len, colors=('m', 'c', 'y'),
                        labels=('X_s', 'Y_s', 'Z_s'),
                        linewidth=self.axis_linewidth, fontsize=self.axis_fontsize)

        # Sonar FOV
        self._draw_sonar_fov(T_sonar_in_cam, sonar, center_axis_len=axis_len)

        # Tight limits
        xlim, ylim, zlim = self._auto_limits(P, default=max(2.0, 0.8 * span))
        self.ax.set_xlim(xlim); self.ax.set_ylim(ylim); self.ax.set_zlim(zlim)

        # Apply camera-level view:
        #   - look ~ along +X (so screen plane is Y–Z)
        #   - invert Y so +Y points DOWN on screen (camera convention)
        self._apply_cam_level_view()
        self._ensure_y_down()

        self.ax.set_xlabel('X (cam)', labelpad=1)
        self.ax.set_ylabel('Y↓ (cam)', labelpad=1)   # make it explicit
        self.ax.set_zlabel('Z (cam)', labelpad=1)
        self.canvas.draw_idle()

    # ---- Internals ----
    def _apply_cam_level_view(self):
        """
        View so that:
          - screen vertical is +Y_cam (we'll flip it to go down),
          - screen horizontal is Z_cam,
          - viewing direction ~ +X_cam.
        In mpl, elev=0, azim=-90 looks toward +X.
        """
        self.ax.view_init(elev=0, azim=-90)

    def _ensure_y_down(self):
        """Invert Y axis if needed so +Y goes downward on screen."""
        y0, y1 = self.ax.get_ylim()
        if y1 > y0:  # normal (upwards)
            self.ax.set_ylim(y1, y0)  # flip so +Y is down

    def _scene_span(self, P, fallback=2.0):
        if P is None or P.size == 0:
            return fallback
        mn = np.nanmin(P, axis=0); mx = np.nanmax(P, axis=0)
        span = float(np.max(mx - mn))
        return max(span, fallback, 1.0)

    def _auto_limits(self, P, default=2.0):
        if P is None or P.size == 0:
            return (-default, default), (-default, default), (-default, default)
        mn = np.min(P, axis=0); mx = np.max(P, axis=0)
        span = float(np.max(mx - mn)); span = max(span, 1.0)
        c = 0.5 * (mn + mx); L = 0.6 * span
        return (c[0]-L, c[0]+L), (c[1]-L, c[1]+L), (c[2]-L, c[2]+L)

    def _draw_axes(self, T, length=1.0, colors=('r', 'g', 'b'), labels=('X', 'Y', 'Z'),
                   linewidth=2.0, fontsize=10):
        o = T[:3, 3]; R = T[:3, :3]
        axes = np.eye(3)
        for i in range(3):
            d = R @ (axes[:, i] * length)
            p = o + d
            self.ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]], color=colors[i], linewidth=linewidth)
            self.ax.text(p[0], p[1], p[2], labels[i], color=colors[i], fontsize=fontsize, zorder=10)

    def _draw_sonar_fov(self, T_sonar_in_cam, sonar, center_axis_len=1.0):
        r = float(sonar.range_max)
        elev = np.deg2rad(float(sonar.elevation_fov_deg)) * 0.5
        a0 = np.deg2rad(float(sonar.sector_start_deg))
        a1 = np.deg2rad(float(sonar.sector_end_deg))
        span = (a1 - a0) % (2*np.pi)
        if span == 0: span = 2*np.pi
        if a1 < a0: a1 = a0 + span

        ori = getattr(sonar, "orientation", ScanOrientation.horizontal)
        is_horizontal = (ori.lower().startswith('h') if isinstance(ori, str) else (ori == ScanOrientation.horizontal))

        na, nb = 40, 10
        alphas = np.linspace(a0, a1, na)
        betas  = np.linspace(-elev, elev, nb)

        verts = []
        if is_horizontal:
            for i in range(na-1):
                for j in range(nb-1):
                    aL, aR = alphas[i], alphas[i+1]
                    bB, bT = betas[j],  betas[j+1]
                    pts = []
                    for a, b in [(aL,bB),(aR,bB),(aR,bT),(aL,bT)]:
                        xs = np.cos(b)*np.cos(a); ys = np.cos(b)*np.sin(a); zs = np.sin(b)
                        ps = r * np.array([xs, ys, zs, 1.0], dtype=float)
                        pc = (T_sonar_in_cam @ ps)[:3]; pts.append(tuple(pc.tolist()))
                    verts.append(pts)
        else:
            for i in range(na-1):
                for j in range(nb-1):
                    bB, bT = alphas[i], alphas[i+1]
                    aL, aR = -elev, +elev
                    pts = []
                    for a, b in [(aL,bB),(aR,bB),(aR,bT),(aL,bT)]:
                        xs = np.cos(b)*np.cos(a); ys = np.cos(b)*np.sin(a); zs = np.sin(b)
                        ps = r * np.array([xs, ys, zs, 1.0], dtype=float)
                        pc = (T_sonar_in_cam @ ps)[:3]; pts.append(tuple(pc.tolist()))
                    verts.append(pts)

        poly = Poly3DCollection(verts, alpha=0.15, facecolor=(0.2,0.7,1.0), edgecolor=(0.2,0.7,1.0), linewidths=0.3)
        self.ax.add_collection3d(poly)

        # central acoustic axis (X_s)
        o = T_sonar_in_cam[:3, 3]
        x_dir_cam = T_sonar_in_cam[:3, :3] @ np.array([1, 0, 0], dtype=float)
        p = o + center_axis_len * x_dir_cam / (np.linalg.norm(x_dir_cam) + 1e-12)
        self.ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]], color=(0.2,0.7,1.0), linewidth=self.axis_linewidth + 1.0)

    # ---- Zoom handlers ----
    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        factor = self._zoom_factor if event.button == 'up' else (1.0 / self._zoom_factor)
        self._zoom_axes(factor)

    def _on_key(self, event):
        if event.key in ('+', '='): self._zoom_axes(self._zoom_factor)
        elif event.key in ('-', '_'): self._zoom_axes(1.0 / self._zoom_factor)
        elif event.key in ('r', 'R'):  # reset to camera-level view (Y down)
            self._apply_cam_level_view()
            self._ensure_y_down()
            self.canvas.draw_idle()

    def _zoom_axes(self, factor):
        def scale(lim):
            # Preserve current orientation (including inverted Y)
            c = 0.5 * (lim[0] + lim[1])
            r0 = 0.5 * (lim[1] - lim[0])
            r1 = r0 * factor
            normal_order = lim[1] > lim[0]
            if normal_order:
                return (c - r1, c + r1)
            else:
                return (c + r1, c - r1)
        self.ax.set_xlim(scale(self.ax.get_xlim()))
        self.ax.set_ylim(scale(self.ax.get_ylim()))
        self.ax.set_zlim(scale(self.ax.get_zlim()))
        self.canvas.draw_idle()
