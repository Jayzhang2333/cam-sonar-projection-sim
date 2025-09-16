from PyQt5 import QtWidgets, QtCore
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class WorldView3D(QtWidgets.QWidget):
    """
    Matplotlib 3D viewer (axes + FOV volumes only).
    Default view:
      - Viewer looks roughly along +X_cam
      - Screen horizontal = Z_cam, screen vertical = +Y_cam (inverted so +Y goes DOWN)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Figure with zero margins
        self.fig = Figure(figsize=(4, 4), facecolor="white")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.set_position([0, 0, 1, 1])

        # Toolbar + Reset
        topbar_w = QtWidgets.QWidget()
        topbar = QtWidgets.QHBoxLayout(topbar_w)
        topbar.setContentsMargins(0, 0, 0, 0)
        topbar.setSpacing(4)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet("QToolBar { margin:0px; padding:0px; spacing:0px; }")
        self.toolbar.setIconSize(QtCore.QSize(14, 14))

        self.btn_reset = QtWidgets.QToolButton()
        self.btn_reset.setText("Reset")
        self.btn_reset.setToolTip("Reset to default camera-aligned view")
        self.btn_reset.setAutoRaise(True)
        self.btn_reset.clicked.connect(self.reset_view)

        topbar.addWidget(self.toolbar); topbar.addStretch(1); topbar.addWidget(self.btn_reset)
        layout.addWidget(topbar_w)
        layout.addWidget(self.canvas)

        # Visual tuning
        self.axis_len_frac = 0.28
        self.axis_len_min = 0.3
        self.axis_linewidth = 3.0
        self.axis_fontsize = 11
        self.ax.tick_params(pad=1, labelsize=7)

        # Zoom handlers
        self._zoom_factor = 0.9
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('key_press_event', self._on_key)

        self._last_span = 3.0

    def reset_view(self):
        self._apply_cam_level_view()
        self._ensure_y_down()
        L = self._last_span
        self.ax.set_xlim(-L, L); self.ax.set_ylim(L, -L); self.ax.set_zlim(-L, L)
        self.canvas.draw_idle()

    def update_scene(self, K, image_size, T_cam_from_sonar, sonar):
        fx, fy, cx, cy = K
        W, H = image_size

        self.ax.clear()
        self.ax.tick_params(pad=1, labelsize=7)
        try: self.ax.set_box_aspect((1, 1, 1))
        except Exception: pass

        # Scene scale from sonar max range
        rmax = float(sonar.range_max)
        span = max(rmax, 2.0)
        self._last_span = span
        axis_len = max(self.axis_len_min, self.axis_len_frac * span)

        # Camera axes (origin)
        self._draw_axes(np.eye(4), length=axis_len, colors=('r', 'g', 'b'),
                        labels=('X', 'Y↓', 'Z'),
                        linewidth=self.axis_linewidth, fontsize=self.axis_fontsize)

        # Camera frustum
        self._draw_camera_frustum(fx, fy, cx, cy, W, H, depth=span*0.9, alpha=0.15,
                                  facecolor=(0.2, 0.6, 1.0), edgecolor=(0.2, 0.6, 1.0))

        # Sonar axes and FOV (centered on +X_s)
        T_sonar_in_cam = T_cam_from_sonar
        self._draw_axes(T_sonar_in_cam, length=axis_len, colors=('m', 'c', 'y'),
                        labels=('X_s', 'Y_s', 'Z_s'),
                        linewidth=self.axis_linewidth, fontsize=self.axis_fontsize)

        self._draw_sonar_fov(T_sonar_in_cam, sonar, alpha=0.18,
                             facecolor=(1.0, 0.5, 0.2), edgecolor=(1.0, 0.5, 0.2))

        # Limits + camera-aligned view
        L = span
        self.ax.set_xlim(-L, L); self.ax.set_ylim(-L, L); self.ax.set_zlim(-L, L)
        self._apply_cam_level_view(); self._ensure_y_down()
        self.ax.set_xlabel('X (cam)', labelpad=1)
        self.ax.set_ylabel('Y↓ (cam)', labelpad=1)
        self.ax.set_zlabel('Z (cam)', labelpad=1)
        self.canvas.draw_idle()

    # ---------- helpers ----------
    def _apply_cam_level_view(self):
        self.ax.view_init(elev=0, azim=-90)

    def _ensure_y_down(self):
        y0, y1 = self.ax.get_ylim()
        if y1 > y0:
            self.ax.set_ylim(y1, y0)

    def _draw_axes(self, T, length=1.0, colors=('r', 'g', 'b'), labels=('X', 'Y', 'Z'),
                   linewidth=2.0, fontsize=10):
        o = T[:3, 3]; R = T[:3, :3]
        axes = np.eye(3)
        for i in range(3):
            d = R @ (axes[:, i] * length)
            p = o + d
            self.ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]], color=colors[i], linewidth=linewidth)
            self.ax.text(p[0], p[1], p[2], labels[i], color=colors[i], fontsize=fontsize, zorder=10)

    def _draw_camera_frustum(self, fx, fy, cx, cy, W, H, depth=3.0, alpha=0.15, facecolor=(0.2,0.6,1.0), edgecolor=(0.2,0.6,1.0)):
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=float)
        rays = []
        for (u, v) in corners:
            x = (u - cx) / fx
            y = (v - cy) / fy
            rays.append(np.array([x, y, 1.0], dtype=float))
        rays = np.stack(rays, axis=0)
        far_pts = depth * rays  # 4x3

        # Faces
        o = np.zeros(3)
        verts = [
            [tuple(o), tuple(far_pts[0]), tuple(far_pts[1])],
            [tuple(o), tuple(far_pts[1]), tuple(far_pts[2])],
            [tuple(o), tuple(far_pts[2]), tuple(far_pts[3])],
            [tuple(o), tuple(far_pts[3]), tuple(far_pts[0])],
            [tuple(far_pts[0]), tuple(far_pts[1]), tuple(far_pts[2]), tuple(far_pts[3])],
        ]
        poly = Poly3DCollection(verts, alpha=alpha, facecolor=facecolor, edgecolor=edgecolor, linewidths=0.6)
        self.ax.add_collection3d(poly)

    def _draw_sonar_fov(self, T_sonar_in_cam, sonar, alpha=0.15, facecolor=(1.0, 0.5, 0.2), edgecolor=(1.0, 0.5, 0.2)):
        """
        Sonar wedge centered on +X_s:
          azimuth sector [a0, a1] around +X_s (0 rad), elevation ±e/2, radius = range_max.
        Sonar frame: +X forward (acoustic axis), +Y right, +Z up.

        NOTE (FIX): Build homogeneous points with w=1 (NOT scaled by r).
        Then Pc = R * (r * dir) + t, so the vector from sonar origin (t) to Pc is R*(r*dir),
        whose Euclidean norm is exactly r because R is orthonormal.
        """
        r = float(sonar.range_max)
        elev = np.deg2rad(float(sonar.elevation_fov_deg)) * 0.5
        a0 = np.deg2rad(float(sonar.sector_start_deg))
        a1 = np.deg2rad(float(sonar.sector_end_deg))
        span = (a1 - a0) % (2*np.pi)
        if span == 0: span = 2*np.pi
        if a1 < a0: a1 = a0 + span

        na, nb = 60, 12
        alphas = np.linspace(a0, a1, na)     # azimuth sweep (0 rad is +X_s)
        betas  = np.linspace(-elev, elev, nb)  # elevation sweep

        # Convenience
        def sonar_dir(a, b):
            # Unit direction in sonar frame: +X forward, +Y right, +Z up
            xs = np.cos(b) * np.cos(a)
            ys = np.cos(b) * np.sin(a)
            zs = np.sin(b)
            return np.array([xs, ys, zs], dtype=float)

        # Sonar origin in camera frame
        o = T_sonar_in_cam[:3, 3]

        # Far surface mesh at radius r (correct homogeneous form)
        surf_pts = []
        for a in alphas:
            col = []
            for b in betas:
                d_s = sonar_dir(a, b)             # unit
                p_s = r * d_s                      # length r in SONAR frame
                p_c = (T_sonar_in_cam @ np.array([p_s[0], p_s[1], p_s[2], 1.0]))[:3]  # w=1
                col.append(tuple(p_c.tolist()))
            surf_pts.append(col)

        # Far surface quads
        quads = []
        for i in range(na - 1):
            for j in range(nb - 1):
                q = [surf_pts[i][j], surf_pts[i+1][j], surf_pts[i+1][j+1], surf_pts[i][j+1]]
                quads.append(q)
        poly = Poly3DCollection(quads, alpha=alpha, facecolor=facecolor, edgecolor=edgecolor, linewidths=0.3)
        self.ax.add_collection3d(poly)

        # Radial context lines (origin -> far points), all should be length ~r
        lines = []
        for a in np.linspace(a0, a1, 9):
            for b in (-elev, 0.0, elev):
                d_s = sonar_dir(a, b)
                p_s = r * d_s
                p_c = (T_sonar_in_cam @ np.array([p_s[0], p_s[1], p_s[2], 1.0]))[:3]
                lines.append([tuple(o.tolist()), tuple(p_c.tolist())])
        lc = Line3DCollection(lines, colors=[edgecolor], linewidths=0.6, alpha=min(1.0, alpha+0.1))
        self.ax.add_collection3d(lc)

        # (Optional) Sanity check: max | ||p_c - o|| - r |
        # Uncomment if you want a quick console check:
        # import numpy as _np
        # diffs = [_np.linalg.norm(_np.array(q[1]) - o) - r for q in lines]
        # print("max radial length error:", max(abs(x) for x in diffs))


    # ---- Zoom / keys ----
    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        factor = self._zoom_factor if event.button == 'up' else (1.0 / self._zoom_factor)
        self._zoom_axes(factor)

    def _on_key(self, event):
        if event.key in ('+', '='): self._zoom_axes(self._zoom_factor)
        elif event.key in ('-', '_'): self._zoom_axes(1.0 / self._zoom_factor)
        elif event.key in ('r', 'R'): self.reset_view()

    def _zoom_axes(self, factor):
        def scale(lim):
            c = 0.5 * (lim[0] + lim[1])
            r0 = 0.5 * (lim[1] - lim[0])
            r1 = r0 * factor
            normal = lim[1] > lim[0]
            return (c - r1, c + r1) if normal else (c + r1, c - r1)
        self.ax.set_xlim(scale(self.ax.get_xlim()))
        self.ax.set_ylim(scale(self.ax.get_ylim()))
        self.ax.set_zlim(scale(self.ax.get_zlim()))
        self.canvas.draw_idle()
