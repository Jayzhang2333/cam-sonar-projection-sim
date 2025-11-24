from PyQt5 import QtWidgets, QtCore
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class WorldView3D(QtWidgets.QWidget):
    """
    Matplotlib 3D viewer (camera + sonar FOV) in WORLD coordinates.

    World frame:
      - (X_w, Y_w, Z_w) is the underlying coordinate system (axes not drawn).

    Camera frame (OpenCV convention):
      - +Z_c : forward (optical axis)
      - +X_c : right
      - +Y_c : down

    Camera pose in world:
      - Camera center at p_cam^w = (1, 1, 1).
      - Orientation:
          Z_cam ∥  X_world   (same direction)
          X_cam ∥ -Y_world   (reversed)
          Y_cam ∥ -Z_world   (reversed)
        i.e.:
          x_cam_w = (0, -1,  0)
          y_cam_w = (0,  0, -1)
          z_cam_w = (1,  0,  0)

    Sonar pose:
      - Given extrinsics T_cam_from_sonar (4x4) mapping sonar->camera,
        sonar pose in world is:
          T_world_from_sonar = T_world_from_cam @ T_cam_from_sonar
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
        self.btn_reset.setToolTip("Reset to default world-view")
        self.btn_reset.setAutoRaise(True)
        self.btn_reset.clicked.connect(self.reset_view)

        topbar.addWidget(self.toolbar)
        topbar.addStretch(1)
        topbar.addWidget(self.btn_reset)
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

    # -------- public API --------
    def reset_view(self):
        self._apply_default_view()
        L = self._last_span
        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_zlim(-L, L)
        self.canvas.draw_idle()

    def update_scene(self, K, image_size, T_cam_from_sonar, sonar):
        """
        K: (fx, fy, cx, cy)
        image_size: (W, H)
        T_cam_from_sonar: 4x4, maps sonar frame -> camera frame
        sonar: SonarModel (has range_max, elevation_fov_deg, sector_start_deg, sector_end_deg)
        """
        fx, fy, cx, cy = K
        W, H = image_size

        self.ax.clear()
        self.ax.tick_params(pad=1, labelsize=7)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        # ---------- Define camera pose in world ----------
        # Camera origin in world:
        t_w_c = np.array([1.0, 1.0, 1.0], dtype=float)

        # Orientation:
        #   Z_cam ∥  X_world
        #   X_cam ∥ -Y_world
        #   Y_cam ∥ -Z_world
        #
        # Columns of R_w_c are camera basis vectors expressed in world coords:
        #   first  column = x_cam in world
        #   second column = y_cam in world
        #   third  column = z_cam in world
        x_cam_w = np.array([0.0, -1.0,  0.0])  # -Y_w
        y_cam_w = np.array([0.0,  0.0, -1.0])  # -Z_w
        z_cam_w = np.array([1.0,  0.0,  0.0])  #  X_w

        R_w_c = np.column_stack([x_cam_w, y_cam_w, z_cam_w])  # 3x3
        T_w_c = np.eye(4, dtype=float)
        T_w_c[:3, :3] = R_w_c
        T_w_c[:3, 3] = t_w_c

        # Sonar pose in world: T_world_from_sonar = T_w_c @ T_cam_from_sonar
        T_w_s = T_w_c @ T_cam_from_sonar

        # ---------- Scene scaling ----------
        rmax = float(sonar.range_max)
        cam_dist = np.linalg.norm(t_w_c)
        span = max(rmax + cam_dist, 2.0)
        self._last_span = span
        axis_len = max(self.axis_len_min, self.axis_len_frac * span)

        # ---------- Draw camera + sonar (world coordinates) ----------
        # Camera axes in world
        self._draw_axes(T_w_c, length=axis_len, colors=('r', 'g', 'b'),
                        labels=('X_c', 'Y_c', 'Z_c'),
                        linewidth=self.axis_linewidth, fontsize=self.axis_fontsize)

        # Camera frustum, in world
        self._draw_camera_frustum(
            fx, fy, cx, cy, W, H,
            depth=span * 0.9,
            T_w_from_c=T_w_c,
            alpha=0.15,
            facecolor=(0.2, 0.6, 1.0),
            edgecolor=(0.2, 0.6, 1.0),
        )

        # Sonar axes and FOV in world
        self._draw_axes(T_w_s, length=axis_len, colors=('m', 'c', 'y'),
                        labels=('X_s', 'Y_s', 'Z_s'),
                        linewidth=self.axis_linewidth, fontsize=self.axis_fontsize)

        self._draw_sonar_fov(
            T_sonar_in_world=T_w_s,
            sonar=sonar,
            alpha=0.18,
            facecolor=(1.0, 0.5, 0.2),
            edgecolor=(1.0, 0.5, 0.2),
        )

        # Limits + default view
        L = span
        self.ax.set_xlim(-L, L)
        self.ax.set_ylim(-L, L)
        self.ax.set_zlim(-L, L)
        self._apply_default_view()

        self.ax.set_xlabel('X (world)', labelpad=1)
        self.ax.set_ylabel('Y (world)', labelpad=1)
        self.ax.set_zlabel('Z (world)', labelpad=1)
        self.canvas.draw_idle()

    # ---------- view helpers ----------
    def _apply_default_view(self):
        """
        Default world-aligned view:
          - Z_w roughly up on screen.
          - X_w/Y_w in the horizontal plane with some azimuth.
        """
        self.ax.view_init(elev=25, azim=-60)

    # ---------- drawing helpers ----------
    def _draw_axes(self, T, length=1.0, colors=('r', 'g', 'b'), labels=('X', 'Y', 'Z'),
                   linewidth=2.0, fontsize=10):
        o = T[:3, 3]
        R = T[:3, :3]
        axes = np.eye(3)
        for i in range(3):
            d = R @ (axes[:, i] * length)
            p = o + d
            self.ax.plot([o[0], p[0]], [o[1], p[1]], [o[2], p[2]],
                         color=colors[i], linewidth=linewidth)
            self.ax.text(p[0], p[1], p[2], labels[i],
                         color=colors[i], fontsize=fontsize, zorder=10)

    def _draw_camera_frustum(self, fx, fy, cx, cy, W, H, depth,
                              T_w_from_c,
                              alpha=0.15, facecolor=(0.2, 0.6, 1.0),
                              edgecolor=(0.2, 0.6, 1.0)):
        """
        Draw the camera frustum in WORLD coordinates.

        K = [fx, fy, cx, cy]
        Image plane in camera frame is at Z_c = 1; we scale rays by 'depth' and
        then transform those 3D points to world via T_w_from_c.
        """
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=float)

        rays_cam = []
        for (u, v) in corners:
            x = (u - cx) / fx
            y = (v - cy) / fy
            # Camera frame: +Z forward, +X right, +Y down
            rays_cam.append(np.array([x, y, 1.0], dtype=float))
        rays_cam = np.stack(rays_cam, axis=0)

        # Far points in camera frame -> world
        far_pts_world = []
        for r_cam in rays_cam:
            p_cam = depth * r_cam
            p_world = (T_w_from_c @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]
            far_pts_world.append(p_world)
        far_pts_world = np.stack(far_pts_world, axis=0)

        # Camera origin in world
        o_world = T_w_from_c[:3, 3]

        verts = [
            [tuple(o_world), tuple(far_pts_world[0]), tuple(far_pts_world[1])],
            [tuple(o_world), tuple(far_pts_world[1]), tuple(far_pts_world[2])],
            [tuple(o_world), tuple(far_pts_world[2]), tuple(far_pts_world[3])],
            [tuple(o_world), tuple(far_pts_world[3]), tuple(far_pts_world[0])],
            [tuple(far_pts_world[0]), tuple(far_pts_world[1]),
             tuple(far_pts_world[2]), tuple(far_pts_world[3])],
        ]
        poly = Poly3DCollection(
            verts,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidths=0.6,
        )
        self.ax.add_collection3d(poly)

    def _draw_sonar_fov(self, T_sonar_in_world, sonar,
                        alpha=0.15, facecolor=(1.0, 0.5, 0.2),
                        edgecolor=(1.0, 0.5, 0.2)):
        """
        Sonar wedge in WORLD coordinates.

        Sonar frame:
          - +X_s : forward (acoustic axis)
          - +Y_s : right
          - +Z_s : up

        T_sonar_in_world: 4x4 mapping sonar frame -> world frame.
        """
        r = float(sonar.range_max)
        elev = np.deg2rad(float(sonar.elevation_fov_deg)) * 0.5
        a0 = np.deg2rad(float(sonar.sector_start_deg))
        a1 = np.deg2rad(float(sonar.sector_end_deg))
        span = (a1 - a0) % (2 * np.pi)
        if span == 0:
            span = 2 * np.pi
        if a1 < a0:
            a1 = a0 + span

        na, nb = 60, 12
        alphas = np.linspace(a0, a1, na)      # azimuth sweep around +X_s
        betas = np.linspace(-elev, elev, nb)  # elevation sweep

        def sonar_dir(a, b):
            # Unit direction in sonar frame: +X forward, +Y right, +Z up
            xs = np.cos(b) * np.cos(a)
            ys = np.cos(b) * np.sin(a)
            zs = np.sin(b)
            return np.array([xs, ys, zs], dtype=float)

        # Sonar origin in world frame
        o_w = T_sonar_in_world[:3, 3]

        # Far surface mesh (radius r) in world
        surf_pts = []
        for a in alphas:
            col = []
            for b in betas:
                d_s = sonar_dir(a, b)   # unit in sonar frame
                p_s = r * d_s          # point at radius r in sonar frame
                p_w = (T_sonar_in_world @ np.array([p_s[0], p_s[1], p_s[2], 1.0]))[:3]
                col.append(tuple(p_w.tolist()))
            surf_pts.append(col)

        # Far surface quads
        quads = []
        for i in range(na - 1):
            for j in range(nb - 1):
                q = [
                    surf_pts[i][j],
                    surf_pts[i + 1][j],
                    surf_pts[i + 1][j + 1],
                    surf_pts[i][j + 1],
                ]
                quads.append(q)
        poly = Poly3DCollection(
            quads,
            alpha=alpha,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidths=0.3,
        )
        self.ax.add_collection3d(poly)

        # Radial lines from sonar origin to far points
        lines = []
        for a in np.linspace(a0, a1, 9):
            for b in (-elev, 0.0, elev):
                d_s = sonar_dir(a, b)
                p_s = r * d_s
                p_w = (T_sonar_in_world @ np.array([p_s[0], p_s[1], p_s[2], 1.0]))[:3]
                lines.append([tuple(o_w.tolist()), tuple(p_w.tolist())])
        lc = Line3DCollection(
            lines,
            colors=[edgecolor],
            linewidths=0.6,
            alpha=min(1.0, alpha + 0.1),
        )
        self.ax.add_collection3d(lc)

    # ---- Zoom / keys ----
    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        factor = self._zoom_factor if event.button == 'up' else (1.0 / self._zoom_factor)
        self._zoom_axes(factor)

    def _on_key(self, event):
        if event.key in ('+', '='):
            self._zoom_axes(self._zoom_factor)
        elif event.key in ('-', '_'):
            self._zoom_axes(1.0 / self._zoom_factor)
        elif event.key in ('r', 'R'):
            self.reset_view()

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
