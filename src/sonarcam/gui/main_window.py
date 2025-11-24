from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np

from ..models.sonar import SonarModel, ScanOrientation
from ..utils.transforms import rpy_to_R, Rt_to_T
from .world_view_3d import WorldView3D
from .projection_view import ProjectionView


def sonar_to_cam_base_R():
    """
    Base rotation aligning sonar axes to camera axes when both 'point forward'.
    Camera: +Z forward, +X right, +Y down
    Sonar:  +X forward, +Y right, +Z up
    Map:    x_s->+Z_c,  y_s->+X_c,  z_s->-Y_c
    """
    ex = np.array([0, 0, 1], dtype=float)   # x_s in cam
    ey = np.array([1, 0, 0], dtype=float)   # y_s in cam
    ez = np.array([0, -1, 0], dtype=float)  # z_s in cam
    return np.column_stack([ex, ey, ez]).astype(np.float32)


def _lab(text, small=False, bold=False, color="#000"):
    lbl = QtWidgets.QLabel(text)
    lbl.setWordWrap(True)
    style = []
    if small:
        style.append("font-size: 12px;")  # increased from 11px
    # Non-small labels inherit the window font size
    if bold:
        style.append("font-weight: 700;")
    style.append(f"color: {color};")
    lbl.setStyleSheet("".join(style))
    return lbl


def _param_row(name, unit, widget, value_width=110, label_width=120):
    w = QtWidgets.QWidget()
    g = QtWidgets.QGridLayout(w)
    g.setContentsMargins(2, 4, 2, 4)
    g.setHorizontalSpacing(4)
    g.setVerticalSpacing(0)

    name_lbl = _lab(name, small=False, bold=False)
    unit_lbl = _lab(unit, small=True, bold=False, color="#444")
    name_lbl.setFixedWidth(label_width)
    unit_lbl.setFixedWidth(label_width)

    widget.setMaximumWidth(value_width)
    widget.setMinimumHeight(26)  # slightly taller
    widget.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    g.addWidget(name_lbl, 0, 0, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    g.addWidget(unit_lbl, 1, 0, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    g.addWidget(widget,   0, 1, 2, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    g.setColumnStretch(0, 0)
    g.setColumnStretch(1, 0)
    return w


def _section(title: str):
    gb = QtWidgets.QGroupBox(title)
    f = gb.font()
    f.setBold(True)
    gb.setFont(f)
    gb.setFlat(False)
    gb.setStyleSheet("""
        QGroupBox {
            border: 1px solid #777;
            border-radius: 6px;
            margin-top: 8px;
            padding-top: 4px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 8px;
            padding: 0px 4px 0px 4px;
        }
    """)
    v = QtWidgets.QVBoxLayout(gb)
    v.setContentsMargins(6, 4, 6, 6)
    v.setSpacing(4)
    return gb, v


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sonar–Camera Simulator (Extrinsics & Elevation Ambiguity)")
        self.resize(1400, 900)

        # Increase base font size for the whole window (more aggressive bump)
        base_font = self.font()
        if base_font.pointSize() > 0:
            base_font.setPointSize(max(base_font.pointSize() + 4, 13))
        else:
            base_font.setPointSize(13)
        self.setFont(base_font)

        # Current GUI values (not yet applied until 'Update config')
        self.image_size = (640, 480)     # (W, H)
        self.K = (600.0, 600.0, 320.0, 240.0)  # fx, fy, cx, cy
        self.sonar_points = []           # {'az','r','color'}

        # Last applied config (used by views)
        self.applied_K = self.K
        self.applied_image_size = self.image_size
        self.applied_T_cam_from_sonar = np.eye(4, dtype=np.float32)
        self.applied_sonar = SonarModel(
            0.5, 30.0, 128, 180, 20.0,
            orientation=ScanOrientation.horizontal,
            sector_start_deg=-60.0, sector_end_deg=60.0
        )

        splitter = QtWidgets.QSplitter(self)
        splitter.setOrientation(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ===== LEFT: scrollable controls column =====
        left_container = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_container)
        left_v.setContentsMargins(6, 6, 6, 6)
        left_v.setSpacing(6)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        host = QtWidgets.QWidget()
        host_v = QtWidgets.QVBoxLayout(host)
        host_v.setContentsMargins(0, 0, 0, 0)
        host_v.setSpacing(6)

        def dspin(minv, maxv, val, decimals=2, step=None):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(minv, maxv)
            s.setDecimals(decimals)
            s.setValue(val)
            if step is not None:
                s.setSingleStep(step)
            s.setMinimumHeight(26)
            return s

        def ispin(minv, maxv, val, step=1):
            s = QtWidgets.QSpinBox()
            s.setRange(minv, maxv)
            s.setValue(val)
            s.setSingleStep(step)
            s.setMinimumHeight(26)
            return s

        # --- CAMERA ---
        cam_box, cam_v = _section("Camera")
        self.fx = dspin(1, 10000, 600.0, decimals=2)
        self.fy = dspin(1, 10000, 600.0, decimals=2)
        self.cx = dspin(0, 8192, 320.0, decimals=2)
        self.cy = dspin(0, 8192, 240.0, decimals=2)
        self.w = ispin(16, 8192, 640)
        self.h = ispin(16, 8192, 480)
        cam_v.addWidget(_param_row("fx", "[px]", self.fx))
        cam_v.addWidget(_param_row("fy", "[px]", self.fy))
        cam_v.addWidget(_param_row("cx", "[px]", self.cx))
        cam_v.addWidget(_param_row("cy", "[px]", self.cy))
        cam_v.addWidget(_param_row("width", "[px]", self.w))
        cam_v.addWidget(_param_row("height", "[px]", self.h))
        host_v.addWidget(cam_box)

        # --- SONAR (FOV only; centered on +X acoustic axis) ---
        sonar_box, sonar_v = _section("Sonar FOV")
        self.range_min = dspin(0.0, 10000.0, 0.5, decimals=2)
        self.range_max = dspin(0.1, 10000.0, 30.0, decimals=2)
        self.elev_fov = dspin(0.1, 180.0, 20.0, decimals=1)
        self.sector_start = dspin(-360.0, 360.0, -60.0, decimals=1)
        self.sector_end = dspin(-360.0, 360.0, +60.0, decimals=1)
        sonar_v.addWidget(_param_row("range min", "[m]", self.range_min))
        sonar_v.addWidget(_param_row("range max", "[m]", self.range_max))
        sonar_v.addWidget(_param_row("elev FOV", "[deg]", self.elev_fov))
        sonar_v.addWidget(_param_row("sector start", "[deg]", self.sector_start))
        sonar_v.addWidget(_param_row("sector end", "[deg]", self.sector_end))
        host_v.addWidget(sonar_box)

        # --- EXTRINSICS (cam frame) ---
        extr_box, extr_v = _section("Extrinsics (cam frame)")
        self.tx = dspin(-10, 10, -0.20, decimals=3)
        self.ty = dspin(-10, 10, 0.0, decimals=3)
        self.tz = dspin(-10, 10, 0.0, decimals=3)
        self.r_roll = dspin(-180.0, 180.0, 0.0, decimals=1)
        self.r_pitch = dspin(-180.0, 180.0, 0.0, decimals=1)
        self.r_yaw = dspin(-180.0, 180.0, 0.0, decimals=1)
        for wdg in (self.tx, self.ty, self.tz, self.r_roll, self.r_pitch, self.r_yaw):
            wdg.setMaximumWidth(110)
        extr_v.addWidget(_param_row("tx", "[m]", self.tx))
        extr_v.addWidget(_param_row("ty", "[m]", self.ty))
        extr_v.addWidget(_param_row("tz", "[m]", self.tz))
        extr_v.addWidget(_param_row("roll", "[deg]", self.r_roll))
        extr_v.addWidget(_param_row("pitch", "[deg]", self.r_pitch))
        extr_v.addWidget(_param_row("yaw", "[deg]", self.r_yaw))
        host_v.addWidget(extr_box)

        # --- APPLY CONFIG BUTTON ---
        self.btn_update = QtWidgets.QPushButton("Update camera/sonar config")
        self.btn_update.setMinimumHeight(30)
        self.btn_update.setMaximumWidth(240)
        host_v.addWidget(self.btn_update)

        # --- SONAR POINTS (az + range) ---
        pts_box, pts_v = _section("Sonar Points → Camera")
        self.az = dspin(-180.0, 180.0, 0.0, decimals=2, step=1.0)
        self.rng = dspin(0.01, 10000.0, 5.0, decimals=2, step=0.1)
        self.color = QtGui.QColor(0, 200, 255)
        self.btn_color = QtWidgets.QPushButton("Color")
        self.btn_color.setMaximumWidth(80)
        self.btn_color.setMinimumHeight(28)
        self.btn_color.clicked.connect(self.on_pick_color)
        self._update_color_button()
        pts_v.addWidget(_param_row("azimuth", "[deg]", self.az))
        pts_v.addWidget(_param_row("range", "[m]", self.rng))
        rowc = QtWidgets.QHBoxLayout()
        rowc.setSpacing(6)
        rowc.addWidget(self.btn_color)
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_add.setMaximumWidth(80)
        self.btn_add.setMinimumHeight(28)
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.setMaximumWidth(80)
        self.btn_clear.setMinimumHeight(28)
        rowc.addWidget(self.btn_add)
        rowc.addWidget(self.btn_clear)
        rowc.addStretch(1)
        pts_v.addLayout(rowc)
        self.list_pts = QtWidgets.QListWidget()
        self.list_pts.setMaximumHeight(180)
        pts_v.addWidget(self.list_pts)
        host_v.addWidget(pts_box)

        host_v.addStretch(1)
        scroll.setWidget(host)
        left_v.addWidget(scroll)
        splitter.addWidget(left_container)

        # Left column width – wider & resizable
        left_container.setMinimumWidth(280)

        # ===== RIGHT: World (top) + Projection (bottom) =====
        right = QtWidgets.QWidget()
        RG = QtWidgets.QGridLayout(right)

        # Tighter margins/spacing to reduce white border around figures
        RG.setContentsMargins(0, 0, 0, 0)
        RG.setHorizontalSpacing(0)
        RG.setVerticalSpacing(0)

        self.world = WorldView3D()
        self.world.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        RG.addWidget(self.world, 0, 0)

        self.proj = ProjectionView()
        self.proj.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        RG.addWidget(self.proj, 1, 0)

        RG.setRowStretch(0, 1)
        RG.setRowStretch(1, 1)
        RG.setColumnStretch(0, 1)

        splitter.addWidget(right)

        # Splitter stretch: give left a reasonable share and right a bit more
        splitter.setStretchFactor(0, 1)  # left
        splitter.setStretchFactor(1, 2)  # right

        # Optional initial sizes (tweak to taste)
        splitter.setSizes([420, 980])

        # Hooks
        self.btn_update.clicked.connect(self.on_update_config)
        self.btn_add.clicked.connect(self.on_add_point)
        self.btn_clear.clicked.connect(self.on_clear_points)

        # Initial apply
        self.on_update_config()

    # --------- UI handlers ----------
    def _update_color_button(self):
        c = self.color
        self.btn_color.setStyleSheet(
            f"QPushButton {{ background-color: {c.name()}; color: #000; border: 1px solid #444; }}"
        )

    def on_pick_color(self):
        c = QtWidgets.QColorDialog.getColor(self.color, self, "Pick Color")
        if c.isValid():
            self.color = c
            self._update_color_button()

    def _refresh_points_list(self):
        """Rebuild list items with current arc length suffixes (if available)."""
        self.list_pts.clear()
        lengths = getattr(self.proj, "last_lengths", [])
        for i, p in enumerate(self.sonar_points):
            suffix = ""
            if i < len(lengths) and np.isfinite(lengths[i]):
                suffix = f" — L={lengths[i]:.1f}px"
            self.list_pts.addItem(f"az={p['az']:.1f}°, r={p['r']:.2f} m{suffix}")

    def on_add_point(self):
        az = float(self.az.value())
        r = float(self.rng.value())
        col = (self.color.red(), self.color.green(), self.color.blue())
        self.sonar_points.append({'az': az, 'r': r, 'color': col})
        self.redraw_projection()

    def on_clear_points(self):
        self.sonar_points.clear()
        self.redraw_projection()

    # --------- Config apply / math helpers ----------
    def _camera_T_from_sonar(self):
        R_base = sonar_to_cam_base_R()
        roll_deg, pitch_deg, yaw_deg = (
            self.r_roll.value(),
            self.r_pitch.value(),
            self.r_yaw.value()
        )
        R_user = rpy_to_R(
            np.deg2rad(roll_deg),
            np.deg2rad(pitch_deg),
            np.deg2rad(yaw_deg)
        )
        R_cs = (R_base @ R_user).astype(np.float32)  # sonar->camera
        t_cs = np.array([self.tx.value(), self.ty.value(), self.tz.value()], dtype=np.float32)
        return Rt_to_T(R_cs, t_cs)

    def _gather_config(self):
        fx = float(self.fx.value())
        fy = float(self.fy.value())
        cx = float(self.cx.value())
        cy = float(self.cy.value())
        W = int(self.w.value())
        H = int(self.h.value())
        K = (fx, fy, cx, cy)
        sonar = SonarModel(
            range_min=self.range_min.value(),
            range_max=self.range_max.value(),
            num_bins=128,
            num_beams=180,
            elevation_fov_deg=self.elev_fov.value(),
            orientation=ScanOrientation.horizontal,  # not used; extrinsics define pose
            sector_start_deg=self.sector_start.value(),
            sector_end_deg=self.sector_end.value(),
        )
        T_cam_from_sonar = self._camera_T_from_sonar()
        return K, (W, H), sonar, T_cam_from_sonar

    def on_update_config(self):
        # Apply current GUI values
        K, image_size, sonar, T_cs = self._gather_config()
        self.applied_K = K
        self.applied_image_size = image_size
        self.applied_sonar = sonar
        self.applied_T_cam_from_sonar = T_cs

        # Update 3D world (axes + volumes only)
        self.world.update_scene(
            K=self.applied_K,
            image_size=self.applied_image_size,
            T_cam_from_sonar=self.applied_T_cam_from_sonar,
            sonar=self.applied_sonar
        )

        # Redraw projection with applied config
        self.redraw_projection()

    def redraw_projection(self):
        W, H = self.applied_image_size
        self.proj.update_projection(
            W=W, H=H, K=self.applied_K,
            T_cam_from_sonar=self.applied_T_cam_from_sonar,
            sonar=self.applied_sonar,
            samples=self.sonar_points
        )
        self._refresh_points_list()
