from PyQt5 import QtWidgets, QtCore, QtGui
import os
import numpy as np
from PIL import Image

from ..models.sonar import SonarModel, ScanOrientation
from ..utils.transforms import rpy_to_R, Rt_to_T, inv_T
from ..utils.sonar_viz import polar_to_fan_image
from ..utils.pointcloud import depth_to_points
from ..renderers.sonar_from_depth import render_sonar_from_depth
from .world_view_3d import WorldView3D


# --------- helpers ---------
def np_to_qpixmap(img):
    """
    Robust NumPy -> QPixmap:
    - accepts HxW (gray) or HxWx3 (RGB)
    - ensures uint8 + contiguous
    - uses a bytes buffer so PyQt5 QImage ctor is happy
    - keeps a reference to the buffer on the QImage to avoid GC
    """
    import numpy as _np
    from PyQt5 import QtGui as _QtGui

    if img.ndim == 2:
        arr = _np.ascontiguousarray(img.astype(_np.uint8, copy=False))
        h, w = arr.shape
        buf = arr.tobytes()
        qimg = _QtGui.QImage(buf, w, h, w, _QtGui.QImage.Format_Grayscale8)
        qimg._buf = buf
        return _QtGui.QPixmap.fromImage(qimg)
    else:
        assert img.ndim == 3 and img.shape[2] in (3, 4), "Expected HxWx3 (or 4) image"
        if img.shape[2] == 4:
            arr = img[:, :, :3]
        else:
            arr = img
        arr = _np.ascontiguousarray(arr.astype(_np.uint8, copy=False))
        h, w, _ = arr.shape
        buf = arr.tobytes()
        qimg = _QtGui.QImage(buf, w, h, 3 * w, _QtGui.QImage.Format_RGB888)
        qimg._buf = buf
        return _QtGui.QPixmap.fromImage(qimg)


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


# small label factory
def _lab(text, small=False, bold=False, color="#000"):
    lbl = QtWidgets.QLabel(text)
    lbl.setWordWrap(True)
    style = []
    if small: style.append("font-size: 10px;")
    if bold:  style.append("font-weight: 700;")
    style.append(f"color: {color};")
    lbl.setStyleSheet("".join(style))
    return lbl


def _param_row(name, unit, widget, value_width=96, label_width=112):
    """
    Two-row label (name + unit) on the left; value widget on the right,
    vertically centered spanning the two rows. Extra tight layout.
    """
    w = QtWidgets.QWidget()
    g = QtWidgets.QGridLayout(w)
    g.setContentsMargins(2, 2, 2, 2)
    g.setHorizontalSpacing(2)
    g.setVerticalSpacing(0)

    name_lbl = _lab(name, small=False, bold=False)
    unit_lbl = _lab(unit, small=True, bold=False, color="#444")
    name_lbl.setFixedWidth(label_width)
    unit_lbl.setFixedWidth(label_width)

    widget.setMaximumWidth(value_width)
    widget.setMinimumHeight(22)
    widget.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

    g.addWidget(name_lbl, 0, 0, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    g.addWidget(unit_lbl, 1, 0, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    g.addWidget(widget,   0, 1, 2, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
    g.setColumnStretch(0, 0)
    g.setColumnStretch(1, 0)
    return w

def _section(title: str):
    """
    Thin-bounded QGroupBox section with tight margins and a BOLD title.
    Uses QFont bold on the group box (more reliable across platforms/styles
    than QSS on QGroupBox::title).
    Returns (group_box, inner_vlayout).
    """
    gb = QtWidgets.QGroupBox(title)
    # Make the title bold via QFont (works even when stylesheets ignore font-weight)
    f = gb.font()
    f.setBold(True)
    gb.setFont(f)

    gb.setFlat(False)
    gb.setStyleSheet("""
        QGroupBox {
            border: 1px solid #777;
            border-radius: 6px;
            margin-top: 8px;      /* space for the title */
            padding-top: 4px;     /* title inset */
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
        self.setWindowTitle("Sonar–Camera Simulator (RGB+Depth)")
        self.resize(1400, 900)

        self.rgb = None                # uint8 HxWx3
        self.depth = None              # float32 meters (NaN invalid)
        self.K = None                  # (fx, fy, cx, cy)
        self.image_size = (0, 0)       # (W, H)

        splitter = QtWidgets.QSplitter(self)
        splitter.setOrientation(QtCore.Qt.Horizontal)
        self.setCentralWidget(splitter)

        # ===== Left: SINGLE, NARROW, SCROLLABLE COLUMN =====
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

        # Small spin helpers
        def dspin(minv, maxv, val, decimals=3, step=None):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(minv, maxv)
            s.setDecimals(decimals)
            s.setValue(val)
            if step is not None: s.setSingleStep(step)
            return s

        def ispin(minv, maxv, val, step=1):
            s = QtWidgets.QSpinBox()
            s.setRange(minv, maxv)
            s.setValue(val)
            s.setSingleStep(step)
            return s

        # ---------- Inputs (boxed) ----------
        inputs_box, inputs_v = _section("Inputs")
        self.btn_rgb = QtWidgets.QPushButton("Load RGB")
        self.btn_rgb.setMaximumWidth(128)
        self.lbl_rgb = _lab("—", small=True, color="#555")
        self.btn_depth = QtWidgets.QPushButton("Load Depth")
        self.btn_depth.setMaximumWidth(128)
        self.lbl_depth = _lab("—", small=True, color="#555")
        # Depth scale: 1 decimal, step 0.1
        self.depth_scale = dspin(1e-6, 1e3, 1.0, decimals=1, step=0.1)
        self.downsample  = ispin(1, 8, 2)

        inputs_v.addWidget(self.btn_rgb)
        inputs_v.addWidget(self.lbl_rgb)
        inputs_v.addWidget(self.btn_depth)
        inputs_v.addWidget(self.lbl_depth)
        inputs_v.addWidget(_param_row("Depth scale", "[m / unit]", self.depth_scale))
        inputs_v.addWidget(_param_row("Downsample", "[px step]", self.downsample))
        host_v.addWidget(inputs_box)

        # ---------- Camera (boxed) ----------
        cam_box, cam_v = _section("Camera")
        self.fx = dspin(1, 10000, 600.0, decimals=2)
        self.fy = dspin(1, 10000, 600.0, decimals=2)
        self.cx = dspin(0, 8192, 320.0, decimals=2)
        self.cy = dspin(0, 8192, 240.0, decimals=2)
        self.w  = ispin(16, 8192, 640)
        self.h  = ispin(16, 8192, 480)
        cam_v.addWidget(_param_row("fx", "[px]", self.fx))
        cam_v.addWidget(_param_row("fy", "[px]", self.fy))
        cam_v.addWidget(_param_row("cx", "[px]", self.cx))
        cam_v.addWidget(_param_row("cy", "[px]", self.cy))
        cam_v.addWidget(_param_row("width", "[px]", self.w))
        cam_v.addWidget(_param_row("height", "[px]", self.h))
        host_v.addWidget(cam_box)

        # ---------- Sonar (boxed) ----------
        sonar_box, sonar_v = _section("Sonar")
        self.range_min = dspin(0.0, 10000.0, 0.5, decimals=3)
        self.range_max = dspin(0.1, 10000.0, 30.0, decimals=3)
        self.num_bins  = ispin(16, 4096, 384)
        self.num_beams = ispin(16, 4096, 540)
        self.elev_fov  = dspin(0.1, 180.0, 20.0, decimals=2)
        self.orientation = QtWidgets.QComboBox(); self.orientation.addItems(["horizontal", "vertical"])
        self.orientation.setMaximumWidth(96)
        self.sector_start = dspin(-360.0, 360.0, -60.0, decimals=2)
        self.sector_end   = dspin(-360.0, 360.0, +60.0, decimals=2)

        sonar_v.addWidget(_param_row("range min", "[m]", self.range_min))
        sonar_v.addWidget(_param_row("range max", "[m]", self.range_max))
        sonar_v.addWidget(_param_row("num bins", "[—]", self.num_bins))
        sonar_v.addWidget(_param_row("num beams", "[—]", self.num_beams))
        sonar_v.addWidget(_param_row("elev FOV", "[deg]", self.elev_fov))
        sonar_v.addWidget(_param_row("orientation", "[mode]", self.orientation))
        sonar_v.addWidget(_param_row("sector start", "[deg]", self.sector_start))
        sonar_v.addWidget(_param_row("sector end", "[deg]", self.sector_end))
        host_v.addWidget(sonar_box)

        # ---------- Extrinsics (boxed) ----------
        extr_box, extr_v = _section("Extrinsics (cam frame)")
        self.tx = dspin(-10, 10, -0.20, decimals=4)  # sonar left of camera
        self.ty = dspin(-10, 10, 0.0, decimals=4)
        self.tz = dspin(-10, 10, 0.0, decimals=4)
        self.r_roll  = dspin(-180.0, 180.0, 0.0, decimals=2)
        self.r_pitch = dspin(-180.0, 180.0, 0.0, decimals=2)
        self.r_yaw   = dspin(-180.0, 180.0, 0.0, decimals=2)
        for wdg in (self.tx, self.ty, self.tz, self.r_roll, self.r_pitch, self.r_yaw):
            wdg.setMaximumWidth(96)

        extr_v.addWidget(_param_row("tx", "[m]", self.tx))
        extr_v.addWidget(_param_row("ty", "[m]", self.ty))
        extr_v.addWidget(_param_row("tz", "[m]", self.tz))
        extr_v.addWidget(_param_row("roll", "[deg]", self.r_roll))
        extr_v.addWidget(_param_row("pitch", "[deg]", self.r_pitch))
        extr_v.addWidget(_param_row("yaw", "[deg]", self.r_yaw))
        host_v.addWidget(extr_box)

        # Actions
        self.btn_render = QtWidgets.QPushButton("Render")
        self.btn_render.setMaximumWidth(128)
        self.btn_render.setMinimumHeight(28)
        host_v.addWidget(self.btn_render)

        host_v.addStretch(1)
        scroll.setWidget(host)
        left_v.addWidget(scroll)
        splitter.addWidget(left_container)

        # Keep the left column narrow
        left_container.setMinimumWidth(210)
        left_container.setMaximumWidth(240)

        # ===== Right: 2×2 GRID (big views) =====
        right = QtWidgets.QWidget()
        RG = QtWidgets.QGridLayout(right)
        RG.setContentsMargins(4, 4, 4, 4)
        RG.setHorizontalSpacing(6)
        RG.setVerticalSpacing(6)

        self.world = WorldView3D()
        RG.addWidget(self.world, 0, 0)

        self.lbl_rgb_view = QtWidgets.QLabel(); self.lbl_rgb_view.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_rgb_view.setStyleSheet("QLabel { border: 1px solid #333; }")
        self.lbl_rgb_view.setScaledContents(True)
        RG.addWidget(self.lbl_rgb_view, 0, 1)

        self.lbl_sonar = QtWidgets.QLabel(); self.lbl_sonar.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_sonar.setStyleSheet("QLabel { border: 1px solid #333; }")
        self.lbl_sonar.setScaledContents(True)
        RG.addWidget(self.lbl_sonar, 1, 0)

        self.lbl_overlay = QtWidgets.QLabel(); self.lbl_overlay.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_overlay.setStyleSheet("QLabel { border: 1px solid #333; }")
        self.lbl_overlay.setScaledContents(True)
        RG.addWidget(self.lbl_overlay, 1, 1)

        RG.setRowStretch(0, 1); RG.setRowStretch(1, 1)
        RG.setColumnStretch(0, 1); RG.setColumnStretch(1, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)  # left fixed-ish
        splitter.setStretchFactor(1, 1)  # right expands

        # Hooks
        self.btn_rgb.clicked.connect(self.on_load_rgb)
        self.btn_depth.clicked.connect(self.on_load_depth)
        self.btn_render.clicked.connect(self.on_render)

    # --------- Loaders ----------
    def _assert_ext(self, path, allowed_exts, kind):
        ext = os.path.splitext(path)[1].lower()
        if ext not in allowed_exts:
            raise ValueError(f"{kind} file must be one of: {', '.join(allowed_exts)} (got '{ext}')")

    def on_load_rgb(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open RGB", "", "Images (*.png *.tif *.tiff)")
        if not path: return
        try:
            self._assert_ext(path, {".png", ".tif", ".tiff"}, "RGB")
            im = Image.open(path).convert("RGB")
            self.rgb = np.array(im, dtype=np.uint8)
            H, W, _ = self.rgb.shape
            self.w.setValue(W); self.h.setValue(H)
            if self.cx.value() == 320.0 and self.cy.value() == 240.0:
                self.cx.setValue(W/2); self.cy.setValue(H/2)
            self.image_size = (W, H)
            self.lbl_rgb.setText(path)
            self.lbl_rgb_view.setPixmap(np_to_qpixmap(self.rgb))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load RGB failed", str(e))

    def on_load_depth(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Depth", "", "Depth (*.npy *.tif *.tiff)")
        if not path: return
        try:
            self._assert_ext(path, {".npy", ".tif", ".tiff"}, "Depth")
            if path.lower().endswith(".npy"):
                D = np.load(path)
            else:
                im = Image.open(path)
                if im.mode not in ("I;16", "I", "F"):
                    im = im.convert("I;16")
                D = np.array(im)
            if D.ndim == 3 and D.shape[2] == 1: D = D[..., 0]
            if D.ndim != 2:
                raise ValueError(f"Depth must be single-channel (HxW). Got shape {D.shape}.")
            D = np.array(D, dtype=np.float32) * float(self.depth_scale.value())
            invalid = ~np.isfinite(D) | (D <= 0.0)
            if np.any(invalid):
                D[invalid] = np.nan
            self.depth = D
            H, W = D.shape
            n_valid = int(np.isfinite(D).sum())
            self.lbl_depth.setText(f"{path}  |  valid: {n_valid:,} / {H*W:,}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load depth failed", str(e))

    # --------- Pipeline ----------
    def on_render(self):
        if self.rgb is None or self.depth is None:
            QtWidgets.QMessageBox.warning(self, "Missing input", "Please load both RGB (png/tif) and Depth (npy/tif).")
            return

        fx, fy, cx, cy = float(self.fx.value()), float(self.fy.value()), float(self.cx.value()), float(self.cy.value())
        W, H = int(self.w.value()), int(self.h.value())
        if (self.rgb.shape[1], self.rgb.shape[0]) != (W, H):
            self.rgb = np.array(Image.fromarray(self.rgb).resize((W, H), Image.BILINEAR))
        if (self.depth.shape[1], self.depth.shape[0]) != (W, H):
            Dimg = Image.fromarray(self.depth, mode="F").resize((W, H), Image.NEAREST)
            self.depth = np.array(Dimg, dtype=np.float32)

        self.K = (fx, fy, cx, cy)
        self.image_size = (W, H)

        # Extrinsics (degrees) with base alignment
        R_base = sonar_to_cam_base_R()
        roll_deg, pitch_deg, yaw_deg = self.r_roll.value(), self.r_pitch.value(), self.r_yaw.value()
        R_user = rpy_to_R(np.deg2rad(roll_deg), np.deg2rad(pitch_deg), np.deg2rad(yaw_deg))
        R_cs = (R_base @ R_user).astype(np.float32)  # sonar->camera
        t_cs = np.array([self.tx.value(), self.ty.value(), self.tz.value()], dtype=np.float32)
        T_cam_from_sonar = Rt_to_T(R_cs, t_cs)       # sonar->camera
        T_cam_to_sonar   = inv_T(T_cam_from_sonar)   # camera->sonar

        # Point cloud from depth (camera frame)
        ds = int(self.downsample.value())
        pts_c, uv = depth_to_points(self.depth, fx, fy, cx, cy, downsample=ds)
        if pts_c.shape[0] == 0:
            QtWidgets.QMessageBox.warning(self, "Empty cloud", "No valid depth points after sanitization.")
            return

        sonar = SonarModel(
            range_min=self.range_min.value(), range_max=self.range_max.value(),
            num_bins=self.num_bins.value(), num_beams=self.num_beams.value(),
            elevation_fov_deg=self.elev_fov.value(),
            orientation=ScanOrientation(self.orientation.currentText()),
            sector_start_deg=self.sector_start.value(), sector_end_deg=self.sector_end.value(),
        )

        polar, used_mask, angle_grid = render_sonar_from_depth(pts_c, uv, sonar, T_cam_to_sonar)

        # Minimal-whitespace fan image
        fan = polar_to_fan_image(
            polar.astype(np.uint8),
            sonar.range_min, sonar.range_max, angle_grid,
            out_size=360, grid_rings=6, angle_step_deg=15, show_labels=True,
            top_pad_px=2, side_pad_px=2, bottom_pad_px=4
        )

        overlay = self.draw_points(self.rgb, uv[used_mask], color=(0, 200, 255), radius=2)

        self.lbl_rgb_view.setPixmap(np_to_qpixmap(self.rgb))
        self.lbl_sonar.setPixmap(np_to_qpixmap(fan))
        self.lbl_overlay.setPixmap(np_to_qpixmap(overlay))

        self.world.update_scene(pts_c, T_cam_to_sonar, sonar)

    # ---- overlay drawing ----
    def draw_points(self, img, uv, color=(255, 0, 0), radius=2):
        try:
            import cv2
            out = img.copy()
            rr, gg, bb = int(color[0]), int(color[1]), int(color[2])
            bgr = (bb, gg, rr)
            for p in uv:
                if not (np.isfinite(p[0]) and np.isfinite(p[1])): 
                    continue
                cv2.circle(out, (int(round(p[0])), int(round(p[1]))), radius, bgr, thickness=-1, lineType=cv2.LINE_AA)
            return out
        except Exception:
            H, W, _ = img.shape
            out = img.copy()
            rr, gg, bb = color
            for p in uv:
                if not (np.isfinite(p[0]) and np.isfinite(p[1])): 
                    continue
                x = int(round(p[0])); y = int(round(p[1]))
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        xx = x + dx; yy = y + dy
                        if 0 <= xx < W and 0 <= yy < H and dx*dx + dy*dy <= radius*radius:
                            out[yy, xx, 0] = rr; out[yy, xx, 1] = gg; out[yy, xx, 2] = bb
            return out
