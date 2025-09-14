from PyQt5 import QtWidgets, QtGui, QtCore
import numpy as np
from ..utils.transforms import inv_T

class WorldView(QtWidgets.QWidget):
    """Compact top-down 2D world preview with zoom."""
    def __init__(self, scn, parent=None):
        super().__init__(parent)
        self._scene = scn
        self._T_world_cam = np.eye(4, dtype=np.float32)
        self._T_world_sonar = np.eye(4, dtype=np.float32)
        self._half_extent = 30.0  # meters (zoomable)
        self._min_half = 5.0
        self._max_half = 200.0

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # Keep this short to give more room to the four images
        self._label = QtWidgets.QLabel("")
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setStyleSheet("QLabel { background: #111; }")
        self._label.setMinimumHeight(110)
        self._label.setMaximumHeight(150)  # << smaller footprint
        self._label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        lay.addWidget(self._label)

        # hint
        self.setToolTip("Mouse wheel: zoom  |  +/- keys: zoom  |  Double-click: reset")

        self.refresh()

    def set_poses(self, T_world_cam, T_world_sonar):
        """Store current sensor poses (world->camera, world->sonar)."""
        self._T_world_cam = T_world_cam.astype(np.float32)
        self._T_world_sonar = T_world_sonar.astype(np.float32)

    # --- zoom controls ---
    def wheelEvent(self, ev: QtGui.QWheelEvent):
        delta = ev.angleDelta().y()
        factor = 0.9 if delta > 0 else 1.1
        self._half_extent = float(np.clip(self._half_extent * factor, self._min_half, self._max_half))
        self.refresh()

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        if ev.key() in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self._half_extent = max(self._min_half, self._half_extent * 0.9)
            self.refresh()
        elif ev.key() in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            self._half_extent = min(self._max_half, self._half_extent * 1.1)
            self.refresh()
        else:
            super().keyPressEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
        self._half_extent = 30.0
        self.refresh()

    def refresh(self):
        # Render to the current label size (width x fixed ~140h)
        W = max(300, self._label.width())
        H = max(110, self._label.height())
        img = QtGui.QImage(W, H, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(16, 16, 16))
        p = QtGui.QPainter(img)
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)

        half = float(self._half_extent)

        def world_to_px(x, y):
            """Top-down: +X right, +Y up."""
            u = int((x + half) / (2 * half) * (W - 1))
            v = int((half - y) / (2 * half) * (H - 1))
            return u, v

        # grid
        pen = QtGui.QPen(QtGui.QColor(60, 60, 60)); pen.setWidth(1); p.setPen(pen)
        ticks = np.linspace(-half, half, 7)
        for t in ticks:
            x0, y0 = world_to_px(t, -half); x1, y1 = world_to_px(t, half);  p.drawLine(x0, y0, x1, y1)
            x0, y0 = world_to_px(-half, t); x1, y1 = world_to_px(half, t); p.drawLine(x0, y0, x1, y1)

        # axis tick labels (meters)
        p.setPen(QtGui.QPen(QtGui.QColor(150, 150, 150)))
        for t in ticks:
            u, _ = world_to_px(t, -half)
            p.drawText(u - 10, H - 4, f"{t:.0f}")
            _, v = world_to_px(-half, t)
            p.drawText(4, v + 4, f"{t:.0f}")

        # world axes legend (X right, Y up, Z out of plane)
        ox, oy = world_to_px(-half + 6.0, -half + 6.0)
        p.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 2))
        p.drawLine(ox, oy, ox + 30, oy); p.drawText(ox + 34, oy + 4, "X")
        p.drawLine(ox, oy, ox, oy - 30); p.drawText(ox + 4, oy - 34, "Y")
        p.drawText(ox - 6, oy + 14, "Z↑")

        # objects
        for obj in getattr(self._scene, "objects", []):
            color = QtGui.QColor(*obj.color)
            p.setBrush(QtGui.QBrush(color))
            p.setPen(QtGui.QPen(QtGui.QColor(230, 230, 230), 1))
            cx, cy = float(obj.position[0]), float(obj.position[1])
            sx, sy = float(obj.size[0]), float(obj.size[1])
            x0, y0 = cx - sx / 2, cy - sy / 2
            x1, y1 = cx + sx / 2, cy + sy / 2
            (u0, v0) = world_to_px(x0, y0); (u1, v1) = world_to_px(x1, y1)
            x, y = min(u0, u1), min(v0, v1)
            w, h = abs(u1 - u0), abs(v1 - v0)
            if obj.shape == "cylinder":
                p.drawEllipse(x, y, w, h)
            elif obj.shape == "tri_prism":
                pA = world_to_px(cx, cy + sy / 2)
                pB = world_to_px(cx - sx / 2, cy - sy / 2)
                pC = world_to_px(cx + sx / 2, cy - sy / 2)
                poly = QtGui.QPolygon([QtCore.QPoint(*pA), QtCore.QPoint(*pB), QtCore.QPoint(*pC)])
                p.drawPolygon(poly)
            else:
                p.drawRect(x, y, w, h)

        # draw camera & sonar poses and pointing directions (+Z projected to XY)
        def draw_sensor(T_world_sensor, name=""):
            T_sw = inv_T(T_world_sensor)  # sensor->world
            R_sw = T_sw[:3, :3]; t_w = T_sw[:3, 3]
            z_dir = R_sw @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
            dir_xy = z_dir[:2]; n = np.linalg.norm(dir_xy)
            if n < 1e-6:
                dir_xy = np.array([1.0, 0.0], dtype=np.float32)
            else:
                dir_xy = dir_xy / n
            u0, v0 = world_to_px(t_w[0], t_w[1])
            scale = 8.0
            u1, v1 = world_to_px(t_w[0] + scale * dir_xy[0], t_w[1] + scale * dir_xy[1])
            color = QtGui.QColor(120, 220, 255) if name.lower().startswith("cam") else QtGui.QColor(255, 210, 90)
            p.setPen(QtGui.QPen(color, 2))
            p.drawLine(u0, v0, u1, v1)
            ah = 8; ang = np.arctan2(v1 - v0, u1 - u0)
            left  = QtCore.QPoint(int(u1 - ah * np.cos(ang - np.pi/6)), int(v1 - ah * np.sin(ang - np.pi/6)))
            right = QtCore.QPoint(int(u1 - ah * np.cos(ang + np.pi/6)), int(v1 - ah * np.sin(ang + np.pi/6)))
            p.drawLine(u1, v1, left.x(), left.y()); p.drawLine(u1, v1, right.x(), right.y())
            p.setBrush(QtGui.QBrush(color)); p.drawEllipse(u0-3, v0-3, 6, 6)
            p.setPen(QtGui.QPen(color)); p.drawText(u0+6, v0-6, name)

        draw_sensor(self._T_world_cam,   "Cam (+Z)")
        draw_sensor(self._T_world_sonar, "Sonar (+Z)")

        p.end()
        self._label.setPixmap(QtGui.QPixmap.fromImage(img))
