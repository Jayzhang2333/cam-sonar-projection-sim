class CameraModel:
    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx=float(fx); self.fy=float(fy); self.cx=float(cx); self.cy=float(cy)
        self.width=int(width); self.height=int(height)
