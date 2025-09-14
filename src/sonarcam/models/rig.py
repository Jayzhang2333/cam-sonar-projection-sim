import numpy as np
class Rig:
    def __init__(self, R_cam_sonar=None, t_cam_sonar=None):
        self.R_cam_sonar = np.eye(3, dtype=np.float32) if R_cam_sonar is None else R_cam_sonar.astype(np.float32)
        self.t_cam_sonar = np.zeros(3, dtype=np.float32) if t_cam_sonar is None else t_cam_sonar.astype(np.float32)
