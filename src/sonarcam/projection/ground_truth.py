import numpy as np
from ..utils.transforms import project_points

def project_hits_to_camera(hit_pts_w, T_world_cam, K_tuple):
    if hit_pts_w is None or len(hit_pts_w)==0:
        return np.zeros((0,2),dtype=np.float32), np.array([], dtype=bool)
    Pw = hit_pts_w
    Pc = (np.c_[Pw, np.ones(len(Pw))] @ T_world_cam.T)[:,:3]
    uv, mask = project_points(K_tuple, Pc)
    return uv, mask
