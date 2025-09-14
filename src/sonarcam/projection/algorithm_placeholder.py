import numpy as np
from ..utils.transforms import inv_T, project_points

def flat_ground_assumption_projection(ranges, azis_rad, T_world_sonar, T_world_cam, K_tuple):
    # Placeholder: assume all hits lie on z=0; sonar sweeps horizontally.
    T_sw = inv_T(T_world_sonar)  # sonar->world
    origin_w = (np.array([0,0,0,1.0],dtype=np.float32) @ T_sw.T)[:3]
    R_sw = T_sw[:3,:3]
    pts=[]
    for r,a in zip(ranges, azis_rad):
        if not np.isfinite(r): 
            pts.append([np.nan,np.nan,np.nan]); 
            continue
        dir_s = np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float32)
        dir_w = R_sw @ dir_s
        # Move from sonar origin down/up to ground z=0, then go r meters in XY along dir_w
        o2 = origin_w.copy(); o2[2]=0.0
        dxy = dir_w[:2]; n = np.linalg.norm(dxy)+1e-12
        p = o2 + np.array([dxy[0]/n * r, dxy[1]/n * r, 0.0], dtype=np.float32)
        pts.append(p)
    Pw = np.array(pts, dtype=np.float32)
    Pc = (np.c_[Pw, np.ones(len(Pw))] @ T_world_cam.T)[:,:3]
    uv, mask = project_points(K_tuple, Pc)
    return uv, mask, Pw
