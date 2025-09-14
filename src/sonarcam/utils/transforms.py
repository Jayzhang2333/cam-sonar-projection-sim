import numpy as np

def rpy_to_R(roll, pitch, yaw):
    cr,cp,cy = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sr,sp,sy = np.sin(roll), np.sin(pitch), np.sin(yaw)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float32)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float32)
    Rz = np.array([[cy,-sy,0],[sy, cy,0],[0,  0,1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def Rt_to_T(R, t):
    T = np.eye(4, dtype=np.float32); T[:3,:3]=R; T[:3,3]=t; return T

def inv_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3,:3]=R.T; Ti[:3,3]= -R.T@t
    return Ti

def project_points(K_tuple, Xc):
    fx,fy,cx,cy = K_tuple
    x,y,z = Xc[:,0], Xc[:,1], Xc[:,2]
    mask = z>1e-6
    u = fx*(x/(z+1e-12)) + cx
    v = fy*(y/(z+1e-12)) + cy
    return np.stack([u,v],axis=1), mask
