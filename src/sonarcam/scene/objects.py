import numpy as np

class SceneObject:
    def __init__(self, shape="cube"):
        self.shape = shape  # "cube","cylinder","tri_prism"
        self.position = np.array([3.0, 0.0, 1.0], dtype=np.float32)
        self.size     = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        self.rpy      = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.color    = (200, 160, 60)

def _cube_tris(size):
    sx, sy, sz = size
    x=sx/2; y=sy/2; z=sz/2
    V = np.array([[-x,-y,-z],[ x,-y,-z],[ x, y,-z],[-x, y,-z],[-x,-y, z],[ x,-y, z],[ x, y, z],[-x, y, z]], dtype=np.float32)
    F = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[1,2,6],[1,6,5],[0,3,7],[0,7,4]], dtype=np.int32)
    return V,F

def _tri_prism_tris(size):
    b,h,d = size
    V = np.array([[0, h/2, -d/2],[-b/2,-h/2,-d/2],[b/2,-h/2,-d/2],
                  [0, h/2,  d/2],[-b/2,-h/2, d/2],[b/2,-h/2, d/2]], dtype=np.float32)
    F = np.array([[0,1,2],[3,5,4],[0,3,4],[0,4,1],[0,2,5],[0,5,3],[1,4,5],[1,5,2]], dtype=np.int32)
    return V,F

def _cyl_tris(size, segments=24):
    r = size[0]*0.5; h=size[2]
    zs = np.array([-h/2, h/2], dtype=np.float32)
    verts=[]; faces=[]
    # side only
    for i in range(segments):
        a0 = 2*np.pi*i/segments; a1=2*np.pi*(i+1)/segments
        x0,y0 = r*np.cos(a0), r*np.sin(a0); x1,y1 = r*np.cos(a1), r*np.sin(a1)
        v0=[x0,y0,zs[0]]; v1=[x1,y1,zs[0]]; v2=[x1,y1,zs[1]]; v3=[x0,y0,zs[1]]
        base=len(verts); verts+= [v0,v1,v2,v3]
        faces+= [[base,base+1,base+2],[base,base+2,base+3]]
    V=np.array(verts,dtype=np.float32); F=np.array(faces,dtype=np.int32)
    return V,F

def make_tris(shape, size):
    if shape=="cube": return _cube_tris(size)
    if shape=="cylinder": return _cyl_tris(size)
    return _tri_prism_tris(size)

def transform_pts(V, position, rpy):
    cr,cp,cy = np.cos(rpy); sr,sp,sy = np.sin(rpy)
    Rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]],dtype=np.float32)
    Ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]],dtype=np.float32)
    Rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]],dtype=np.float32)
    R=(Rz@Ry@Rx).astype(np.float32)
    return (V@R.T + position[None,:]).astype(np.float32)
