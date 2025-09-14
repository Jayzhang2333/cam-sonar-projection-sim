import numpy as np

def ray_triangle_intersect(orig, dir, v0, v1, v2, eps=1e-8):
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(dir, edge2)
    a = np.dot(edge1, h)
    if -eps < a < eps:
        return False, None
    f = 1.0 / a
    s = orig - v0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return False, None
    q = np.cross(s, edge1)
    v = f * np.dot(dir, q)
    if v < 0.0 or u + v > 1.0:
        return False, None
    t = f * np.dot(edge2, q)
    if t > eps:
        return True, t
    return False, None

def project_points_pinhole(P_c, fx, fy, cx, cy):
    z = P_c[:,2]
    u = fx * (P_c[:,0] / z) + cx
    v = fy * (P_c[:,1] / z) + cy
    return np.stack([u, v], axis=1)

def triangles_to_image(tris_cam, colors, K, width, height):
    fx, fy, cx, cy = K
    H, W = height, width
    color_img = np.full((H, W, 3), 200, dtype=np.uint8)
    depth = np.full((H, W), np.inf, dtype=np.float32)

    for tri, col in zip(tris_cam, colors):
        if np.any(tri[:,2] <= 1e-6):
            continue
        uv = project_points_pinhole(tri, fx, fy, cx, cy)
        minx = int(np.floor(max(0, np.min(uv[:,0]))))
        maxx = int(np.ceil(min(W-1, np.max(uv[:,0]))))
        miny = int(np.floor(max(0, np.min(uv[:,1]))))
        maxy = int(np.ceil(min(H-1, np.max(uv[:,1]))))
        if minx>=W or maxx<0 or miny>=H or maxy<0 or minx>maxx or miny>maxy:
            continue
        x0,y0 = uv[0]; x1,y1 = uv[1]; x2,y2 = uv[2]
        denom = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
        if abs(denom) < 1e-9:
            continue
        z0,z1,z2 = tri[:,2]
        for y in range(miny, maxy+1):
            for x in range(minx, maxx+1):
                w0 = ((y1 - y2)*(x - x2) + (x2 - x1)*(y - y2)) / denom
                w1 = ((y2 - y0)*(x - x2) + (x0 - x2)*(y - y2)) / denom
                w2 = 1.0 - w0 - w1
                if (w0 < 0) or (w1 < 0) or (w2 < 0):
                    continue
                z = 1.0 / (w0*(1.0/z0) + w1*(1.0/z1) + w2*(1.0/z2))
                if z < depth[y,x]:
                    depth[y,x] = z
                    color_img[y,x,:] = col
    return color_img, depth
