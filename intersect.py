#shape intersection algorithms

def sphere_intersect(o, d, r, center):
    b = 2 * np.dot(d, o - center)
    c = np.linalg.norm(o - center) ** 2 - r ** 2
    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)
    return None

def cylinder_intersect(o, d, cylinder):    #cylinders of +y orientation only
    a = cylinder['a']   #lower
    b = cylinder['b']   #higher
    r = cylinder['radius']
    cdir = normalize(a-b)   #cylinder axis
    center = (a+b)/2    #assuming cylinder is in the y direction
    h = b[1]-a[1]
    
    
    oc = o - center
    pdot = np.dot(d,cdir)
    if np.abs(pdot) < 1e-6:
        return None
    
    #compute torso intersection using discriminant method
    a = np.dot(d - pdot * cdir, d - pdot * cdir)
    b = 2 * np.dot(d - pdot * cdir, oc - np.dot(oc, cdir) * cdir)
    c = np.dot(oc - np.dot(oc, cdir) * cdir, oc - np.dot(oc, cdir) * cdir) - r ** 2
    
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    t1 = (-b - np.sqrt(disc)) / (2 * a)
    t2 = (-b + np.sqrt(disc)) / (2 * a)
    
    #checking for intersections with finite ends
    intersect1 = o + t1 * d
    intersect2 = o + t2 * d
    
    axis_proj1 = np.dot(intersect1 - center, cdir)
    axis_proj2 = np.dot(intersect2 - center, cdir)
    
    if 0 <= axis_proj1 <= h or 0 <= axis_proj2 <= h:
    #if not (0 <= axis_proj1 <= h or 0 <= axis_proj2 <= h):
    #    return None
        if t1>=0 and t2>=0:
            return min(t1,t2)
        elif t1<0 and t2>=0:
            return max(t1,t2)
        else:
            return None
    #returns minimum t value above 0.

#plane(p, normal n)
def plane_intersect(o, d, p, n):
    denominator = np.dot(d,n)
    if np.abs(denominator) <1e-6:
        return None
    t = -(np.dot((o-p),n) / denominator)
    if t < 0:
        return None
    return t

def triangle_intersect(o,d,triangle):
    #defining vertices as v_, edge_, and other useful values to find solution for ray parameter t.
    v0 = triangle['v0']
    v1 = triangle['v1']
    v2 = triangle['v2']
    
    edge1 = v1-v0
    edge2 = v2-v0
    
    norm = np.cross(edge1,edge2)
    T = o-v0
    Q = np.cross(T,edge1)
    P = np.cross(d, edge2)
    det = np.dot(P,edge1)

    if det == 0:
        return None
    inv_det = 1.0 / det  #saves float divisions and opt for more multiplications to lower cost

    #calculating barycentric u 
    u = np.dot(T,P) * inv_det
    if u < 0 or u > 1:
            return None

    #calculating barycentric v
    v = np.dot(d,Q) * inv_det
    if v < 0 or (u + v) > 1:
        return None

    #calculating final barycentric t. (If the code got to here, then a ray has hit the triangle.)
    matrix_t = np.abs(np.dot(edge2,Q)) * inv_det
    if matrix_t > 0:
            return matrix_t
