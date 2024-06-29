# Cylinder Attempt
#
#6/5/24 Adding find_normal function
#
#
import numpy as np
import matplotlib.pyplot as plt



light = { 'position': np.array([3, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
    {'type': 'cylinder', 'radius': 0.1, 'a': np.array([0,-0.5,-0.5]), 'b': np.array([0,1,-0.5]),'ambient': np.array([0,0,0]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0 },    
    { 'type': 'sphere', 'center': np.array([0, 0.5, -0.5]), 'radius': 0.25, 'ambient': np.array([1/2, 1/2, .8784/2]), 'diffuse': np.array([0.5, 0.5, 0.5]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    { 'type': 'plane', 'center': np.array([0,-0.5,1]), 'normal': np.array([0,1,0]),'ambient': np.array([0.2,0.2,0.2]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0 },
]




width = 150
height = 100

max_depth = 3

camera = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        origin = camera
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_depth):
            # check for intersections
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            intersection = origin + min_distance * direction
            normal_to_surface = find_normal(intersection, nearest_object)
            shifted_point = intersection + 1e-6 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed:
                continue

            illumination = np.zeros((3))

            # ambiant
            illumination += nearest_object['ambient'] * light['ambient']

            # diffuse
            illumination += nearest_object['diffuse'] * light['diffuse'] * (np.dot(intersection_to_light, normal_to_surface))

            # specular
            intersection_to_camera = normalize(camera - intersection)
            H = normalize(intersection_to_light + intersection_to_camera)
            illumination += nearest_object['specular'] * light['specular'] * (np.dot(normal_to_surface, H)) ** (nearest_object['shininess'] / 4)

            # reflection
            color += reflection * illumination
            reflection *= nearest_object['reflection']

            origin = shifted_point
            direction = reflect(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)
plt.imsave('image.png',image)
img = plt.imread('image.png')
imgplot = plt.imshow(img)
plt.show()


# In[11]:


#vector functions

import numpy as np
import matplotlib.pyplot as plt

#normalizes input vector by dividing by magnitude
def normalize(v):
    if np.linalg.norm(v) == 0:
        return v
    else:
        return v / np.linalg.norm(v)

#takes a direction and axis. **only meant for directions
def reflect(d, ax):
    return d - 2 * np.dot(d, ax) * ax

#computes normal for various objects. Each object calls for a different algorithm
def find_normal(intersection, obj):
    if obj['type'] == 'sphere':
        return normalize(intersection - obj['center'])
    if obj['type'] == 'plane':
        return normalize(obj['normal'])
    if obj['type'] == 'triangle':
        v0 = obj['v0']
        v1 = obj['v1']
        v2 = obj['v2']
        return normalize(np.cross(v2-v0,v1-v0))
    if obj['type'] == 'cylinder':
        a = obj['a']   #lower
        b = obj['b']   #higher
        r = obj['radius']
        cdir = normalize(a-b)
        if np.linalg.norm(intersection - b) < r: #top
            return cdir
        elif np.linalg.norm(intersection - a) < r:
            return -1*cdir
        else:
            t = np.dot(intersection - a,cdir)
            pt = a + t*cdir
            return normalize(intersection - pt)


# In[33]:


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

def cylinder_intersect(o, d, cylinder):
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


# In[9]:


#general intersect algorithms

def intersect(ray_origin,ray_direction,obj):
    if obj['type'] == 'plane':
        return plane_intersect(ray_origin,ray_direction,obj['center'],obj['normal'])
    if obj['type'] == 'sphere':
        return sphere_intersect(ray_origin,ray_direction,obj['radius'],obj['center'])
    if obj['type'] == 'triangle':
        return triangle_intersect(ray_origin, ray_direction, obj)
    if obj['type'] == 'cylinder':
        return cylinder_intersect(ray_origin, ray_direction, obj)
    
def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [intersect(ray_origin,ray_direction,obj) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance


# In[40]:


#
# Ray Tracer
# For spheres, planes, triangle planes, and cylinders.
# Uses blinnphong light model
# Needs numpy, matplotlib to run
#
import numpy as np
import matplotlib.pyplot as plt



light = { 'position': np.array([3, 5, 5]), 'ambient': np.array([1, 1, 1]), 'diffuse': np.array([1, 1, 1]), 'specular': np.array([1, 1, 1]) }

objects = [
    {'type': 'cylinder', 'radius': 0.3, 'a': np.array([0,-0.4,-0.5]), 'b': np.array([0,-0.1,-0.5]),'ambient': np.array([0.1,0,0]), 'diffuse': np.array([0.6, 0, 0]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },    
    { 'type': 'sphere', 'center': np.array([0, 0.5, -0.5]), 'radius': 0.25, 'ambient': np.array([1/4, 1/4, .8784/4]), 'diffuse': np.array([0.4, 0.4, 0.4]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0.5 },
    #{ 'type': 'plane', 'center': np.array([0,-0.5,1]), 'normal': np.array([0,1,0]),'ambient': np.array([0.2,0.2,0.2]), 'diffuse': np.array([0.6, 0.6, 0.6]), 'specular': np.array([1, 1, 1]), 'shininess': 100, 'reflection': 0 },
]




width = 300
height = 200

#amount of time the rays will reflect, if any do
max_reflections = 3

home = np.array([0, 0, 1])
ratio = float(width) / height
screen = (-1, 1 / ratio, 1, -1 / ratio) # left, top, right, bottom

image = np.zeros((height, width, 3))
for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
    for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
        # screen is on origin
        pixel = np.array([x, y, 0])
        origin = home
        direction = normalize(pixel - origin)

        color = np.zeros((3))
        reflection = 1

        for k in range(max_reflections):    #trace ray maximum of max_reflections times
            # check for intersections
            nearest_object, min_distance = nearest_intersected_object(objects, origin, direction)
            if nearest_object is None:
                break

            intersection = origin + min_distance * direction
            normal_to_surface = find_normal(intersection, nearest_object)
            shifted_point = intersection + 1e-6 * normal_to_surface
            intersection_to_light = normalize(light['position'] - shifted_point)

            _, min_distance = nearest_intersected_object(objects, shifted_point, intersection_to_light)
            intersection_to_light_distance = np.linalg.norm(light['position'] - intersection)
            is_shadowed = min_distance < intersection_to_light_distance

            if is_shadowed == True:
                continue

            #illum matrix
            illum = np.zeros((3))
            magLplusV = normalize(intersection_to_light + intersection_to_camera)
            
            #ambient and diffuse
            LdotN = np.dot(intersection_to_light, normal_to_surface)
            illum += nearest_object['ambient'] * light['ambient'] + (nearest_object['diffuse'] * light['diffuse'] * LdotN)

            # specular
            intersection_to_camera = normalize(home - intersection)
            illum += nearest_object['specular'] * light['specular'] * (np.dot(normal_to_surface, magLplusV)) ** (nearest_object['shininess'] / 4)

            # reflection
            color += reflection * illum
            reflection *= nearest_object['reflection']

            origin = shifted_point
            direction = reflect(direction, normal_to_surface)

        image[i, j] = np.clip(color, 0, 1)    #describes all color ranges from 0 to 1 instead of 0 to 256
plt.imsave('image.png',image)
img = plt.imread('image.png')
imgplot = plt.imshow(img)
plt.show()


# In[ ]:




