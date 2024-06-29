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



#resolution
width = 640
height = 480

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
