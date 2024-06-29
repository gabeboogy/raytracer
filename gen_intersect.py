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
    
def nearest_intersected_object(objects, ray_origin, ray_direction):     #ray tracer may run into multiple objects, determines which is closest
    distances = [intersect(ray_origin,ray_direction,obj) for obj in objects]
    nearest_object = None
    min_distance = np.inf
    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]
    return nearest_object, min_distance
