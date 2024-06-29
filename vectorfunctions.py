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
