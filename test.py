# Test to see up to which point is the Runge-Kutta method accurate in the roation model

import numpy as np
from tqdm import tqdm
from math import pi, cos, sin

def rotate(v, r = 1, alpha = 0): 
    return -alpha * v  + np.linalg.norm(v) ** r * np.array([-v[1], v[0]])

def HopfSDE4RK(v, drift, delta, endtime):
    path = [v]
    for _ in range(int(endtime / delta)): 
        prev = path[-1]
        k1 = drift(prev)
        k2 = drift(prev + k1 * delta / 2)
        k3 = drift(prev + k2 * delta / 2)
        k4 = drift(prev + k3 * delta)
        next = prev + (k1 + 2 * k2 + 2 * k3 + k4) * delta / 6
        if np.linalg.norm(next) > 10e1:
            print('Hit the boundary')
            path.append(np.array([100, 0]))
        else:
            path.append(next)
    return path

# Generates points on the cicle
def PointsInCircum(radius, n = 100):
    return [np.array([cos(2 * pi / n * x) * radius, sin(2 * pi / n * x) * radius]) for x in range(0, n)]

r = 2.5
delta = 0.0005
endtime = 16
maxnorm = 100
n = 10
animate = True

HopfSDE4RKrotate = lambda v: HopfSDE4RK(v, lambda v: rotate(v, r = r), delta, endtime)

for norm in range(1, maxnorm):
    paths = np.array([HopfSDE4RKrotate(pt) for pt in 
                        tqdm(PointsInCircum(norm, n = n), leave = False, unit = 'pt', 
                             desc = f'Simulating path for norm = {norm}')]) 
    pathsT = np.stack(paths, axis = 1)

    # Maximum diff at a given time:
    maxdiff = max([abs(max([np.linalg.norm(pt) for pt in time]) - norm) for time in pathsT])
    if maxdiff > 0.1:
        print(f'At norm = {norm}, maxdiff = {maxdiff}')
        break
    