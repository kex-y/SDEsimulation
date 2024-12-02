# Simulate (via Euler-Maruyama) the Hopf bifurcation with additive noise and see the rate 
# at which the boundary grows.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

def dWs(seed, delta, T, dim = 1):
    np.random.seed(seed)
    return list(zip(*[np.random.normal(0, np.sqrt(delta), int(T / delta)) for _ in range(dim)]))

def rotate(v, r = 1, alpha = 0): 
    return -alpha * v  + np.linalg.norm(v) ** r * np.array([-v[1], v[0]])

# Euler-Maruyama method
def HopfSDE(v, drift, delta, noise):
    path = [v]
    for dW in noise: 
        prev = path[-1]
        if np.linalg.norm(prev) > 10e2:
            path.append(np.array([1000, 0]))
        else:
            path.append(prev + drift(prev) * delta + dW)
    return path

# Euler-Maruyama method with renormalization after each step
def HopfSDErenormalized(v, drift, delta, noise):
    path = [v]
    for dW in noise: 
        prev = path[-1]
        nextNoiseless = prev + drift(prev) * delta
        path.append(nextNoiseless / np.linalg.norm(nextNoiseless) * np.linalg.norm(prev) + dW)
    return path

# AB 2-step method
def HopfSDE2step(v, drift, delta, noise):
    path = [v, v + drift(v) * delta]
    for dW in noise: 
        prev = path[-1]
        prev2 = path[-2]
        if np.linalg.norm(prev) > 10e2 or np.linalg.norm(prev2) > 10e2:
            path.append(np.array([1000, 0]))
        else:
            path.append(prev + (3 * drift(prev) - drift(prev2)) * delta / 2 + dW)
    return path

# Runge-Kutta 4th order method
def HopfSDE4RK(v, drift, delta, noise):
    path = [v]
    hasHit = False
    for dW in noise:
        if hasHit: 
            path.append(np.array([100, 0]))
            continue
        prev = path[-1]
        k1 = drift(prev)
        k2 = drift(prev + k1 * delta / 2)
        k3 = drift(prev + k2 * delta / 2)
        k4 = drift(prev + k3 * delta)
        next = prev + (k1 + 2 * k2 + 2 * k3 + k4) * delta / 6 + dW
        if np.linalg.norm(next) > 10e1:
            if not hasHit:
                # print('Hit the boundary')
                hasHit = True
            path.append(np.array([100, 0]))
        else:
            path.append(next)
    return path
