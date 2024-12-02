from functions import *

# make these smaller to increase the resolution
dx, dy = 0.05, 0.05
seed = 5
delta = 0.01
endtime = 5
meshsize = 0.005
r = 3
alpha = 0
fps = 24
size = 2
method = HopfSDE4RK

# Take a realization of the Brownian motion
realizeddBM = dWs(seed, delta, endtime, 2)
# Brownian motion (scaled for visibility)
BM = (0.005 / delta) * np.cumsum(realizeddBM, axis = 0)

# generate 2 2d grids for the x & y bounds
y, x = np.mgrid[slice(1, 5 + dy, dy),
                slice(1, 5 + dx, dx)]

def HopfSDE4RKs(x, y, drift, delta, noise):
    path = [(x, y)]
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
    return path[-1]

z = np.vectorize(HopfSDE4RKs)(x, y, lambda v: rotate(v, r = r, alpha = alpha), delta, realizeddBM)

# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]

# pick the desired colormap, sensible levels, and define a normalization
# instance which takes data values and translates those into levels.
cmap = plt.colormaps['PiYG']

fig, ax0 = plt.subplots()

im = ax0.pcolormesh(x, y, z, cmap=cmap)
fig.colorbar(im, ax=ax0)
ax0.set_title('pcolormesh with levels')
plt.show()