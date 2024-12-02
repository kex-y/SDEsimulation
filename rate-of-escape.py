from functions import *

seed = 5
delta = 0.01
endtime = 7
meshsize = 0.01
r = 5
alpha = 0
fps = 24
size = 5
method = HopfSDE4RK

# Take a realization of the Brownian motion
# realizeddBM = dWs(seed, delta, endtime, 2)
realizeddBM = [(0, 0) for _ in range(int(endtime / delta))]
# Brownian motion (scaled for visibility)
BM = (0.005 / delta) * np.cumsum(realizeddBM, axis = 0)

# Creating initial conditions of points in the unit circle
xs = np.arange(-size, size, meshsize)
ys = np.arange(-size, size, meshsize)
# init = np.array([np.array([x, y]) 
#             for x in np.arange(-size, size, meshsize) 
#             for y in np.arange(-size, size, meshsize)])# and x ** 2 + y ** 2 >= 0.25]

finalpos = np.array([[np.linalg.norm(method(np.array([x, y]), lambda v: rotate(v, r = r, alpha = alpha), delta, realizeddBM)[-1]) 
            for x in xs] for y in tqdm(ys, desc = 'Simulating paths', unit = 'pt')])

X, Y = np.meshgrid(xs, ys)
# Plot the norm at the end time via a color gradient
# hist, xedges, yedges = np.histogram2d(init[:,0], init[:,1], weights = finalpos)
plt.pcolormesh(X, Y, finalpos)
plt.show()