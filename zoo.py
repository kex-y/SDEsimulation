from functions import *

seed = 1
delta = 0.01
endtime = 7
meshsize = 0.1
r = 5
alpha = 0
fps = 24
initradsq = 10
method = HopfSDE

delete = True
animate = True

# Take a realization of the Brownian motion
# realizeddBM = dWs(seed, delta, endtime, 2)
realizeddBM = [(0, 0) for _ in range(int(endtime / delta))]
# Brownian motion (scaled for visibility)
BM = (0.005 / delta) * np.cumsum(realizeddBM, axis = 0)
BMlength = np.cumsum(np.abs(realizeddBM), axis = 0)

# Creating initial conditions of points in the unit circle
init = [np.array([x, y]) 
            for x in np.arange(-initradsq, initradsq, meshsize) 
            for y in np.arange(-initradsq, initradsq, meshsize) 
                if x ** 2 + y ** 2 <= initradsq]# and x ** 2 + y ** 2 >= 0.25]

paths = np.array([method(v, lambda v: rotate(v, r = r, alpha = alpha), delta, realizeddBM) 
            for v in tqdm(init, desc = 'Simulating paths', unit = 'pt')])
pathsT = np.stack(paths, axis = 1)

maxval = max([abs(coord) for path in paths for pt in path for coord in pt])
maxvalBM = max([abs(coord) for coord in BM.flatten()])

# Maximum norm at a given time:
maxnorms = np.array([max([np.linalg.norm(pt) for pt in time]) for time in pathsT])

# Median norm at a given time:
avgnorms = np.array([np.median([np.linalg.norm(pt) for pt in time]) for time in pathsT])

# Time
times = np.arange(0, delta * len(avgnorms), delta)

# Delete frames so the animate process is faster
def deleteFrames(arr, keep):
    return arr[np.round(np.linspace(0, len(arr) - 1, keep)).astype(int)]

# No need to delete frames if we're not animating
if delete and animate:
    
    maxnorms = deleteFrames(maxnorms, endtime * fps)
    avgnorms = deleteFrames(avgnorms, endtime * fps)
    times = deleteFrames(times, endtime * fps)
    pathsT = deleteFrames(pathsT, endtime * fps)

# Animate
if animate:
    
    axislim = min(max(maxval, maxvalBM), 10)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.set_size_inches(5,5)

    def animate(i, points):

        ax1.clear()
        x, y = np.transpose(points[i])
        ax1.scatter(x, y, color='green', label='Evolution of Ball', marker='o', s = 2)
        xB, yB = np.transpose(BM[:i])
        ax1.plot(xB, yB, color = 'black', label = 'Brownian motion')
        # ax1.legend('Evolution of Ball', 'Brownian motion')
        ax1.set_xlim([-axislim, axislim])
        ax1.set_ylim([-axislim, axislim])

        t, maxn, avgn = times[:i], maxnorms[:i], avgnorms[:i]
        ax2.plot(t, maxn, label = 'Max norm', color = 'red')
        ax2.plot(t, avgn, label = 'Average norm', color = 'blue')
        ax2.set_xlim([0, endtime])
        ax2.set_ylim([0, min(30, maxval)])
        # ax2.set_yscale('symlog', base = 2)
        # ax2.set_xscale('symlog', base = 2)
        # ax2.set_yscale('log', base = 2)

        # Align the two plots
        ax1.set_box_aspect(1)
        ax2.set_box_aspect(1)

    ani = FuncAnimation(fig, lambda i: animate(i, pathsT), 
                        frames = tqdm(range(len(pathsT)), desc = 'Animating frames', unit = 'f'),
                        interval = 500, repeat = True)
    plt.close()

    # print(path)
    # Save the animation as an animated GIF
    ani.save("simple_animation.gif", dpi = 300,
            writer = PillowWriter(fps = fps))