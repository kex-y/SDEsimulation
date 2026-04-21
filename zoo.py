# Force non-interactive backend BEFORE any matplotlib import (incl. transitively via
# functions.py). This avoids the macOS Cocoa runtime being initialised at all, which
# is the usual source of segfaults at interpreter teardown.
import os
os.environ["MPLBACKEND"] = "Agg"

# Silence the harmless `multiprocessing.resource_tracker: leaked semaphore` warning
# that numpy/Pillow occasionally produce on macOS. The tracker is a separate Python
# subprocess, so an in-process `warnings.filterwarnings` does not reach it; setting
# PYTHONWARNINGS in the environment before the subprocess is spawned does.
os.environ["PYTHONWARNINGS"] = (
    os.environ.get("PYTHONWARNINGS", "")
    + ("," if os.environ.get("PYTHONWARNINGS") else "")
    + "ignore::UserWarning:multiprocessing.resource_tracker"
)

import matplotlib
matplotlib.use("Agg", force=True)

import gc
import sys
import warnings
import hashlib
import json

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import patheffects as pe
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

from functions import (
    HopfSDE4RK,
    HopfSDE4RK_batched,
    dWs,
    rotate,
    rotate_batched,
)

# Also filter it in-process, in case the tracker runs inside the parent.
warnings.filterwarnings(
    "ignore", category=UserWarning, module=r"multiprocessing\.resource_tracker"
)

# ---------------- Parameters ----------------
seed        = 1
delta       = 0.01
endtime     = 10
meshsize    = 0.001
r           = 3
alpha       = 0
size        = 1           # half-width of the square [-size, size]^2 of initial conditions
axislim     = 5.0         # half-width of the viewing window for panel 1 (flow image)
method      = HopfSDE4RK
noise       = True       # toggle Brownian noise in the SDE on/off
show_brownian_path = False  # draw the realized W_t overlay on panel 1
downsample  = True
animate_on  = True
out_path    = "simple_animation.gif"

# ---- Animation performance knobs ----
# Upper bound on the number of scatter markers drawn per frame in panel 1.
# For huge grids (millions of initial conditions) scatter points overplot into
# a solid mass anyway, so randomly subsampling to ~5e5 is visually identical
# but roughly 20x cheaper to render. Set to None to draw every trajectory.
scat_subsample_max = 500_000
# Save resolution. 100 dpi on a 15x5 figure gives 1500x520 px which is plenty
# for a GIF; 150 dpi was ~2.25x more pixels for no visible quality gain.
save_dpi           = 100

# ---- Playback configuration ----
# Two distinct quantities, separated:
#
#   * `fps` = SIMULATION DENSITY -- how many frames per second of simulation
#     time are persisted to the cache. Auto-derived from GIF_MAX_FPS and
#     time_scale below; higher = smoother motion, bigger cache.
#
#   * `playback_fps` = PLAYBACK RATE -- how many of those frames are shown
#     per second when the GIF is viewed. Must be <= GIF_MAX_FPS because GIF
#     stores per-frame delay in centiseconds (1/100 s), so rates above ~50
#     round to identical delays, and many viewers (macOS QuickLook, older
#     browsers) clamp short delays to ~10 fps.
#
#   * `time_scale` = playback duration / real duration. Defined so that
#     nframes/playback_fps = endtime*time_scale. `time_scale > 1` plays back
#     slower than real time; `= 1` is real-time.
GIF_MAX_FPS = 50
time_scale  = 1.5

# Auto-derivation. We always play back at the maximum GIF-safe rate and bake
# the time_scale into how many frames we simulate, so slow-motion stays
# smooth (no repeated frames / step-y look). Override `fps` below the block
# if you want to manually trade smoothness for cache size.
_is_gif      = out_path.lower().endswith(".gif")
_base_pfs    = GIF_MAX_FPS if _is_gif else GIF_MAX_FPS
fps          = int(round(_base_pfs * time_scale))
playback_fps = fps / time_scale

# ---------------- Styling ----------------
plt.style.use("dark_background")
mpl.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "axes.edgecolor":   "#777777",
    "axes.linewidth":   0.8,
    "xtick.color":      "#bbbbbb",
    "ytick.color":      "#bbbbbb",
    "axes.labelcolor":  "#dddddd",
    "axes.titlecolor":  "#ffffff",
    "figure.facecolor": "#0a0a0a",
    "axes.facecolor":   "#111111",
    "savefig.facecolor":"#0a0a0a",
})



# ---------------- Square grid of initial conditions ----------------
xs = np.arange(-size, size + meshsize, meshsize)
ys = np.arange(-size, size + meshsize, meshsize)
nx, ny = len(xs), len(ys)
XX, YY = np.meshgrid(xs, ys)
init_flat = np.stack([XX.ravel(), YY.ravel()], axis=1)   # shape (N, 2)

# ---------------- Simulation cache (resumable, memory-mapped) ----------------
# The expensive simulation is cached to disk so iterating on styling does not
# trigger a recompute. The cache is keyed on every parameter that affects the
# numerical result; styling / output-path / animate_on are excluded. NOTE:
# edits to the math in functions.py (drift, integrator, noise generator) are
# NOT detected -- delete the `.cache/` directory manually in that case.
#
# For large grids (millions of initial conditions) two design decisions matter:
#
#   * We only ever keep the DOWNSAMPLED trajectory frames on disk (nframes_total
#     per initial condition, not nsteps+1). Since the animation only ever shows
#     these frames, discarding the per-step history cuts storage & RAM by the
#     factor nsteps / nframes_total (typically 4x-25x).
#
#   * The trajectory array lives in a memory-mapped .npy file. Each trajectory
#     is written into its column as soon as it is simulated, and a small JSON
#     progress file is flushed periodically. If the process is killed (OOM or
#     otherwise), the next run picks up exactly where it left off.
#
# fps/downsample affect the cached frame resolution, so they are part of the key.
_cache_payload = {
    "seed":       seed,
    "delta":      delta,
    "endtime":    endtime,
    "meshsize":   meshsize,
    "size":       size,
    "r":          r,
    "alpha":      alpha,
    "noise":      bool(noise),
    "method":     method.__name__,
    "drift":      "rotate",
    "fps":        fps,
    "downsample": bool(downsample),
}
_cache_key  = hashlib.sha1(json.dumps(_cache_payload, sort_keys=True).encode()).hexdigest()[:16]
_cache_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
os.makedirs(_cache_dir, exist_ok=True)
_paths_npy     = os.path.join(_cache_dir, f"pathsT_{_cache_key}.npy")
_progress_json = os.path.join(_cache_dir, f"progress_{_cache_key}.json")
_bm_npy        = os.path.join(_cache_dir, f"dBM_{_cache_key}.npy")
_stats_npz     = os.path.join(_cache_dir, f"stats_{_cache_key}.npz")

N        = len(init_flat)
nsteps   = int(endtime / delta)
nframes_total = int(endtime * fps) if downsample else (nsteps + 1)
# Indices (into the 0..nsteps step grid) that we persist to disk.
idx_frames = np.round(np.linspace(0, nsteps, nframes_total)).astype(int)

# ---- Brownian increments (small, cached as its own file) ----
if os.path.exists(_bm_npy):
    realizeddBM = np.load(_bm_npy)
else:
    if noise:
        realizeddBM = np.asarray(dWs(seed, delta, endtime, 2), dtype=float)
    else:
        realizeddBM = np.zeros((nsteps, 2), dtype=float)
    np.save(_bm_npy, realizeddBM)
BM = (0.005 / delta) * np.cumsum(realizeddBM, axis=0)

_state_npz = os.path.join(_cache_dir, f"state_{_cache_key}.npz")

def _read_progress():
    if os.path.exists(_progress_json):
        try:
            with open(_progress_json) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _write_progress(completed_steps, done=False):
    """`completed_steps` is the step counter for the vectorized integrator
    (0 <= completed_steps <= nsteps). The serial fallback stores the
    per-trajectory count in the same field; the two representations never
    coexist for a given key because `method` is part of the cache key."""
    with open(_progress_json, "w") as f:
        json.dump({
            "completed_steps": int(completed_steps),
            "N":               int(N),
            "nframes":         int(nframes_total),
            "nsteps":          int(nsteps),
            "done":            bool(done),
        }, f)

# ---- Dispatch: prefer the vectorized integrator when available ----
# The serial integrators in functions.py accept a single (2,) state. We
# have a batched RK4 that advances all N trajectories in one numpy call;
# use it whenever the chosen method has a matching batched implementation,
# otherwise fall back to the per-trajectory loop.
_BATCHED_IMPL = {
    "HopfSDE4RK": HopfSDE4RK_batched,
}
_batched_method = _BATCHED_IMPL.get(method.__name__)

# ---- Trajectory memmap: fresh, resume partial, or reuse completed ----
_prog     = _read_progress()
_shape_ok = (_prog.get("N") == N and _prog.get("nframes") == nframes_total)
_complete = bool(_prog.get("done")) and _shape_ok and os.path.exists(_paths_npy)

if _complete:
    pathsT = np.load(_paths_npy, mmap_mode="r")
    print(f"[cache] Loaded simulation from {_paths_npy}  "
          f"(N={N}, nframes={nframes_total})")
else:
    reuse = _shape_ok and os.path.exists(_paths_npy)
    if reuse:
        pathsT = np.lib.format.open_memmap(_paths_npy, mode="r+")
    else:
        # Stale or missing file -- recreate from scratch.
        for p in (_paths_npy, _stats_npz, _state_npz):
            if os.path.exists(p):
                os.remove(p)
        pathsT = np.lib.format.open_memmap(
            _paths_npy, mode="w+",
            dtype=np.float64, shape=(nframes_total, N, 2),
        )
        _write_progress(0, done=False)

    try:
        if _batched_method is not None:
            # ---- Vectorized (numpy-batched) RK4 ----
            # Advances all N trajectories simultaneously; Python-level loop is
            # over time steps only, so cost scales as nsteps, not nsteps*N.
            SNAPSHOT_EVERY = 20                     # steps between flushes
            start_step = 0
            state_in   = init_flat
            hasHit_in  = None
            if reuse and os.path.exists(_state_npz):
                _snap = np.load(_state_npz)
                if _snap["state"].shape == (N, 2):
                    state_in   = _snap["state"]
                    hasHit_in  = _snap["hasHit"]
                    start_step = int(_snap["step"])
                    print(f"[cache] Resuming vectorized integrator at step "
                          f"{start_step}/{nsteps}")
                _snap.close()

            with tqdm(initial=start_step, total=nsteps,
                      desc="Simulating (vectorized)", unit="step") as pbar:
                def _snap_cb(step, state, hasHit):
                    pathsT.flush()
                    np.savez(_state_npz, state=state, hasHit=hasHit, step=step)
                    _write_progress(step, done=False)

                state_in_arr = (state_in if state_in is not init_flat
                                else np.asarray(init_flat, dtype=np.float64))
                _batched_method(
                    init=init_flat,
                    drift_batched=lambda V: rotate_batched(V, r=r, alpha=alpha),
                    delta=delta,
                    noise=realizeddBM,
                    idx_frames=idx_frames,
                    out=pathsT,
                    state=(state_in_arr if start_step > 0 else None),
                    hasHit=hasHit_in,
                    start_step=start_step,
                    snapshot_cb=_snap_cb,
                    snapshot_every=SNAPSHOT_EVERY,
                    progress=pbar,
                )
        else:
            # ---- Serial fallback (per-trajectory Python loop) ----
            FLUSH_EVERY = 2048
            start_j = int(_prog.get("completed_steps", 0)) if reuse else 0
            if reuse:
                print(f"[cache] Resuming serial simulation at trajectory {start_j}/{N}")
            last_j = start_j
            with tqdm(initial=start_j, total=N,
                      desc="Simulating paths", unit="pt") as pbar:
                for j in range(start_j, N):
                    traj = method(init_flat[j],
                                  lambda v: rotate(v, r=r, alpha=alpha),
                                  delta, realizeddBM)
                    pathsT[:, j, :] = np.asarray(traj)[idx_frames]
                    last_j = j + 1
                    pbar.update(1)
                    if last_j % FLUSH_EVERY == 0:
                        pathsT.flush()
                        _write_progress(last_j, done=False)

        pathsT.flush()
        _write_progress(nsteps if _batched_method else N, done=True)
        if os.path.exists(_state_npz):
            os.remove(_state_npz)
        print(f"[cache] Saved simulation to {_paths_npy}")
    except BaseException:
        # KeyboardInterrupt, OOM, anything -- persist progress before re-raising.
        try:
            pathsT.flush()
            # Best-effort snapshot: the batched path already snapshots
            # periodically, the serial path updates progress in-loop.
        except Exception:
            pass
        raise

    # Reopen read-only for downstream use (frees the r+ handle).
    del pathsT
    pathsT = np.load(_paths_npy, mmap_mode="r")

times  = idx_frames * delta
idx_BM = np.clip(idx_frames, 0, max(len(BM) - 1, 0))

# ---- Per-frame summary statistics (cheap; cached once) ----
if os.path.exists(_stats_npz):
    _st = np.load(_stats_npz)
    maxnorms = _st["maxnorms"]
    avgnorms = _st["avgnorms"]
    p95norms = _st["p95norms"]
    _st.close()
else:
    maxnorms = np.empty(nframes_total, dtype=np.float64)
    avgnorms = np.empty(nframes_total, dtype=np.float64)
    p95norms = np.empty(nframes_total, dtype=np.float64)
    for i in tqdm(range(nframes_total), desc="Computing norm stats", unit="f"):
        ni = np.linalg.norm(pathsT[i], axis=1)
        maxnorms[i] = float(np.nanmax(ni))
        avgnorms[i] = float(np.nanmedian(ni))
        p95norms[i] = float(np.nanpercentile(ni, 95))
    np.savez(_stats_npz, maxnorms=maxnorms, avgnorms=avgnorms, p95norms=p95norms)

# ---------------- Animate ----------------
if animate_on:

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15, 5.2),
        gridspec_kw={"width_ratios": [1, 1.15, 1]},
    )

    # Scatter (panel 1): each particle is colored by the local particle
    # density in the viewing window. We bin the (subsampled) positions onto
    # a DENSITY_BINS x DENSITY_BINS regular grid once per frame, Gaussian-
    # smooth the count field, and each particle reads back the *smoothed*
    # density at its bin. The smoothing is essential: raw histogram counts
    # produce visible circular "blobs" of uniform color where all particles
    # sharing one bin get identical density. A separable Gaussian kernel
    # with sigma ~= DENSITY_SMOOTH_SIGMA bins blurs across that scale so
    # density varies continuously in screen space.
    DENSITY_BINS         = 200
    DENSITY_SMOOTH_SIGMA = 2.5           # bins
    _hist_range          = [[-axislim, axislim], [-axislim, axislim]]

    # Precompute 1D Gaussian kernel used for the separable blur.
    _blur_radius = int(3 * DENSITY_SMOOTH_SIGMA) + 1
    _blur_x      = np.arange(-_blur_radius, _blur_radius + 1, dtype=np.float64)
    _blur_kernel = np.exp(-0.5 * (_blur_x / DENSITY_SMOOTH_SIGMA) ** 2)
    _blur_kernel /= _blur_kernel.sum()

    def _gaussian_blur_2d(img):
        # Separable 2D Gaussian: convolve along rows, then columns.
        tmp = np.apply_along_axis(lambda a: np.convolve(a, _blur_kernel, mode="same"), 0, img)
        return  np.apply_along_axis(lambda a: np.convolve(a, _blur_kernel, mode="same"), 1, tmp)

    def _density_at(pts):
        counts, _, _ = np.histogram2d(
            pts[:, 0], pts[:, 1],
            bins=DENSITY_BINS, range=_hist_range,
        )
        counts = _gaussian_blur_2d(counts)
        # Bin index of every particle (floor); points outside the view
        # window get density 0 so they don't artificially inflate edge bins.
        x_rel = (pts[:, 0] + axislim) / (2.0 * axislim)
        y_rel = (pts[:, 1] + axislim) / (2.0 * axislim)
        ix = np.floor(x_rel * DENSITY_BINS).astype(np.int64)
        iy = np.floor(y_rel * DENSITY_BINS).astype(np.int64)
        out = (ix < 0) | (ix >= DENSITY_BINS) | (iy < 0) | (iy >= DENSITY_BINS)
        np.clip(ix, 0, DENSITY_BINS - 1, out=ix)
        np.clip(iy, 0, DENSITY_BINS - 1, out=iy)
        d = counts[ix, iy]
        d[out] = 0.0
        return d

    # Heatmap (panel 2) uses a blue -> white ramp only (no red). Values above
    # vmax simply saturate at white; yellow is reserved for initial conditions
    # whose trajectory has hit the blow-up clamp in the integrator (|X_t| ==
    # 100) or degenerated to NaN. The masking to NaN is done at draw time so
    # the clamp color is picked up by the colormap's `bad` slot.
    heat_vmax        = 10.0
    BLOWUP_THRESHOLD = 100.0                  # matches HopfSDE4RK's clamp
    # Shared accent yellow: blow-up cells in the heatmap, the Brownian-path
    # overlay, and the max-norm line in panel 3 all use this color so that
    # "norm has grown" events are visually unified across panels.
    ACCENT_COLOR     = "#ffdf33"
    BLOWUP_COLOR     = ACCENT_COLOR
    heat_norm = Normalize(vmin=0.0, vmax=heat_vmax)
    heat_cmap = mpl.colormaps["Blues_r"].with_extremes(
        over="white",          # 10 < |X_t| < 100 clips to white
        bad=BLOWUP_COLOR,      # |X_t| >= 100 (masked to NaN below) or real NaN
    )

    def _heat_values(ni):
        """Mask blown-up cells to NaN so the `bad` color is applied."""
        return np.where(ni >= BLOWUP_THRESHOLD, np.nan, ni)

    # -------- Tick spacing (equal gap on x and y for every panel) --------
    # Heatmap tick step is chosen from the initial-condition size; the flow-image
    # panel tick step is derived from axislim so the two stay in sync when you
    # change the viewing window.
    if size <= 1:
        step_heat = 0.5
    elif size <= 3:
        step_heat = 1.0
    else:
        step_heat = max(1.0, round(size / 3))
    # Aim for ~5 major ticks per side in the flow-image panel.
    step_cloud = max(0.5, round(axislim / 5))

    # -------- Panel 1: particle cloud + BM path --------
    ax1.set_xlim(-axislim, axislim)
    ax1.set_ylim(-axislim, axislim)
    ax1.set_box_aspect(1)
    ax1.xaxis.set_major_locator(MultipleLocator(step_cloud))
    ax1.yaxis.set_major_locator(MultipleLocator(step_cloud))
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title("Particle cloud")
    ax1.grid(False)

    # Random subsample of trajectory indices for the scatter layer. The
    # heatmap still uses every trajectory (full resolution); the scatter
    # just skips a consistent subset of particles to keep rendering cheap.
    if scat_subsample_max is not None and N > scat_subsample_max:
        _scat_idx = np.random.default_rng(0).choice(
            N, scat_subsample_max, replace=False
        )
        _scat_idx.sort()
    else:
        _scat_idx = np.arange(N)

    _pts0 = pathsT[0][_scat_idx]
    _d0   = _density_at(_pts0)
    # Custom white -> sky blue -> deep blue ramp. The range is traversed
    # by adaptive per-frame normalization below -- with a uniform initial
    # grid, absolute densities barely vary, so a fixed norm would leave
    # every particle the same color. Mapping instead to the current
    # frame's 5th..95th percentile stretches even tiny variations across
    # the full cmap, making density structure visible from frame 1.
    #
    # High-density endpoint (#08306b) is the darkest color of matplotlib's
    # "Blues" colormap -- the same shade as the low end of the heatmap's
    # "Blues_r" ramp, so the two panels reinforce each other visually.
    density_cmap = LinearSegmentedColormap.from_list(
        "density_blue",
        ["white", "#4292c6", "#08306b"],
    )
    _p5,  _p95 = np.percentile(_d0, [5, 95])
    scat = ax1.scatter(_pts0[:, 0], _pts0[:, 1],
                       c=_d0, cmap=density_cmap,
                       vmin=float(_p5), vmax=float(max(_p95, _p5 + 1)),
                       s=2, edgecolors="none", alpha=0.9)
    if show_brownian_path:
        bm_line, = ax1.plot([], [], color=ACCENT_COLOR, lw=1.0, alpha=0.95,
                            label=r"$W_t$")
        # Thin black halo so the yellow line stays readable against the
        # bright white regions of the density scatter. `Stroke` draws a
        # wider black line first, `Normal` then renders the original
        # yellow line on top.
        bm_line.set_path_effects([
            pe.Stroke(linewidth=2, foreground="black"),
            pe.Normal(),
        ])
        ax1.legend(loc="upper right", frameon=False, fontsize=9)
    else:
        bm_line = None

    # -------- Panel 2: heatmap over initial-condition square --------
    # imshow (single texture upload) is substantially faster than pcolormesh
    # (per-quad path rendering) for large regular grids. `extent` is padded
    # by meshsize/2 so cell centers sit on the mesh points, matching the
    # previous pcolormesh output pixel-for-pixel.
    _ext = (xs[0] - meshsize / 2, xs[-1] + meshsize / 2,
            ys[0] - meshsize / 2, ys[-1] + meshsize / 2)
    _n0_all = np.linalg.norm(pathsT[0], axis=1)
    heat = ax2.imshow(_heat_values(_n0_all).reshape(ny, nx),
                      cmap=heat_cmap, norm=heat_norm,
                      origin="lower", extent=_ext,
                      interpolation="nearest", aspect="equal")
    ax2.set_box_aspect(1)
    ax2.xaxis.set_major_locator(MultipleLocator(step_heat))
    ax2.yaxis.set_major_locator(MultipleLocator(step_heat))
    ax2.set_xlabel(r"$x_0$")
    ax2.set_ylabel(r"$y_0$")
    ax2.set_title(r"$|X_t(x_0)|$ as a function of $x_0$")
    cbar = fig.colorbar(heat, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label(r"$|X_t|$", color="#dddddd")
    cbar.ax.yaxis.set_tick_params(color="#bbbbbb")
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("#777777")

    # Legend: make the "yellow = blown up" convention explicit.
    blowup_handle = Patch(facecolor=BLOWUP_COLOR, edgecolor="none",
                          label=fr"blew up ($|X_t| \geq {BLOWUP_THRESHOLD:g}$)")
    ax2.legend(handles=[blowup_handle], loc="upper right",
               frameon=False, fontsize=9, labelcolor="#dddddd")

    # -------- Panel 3: max / median norm vs time --------
    # Round y_upper up to the next multiple of the tick step so both x and y end
    # on a labeled tick; pick a single step used for both axes so the gap is equal.
    y_upper_raw = float(np.nanmax(maxnorms))
    y_upper_raw = max(min(y_upper_raw * 1.1, 30.0), 5.0)
    step_time = max(1.0, round(max(endtime, y_upper_raw) / 7))
    y_upper = step_time * np.ceil(y_upper_raw / step_time)

    ax3.set_xlim(0, endtime)
    ax3.set_ylim(0, y_upper)
    ax3.set_box_aspect(1)
    ax3.xaxis.set_major_locator(MultipleLocator(1.0))   # one tick per second
    ax3.yaxis.set_major_locator(MultipleLocator(step_time))
    ax3.set_xlabel(r"time $t$")
    ax3.set_ylabel("norm")
    ax3.set_title("Ensemble norm statistics")
    ax3.grid(False)
    max_line, = ax3.plot([], [], color=ACCENT_COLOR, lw=1.8, label=r"$\max_{x_0}|X_t|$")
    avg_line, = ax3.plot([], [], color="#4dd0ff", lw=1.8, label=r"median $|X_t|$")
    ax3.legend(loc="upper left", frameon=False, fontsize=9)

    # Super title with SDE + time
    suptitle = fig.suptitle("", fontsize=13, color="#ffffff", y=0.98)

    def update(i):
        pts_all = pathsT[i]                               # memmap slice (N, 2)
        ni_all  = np.linalg.norm(pts_all, axis=1)         # (N,)
        # Scatter positions change every frame; recolor each particle by
        # the local density in the current frame's binning.
        pts_sub = pts_all[_scat_idx]
        d_sub   = _density_at(pts_sub)
        scat.set_offsets(pts_sub)
        scat.set_array(d_sub)
        # Rescale the color norm to the current frame's density spread so
        # the full cmap range is always used (see init-time comment).
        p5, p95 = np.percentile(d_sub, [5, 95])
        scat.set_clim(float(p5), float(max(p95, p5 + 1)))
        # Heatmap still uses every initial condition (full resolution).
        heat.set_data(_heat_values(ni_all).reshape(ny, nx))
        if bm_line is not None:
            bm_line.set_data(BM[:idx_BM[i], 0], BM[:idx_BM[i], 1])
        max_line.set_data(times[:i + 1], maxnorms[:i + 1])
        avg_line.set_data(times[:i + 1], avgnorms[:i + 1])
        suptitle.set_text(
            rf"$dX_t = |X_t|^{{{r}}}\, J\, X_t\, dt + dW_t$"
            rf"$\qquad t = {times[i]:.2f}$"
        )
        artists = [scat, heat, max_line, avg_line]
        if bm_line is not None:
            artists.append(bm_line)
        return tuple(artists)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    # Do NOT pass a tqdm iterator as `frames=` (keeps references that break clean
    # shutdown on macOS). Feed a plain range and drive tqdm via the save callback.
    nframes_total = len(pathsT)
    ani = FuncAnimation(
        fig, update,
        frames=range(nframes_total),
        interval=1000 / playback_fps, blit=False, repeat=False,
    )

    with tqdm(total=nframes_total, desc="Animating frames", unit="f") as pbar:
        ani.save(
            out_path, dpi=save_dpi, writer=PillowWriter(fps=int(round(playback_fps))),
            progress_callback=lambda i, n: pbar.update(1),
        )

    # Drop references and tear everything down before Python exits. Then skip the
    # default atexit path, which is where macOS matplotlib builds occasionally
    # segfault even with the Agg backend.
    del ani
    plt.close(fig)
    plt.close("all")
    gc.collect()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
