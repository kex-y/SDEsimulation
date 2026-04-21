# Simulate (via Euler-Maruyama) the Hopf bifurcation with additive noise and see the rate
# at which the boundary grows.

import numpy as np

def dWs(seed, delta, T, dim = 1):
    np.random.seed(seed)
    return list(zip(*[np.random.normal(0, np.sqrt(delta), int(T / delta)) for _ in range(dim)]))

def rotate(v, r = 1, alpha = 0): 
    return -alpha * v  + np.linalg.norm(v) ** r * np.array([-v[1], v[0]])

def rotate_batched(V, r = 1, alpha = 0):
    """Vectorized drift: V is (N, 2) -> returns (N, 2).

    Matches `rotate` elementwise over the first axis.
    """
    scale = np.linalg.norm(V, axis=1) ** r        # (N,)
    rot = np.empty_like(V)
    rot[:, 0] = -V[:, 1]
    rot[:, 1] =  V[:, 0]
    return -alpha * V + scale[:, None] * rot

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


def HopfSDE4RK_batched(init, drift_batched, delta, noise,
                       idx_frames, out,
                       state=None, hasHit=None, start_step=0,
                       snapshot_cb=None, snapshot_every=0,
                       progress=None):
    """Vectorized RK4 SDE integrator (additive shared noise).

    Advances N trajectories simultaneously using a single (N, 2)-valued
    numpy RK4 step. Equivalent to looping `HopfSDE4RK` over N initial
    conditions and the same Brownian path, but ~O(N) faster because the
    Python-level loop runs only over time steps, not trajectories.

    Parameters
    ----------
    init : (N, 2) ndarray
        Initial states. Unused if `state` is passed (resume mode).
    drift_batched : callable (N, 2) -> (N, 2)
    delta : float
    noise : (nsteps, 2) ndarray
        Brownian increments, broadcast across all trajectories.
    idx_frames : (nframes,) ndarray[int]
        Step indices whose state should be written to `out`. Must be
        increasing, start at 0, and end at nsteps.
    out : array-like of shape (nframes, N, 2)
        Writable output (typically a numpy memmap).
    state, hasHit, start_step : resume-from-snapshot inputs. If `state`
        is None the run starts fresh from `init`.
    snapshot_cb : optional callable(step, state, hasHit) invoked at the
        end of every `snapshot_every` steps. Used for persistent
        progress checkpoints so the integrator can be resumed.
    snapshot_every : int, 0 disables periodic snapshots.
    progress : optional tqdm-like object with an `.update(n)` method.

    Returns
    -------
    (state, hasHit) : the final state and blow-up mask.
    """
    nsteps = len(noise)
    N = (state if state is not None else init).shape[0]

    # Build step -> frame map for O(1) dispatch inside the hot loop.
    step_to_frame = {int(s): i for i, s in enumerate(idx_frames)}

    if state is None:
        state = np.asarray(init, dtype=np.float64).copy()
        hasHit = np.zeros(N, dtype=bool)
        start_step = 0
        if 0 in step_to_frame:
            out[step_to_frame[0], :, :] = state
    else:
        state = np.asarray(state, dtype=np.float64)
        if hasHit is None:
            hasHit = np.zeros(N, dtype=bool)

    cap = np.array([100.0, 0.0], dtype=np.float64)
    half = delta / 2.0
    sixth = delta / 6.0

    # Blow-up of individual trajectories produces legitimate float overflow
    # warnings in the drift computation for the rows we are about to clamp.
    # Silence them inside the hot loop; the clamp below keeps the output finite.
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(start_step, nsteps):
            dW = noise[step]
            k1 = drift_batched(state)
            k2 = drift_batched(state + k1 * half)
            k3 = drift_batched(state + k2 * half)
            k4 = drift_batched(state + k3 * delta)
            nxt = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * sixth + dW
            # Blow-up detection. Using `~(norm <= 100)` rather than `norm > 100`
            # so that NaN norms (from sign-cancellation of infinities inside the
            # RK4 stages) are also flagged as blown up. Matches HopfSDE4RK's
            # `> 10e1` threshold for finite values.
            norms = np.linalg.norm(nxt, axis=1)
            big = ~(norms <= 100.0)
            if big.any() or hasHit.any():
                hasHit |= big
                nxt[hasHit] = cap       # boolean-index row assign, broadcasts cap
            state = nxt

            done_step = step + 1
            if done_step in step_to_frame:
                out[step_to_frame[done_step], :, :] = state
            if progress is not None:
                progress.update(1)
            if snapshot_every and snapshot_cb is not None and done_step % snapshot_every == 0:
                snapshot_cb(done_step, state, hasHit)

    return state, hasHit

