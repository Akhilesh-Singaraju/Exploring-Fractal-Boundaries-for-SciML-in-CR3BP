import numpy as np
from numba import njit

MU = 0.0121505856
R_EARTH = 6378.1363 / 384400.0
R_MOON = 1737.4 / 384400.0
R_ESCAPE = 5.0


@njit(fastmath=True)
def distances(x, y):
    dx1 = x + MU
    dx2 = x - 1.0 + MU
    r1 = np.sqrt(dx1 * dx1 + y * y)
    r2 = np.sqrt(dx2 * dx2 + y * y)
    return r1, r2


@njit(fastmath=True)
def omega(x, y):
    r1, r2 = distances(x, y)
    return 0.5 * (x * x + y * y) + (1.0 - MU) / r1 + MU / r2


@njit(fastmath=True)
def jacobi(x, y, vx, vy):
    return 2.0 * omega(x, y) - (vx * vx + vy * vy)


@njit(fastmath=True)
def rhs_numba(state, out):
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    dx1 = x + MU
    dx2 = x - 1.0 + MU
    r1_sq = dx1 * dx1 + y * y
    r2_sq = dx2 * dx2 + y * y

    r1_3 = r1_sq * np.sqrt(r1_sq)
    r2_3 = r2_sq * np.sqrt(r2_sq)

    dOmega_dx = x - (1.0 - MU) * dx1 / r1_3 - MU * dx2 / r2_3
    dOmega_dy = y - (1.0 - MU) * y / r1_3 - MU * y / r2_3

    out[0] = vx
    out[1] = vy
    out[2] = dOmega_dx + 2.0 * vy
    out[3] = dOmega_dy - 2.0 * vx


@njit(fastmath=True)
def initial_state_from_x_vx(x0, vx0, C, vy_sign=1.0):
    y0 = 0.0
    vy2 = 2.0 * omega(x0, y0) - C - vx0 * vx0
    if vy2 < 0.0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64), False
    vy0 = vy_sign * np.sqrt(vy2)
    return np.array([x0, y0, vx0, vy0], dtype=np.float64), True

OUTCOME_MAP = {
    0: "INVALID_ENERGY",
    1: "IMPACT_EARTH",
    2: "IMPACT_MOON",
    3: "ESCAPE",
    4: "MAX_TIME",
}

@njit(fastmath=True)
def rk4_step(state, dt):
    k1 = np.empty(4, dtype=np.float64)
    k2 = np.empty(4, dtype=np.float64)
    k3 = np.empty(4, dtype=np.float64)
    k4 = np.empty(4, dtype=np.float64)

    rhs_numba(state, k1)
    rhs_numba(state + 0.5 * dt * k1, k2)
    rhs_numba(state + 0.5 * dt * k2, k3)
    rhs_numba(state + dt * k3, k4)

    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(fastmath=True)
def integrate_single_numba(x0, vx0, C, t_max, dt=0.01):
    state, valid = initial_state_from_x_vx(x0, vx0, C)
    if not valid:
        return 0, 0.0, 0, 0.0, 0.0, 0.0

    t = 0.0
    steps = 0
    min_r1 = 1e9
    min_r2 = 1e9
    max_r = 0.0

    while t < t_max:
        r1, r2 = distances(state[0], state[1])
        r_center = np.sqrt(state[0] ** 2 + state[1] ** 2)

        if r1 < min_r1:
            min_r1 = r1
        if r2 < min_r2:
            min_r2 = r2
        if r_center > max_r:
            max_r = r_center

        if r1 < R_EARTH:
            return 1, t, steps, min_r1, min_r2, max_r
        if r2 < R_MOON:
            return 2, t, steps, min_r1, min_r2, max_r
        if r_center > R_ESCAPE:
            return 3, t, steps, min_r1, min_r2, max_r

        state = rk4_step(state, dt)
        t += dt
        steps += 1

    return 4, t_max, steps, min_r1, min_r2, max_r