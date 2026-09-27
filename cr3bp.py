import numpy as np

MU = 0.0121505856
R_EARTH = 6378.1363 / 384400.0
R_MOON = 1737.4 / 384400.0
R_ESCAPE = 5.0


def distances(x, y):
    r1 = np.sqrt((x + MU) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y ** 2)
    return r1, r2


def omega(x, y):
    r1, r2 = distances(x, y)
    return 0.5 * (x * x + y * y) + (1.0 - MU) / r1 + MU / r2


def jacobi(x, y, vx, vy):
    return 2.0 * omega(x, y) - (vx * vx + vy * vy)


def rhs(t, state):
    x, y, vx, vy = state
    r1, r2 = distances(x, y)

    dOmega_dx = (
        x
        - (1.0 - MU) * (x + MU) / r1**3
        - MU * (x - 1.0 + MU) / r2**3
    )
    dOmega_dy = (
        y
        - (1.0 - MU) * y / r1**3
        - MU * y / r2**3
    )

    ax = dOmega_dx + 2.0 * vy
    ay = dOmega_dy - 2.0 * vx

    return np.array([vx, vy, ax, ay], dtype=float)


def initial_state_from_x_vx(x0, vx0, C, vy_sign=1.0):
    y0 = 0.0
    vy2 = 2.0 * omega(x0, y0) - C - vx0 * vx0
    if vy2 < 0.0:
        return None
    vy0 = vy_sign * np.sqrt(max(vy2, 0.0))
    return np.array([x0, y0, vx0, vy0], dtype=float)
