import numpy as np
from numba import njit
from cr3bp import (
    MU,
    R_EARTH,
    R_MOON,
    R_ESCAPE,
    omega,
    rhs_numba,
    initial_state_from_x_vx,
)

OUTCOME_INVALID = 0
OUTCOME_EARTH = 1
OUTCOME_MOON = 2
OUTCOME_ESCAPE = 3
OUTCOME_UNRESOLVED = 4

OUTCOME_MAP = {
    OUTCOME_INVALID: "Invalid",
    OUTCOME_EARTH: "Earth",
    OUTCOME_MOON: "Moon",
    OUTCOME_ESCAPE: "Escape",
    OUTCOME_UNRESOLVED: "Unresolved",
}


@njit(fastmath=True)
def integrate_single_numba(x0, vx0, C, t_max=100.0, rtol=1e-8, atol=1e-10):
    state, valid = initial_state_from_x_vx(x0, vx0, C)
    if not valid:
        return OUTCOME_INVALID, np.nan, 0, np.nan, np.nan, np.nan

    t = 0.0
    dt = 0.01
    n_steps = 0

    min_r1 = np.sqrt((x0 + MU) ** 2)
    min_r2 = np.sqrt((x0 - 1.0 + MU) ** 2)
    max_r = np.abs(x0)

    c2, c3, c4, c5 = 1 / 5, 3 / 10, 4 / 5, 8 / 9
    a21 = 1 / 5
    a31, a32 = 3 / 40, 9 / 40
    a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
    a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    a61, a62, a63, a64, a65 = (
        9017 / 3168,
        -355 / 33,
        46732 / 5247,
        49 / 176,
        -5103 / 18656,
    )
    b1, b3, b4, b5, b6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    e1, e3, e4, e5, e6, e7 = (
        71 / 57600,
        -71 / 16695,
        71 / 1920,
        -17253 / 339200,
        22 / 525,
        -1 / 40,
    )

    k1, k2, k3, k4, k5, k6, k7 = (
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
        np.zeros(4),
    )
    tmp_state = np.zeros(4)

    while t < t_max:
        x, y = state[0], state[1]
        r1 = np.sqrt((x + MU) ** 2 + y * y)
        r2 = np.sqrt((x - 1.0 + MU) ** 2 + y * y)
        r = np.sqrt(x * x + y * y)

        if r1 < min_r1:
            min_r1 = r1
        if r2 < min_r2:
            min_r2 = r2
        if r > max_r:
            max_r = r

        if r1 <= R_EARTH:
            return OUTCOME_EARTH, t, n_steps, min_r1, min_r2, max_r
        if r2 <= R_MOON:
            return OUTCOME_MOON, t, n_steps, min_r1, min_r2, max_r
        if r >= R_ESCAPE:
            return OUTCOME_ESCAPE, t, n_steps, min_r1, min_r2, max_r

        rhs_numba(state, k1)

        tmp_state[:] = state + dt * a21 * k1
        rhs_numba(tmp_state, k2)

        tmp_state[:] = state + dt * (a31 * k1 + a32 * k2)
        rhs_numba(tmp_state, k3)

        tmp_state[:] = state + dt * (a41 * k1 + a42 * k2 + a43 * k3)
        rhs_numba(tmp_state, k4)

        tmp_state[:] = state + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        rhs_numba(tmp_state, k5)

        tmp_state[:] = state + dt * (
            a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5
        )
        rhs_numba(tmp_state, k6)

        new_state = state + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        rhs_numba(new_state, k7)

        err = np.max(
            np.abs(
                dt
                * (
                    e1 * k1
                    + e3 * k3
                    + e4 * k4
                    + e5 * k5
                    + e6 * k6
                    + e7 * k7
                )
            )
        )
        scale = atol + rtol * np.max(np.abs(state))

        if err <= scale or dt <= 1e-12:
            t += dt
            state[:] = new_state
            n_steps += 1
            dt = max(
                1e-12,
                min(
                    dt * min(5.0, max(0.2, 0.9 * (scale / (err + 1e-16)) ** 0.2)),
                    0.5,
                ),
            )
        else:
            dt = max(1e-12, dt * max(0.1, 0.9 * (scale / (err + 1e-16)) ** 0.2))

    return OUTCOME_UNRESOLVED, t_max, n_steps, min_r1, min_r2, max_r


def integrate_state(x0, vx0, C, t_max=100.0, rtol=1e-8, atol=1e-10):
    outcome_code, t_event, n_steps, min_r1, min_r2, max_r = integrate_single_numba(
        x0, vx0, C, t_max, rtol, atol
    )
    state0, valid = initial_state_from_x_vx(x0, vx0, C)
    return {
        "outcome": OUTCOME_MAP[outcome_code],
        "t_event": t_event,
        "state0": state0 if valid else None,
        "solver_success": True,
        "n_steps": n_steps,
        "min_r_earth": min_r1,
        "min_r_moon": min_r2,
        "max_r": max_r,
    }