import numpy as np
from scipy.integrate import solve_ivp

from cr3bp import (
    MU,
    R_EARTH,
    R_MOON,
    R_ESCAPE,
    rhs,
    initial_state_from_x_vx,
    distances,
)


def earth_event(t, state):
    x, y, *_ = state
    r1, _ = distances(x, y)
    return r1 - R_EARTH

earth_event.terminal = True
earth_event.direction = -1


def moon_event(t, state):
    x, y, *_ = state
    _, r2 = distances(x, y)
    return r2 - R_MOON

moon_event.terminal = True
moon_event.direction = -1


def escape_event(t, state):
    x, y, vx, vy = state
    r = np.hypot(x, y)
    return r - R_ESCAPE

escape_event.terminal = True
escape_event.direction = 1


def integrate_state(x0, vx0, C, t_max=100.0, rtol=1e-10, atol=1e-12, max_step=0.05):
    state0 = initial_state_from_x_vx(x0, vx0, C)
    if state0 is None:
        return {
            "outcome": "Invalid",
            "t_event": np.nan,
            "state0": None,
            "solution": None,
        }

    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        state0,
        events=[earth_event, moon_event, escape_event],
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )

    outcome = "Unresolved"
    t_event = np.nan

    if len(sol.t_events[0]) > 0:
        outcome = "Earth"
        t_event = sol.t_events[0][0]
    elif len(sol.t_events[1]) > 0:
        outcome = "Moon"
        t_event = sol.t_events[1][0]
    elif len(sol.t_events[2]) > 0:
        outcome = "Escape"
        t_event = sol.t_events[2][0]

    if sol.y.shape[1] > 0:
        x = sol.y[0]
        y = sol.y[1]
        r1, r2 = distances(x, y)
        r = np.hypot(x, y)
        min_r1 = float(np.min(r1))
        min_r2 = float(np.min(r2))
        max_r = float(np.max(r))
    else:
        min_r1 = np.nan
        min_r2 = np.nan
        max_r = np.nan

    return {
        "outcome": outcome,
        "t_event": t_event,
        "state0": state0,
        "solution": sol,
        "solver_success": bool(sol.success),
        "solver_message": sol.message,
        "n_steps": int(sol.t.size),
        "min_r_earth": min_r1,
        "min_r_moon": min_r2,
        "max_r": max_r,
    }
