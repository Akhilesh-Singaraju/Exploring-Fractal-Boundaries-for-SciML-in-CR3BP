import numpy as np
from numba import njit, prange
import pandas as pd
from scipy.integrate import solve_ivp

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


@njit(parallel=True, fastmath=True)
def generate_basin_grid_numba(x_arr, vx_arr, C, t_max, dt=0.01):
    nx = len(x_arr)
    nvx = len(vx_arr)
    
    labels = np.empty((nvx, nx), dtype=np.int32)
    t_events = np.empty((nvx, nx), dtype=np.float64)
    n_steps_arr = np.empty((nvx, nx), dtype=np.int32)
    min_r1_arr = np.empty((nvx, nx), dtype=np.float64)
    min_r2_arr = np.empty((nvx, nx), dtype=np.float64)
    max_r_arr = np.empty((nvx, nx), dtype=np.float64)

    for i in prange(nvx):
        vx0 = vx_arr[i]
        for j in range(nx):
            x0 = x_arr[j]
            label, t_ev, steps, r1, r2, rmax = integrate_single_numba(
                x0, vx0, C, t_max, dt
            )
            labels[i, j] = label
            t_events[i, j] = t_ev
            n_steps_arr[i, j] = steps
            min_r1_arr[i, j] = r1
            min_r2_arr[i, j] = r2
            max_r_arr[i, j] = rmax

    return labels, t_events, n_steps_arr, min_r1_arr, min_r2_arr, max_r_arr

def cr3bp_rhs_scipy(t, state):
    x, y, vx, vy = state
    dx1 = x + MU
    dx2 = x - 1.0 + MU
    r1_sq = dx1 * dx1 + y * y
    r2_sq = dx2 * dx2 + y * y

    r1_3 = r1_sq * np.sqrt(r1_sq)
    r2_3 = r2_sq * np.sqrt(r2_sq)

    dOmega_dx = x - (1.0 - MU) * dx1 / r1_3 - MU * dx2 / r2_3
    dOmega_dy = y - (1.0 - MU) * y / r1_3 - MU * y / r2_3

    return [vx, vy, dOmega_dx + 2.0 * vy, dOmega_dy - 2.0 * vx]

def event_earth(t, state):
    r1 = np.sqrt((state[0] + MU)**2 + state[1]**2)
    return r1 - R_EARTH
event_earth.terminal = True
event_earth.direction = -1

def event_moon(t, state):
    r2 = np.sqrt((state[0] - 1.0 + MU)**2 + state[1]**2)
    return r2 - R_MOON
event_moon.terminal = True
event_moon.direction = -1

def event_escape(t, state):
    r_center = np.sqrt(state[0]**2 + state[1]**2)
    return r_center - R_ESCAPE
event_escape.terminal = True
event_escape.direction = 1


def generate_and_compare(C, t_max, res_x=50, res_vx=50, x_bounds=(-1.5, 1.5), vx_bounds=(-3.0, 3.0), dt=0.01):
    x_arr = np.linspace(x_bounds[0], x_bounds[1], res_x)
    vx_arr = np.linspace(vx_bounds[0], vx_bounds[1], res_vx)

    print("Running Numba RK4...")
    labels_rk4, t_events_rk4, _, _, _, _ = generate_basin_grid_numba(
        x_arr, vx_arr, C, t_max, dt
    )

    print(f"Running SciPy DOP853 for {res_x * res_vx} points (this may take a minute)...")
    labels_dop = np.empty((res_vx, res_x), dtype=np.int32)
    t_events_dop = np.empty((res_vx, res_x), dtype=np.float64)

    for i in range(res_vx):
        vx0 = vx_arr[i]
        for j in range(res_x):
            x0 = x_arr[j]
            state0, valid = initial_state_from_x_vx(x0, vx0, C)
            
            if not valid:
                labels_dop[i, j] = 0
                t_events_dop[i, j] = 0.0
                continue
                
            sol = solve_ivp(
                cr3bp_rhs_scipy,
                (0.0, t_max),
                state0,
                method='DOP853',
                events=[event_earth, event_moon, event_escape],
                rtol=1e-10,
                atol=1e-10
            )
            
            if sol.status == 1:
                if len(sol.t_events[0]) > 0:
                    label = 1
                elif len(sol.t_events[1]) > 0:
                    label = 2
                elif len(sol.t_events[2]) > 0:
                    label = 3
                else:
                    label = 4
            else:
                label = 4
                
            labels_dop[i, j] = label
            t_events_dop[i, j] = sol.t[-1]

    x_mesh, vx_mesh = np.meshgrid(x_arr, vx_arr)
    
    df_compare = pd.DataFrame({
        "x0": x_mesh.ravel(),
        "vx0": vx_mesh.ravel(),
        "C": C,
        "label_rk4": labels_rk4.ravel(),
        "label_dop": labels_dop.ravel(),
        "outcome_rk4": [OUTCOME_MAP[l] for l in labels_rk4.ravel()],
        "outcome_dop": [OUTCOME_MAP[l] for l in labels_dop.ravel()],
        "t_event_rk4": t_events_rk4.ravel(),
        "t_event_dop": t_events_dop.ravel(),
    })
    
    valid_mask = df_compare["label_rk4"] != 0
    df_valid = df_compare[valid_mask]
    
    matches = df_valid["label_rk4"] == df_valid["label_dop"]
    match_pct = matches.mean() * 100
    
    t_diff = np.abs(df_valid.loc[matches, "t_event_rk4"] - df_valid.loc[matches, "t_event_dop"])
    
    print("\n=== Validation Results ===")
    print(f"Total Valid Trajectories: {len(df_valid)}")
    print(f"Label Agreement (RK4 vs DOP853): {match_pct:.2f}%")
    
    if not t_diff.empty:
        print(f"Mean Time Difference (Matching Labels): {t_diff.mean():.6f}")
        print(f"Max Time Difference (Matching Labels): {t_diff.max():.6f}")
        
    mismatches = df_valid[~matches]
    if not mismatches.empty:
        print(f"\nNote: Found {len(mismatches)} mismatches. These usually occur on fractal basin boundaries where tiny trajectory differences lead to entirely different outcomes.")
        
    return df_compare

if __name__ == "__main__":
    df_results = generate_and_compare(C=3.1, t_max=10.0, res_x=50, res_vx=50, dt=0.01)