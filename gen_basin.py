import time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

from integrate import integrate_single_numba, OUTCOME_MAP


@njit(parallel=True, fastmath=True)
def run_grid_numba(x_vals, vx_vals, C, t_max):
    nx = len(x_vals)
    nvx = len(vx_vals)
    n_total = nx * nvx

    outcomes = np.zeros(n_total, dtype=np.int32)
    t_events = np.zeros(n_total, dtype=np.float64)
    n_steps = np.zeros(n_total, dtype=np.int32)
    min_r1 = np.zeros(n_total, dtype=np.float64)
    min_r2 = np.zeros(n_total, dtype=np.float64)
    max_r = np.zeros(n_total, dtype=np.float64)

    for i in prange(nx):
        for j in range(nvx):
            idx = i * nvx + j
            res = integrate_single_numba(x_vals[i], vx_vals[j], C, t_max)
            outcomes[idx] = res[0]
            t_events[idx] = res[1]
            n_steps[idx] = res[2]
            min_r1[idx] = res[3]
            min_r2[idx] = res[4]
            max_r[idx] = res[5]

    return outcomes, t_events, n_steps, min_r1, min_r2, max_r


def generate_basin(C, x_min, x_max, vx_min, vx_max, nx, nvx, output, t_max=100.0):
    x_values = np.linspace(x_min, x_max, nx)
    vx_values = np.linspace(vx_min, vx_max, nvx)

    print("Warming up Numba JIT compiler...")
    _ = integrate_single_numba(0.5, 0.0, C, 1.0)

    start = time.time()
    print(f"Integrating {nx * nvx:,} trajectories across CPU cores...")
    outcomes, t_events, n_steps, min_r1, min_r2, max_r = run_grid_numba(
        x_values, vx_values, C, t_max
    )
    print(f"Completed integration in {time.time() - start:.2f} seconds.")

    X, VX = np.meshgrid(x_values, vx_values, indexing="ij")
    df = pd.DataFrame(
        {
            "C": C,
            "x0": X.ravel(),
            "y0": 0.0,
            "vx0": VX.ravel(),
            "outcome": [OUTCOME_MAP[o] for o in outcomes],
            "t_event": t_events,
            "n_steps": n_steps,
            "min_r_earth": min_r1,
            "min_r_moon": min_r2,
            "max_r": max_r,
        }
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")


if __name__ == "__main__":
    generate_basin(
        C=3.14,
        x_min=-0.2,
        x_max=1.2,
        vx_min=-1.5,
        vx_max=1.5,
        nx=100,
        nvx=100,
        output='files/basin_C3.14.csv',
        t_max=100.0,
    )