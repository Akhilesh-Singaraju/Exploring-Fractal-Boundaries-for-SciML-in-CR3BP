import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

from cr3bp import omega
from integrate import integrate_state


def eval_single_point(args):
    x0, vx0, C, t_max = args
    vy2 = 2.0 * omega(x0, 0.0) - C - vx0 * vx0

    if vy2 < 0:
        return (
            C,
            x0,
            0.0,
            vx0,
            np.nan,
            "Invalid",
            np.nan,
            True,
            0,
            np.nan,
            np.nan,
            np.nan,
        )

    result = integrate_state(x0, vx0, C, t_max=t_max)
    state0 = result["state0"]

    return (
        C,
        x0,
        0.0,
        vx0,
        state0[3],
        result["outcome"],
        result["t_event"],
        result["solver_success"],
        result["n_steps"],
        result["min_r_earth"],
        result["min_r_moon"],
        result["max_r"],
    )


def generate_basin_parallel(
    C, x_min, x_max, vx_min, vx_max, nx, nvx, output, t_max=100.0, num_workers=None
):
    if num_workers is None:
        num_workers = os.cpu_count() or 4

    x_values = np.linspace(x_min, x_max, nx)
    vx_values = np.linspace(vx_min, vx_max, nvx)

    # Pre-generate parameters list
    param_list = [
        (x0, vx0, C, t_max) for x0 in x_values for vx0 in vx_values
    ]

    print(
        f"Integrating {len(param_list):,} trajectories across {num_workers} CPU cores..."
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(eval_single_point, param_list, chunksize=100))

    columns = [
        "C",
        "x0",
        "y0",
        "vx0",
        "vy0",
        "outcome",
        "t_event",
        "solver_success",
        "n_steps",
        "min_r_earth",
        "min_r_moon",
        "max_r",
    ]

    df = pd.DataFrame(results, columns=columns)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")


if __name__ == "__main__":
    generate_basin_parallel(
        C=3.14,
        x_min=-0.2,
        x_max=1.2,
        vx_min=-1.5,
        vx_max=1.5,
        nx=100,
        nvx=100,
        output="files/basin.csv",
        t_max=100.0,
    )