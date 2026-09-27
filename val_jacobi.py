import numpy as np
from integrate import integrate_state
from cr3bp import jacobi


def check_points(C, points):
    for x0, vx0 in points:
        result = integrate_state(x0, vx0, C)
        sol = result["solution"]
        if sol is None:
            print("Invalid point:", x0, vx0)
            continue
        C_values = jacobi(sol.y[0], sol.y[1], sol.y[2], sol.y[3])
        err = np.max(np.abs(C_values - C))
        print(
            f"x={x0:.6f}, vx={vx0:.6f}, outcome={result['outcome']}, "
            f"max |dC|={err:.3e}, steps={result['n_steps']}"
        )


if __name__ == "__main__":
    rng = np.random.default_rng(123)
    C = 3.14
    pts = [(float(rng.uniform(-0.1, 1.1)), float(rng.uniform(-1.0, 1.0))) for _ in range(20)]
    check_points(C, pts)
