import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
from tqdm import tqdm


def hh_derivs(t, state):
    x, y, px, py = state
    return [px, py, -x - 2*x*y, -y - (x**2 - y**2)]


def esc_event(t, state):
    return state[0]**2 + state[1]**2 - 25.0


esc_event.terminal = True
esc_event.direction = 1


def init_py(x, y, E=0.25):
    v = 2.0 * E - (x**2 + y**2) - 2.0 * (x**2 * y - (y**3) / 3.0)
    return np.sqrt(v) if v >= 0 else -1


def get_exit(x, y):
    ang = np.arctan2(y, x) % (2 * np.pi)
    if np.pi / 6 <= ang < 5 * np.pi / 6:
        return 1
    elif 5 * np.pi / 6 <= ang < 3 * np.pi / 2:
        return 2
    else:
        return 3


def sim_point(args):
    x, y = args
    py = init_py(x, y)

    if py == -1:
        return 0

    sol = solve_ivp(
        hh_derivs,
        [0, 200],
        [x, y, 0.0, py],
        method='DOP853',
        rtol=1e-9,
        atol=1e-9,
        events=esc_event
    )

    if sol.status == 1:
        return get_exit(sol.y[0, -1], sol.y[1, -1])

    return -1


def gen_grid(n):
    xs = np.linspace(-0.8, 0.8, n)
    ys = np.linspace(-0.8, 0.8, n)

    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = list(zip(grid_x.ravel(), grid_y.ravel()))

    with mp.Pool() as pool:
        res = list(
            tqdm(
                pool.imap(sim_point, pts, chunksize=500),
                total=len(pts),
                desc=f"Generating {n}x{n} grid"
            )
        )

    return np.array(res).reshape((n, n))


if __name__ == '__main__':
    val_grid = gen_grid(500)
    np.save('val_grid.npy', val_grid)

    test_grid = gen_grid(1000)
    np.save('test_grid.npy', test_grid)