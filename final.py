import os
import jax
import jax.numpy as jnp
import diffrax
import numpy as np
import pandas as pd

jax.config.update("jax_enable_x64", True)

MU = 0.0121505856
R_EARTH = 6378.1363 / 384400.0
R_MOON = 1737.4 / 384400.0
R_ESCAPE = 5.0

OUTCOME_MAP = {
    0: "INVALID_ENERGY",
    1: "IMPACT_EARTH",
    2: "IMPACT_MOON",
    3: "ESCAPE",
    4: "MAX_TIME",
}


def cr3bp_rhs(t, state, args):
    x, y, vx, vy = state
    dx1 = x + MU
    dx2 = x - 1.0 + MU
    r1_sq = dx1 * dx1 + y * y
    r2_sq = dx2 * dx2 + y * y

    r1_3 = r1_sq * jnp.sqrt(r1_sq)
    r2_3 = r2_sq * jnp.sqrt(r2_sq)

    dOmega_dx = x - (1.0 - MU) * dx1 / r1_3 - MU * dx2 / r2_3
    dOmega_dy = y - (1.0 - MU) * y / r1_3 - MU * y / r2_3

    return jnp.array([vx, vy, dOmega_dx + 2.0 * vy, dOmega_dy - 2.0 * vx])


def event_cond(t, state, args, **kwargs):
    x, y, _, _ = state
    r1_sq = (x + MU) ** 2 + y**2
    r2_sq = (x - 1.0 + MU) ** 2 + y**2
    rc_sq = x**2 + y**2
    return (r1_sq <= R_EARTH**2) | (r2_sq <= R_MOON**2) | (rc_sq >= R_ESCAPE**2)


@jax.jit
def _solve_single(x0, vx0, C, t_max):
    r1_0 = jnp.abs(x0 + MU)
    r2_0 = jnp.abs(x0 - 1.0 + MU)
    
    omega0 = 0.5 * (x0**2) + (1.0 - MU) / r1_0 + MU / r2_0
    vy2 = 2.0 * omega0 - C - vx0**2

    valid = vy2 >= 0.0
    vy0 = jnp.where(valid, jnp.sqrt(jnp.maximum(0.0, vy2)), 0.0)
    y0 = jnp.array([x0, 0.0, vx0, vy0])

    term = diffrax.ODETerm(cr3bp_rhs)
    solver = diffrax.Dopri8()
    stepsize_controller = diffrax.PIDController(rtol=1e-10, atol=1e-10)
    event = diffrax.Event(event_cond)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_max,
        dt0=0.01,
        y0=y0,
        stepsize_controller=stepsize_controller,
        event=event,
        max_steps=200000,
        throw=False,
        saveat=diffrax.SaveAt(t1=True),
    )

    final_state = sol.ys[0]
    final_t = sol.ts[0]

    x_f, y_f, _, _ = final_state
    r1_f = jnp.sqrt((x_f + MU) ** 2 + y_f**2)
    r2_f = jnp.sqrt((x_f - 1.0 + MU) ** 2 + y_f**2)
    rc_f = jnp.sqrt(x_f**2 + y_f**2)

    hit_earth = r1_f <= R_EARTH
    hit_moon = r2_f <= R_MOON
    escaped = rc_f >= R_ESCAPE

    label = jnp.select(
        [~valid, hit_earth, hit_moon, escaped], [0, 1, 2, 3], default=4
    )

    return (
        label,
        final_t,
        sol.stats["num_accepted_steps"],
        r1_f,
        r2_f,
        rc_f,
    )


@jax.jit
def _solve_batch(x0_arr, vx0_arr, C, t_max):
    return jax.vmap(lambda x, vx: _solve_single(x, vx, C, t_max))(
        x0_arr, vx0_arr
    )


def generate_basin_grid(
    C,
    t_max,
    res_x=300,
    res_vx=300,
    x_bounds=(-1.5, 1.5),
    vx_bounds=(-3.0, 3.0),
    batch_size=5000,
):
    x_arr = jnp.linspace(x_bounds[0], x_bounds[1], res_x)
    vx_arr = jnp.linspace(vx_bounds[0], vx_bounds[1], res_vx)
    x_mesh, vx_mesh = jnp.meshgrid(x_arr, vx_arr)

    x_flat = x_mesh.ravel()
    vx_flat = vx_mesh.ravel()
    total_points = len(x_flat)

    results_list = []
    for i in range(0, total_points, batch_size):
        x_b = x_flat[i : i + batch_size]
        vx_b = vx_flat[i : i + batch_size]
        res = _solve_batch(x_b, vx_b, C, t_max)
        results_list.append([np.array(r) for r in res])

    labels = np.concatenate([r[0] for r in results_list])
    t_events = np.concatenate([r[1] for r in results_list])
    n_steps = np.concatenate([r[2] for r in results_list])
    min_r1 = np.concatenate([r[3] for r in results_list])
    min_r2 = np.concatenate([r[4] for r in results_list])
    max_r = np.concatenate([r[5] for r in results_list])

    return pd.DataFrame({
        "x0": np.array(x_flat),
        "vx0": np.array(vx_flat),
        "C": C,
        "label_id": labels,
        "outcome": [OUTCOME_MAP[int(l)] for l in labels],
        "t_event": t_events,
        "n_steps": n_steps,
        "min_r_earth": min_r1,
        "min_r_moon": min_r2,
        "max_r": max_r,
    })


def analyze_convergence(filepath, max_samples=100):
    df = pd.read_csv(filepath)
    unresolved = df[df["outcome"] == "MAX_TIME"]
    sample = unresolved.sample(
        n=min(max_samples, len(unresolved)), random_state=42
    )

    x0_j = jnp.array(sample["x0"].values)
    vx0_j = jnp.array(sample["vx0"].values)
    C_val = float(sample["C"].iloc[0])

    for t_max in [100.0, 300.0, 1000.0, 3000.0, 6000.0, 9000.0, 12000.0]:
        labels, _, _, _, _, _ = _solve_batch(x0_j, vx0_j, C_val, t_max)
        labels_np = np.array(labels)

        counts = {}
        for l in labels_np:
            outcome = OUTCOME_MAP[int(l)]
            counts[outcome] = counts.get(outcome, 0) + 1

        print(f"\nt_max = {t_max}")
        total = sum(counts.values())
        for outcome, count in sorted(counts.items()):
            print(f"{outcome:20s} {count:4d} ({count / total:.1%})")

    labels, t_events, steps, r1, r2, rmax = _solve_batch(
        x0_j, vx0_j, C_val, 12000.0
    )
    long_lived = []
    for i in range(len(sample)):
        if int(labels[i]) == 4:
            long_lived.append({
                "x0": float(x0_j[i]),
                "vx0": float(vx0_j[i]),
                "outcome": "MAX_TIME",
                "t_event": float(t_events[i]),
                "n_steps": int(steps[i]),
                "min_r_earth": float(r1[i]),
                "min_r_moon": float(r2[i]),
                "max_r": float(rmax[i]),
            })

    result_df = pd.DataFrame(long_lived)
    print(f"\nRemaining MAX_TIME trajectories: {len(result_df)}")
    if not result_df.empty:
        print(result_df.to_string(index=False))
        print(
            result_df[[
                "min_r_earth",
                "min_r_moon",
                "max_r",
                "n_steps",
            ]].describe()
        )


def run_batch_generation(T_val):
    C_VALUES = [3.00, 3.05, 3.10, 3.14, 3.20]
    GRID_RESOLUTION = 300

    os.makedirs("files/data", exist_ok=True)

    for C in C_VALUES:
        df = generate_basin_grid(
            C=C, t_max=T_val, res_x=GRID_RESOLUTION, res_vx=GRID_RESOLUTION
        )
        output_path = f"files/data/basin_C{C:.2f}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

def pre_test_convergence(C, t_max_list, n_samples=100):
    print(f"\nRunning Pre-Test for C={C}")
    np.random.seed(42)
    x0_sample = jnp.array(np.random.uniform(-1.5, 1.5, n_samples))
    vx0_sample = jnp.array(np.random.uniform(-3.0, 3.0, n_samples))

    for t in t_max_list:
        labels, _, _, _, _, _ = _solve_batch(x0_sample, vx0_sample, C, t)
        
        counts = {name: 0 for name in OUTCOME_MAP.values()}
        for l in np.array(labels):
            counts[OUTCOME_MAP[int(l)]] += 1
            
        print(f"t_max = {t:<6} | " + " | ".join(f"{k}: {v:3d}" for k, v in counts.items()))

if __name__ == "__main__":
    # TEST_C = 3.05
    # TEST_TIMES = [100.0, 1000.0, 5000.0, 12000.0]
    # pre_test_convergence(TEST_C, TEST_TIMES, n_samples=200)
    run_batch_generation(5000.0)