import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

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


def cr3bp_rhs(t, s):
    x, y, vx, vy = s
    dx1, dx2 = x + MU, x - 1.0 + MU
    r1_3 = (dx1**2 + y**2) ** 1.5
    r2_3 = (dx2**2 + y**2) ** 1.5
    return [
        vx,
        vy,
        x - (1 - MU) * dx1 / r1_3 - MU * dx2 / r2_3 + 2 * vy,
        y - (1 - MU) * y / r1_3 - MU * y / r2_3 - 2 * vx,
    ]


def make_events():
    def hit_earth(t, s):
        return np.hypot(s[0] + MU, s[1]) - R_EARTH

    def hit_moon(t, s):
        return np.hypot(s[0] - 1 + MU, s[1]) - R_MOON

    def escape(t, s):
        return np.hypot(s[0], s[1]) - R_ESCAPE

    hit_earth.direction, hit_moon.direction, escape.direction = -1, -1, 1
    hit_earth.terminal = hit_moon.terminal = escape.terminal = True
    return [hit_earth, hit_moon, escape]


def scipy_classify(x0, vx0, C, t_max, method="RK45"):
    r1_0 = abs(x0 + MU)
    r2_0 = abs(x0 - 1 + MU)
    omega0 = 0.5 * x0**2 + (1 - MU) / r1_0 + MU / r2_0
    vy2 = 2 * omega0 - C - vx0**2
    if vy2 < 0:
        return 0, 0.0, 0, r1_0, r2_0, np.hypot(x0, 0.0)
    y0 = [x0, 0.0, vx0, np.sqrt(vy2)]

    sol = solve_ivp(
        cr3bp_rhs,
        (0.0, t_max),
        y0,
        method=method,
        rtol=1e-10,
        atol=1e-10,
        events=make_events(),
        dense_output=True,
        max_step=np.inf,
    )

    fired = [i for i in range(3) if sol.t_events[i].size > 0]
    if fired:
        i = min(fired, key=lambda k: sol.t_events[k][0])
        te = sol.t_events[i][0]
        yf = sol.sol(te)
        label = {0: 1, 1: 2, 2: 3}[i]
    else:
        te, yf = sol.t[-1], sol.y[:, -1]
        label = 4

    r1 = np.hypot(yf[0] + MU, yf[1])
    r2 = np.hypot(yf[0] - 1 + MU, yf[1])
    rc = np.hypot(yf[0], yf[1])

    ts = np.linspace(0.0, te, 4001)
    ys = sol.sol(ts)
    min_r1 = np.min(np.hypot(ys[0] + MU, ys[1]))
    min_r2 = np.min(np.hypot(ys[0] - 1 + MU, ys[1]))
    max_rc = np.max(np.hypot(ys[0], ys[1]))

    return label, te, sol.nfev // 7, min_r1, min_r2, max_rc


print("RK45 vs DOP853")
rng = np.random.default_rng(0)
C, T_MAX = 3.05, 5000.0
print(
    f"{'x0':>8} {'vx0':>8} | {'RK45 label':>10} {'t_ev':>10} | "
    f"{'DOP853 label':>12} {'t_ev':>10}"
)
n_agree = 0
for _ in range(30):
    x0, vx0 = rng.uniform(-1.5, 1.5), rng.uniform(-3.0, 3.0)
    l1, t1, *_ = scipy_classify(x0, vx0, C, T_MAX, "RK45")
    l2, t2, *_ = scipy_classify(x0, vx0, C, T_MAX, "DOP853")
    agree = (l1 == l2) and (l1 in (0, 4) or abs(t1 - t2) < 1e-4)
    n_agree += agree
    print(
        f"{x0:8.4f} {vx0:8.4f} | {OUTCOME_MAP[l1]:>14} {t1:10.4f} | "
        f"{OUTCOME_MAP[l2]:>14} {t2:10.4f}  {'OK' if agree else 'MISMATCH'}"
    )
print(f"\nRK45 vs DOP853 agreement: {n_agree}/30")

rng = np.random.default_rng(0)
mismatches = []
for _ in range(30):
    x0, vx0 = rng.uniform(-1.5, 1.5), rng.uniform(-3.0, 3.0)
    l1, t1, _, mr1, mr2, mrc = scipy_classify(x0, vx0, C, T_MAX, "RK45")
    l2, t2, *_ = scipy_classify(x0, vx0, C, T_MAX, "DOP853")
    if l1 != l2:
        margin = min(abs(mr1 - R_EARTH), abs(mr2 - R_MOON), abs(mrc - R_ESCAPE))
        mismatches.append((x0, vx0, OUTCOME_MAP[l1], OUTCOME_MAP[l2], margin))

for m in mismatches:
    print(f"x0={m[0]:.5f} vx0={m[1]:.5f}  {m[2]} vs {m[3]}  margin={m[4]:.2e}")

print("\nCompare with Rust CSV output")
df = pd.read_csv("files/data/basin_C3.05.csv")
sample = df.sample(50, random_state=42)

mismatch_label, worst_dt, worst_dr = 0, 0.0, 0.0
for _, row in sample.iterrows():
    l_ref, te_ref, _, mr1, mr2, mrc = scipy_classify(
        row["x0"], row["vx0"], row["C"], 5000.0, "RK45"
    )
    if l_ref != row["label_id"]:
        mismatch_label += 1
        print(
            f"  LABEL MISMATCH x0={row['x0']:.6f} vx0={row['vx0']:.6f}: "
            f"rust={OUTCOME_MAP[row['label_id']]} scipy={OUTCOME_MAP[l_ref]}"
        )
    elif l_ref in (1, 2, 3):
        worst_dt = max(worst_dt, abs(te_ref - row["t_event"]))
    worst_dr = max(
        worst_dr,
        abs(mr1 - row["min_r_earth"]),
        abs(mr2 - row["min_r_moon"]),
        abs(mrc - row["max_r"]),
    )

print(f"label mismatches: {mismatch_label}/50")
print(f"worst event-time diff:  {worst_dt:.3e}")
print(f"worst distance-metric diff: {worst_dr:.3e}")
print(
    "(distance metrics use dense output on both sides, so they should agree "
    "\n to ~1e-6; event times to ~1e-8. Label flips only possible on "
    "\n knife-edge boundary trajectories — chaotic grazing cases.)"
)
