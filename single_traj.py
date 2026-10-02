import numpy as np
from scipy.integrate import solve_ivp

MU = 0.0121505856
R_EARTH = 6378.1363 / 384400.0
R_MOON = 1737.4 / 384400.0
R_ESCAPE = 5.0

def cr3bp_rhs(t, s):
    x, y, vx, vy = s
    dx1, dx2 = x + MU, x - 1.0 + MU
    r1_3 = (dx1**2 + y**2) ** 1.5
    r2_3 = (dx2**2 + y**2) ** 1.5
    return [
        vx, vy,
        x - (1 - MU) * dx1 / r1_3 - MU * dx2 / r2_3 + 2 * vy,
        y - (1 - MU) * y / r1_3 - MU * y / r2_3 - 2 * vx,
    ]

def make_events():
    def hit_earth(t, s): return np.hypot(s[0] + MU, s[1]) - R_EARTH
    def hit_moon(t, s):  return np.hypot(s[0] - 1 + MU, s[1]) - R_MOON
    def escape(t, s):    return np.hypot(s[0], s[1]) - R_ESCAPE
    hit_earth.direction, hit_moon.direction, escape.direction = -1, -1, 1
    hit_earth.terminal = hit_moon.terminal = escape.terminal = True
    return [hit_earth, hit_moon, escape]

def diagnose_scipy(x0, vx0, C, t_max):
    r1_0 = abs(x0 + MU)
    r2_0 = abs(x0 - 1 + MU)
    omega0 = 0.5 * x0**2 + (1 - MU) / r1_0 + MU / r2_0
    vy2 = 2 * omega0 - C - vx0**2
    if vy2 < 0:
        print("INVALID_ENERGY"); return
    y0 = [x0, 0.0, vx0, np.sqrt(vy2)]

    sol = solve_ivp(
        cr3bp_rhs, (0.0, t_max), y0,
        method="DOP853",
        rtol=1e-11, atol=1e-13,
        events=make_events(),
        dense_output=True,
        max_step=0.01,
    )

    fired = [i for i in range(3) if sol.t_events[i].size > 0]
    label_map = {0: "IMPACT_EARTH", 1: "IMPACT_MOON", 2: "ESCAPE"}
    if fired:
        i = min(fired, key=lambda k: sol.t_events[k][0])
        te = sol.t_events[i][0]
        yf = sol.sol(te)
        label = label_map[i]
    else:
        te, yf = sol.t[-1], sol.y[:, -1]
        label = "MAX_TIME"

    ts = np.linspace(0.0, te, 4001)
    ys = sol.sol(ts)
    min_r1 = np.min(np.hypot(ys[0] + MU, ys[1]))
    min_r2 = np.min(np.hypot(ys[0] - 1 + MU, ys[1]))
    max_rc = np.max(np.hypot(ys[0], ys[1]))

    jacobi_init = 2 * omega0 - (vx0**2 + y0[3]**2)
    omega_f = 0.5*(yf[0]**2 + yf[1]**2) + (1-MU)/np.hypot(yf[0]+MU, yf[1]) + MU/np.hypot(yf[0]-1+MU, yf[1])
    jacobi_final = 2*omega_f - (yf[2]**2 + yf[3]**2)

    print("\n=== SCIPY DIAGNOSTIC ===")
    print(f"Label:       {label}")
    print(f"t_event:     {te:.17e}")
    print(f"n_steps:     {sol.nfev // 7}")
    print(f"min_r_earth: {min_r1:.17e}")
    print(f"min_r_moon:  {min_r2:.17e}")
    print(f"max_r:       {max_rc:.17e}")
    print(f"Jacobi init: {jacobi_init:.17e}")
    print(f"Jacobi final:{jacobi_final:.17e}")
    print(f"Jacobi error:{abs(jacobi_final - jacobi_init):.3e}")
    print("========================\n")

diagnose_scipy(0.065217, -2.177258, 3.05, 5000.0)