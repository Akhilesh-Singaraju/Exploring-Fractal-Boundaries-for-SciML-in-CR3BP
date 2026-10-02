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

def jacobi_constant(s):
    x, y, vx, vy = s
    dx1, dx2 = x + MU, x - 1 + MU
    r1 = np.hypot(dx1, y)
    r2 = np.hypot(dx2, y)
    omega = 0.5 * (x**2 + y**2) + (1 - MU) / r1 + MU / r2
    return 2 * omega - (vx**2 + vy**2)

x0, vx0, C, t_max = 0.065217, -2.177258, 3.05, 5000.0

r1_0 = abs(x0 + MU)
r2_0 = abs(x0 - 1 + MU)
omega0 = 0.5 * x0**2 + (1 - MU) / r1_0 + MU / r2_0
vy2 = 2 * omega0 - C - vx0**2
y0 = [x0, 0.0, vx0, np.sqrt(vy2)]

sol = solve_ivp(
    cr3bp_rhs,
    (0.0, t_max),
    y0,
    method="DOP853",       
    rtol=1e-11,            
    atol=1e-13,            
    max_step=0.01,         
    events=make_events(),
    dense_output=True,
)

fired = [i for i in range(3) if sol.t_events[i].size > 0]
label_map = {0: "IMPACT_EARTH", 1: "IMPACT_MOON", 2: "ESCAPE"}
if fired:
    i = min(fired, key=lambda k: sol.t_events[k][0])
    te = sol.t_events[i][0]
    label = label_map[i]
else:
    te = sol.t[-1]
    label = "MAX_TIME"

print(f"\n=== SCIPY DIAGNOSTIC ===")
print(f"Label:       {label}")
print(f"t_event:     {te:.17e}")
print(f"n_steps:     {sol.nfev // 7}")
print(f"n_accepted:  {len(sol.t)}")

print(f"\n--- SCIPY TRAJECTORY SAMPLES ---")
print(f"{'t':>8}  {'x':>24}  {'y':>24}  {'vx':>24}  {'vy':>24}  {'C':>24}")
for t_check in [0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 300.0]:
    if t_check <= te:
        s = sol.sol(t_check)
        c_now = jacobi_constant(s)
        print(
            f"{t_check:8.3f}  "
            f"{s[0]:+.15e}  {s[1]:+.15e}  "
            f"{s[2]:+.15e}  {s[3]:+.15e}  "
            f"{c_now:.15e}"
        )
print("========================\n")