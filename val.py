import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


CSV_PATH = "files/data/3.05/300x300.csv"
OUT_DIR = "files/val"

MU = 0.0121505856
R_EARTH = 6378.1363 / 384400.0
R_MOON = 1737.4 / 384400.0
R_ESCAPE = 5.0

T_MAX = 5000.0

LABEL_INVALID = 0
LABEL_IMPACT_EARTH = 1
LABEL_IMPACT_MOON = 2
LABEL_ESCAPE = 3
LABEL_MAX_TIME = 4
LABEL_INTEGRATOR_ERROR = -1

LABEL_NAMES = {
    0: "INVALID_ENERGY",
    1: "IMPACT_EARTH",
    2: "IMPACT_MOON",
    3: "ESCAPE",
    4: "MAX_TIME",
    -1: "INTEGRATOR_ERROR",
}

SHORT_LABEL_NAMES = {
    0: "invalid",
    1: "Earth",
    2: "Moon",
    3: "escape",
    4: "max_time",
    -1: "error",
}

TERMINAL_LABELS = {1, 2, 3}
TERMINAL_LABEL_LIST = [1, 2, 3]

VY_SIGN = 1.0

ENERGY_EPS = 1e-12
TIME_ATOL = 1e-6
TIME_RTOL = 1e-6
TMAX_BOUNDARY_TOL = 1e-3
CHAOTIC_TIME_THRESHOLD = 100.0

METHOD = "DOP853"
RTOL = 1e-10
ATOL = 1e-12

MAX_STEP = None


N_SAMPLES = 100
RANDOM_SEED = 1234
STRATIFY = True

PROBE_MISMATCHES = True
MAX_PROBE_MISMATCHES = 12

PROBE_EPS_X = 1e-6
PROBE_EPS_VX = 1e-6
PROBE_MULTIPLIERS = (0.01, 1.0, 100.0)
PROBE_DIRECTIONS = (
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)

PROBE_TIME_MARGIN = 10.0
PROBE_RTOL = 1e-9
PROBE_ATOL = 1e-11
PROBE_MAX_STEP = MAX_STEP

JACOBI_WARN_THRESHOLD = 1e-8

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def sfloat(value, default=0.0):
    try:
        out = float(value)
        if np.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def series_or(df, name, default=0.0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").astype(float)
    return pd.Series(default, index=df.index, dtype=float)



def cr3bp_rhs(t, s):
    x, y, z, vx, vy, vz = s

    r1 = np.sqrt((x + MU) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y * y + z * z)

    r13 = max(r1 * r1 * r1, 1e-30)
    r23 = max(r2 * r2 * r2, 1e-30)

    ax = (
        x
        + 2.0 * vy
        - (1.0 - MU) * (x + MU) / r13
        - MU * (x - 1.0 + MU) / r23
    )

    ay = (
        y
        - 2.0 * vx
        - (1.0 - MU) * y / r13
        - MU * y / r23
    )

    az = (
        - (1.0 - MU) * z / r13
        - MU * z / r23
    )

    return [vx, vy, vz, ax, ay, az]


def jacobi_constant(s):
    x, y, z, vx, vy, vz = s

    r1 = np.sqrt((x + MU) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y * y + z * z)

    r1_safe = r1 if np.isfinite(r1) and r1 > 1e-12 else 1e-12
    r2_safe = r2 if np.isfinite(r2) and r2 > 1e-12 else 1e-12

    omega = (
        0.5 * (x * x + y * y)
        + (1.0 - MU) / r1_safe
        + MU / r2_safe
    )

    v2 = vx * vx + vy * vy + vz * vz
    return 2.0 * omega - v2


def event_earth(t, s):
    x, y, z = s[0], s[1], s[2]
    r1 = np.sqrt((x + MU) ** 2 + y * y + z * z)
    return r1 - R_EARTH


event_earth.terminal = True
event_earth.direction = -1.0


def event_moon(t, s):
    x, y, z = s[0], s[1], s[2]
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y * y + z * z)
    return r2 - R_MOON


event_moon.terminal = True
event_moon.direction = -1.0


def event_escape(t, s):
    x, y, z = s[0], s[1], s[2]
    r = np.sqrt(x * x + y * y + z * z)
    return r - R_ESCAPE


event_escape.terminal = True
event_escape.direction = 1.0


EVENTS = [event_earth, event_moon, event_escape]


def diagnostics_from_states(states):
    if states.size == 0:
        return np.nan, np.nan, np.nan

    x = states[0]
    y = states[1]
    z = states[2]

    r1 = np.sqrt((x + MU) ** 2 + y * y + z * z)
    r2 = np.sqrt((x - 1.0 + MU) ** 2 + y * y + z * z)
    rb = np.sqrt(x * x + y * y + z * z)

    return float(np.min(r1)), float(np.min(r2)), float(np.max(rb))


def make_result(
    label_id,
    t_event,
    min_r_earth,
    min_r_moon,
    max_r,
    jacobi_initial,
    jacobi_final,
    n_steps=0,
    success=True,
    message="",
):
    return {
        "label_id": int(label_id),
        "t_event": float(t_event),
        "min_r_earth": float(min_r_earth),
        "min_r_moon": float(min_r_moon),
        "max_r": float(max_r),
        "jacobi_initial": float(jacobi_initial),
        "jacobi_final": float(jacobi_final),
        "jacobi_error": float(abs(jacobi_final - jacobi_initial)),
        "n_steps": int(n_steps),
        "success": bool(success),
        "message": str(message),
    }


def simulate_initial(
    x0,
    vx0,
    C,
    vy0=None,
    y0=0.0,
    z0=0.0,
    vz0=0.0,
    vy_sign=None,
    t_max=T_MAX,
    method=METHOD,
    rtol=RTOL,
    atol=ATOL,
    max_step=MAX_STEP,
):
    if vy_sign is None:
        vy_sign = VY_SIGN

    sign = 1.0 if isinstance(vy_sign, str) else float(vy_sign)

    x0 = float(x0)
    y0 = float(y0)
    z0 = float(z0)
    vx0 = float(vx0)
    vz0 = float(vz0)
    C = float(C)

    if not np.isfinite(C):
        return make_result(
            LABEL_INTEGRATOR_ERROR,
            0.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            0,
            False,
            "C missing or non-finite",
        )

    r1 = np.sqrt((x0 + MU) ** 2 + y0 * y0 + z0 * z0)
    r2 = np.sqrt((x0 - 1.0 + MU) ** 2 + y0 * y0 + z0 * z0)
    rb = np.sqrt(x0 * x0 + y0 * y0 + z0 * z0)

    r1_safe = r1 if np.isfinite(r1) and r1 > 1e-12 else 1e-12
    r2_safe = r2 if np.isfinite(r2) and r2 > 1e-12 else 1e-12

    omega0 = (
        0.5 * (x0 * x0 + y0 * y0)
        + (1.0 - MU) / r1_safe
        + MU / r2_safe
    )

    if vy0 is None:
        vy2 = 2.0 * omega0 - C - vx0 * vx0 - vz0 * vz0

        if not np.isfinite(vy2):
            return make_result(
                LABEL_INTEGRATOR_ERROR,
                0.0,
                r1,
                r2,
                rb,
                C,
                C,
                0,
                False,
                "non-finite vy2",
            )

        if vy2 < -ENERGY_EPS:
            return make_result(
                LABEL_INVALID,
                0.0,
                r1,
                r2,
                rb,
                C,
                C,
                0,
                True,
                "invalid_energy",
            )

        vy0 = sign * np.sqrt(max(0.0, vy2))

    state0 = np.array([x0, y0, z0, vx0, float(vy0), vz0], dtype=float)
    C0 = jacobi_constant(state0)

    if r1 <= R_EARTH:
        return make_result(
            LABEL_IMPACT_EARTH,
            0.0,
            r1,
            r2,
            rb,
            C0,
            C0,
            0,
            True,
            "initial Earth impact",
        )

    if r2 <= R_MOON:
        return make_result(
            LABEL_IMPACT_MOON,
            0.0,
            r1,
            r2,
            rb,
            C0,
            C0,
            0,
            True,
            "initial Moon impact",
        )

    if rb >= R_ESCAPE:
        return make_result(
            LABEL_ESCAPE,
            0.0,
            r1,
            r2,
            rb,
            C0,
            C0,
            0,
            True,
            "initial escape",
        )

    solver_kwargs = {
        "method": method,
        "rtol": rtol,
        "atol": atol,
        "events": EVENTS,
    }

    if max_step is not None and np.isfinite(max_step):
        solver_kwargs["max_step"] = max_step

    try:
        sol = solve_ivp(
            cr3bp_rhs,
            (0.0, float(t_max)),
            state0,
            **solver_kwargs,
        )
    except Exception as exc:
        return make_result(
            LABEL_INTEGRATOR_ERROR,
            0.0,
            r1,
            r2,
            rb,
            C0,
            C0,
            0,
            False,
            f"solve_ivp exception: {exc}",
        )

    if not sol.success:
        states0 = state0.reshape(6, 1)
        min_e, min_m, max_r = diagnostics_from_states(states0)
        return make_result(
            LABEL_INTEGRATOR_ERROR,
            sol.t[-1] if len(sol.t) else 0.0,
            min_e,
            min_m,
            max_r,
            C0,
            C0,
            int(sol.t.size),
            False,
            sol.message,
        )

    event_label = None
    event_time = None
    event_state = None

    if sol.status == 1:
        best_time = np.inf

        for i, lab in enumerate((LABEL_IMPACT_EARTH, LABEL_IMPACT_MOON, LABEL_ESCAPE)):
            te = sol.t_events[i]
            if len(te) > 0:
                t = float(te[0])
                if t < best_time:
                    best_time = t
                    event_label = lab
                    event_time = t
                    event_state = np.asarray(sol.y_events[i][0], dtype=float)

    if event_label is not None:
        label = event_label
        t_event = event_time
    else:
        label = LABEL_MAX_TIME
        t_event = float(t_max)

    if sol.y.size == 0:
        states = state0.reshape(6, 1)
    else:
        states = sol.y

    if event_state is not None:
        states = np.column_stack((states, event_state))

    min_e, min_m, max_r = diagnostics_from_states(states)

    final_state = event_state if event_state is not None else states[:, -1]
    C_final = jacobi_constant(final_state)

    return make_result(
        label,
        t_event,
        min_e,
        min_m,
        max_r,
        C0,
        C_final,
        int(sol.t.size),
        True,
        "",
    )

def add_initial_geometry(df):
    df = df.copy()

    x0 = series_or(df, "x0", 0.0)
    y0 = series_or(df, "y0", 0.0)
    z0 = series_or(df, "z0", 0.0)

    vx0 = series_or(df, "vx0", 0.0)
    vz0 = series_or(df, "vz0", 0.0)

    if "C" in df.columns:
        C = pd.to_numeric(df["C"], errors="coerce")
    elif "jacobi_initial" in df.columns:
        C = pd.to_numeric(df["jacobi_initial"], errors="coerce")
    else:
        C = pd.Series(np.nan, index=df.index, dtype=float)

    r1 = np.sqrt((x0 + MU) ** 2 + y0 * y0 + z0 * z0)
    r2 = np.sqrt((x0 - 1.0 + MU) ** 2 + y0 * y0 + z0 * z0)
    rb = np.sqrt(x0 * x0 + y0 * y0 + z0 * z0)

    r1_safe = np.where(np.isfinite(r1) & (r1 > 1e-12), r1, 1e-12)
    r2_safe = np.where(np.isfinite(r2) & (r2 > 1e-12), r2, 1e-12)

    omega0 = (
        0.5 * (x0 * x0 + y0 * y0)
        + (1.0 - MU) / r1_safe
        + MU / r2_safe
    )

    vy2 = 2.0 * omega0 - C - vx0 * vx0 - vz0 * vz0

    if "vy0" in df.columns:
        vy0 = pd.to_numeric(df["vy0"], errors="coerce")
    else:
        sign = 1.0 if isinstance(VY_SIGN, str) else float(VY_SIGN)
        vy0 = sign * np.sqrt(np.clip(vy2, 0.0, None))

    v2 = vx0 * vx0 + vy0 * vy0 + vz0 * vz0
    jacobi_state = 2.0 * omega0 - v2

    df["x0"] = x0
    df["y0"] = y0
    df["z0"] = z0
    df["vx0"] = vx0
    df["vz0"] = vz0
    df["vy0_used"] = vy0

    df["C_used"] = C
    df["r1_0"] = r1
    df["r2_0"] = r2
    df["rb_0"] = rb
    df["omega0"] = omega0
    df["vy2"] = vy2
    df["jacobi_state"] = jacobi_state

    df["expected_invalid"] = vy2 < 0.0

    return df


def static_report(df):
    df = add_initial_geometry(df)

    print("=" * 78)
    print("1) Mathematical/static CSV validation")
    print("=" * 78)

    print(f"rows: {len(df)}")

    if "label_id" not in df.columns:
        raise ValueError("CSV must contain a label_id column.")

    label = df["label_id"]

    expected_invalid = df["expected_invalid"].fillna(False)
    actual_invalid = label == LABEL_INVALID
    valid_C = df["C_used"].notna()

    bad_energy = int(((expected_invalid != actual_invalid) & valid_C).sum())

    print(f"INVALID_ENERGY classification errors: {bad_energy} / {len(df)}")
    print(f"rows with missing C: {int((~valid_C).sum())}")
    print(
        f"near-zero vy2 rows (|vy2| < {ENERGY_EPS}): "
        f"{int((df['vy2'].abs() < ENERGY_EPS).sum())}"
    )

    if "t_event" in df.columns:
        t_event = pd.to_numeric(df["t_event"], errors="coerce")

        print("t_event > t_max rows:", int((t_event > T_MAX + 1e-6).sum()))
        print("negative t_event rows:", int((t_event < 0.0).sum()))

        print(
            "terminal rows with t_event=0:",
            int((label.isin(TERMINAL_LABEL_LIST) & (t_event <= 0.0)).sum()),
        )

        print(
            "MAX_TIME rows with t_event < t_max - 1e-6:",
            int(((label == LABEL_MAX_TIME) & (t_event < T_MAX - 1e-6)).sum()),
        )
    else:
        t_event = None
        print("t_event column missing")

    viol = {}

    if "min_r_earth" in df.columns:
        min_r_earth = pd.to_numeric(df["min_r_earth"], errors="coerce")
        viol["IMPACT_EARTH rows with min_r_earth > R_EARTH"] = int(
            ((label == LABEL_IMPACT_EARTH) & (min_r_earth > R_EARTH)).sum()
        )
        viol["MAX_TIME rows that graze Earth (min_r_earth <= R_EARTH)"] = int(
            ((label == LABEL_MAX_TIME) & (min_r_earth <= R_EARTH)).sum()
        )
    else:
        min_r_earth = None

    if "min_r_moon" in df.columns:
        min_r_moon = pd.to_numeric(df["min_r_moon"], errors="coerce")
        viol["IMPACT_MOON rows with min_r_moon > R_MOON"] = int(
            ((label == LABEL_IMPACT_MOON) & (min_r_moon > R_MOON)).sum()
        )
        viol["MAX_TIME rows that graze Moon (min_r_moon <= R_MOON)"] = int(
            ((label == LABEL_MAX_TIME) & (min_r_moon <= R_MOON)).sum()
        )
    else:
        min_r_moon = None

    if "max_r" in df.columns:
        max_r = pd.to_numeric(df["max_r"], errors="coerce")
        viol["ESCAPE rows with max_r < R_ESCAPE"] = int(
            ((label == LABEL_ESCAPE) & (max_r < R_ESCAPE)).sum()
        )
        viol["MAX_TIME rows with max_r >= R_ESCAPE"] = int(
            ((label == LABEL_MAX_TIME) & (max_r >= R_ESCAPE)).sum()
        )
    else:
        max_r = None

    for k, v in viol.items():
        print(f"{k}: {v}")

    if "n_steps" in df.columns:
        n_steps = pd.to_numeric(df["n_steps"], errors="coerce")
        print(
            "\nn_steps: min/median/max =",
            n_steps.min(),
            n_steps.median(),
            n_steps.max(),
        )
        print(
            "terminal rows with n_steps = 0:",
            int((label.isin(TERMINAL_LABEL_LIST) & (n_steps == 0)).sum()),
        )
    else:
        print("\nn_steps column missing")

    if "jacobi_error" in df.columns:
        je = pd.to_numeric(df["jacobi_error"], errors="coerce").abs()
        print(
            "\nCSV jacobi_error min/median/max =",
            je.min(),
            je.median(),
            je.max(),
        )
        print(
            f"CSV rows with jacobi_error > {JACOBI_WARN_THRESHOLD}:",
            int((je > JACOBI_WARN_THRESHOLD).sum()),
        )

    print("any NaNs:", int(df.isna().sum().sum()))

    if "outcome" in df.columns:
        known = label.isin(LABEL_NAMES)
        expected_outcome = label.loc[known].map(LABEL_NAMES)
        bad_outcome = int((df.loc[known, "outcome"] != expected_outcome).sum())
        print(f"outcome-string mismatches: {bad_outcome} / {int(known.sum())}")

    return df


def choose_samples(df, n, seed):
    if n <= 0:
        return df.iloc[0:0]

    valid = df[
        (df["label_id"] != LABEL_INVALID)
        & (~df["expected_invalid"].fillna(False))
        & df["C_used"].notna()
    ].copy()

    if valid.empty:
        return valid

    if len(valid) <= n:
        return valid.sample(frac=1.0, random_state=seed)

    if not STRATIFY:
        return valid.sample(n=n, random_state=seed)

    counts = valid["label_id"].value_counts()
    labels = counts.index.tolist()

    raw = (counts / counts.sum()) * n
    
    alloc = {lab: int(np.floor(raw.get(lab, 0))) for lab in labels}
    remainder = int(n - sum(alloc.values()))

    frac = (raw - pd.Series({lab: np.floor(raw.get(lab, 0)) for lab in labels})).sort_values(ascending=False)

    for lab in frac.index:
        if remainder <= 0:
            break
        if alloc.get(lab, 0) < counts.get(lab, 0):
            alloc[lab] = alloc.get(lab, 0) + 1
            remainder -= 1

    parts = []
    for lab, k in alloc.items():
        if k <= 0:
            continue
        group = valid[valid["label_id"] == lab]
        k = min(k, len(group))
        if k > 0:
            parts.append(group.sample(n=k, random_state=seed))

    if not parts:
        return valid.sample(n=min(n, len(valid)), random_state=seed)

    out = pd.concat(parts)
    return out.sample(frac=1.0, random_state=seed)


def classify_basic(label_rust, label_scipy, t_rust, t_scipy):
    if label_scipy == LABEL_INTEGRATOR_ERROR:
        return "INTEGRATOR_ERROR", "SciPy integration failed"

    if label_rust == LABEL_INTEGRATOR_ERROR:
        return "INTEGRATOR_ERROR", "Rust label is integrator error"

    if label_rust == label_scipy:
        if label_rust in TERMINAL_LABELS and np.isfinite(t_rust) and np.isfinite(t_scipy):
            dt = abs(float(t_scipy) - float(t_rust))
            tol = TIME_ATOL + TIME_RTOL * max(abs(float(t_rust)), abs(float(t_scipy)), 1.0)

            if dt <= tol:
                return "OK", ""

            if min(float(t_rust), float(t_scipy)) > CHAOTIC_TIME_THRESHOLD:
                return "TIME_MISMATCH_POSSIBLY_CHAOTIC", f"dt={dt:.3e}"

            return "TIME_MISMATCH", f"dt={dt:.3e}"

        return "OK", ""

    notes = (
        f"rust={LABEL_NAMES.get(label_rust, label_rust)}, "
        f"scipy={LABEL_NAMES.get(label_scipy, label_scipy)}"
    )

    if (
        label_rust == LABEL_MAX_TIME
        and np.isfinite(t_scipy)
        and t_scipy >= T_MAX - TMAX_BOUNDARY_TOL
    ):
        return "LABEL_MISMATCH_NEAR_TMAX", notes

    if (
        label_scipy == LABEL_MAX_TIME
        and np.isfinite(t_rust)
        and t_rust >= T_MAX - TMAX_BOUNDARY_TOL
    ):
        return "LABEL_MISMATCH_NEAR_TMAX", notes

    if label_rust in TERMINAL_LABELS and label_scipy in TERMINAL_LABELS:
        return "LABEL_MISMATCH_TERMINAL", notes

    if label_rust == LABEL_INVALID or label_scipy == LABEL_INVALID:
        return "LABEL_MISMATCH_ENERGY", notes

    return "LABEL_MISMATCH", notes


def compare_samples(samples):
    records = []

    if samples.empty:
        return pd.DataFrame()

    for i, (idx, row) in enumerate(samples.iterrows()):
        x0 = sfloat(row.get("x0"), 0.0)
        vx0 = sfloat(row.get("vx0"), 0.0)
        y0 = sfloat(row.get("y0"), 0.0)
        z0 = sfloat(row.get("z0"), 0.0)
        vz0 = sfloat(row.get("vz0"), 0.0)
        C = sfloat(row.get("C_used"), np.nan)

        vy0_raw = row.get("vy0", None)
        vy0 = None
        if vy0_raw is not None and not pd.isna(vy0_raw):
            vy0 = float(vy0_raw)

        res = simulate_initial(
            x0=x0,
            vx0=vx0,
            C=C,
            vy0=vy0,
            y0=y0,
            z0=z0,
            vz0=vz0,
        )

        label_rust = int(row["label_id"])
        label_scipy = int(res["label_id"])

        t_rust = sfloat(row.get("t_event"), np.nan)
        t_scipy = float(res["t_event"])

        status, notes = classify_basic(label_rust, label_scipy, t_rust, t_scipy)

        outcome_rust = row.get("outcome", "")
        if pd.isna(outcome_rust):
            outcome_rust = ""

        rec = {
            "row_index": idx,
            "x0": x0,
            "vx0": vx0,
            "C": C,
            "label_rust": label_rust,
            "outcome_rust": outcome_rust,
            "t_rust": t_rust,
            "n_steps_rust": sfloat(row.get("n_steps"), np.nan),
            "jacobi_error_rust": sfloat(row.get("jacobi_error"), np.nan),
            "label_scipy": label_scipy,
            "outcome_scipy": LABEL_NAMES.get(label_scipy, str(label_scipy)),
            "t_scipy": t_scipy,
            "min_r_earth_scipy": res["min_r_earth"],
            "min_r_moon_scipy": res["min_r_moon"],
            "max_r_scipy": res["max_r"],
            "jacobi_error_scipy": res["jacobi_error"],
            "scipy_success": res["success"],
            "scipy_message": res["message"],
            "status_initial": status,
            "status": status,
            "notes": notes,
        }

        records.append(rec)

        if (i + 1) % 10 == 0:
            print(f"  SciPy comparison: {i + 1}/{len(samples)} rows done")

    return pd.DataFrame(records)

def probe_boundary(
    x0,
    vx0,
    C,
    label_scipy,
    label_rust,
    t_probe=T_MAX,
):
    if not np.isfinite(C):
        return False, {label_scipy}

    labels = set()

    for mult in PROBE_MULTIPLIERS:
        eps_x = PROBE_EPS_X * mult
        eps_vx = PROBE_EPS_VX * mult

        for dx, dvx in PROBE_DIRECTIONS:
            xp = x0 + dx * eps_x
            vxp = vx0 + dvx * eps_vx

            res = simulate_initial(
                x0=xp,
                vx0=vxp,
                C=C,
                vy0=None,
                t_max=t_probe,
                rtol=PROBE_RTOL,
                atol=PROBE_ATOL,
                max_step=PROBE_MAX_STEP,
            )

            lab = int(res["label_id"])

            if lab != LABEL_INTEGRATOR_ERROR:
                labels.add(lab)

                if len(labels) > 1:
                    return True, labels

    if not labels:
        return False, {label_scipy}

    sensitive = (
        len(labels) > 1
        or label_scipy not in labels
        or ((label_rust != label_scipy) and (label_rust in labels))
    )

    return sensitive, labels


def probe_mismatches(cmp):
    if cmp.empty or not PROBE_MISMATCHES:
        return cmp

    mismatch_mask = cmp["status"].str.startswith("LABEL_MISMATCH")
    idxs = cmp.index[mismatch_mask]

    if len(idxs) == 0:
        return cmp

    if len(idxs) > MAX_PROBE_MISMATCHES:
        print(
            f"Probing first {MAX_PROBE_MISMATCHES} label mismatches "
            f"out of {len(idxs)}."
        )
        idxs = idxs[:MAX_PROBE_MISMATCHES]
    else:
        print(f"Probing {len(idxs)} label mismatches.")

    for idx in idxs:
        x0 = float(cmp.at[idx, "x0"])
        vx0 = float(cmp.at[idx, "vx0"])
        C = float(cmp.at[idx, "C"])

        label_rust = int(cmp.at[idx, "label_rust"])
        label_scipy = int(cmp.at[idx, "label_scipy"])

        t_rust = sfloat(cmp.at[idx, "t_rust"], T_MAX)
        t_scipy = sfloat(cmp.at[idx, "t_scipy"], T_MAX)

        t_probe = min(T_MAX, max(t_rust, t_scipy) + PROBE_TIME_MARGIN)

        sensitive, labels = probe_boundary(
            x0=x0,
            vx0=vx0,
            C=C,
            label_scipy=label_scipy,
            label_rust=label_rust,
            t_probe=t_probe,
        )

        old_status = cmp.at[idx, "status"]
        label_str = f"probe_labels={sorted(labels)}"

        if sensitive:
            cmp.at[idx, "status"] = "CHAOTIC_BOUNDARY"
            cmp.at[idx, "notes"] = f"{cmp.at[idx, 'notes']}; {label_str}"
        else:
            if old_status == "LABEL_MISMATCH_NEAR_TMAX":
                cmp.at[idx, "status"] = "TMAX_EDGE_NEEDS_REVIEW"
            else:
                cmp.at[idx, "status"] = "GENUINE_MISMATCH"

            cmp.at[idx, "notes"] = f"{cmp.at[idx, 'notes']}; {label_str}"

    return cmp


def plot_basin_map(df, cmp):
    if "x0" not in df.columns or "vx0" not in df.columns:
        return

    max_points = 200000
    d = df

    if len(d) > max_points:
        d = d.sample(n=max_points, random_state=RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    cmap = plt.get_cmap("tab10")
    labels = sorted(d["label_id"].dropna().unique())

    for lab in labels:
        sub = d[d["label_id"] == lab]
        ax.scatter(
            sub["x0"],
            sub["vx0"],
            s=1.2,
            alpha=0.35,
            color=cmap(int(lab) % 10),
            label=LABEL_NAMES.get(int(lab), str(lab)),
        )

    ax.set_xlabel("x0")
    ax.set_ylabel("vx0")
    ax.set_title("Basin labels from CSV")
    ax.legend(markerscale=6, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "basin_labels.png"), dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cmp):
    if cmp.empty:
        return

    preferred = [0, 1, 2, 3, 4, -1]
    present = set(cmp["label_rust"].unique()) | set(cmp["label_scipy"].unique())

    labels = [lab for lab in preferred if lab in present]
    labels += sorted(present - set(preferred))

    if not labels:
        return

    ct = pd.crosstab(cmp["label_rust"], cmp["label_scipy"])
    ct = ct.reindex(index=labels, columns=labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(ct.values, cmap="Blues", aspect="auto")

    tick_labels = [SHORT_LABEL_NAMES.get(lab, str(lab)) for lab in labels]

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("SciPy label")
    ax.set_ylabel("Rust label")
    ax.set_title("Rust vs SciPy outcome confusion matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(ct.values[i, j])
            if val > 0:
                ax.text(
                    j,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=160)
    plt.close(fig)


def plot_time_comparison(cmp):
    if cmp.empty:
        return

    sub = cmp[
        (cmp["label_rust"] == cmp["label_scipy"])
        & cmp["label_rust"].isin(TERMINAL_LABEL_LIST)
    ].copy()

    if sub.empty:
        return

    finite = np.isfinite(sub["t_rust"]) & np.isfinite(sub["t_scipy"])
    sub = sub.loc[finite]

    if sub.empty:
        return

    dt = sub["t_scipy"] - sub["t_rust"]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    axes[0].hist(dt, bins=50, color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("t_event SciPy - t_event Rust")
    axes[0].set_ylabel("count")
    axes[0].set_title("Event-time difference, same terminal label")
    axes[0].axvline(0.0, color="k", linestyle="--", linewidth=1)

    axes[1].scatter(sub["t_rust"], sub["t_scipy"], s=8, alpha=0.7)
    lim_min = min(sub["t_rust"].min(), sub["t_scipy"].min())
    lim_max = max(sub["t_rust"].max(), sub["t_scipy"].max())

    if np.isfinite(lim_min) and np.isfinite(lim_max):
        pad = max(1e-6, 0.02 * (lim_max - lim_min))
        lims = [lim_min - pad, lim_max + pad]
        axes[1].plot(lims, lims, "k--", linewidth=1)

    axes[1].set_xlabel("t_event Rust")
    axes[1].set_ylabel("t_event SciPy")
    axes[1].set_title("Event-time comparison")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "time_event_comparison.png"), dpi=160)
    plt.close(fig)


def plot_mismatch_map(cmp):
    if cmp.empty or "x0" not in cmp.columns or "vx0" not in cmp.columns:
        return

    fig, ax = plt.subplots(figsize=(9.0, 6.8))

    ok = cmp["status"] == "OK"

    if ok.any():
        ax.scatter(
            cmp.loc[ok, "x0"],
            cmp.loc[ok, "vx0"],
            s=14,
            color="lightgray",
            alpha=0.7,
            label="OK",
        )

    bad = ~ok

    if bad.any():
        statuses = sorted(cmp.loc[bad, "status"].unique())
        cmap = plt.get_cmap("tab10")

        for i, status in enumerate(statuses):
            sub = cmp.loc[bad & (cmp["status"] == status)]
            ax.scatter(
                sub["x0"],
                sub["vx0"],
                s=38,
                marker="x",
                color=cmap(i % 10),
                label=status,
            )

    ax.set_xlabel("x0")
    ax.set_ylabel("vx0")
    ax.set_title("Validation status map")
    ax.legend(loc="best", fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "mismatch_map.png"), dpi=160)
    plt.close(fig)


def plot_status_summary(cmp):
    if cmp.empty:
        return

    counts = cmp["status"].value_counts()

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.bar(counts.index.astype(str), counts.values, color="tab:blue", alpha=0.85)

    ax.set_ylabel("count")
    ax.set_title("Validation status summary")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "status_summary.png"), dpi=160)
    plt.close(fig)


def plot_jacobi_error(df, cmp):
    has_csv = "jacobi_error" in df.columns
    has_scipy = (cmp is not None) and (not cmp.empty) and ("jacobi_error_scipy" in cmp.columns)

    if not has_csv and not has_scipy:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    plotted = False

    if has_csv:
        csv_err = pd.to_numeric(df["jacobi_error"], errors="coerce").abs().dropna()
        if len(csv_err) > 0:
            ax.hist(csv_err, bins=50, alpha=0.55, label="CSV jacobi_error")
            plotted = True

    if has_scipy:
        sci_err = pd.to_numeric(cmp["jacobi_error_scipy"], errors="coerce").abs().dropna()
        if len(sci_err) > 0:
            ax.hist(sci_err, bins=50, alpha=0.55, label="SciPy jacobi_error")
            plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.axvline(JACOBI_WARN_THRESHOLD, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("abs Jacobi error")
    ax.set_ylabel("count")
    ax.set_title("Jacobi conservation error")
    ax.set_yscale("symlog")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "jacobi_error.png"), dpi=160)
    plt.close(fig)


def plot_all(df, cmp):
    ensure_dir(OUT_DIR)

    if cmp is None:
        cmp = pd.DataFrame()

    plot_basin_map(df, cmp)
    plot_confusion_matrix(cmp)
    plot_time_comparison(cmp)
    plot_mismatch_map(cmp)
    plot_status_summary(cmp)
    plot_jacobi_error(df, cmp)

def print_summary(cmp):
    if cmp.empty:
        print("\nNo SciPy comparison rows were processed.")
        return

    print("\n" + "=" * 78)
    print("2) SciPy comparison summary")
    print("=" * 78)

    print(cmp["status"].value_counts().to_string())

    match_rate = float((cmp["label_rust"] == cmp["label_scipy"]).mean())
    print(f"\nlabel match rate: {match_rate:.4%}")

    genuine = cmp[cmp["status"] == "GENUINE_MISMATCH"]
    chaotic = cmp[cmp["status"] == "CHAOTIC_BOUNDARY"]
    tmax_edge = cmp[cmp["status"] == "TMAX_EDGE_NEEDS_REVIEW"]

    print(f"genuine mismatches: {len(genuine)}")
    print(f"chaotic-boundary mismatches: {len(chaotic)}")
    print(f"t_max edge cases needing review: {len(tmax_edge)}")

    if not genuine.empty:
        print("\nFirst genuine mismatches:")
        cols = [
            "x0",
            "vx0",
            "label_rust",
            "label_scipy",
            "t_rust",
            "t_scipy",
            "notes",
        ]
        cols = [c for c in cols if c in genuine.columns]
        print(genuine[cols].head(10).to_string(index=False))

def main():
    global CSV_PATH, OUT_DIR

    if len(sys.argv) > 1:
        CSV_PATH = sys.argv[1]

    if len(sys.argv) > 2:
        OUT_DIR = sys.argv[2]

    ensure_dir(OUT_DIR)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    print(f"Reading basin CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    df = static_report(df)

    if N_SAMPLES > 0:
        samples = choose_samples(df, N_SAMPLES, RANDOM_SEED)

        print("\n" + "=" * 78)
        print("2) SciPy random-sample comparison")
        print("=" * 78)
        print(f"selected {len(samples)} valid rows for SciPy re-integration")

        cmp = compare_samples(samples)

        if not cmp.empty:
            print("\n" + "=" * 78)
            print("3) Mismatch boundary probing")
            print("=" * 78)

            cmp = probe_mismatches(cmp)

            cmp_path = os.path.join(OUT_DIR, "scipy_comparison.csv")
            cmp.to_csv(cmp_path, index=False)
            print(f"\nWrote detailed comparison table: {cmp_path}")

            print_summary(cmp)
        else:
            print("\nNo valid rows available for SciPy comparison.")
            cmp = pd.DataFrame()
    else:
        cmp = pd.DataFrame()

    plot_all(df, cmp)

    print(f"\nOutputs written to: {OUT_DIR}/")


if __name__ == "__main__":
    main()