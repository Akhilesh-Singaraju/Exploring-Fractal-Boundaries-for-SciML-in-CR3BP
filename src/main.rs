use rayon::prelude::*;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};

const MU: f64 = 0.0121505856;
const R_EARTH: f64 = 6378.1363 / 384400.0;
const R_MOON: f64 = 1737.4 / 384400.0;
const R_ESCAPE: f64 = 5.0;

const OUTCOMES: [&str; 5] = [
    "INVALID_ENERGY",
    "IMPACT_EARTH",
    "IMPACT_MOON",
    "ESCAPE",
    "MAX_TIME",
];

const RTOL: f64 = 1e-10;
const ATOL: f64 = 1e-10;
const MAX_STEPS: u64 = 200_000;

#[inline]
fn cr3bp_rhs(_t: f64, y: &[f64; 4]) -> [f64; 4] {
    let (x, yy, vx, vy) = (y[0], y[1], y[2], y[3]);
    let dx1 = x + MU;
    let dx2 = x - 1.0 + MU;
    let r1_sq = dx1 * dx1 + yy * yy;
    let r2_sq = dx2 * dx2 + yy * yy;

    let r1_3 = r1_sq * r1_sq.sqrt();
    let r2_3 = r2_sq * r2_sq.sqrt();

    let domega_dx = x - (1.0 - MU) * dx1 / r1_3 - MU * dx2 / r2_3;
    let domega_dy = yy - (1.0 - MU) * yy / r1_3 - MU * yy / r2_3;

    [vx, vy, domega_dx + 2.0 * vy, domega_dy - 2.0 * vx]
}

#[inline]
fn event_triggered(y: &[f64; 4]) -> bool {
    let (x, yy) = (y[0], y[1]);
    let r1_sq = (x + MU) * (x + MU) + yy * yy;
    let r2_sq = (x - 1.0 + MU) * (x - 1.0 + MU) + yy * yy;
    let rc_sq = x * x + yy * yy;
    r1_sq <= R_EARTH * R_EARTH || r2_sq <= R_MOON * R_MOON || rc_sq >= R_ESCAPE * R_ESCAPE
}

#[inline]
fn update_extremes(y: &[f64; 4], min_r1: &mut f64, min_r2: &mut f64, max_rc: &mut f64) {
    let (x, yy) = (y[0], y[1]);
    let r1 = ((x + MU) * (x + MU) + yy * yy).sqrt();
    let r2 = ((x - 1.0 + MU) * (x - 1.0 + MU) + yy * yy).sqrt();
    let rc = (x * x + yy * yy).sqrt();
    if r1 < *min_r1 {
        *min_r1 = r1;
    }
    if r2 < *min_r2 {
        *min_r2 = r2;
    }
    if rc > *max_rc {
        *max_rc = rc;
    }
}

#[allow(clippy::excessive_precision)]
fn dopri5_step(t: f64, y: &[f64; 4], k1: &[f64; 4], dt: f64) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let mut ys = [0.0f64; 4];
    let mut k = [[0.0f64; 4]; 7];
    k[0] = *k1;

    for i in 0..4 {
        ys[i] = y[i] + dt * 0.2 * k[0][i];
    }
    k[1] = cr3bp_rhs(t + 0.2 * dt, &ys);

    for i in 0..4 {
        ys[i] = y[i] + dt * (3.0 / 40.0 * k[0][i] + 9.0 / 40.0 * k[1][i]);
    }
    k[2] = cr3bp_rhs(t + 0.3 * dt, &ys);

    for i in 0..4 {
        ys[i] = y[i]
            + dt * (44.0 / 45.0 * k[0][i] - 56.0 / 15.0 * k[1][i] + 32.0 / 9.0 * k[2][i]);
    }
    k[3] = cr3bp_rhs(t + 0.8 * dt, &ys);

    for i in 0..4 {
        ys[i] = y[i]
            + dt * (19372.0 / 6561.0 * k[0][i] - 25360.0 / 2187.0 * k[1][i]
                + 64448.0 / 6561.0 * k[2][i]
                - 212.0 / 729.0 * k[3][i]);
    }
    k[4] = cr3bp_rhs(t + (8.0 / 9.0) * dt, &ys);

    for i in 0..4 {
        ys[i] = y[i]
            + dt * (9017.0 / 3168.0 * k[0][i] - 355.0 / 33.0 * k[1][i]
                + 46732.0 / 5247.0 * k[2][i]
                + 49.0 / 176.0 * k[3][i]
                - 5103.0 / 18656.0 * k[4][i]);
    }
    k[5] = cr3bp_rhs(t + dt, &ys);

    for i in 0..4 {
        ys[i] = y[i]
            + dt * (35.0 / 384.0 * k[0][i] + 500.0 / 1113.0 * k[2][i]
                + 125.0 / 192.0 * k[3][i]
                - 2187.0 / 6784.0 * k[4][i]
                + 11.0 / 84.0 * k[5][i]);
    }
    k[6] = cr3bp_rhs(t + dt, &ys);

    let y_new = ys;

    let mut err = [0.0f64; 4];
    for i in 0..4 {
        err[i] = dt * (71.0 / 57600.0 * k[0][i]
            - 71.0 / 16695.0 * k[2][i]
            + 71.0 / 1920.0 * k[3][i]
            - 17253.0 / 339200.0 * k[4][i]
            + 22.0 / 525.0 * k[5][i]
            - 1.0 / 40.0 * k[6][i]);
    }
    (y_new, k[6], err)
}

fn error_norm(y: &[f64; 4], y_new: &[f64; 4], err: &[f64; 4]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let sc = ATOL + RTOL * y[i].abs().max(y_new[i].abs());
        sum += (err[i] / sc).powi(2);
    }
    (sum / 4.0).sqrt()
}

fn hermite(y0: &[f64; 4], f0: &[f64; 4], y1: &[f64; 4], f1: &[f64; 4], h: f64, s: f64) -> [f64; 4] {
    let h00 = 2.0 * s * s * s - 3.0 * s * s + 1.0;
    let h10 = s * s * s - 2.0 * s * s + s;
    let h01 = -2.0 * s * s * s + 3.0 * s * s;
    let h11 = s * s * s - s * s;
    let mut out = [0.0f64; 4];
    for i in 0..4 {
        out[i] = h00 * y0[i] + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i];
    }
    out
}

fn refine_event(
    t0: f64,
    t1: f64,
    y0: &[f64; 4],
    f0: &[f64; 4],
    y1: &[f64; 4],
    f1: &[f64; 4],
) -> ([f64; 4], f64) {
    let h = t1 - t0;
    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
    let mut mid = 0.5f64;
    for _ in 0..60 {
        mid = 0.5 * (lo + hi);
        let ym = hermite(y0, f0, y1, f1, h, mid);
        if event_triggered(&ym) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    (hermite(y0, f0, y1, f1, h, hi), t0 + hi * h)
}


#[derive(Clone, Copy, Debug)]
pub struct Solution {
    pub label: u8,
    pub t_event: f64,
    pub n_steps: u64,
    pub min_r1: f64,
    pub min_r2: f64,
    pub max_rc: f64,
}

#[inline]
fn classify(final_state: &[f64; 4], valid: bool) -> u8 {
    let (x_f, y_f) = (final_state[0], final_state[1]);
    let r1_f = ((x_f + MU) * (x_f + MU) + y_f * y_f).sqrt();
    let r2_f = ((x_f - 1.0 + MU) * (x_f - 1.0 + MU) + y_f * y_f).sqrt();
    let rc_f = (x_f * x_f + y_f * y_f).sqrt();

    if !valid {
        0
    } else if r1_f <= R_EARTH {
        1
    } else if r2_f <= R_MOON {
        2
    } else if rc_f >= R_ESCAPE {
        3
    } else {
        4
    }
}

pub fn solve_single(x0: f64, vx0: f64, jacobi_c: f64, t_max: f64) -> Solution {
    let r1_0 = (x0 + MU).abs();
    let r2_0 = (x0 - 1.0 + MU).abs();

    let omega0 = 0.5 * x0 * x0 + (1.0 - MU) / r1_0 + MU / r2_0;
    let vy2 = 2.0 * omega0 - jacobi_c - vx0 * vx0;
    let valid = vy2 >= 0.0;
    let vy0 = if valid { vy2.max(0.0).sqrt() } else { 0.0 };

    let mut y = [x0, 0.0, vx0, vy0];
    let mut t = 0.0f64;
    let mut dt = 0.01f64;
    let mut f = cr3bp_rhs(t, &y);
    let mut n_accepted: u64 = 0;

    let mut min_r1 = f64::INFINITY;
    let mut min_r2 = f64::INFINITY;
    let mut max_rc = 0.0f64;
    update_extremes(&y, &mut min_r1, &mut min_r2, &mut max_rc);

    let mut final_state = y;
    let mut final_t = 0.0f64;
    let mut terminated_by_event = event_triggered(&y);

    if !terminated_by_event {
        let mut iterations: u64 = 0;
        while n_accepted < MAX_STEPS && iterations < 4 * MAX_STEPS {
            iterations += 1;
            if t >= t_max {
                break;
            }
            let mut h = dt;
            if t + h > t_max {
                h = t_max - t;
            }
            if h < 1e-14 {
                break;
            }

            let (y_new, f_new, err) = dopri5_step(t, &y, &f, h);

            if !y_new.iter().all(|v| v.is_finite()) || !err.iter().all(|v| v.is_finite()) {
                dt = h * 0.1;
                continue;
            }

            let en = error_norm(&y, &y_new, &err);

            if en <= 1.0 {
                n_accepted += 1;
                if event_triggered(&y_new) {
                    let (yc, tc) = refine_event(t, t + h, &y, &f, &y_new, &f_new);
                    final_state = yc;
                    final_t = tc;
                    update_extremes(&yc, &mut min_r1, &mut min_r2, &mut max_rc);
                    terminated_by_event = true;
                    break;
                }
                t += h;
                y = y_new;
                f = f_new;
                update_extremes(&y, &mut min_r1, &mut min_r2, &mut max_rc);
                let fac = if en == 0.0 {
                    5.0
                } else {
                    (0.9 * en.powf(-0.2)).clamp(0.2, 5.0)
                };
                dt = h * fac;
            } else {
                let fac = if en.is_finite() {
                    (0.9 * en.powf(-0.2)).max(0.1)
                } else {
                    0.1
                };
                dt = h * fac;
            }
        }
        if !terminated_by_event {
            final_state = y;
            final_t = t;
        }
    }

    let label = classify(&final_state, valid);
    Solution {
        label,
        t_event: final_t,
        n_steps: n_accepted,
        min_r1,
        min_r2,
        max_rc,
    }
}

pub fn solve_batch(xs: &[f64], vxs: &[f64], c: f64, t_max: f64) -> Vec<Solution> {
    xs.par_iter()
        .zip(vxs.par_iter())
        .map(|(&x, &vx)| solve_single(x, vx, c, t_max))
        .collect()
}

#[derive(Clone, Debug)]
struct GridRow {
    x0: f64,
    vx0: f64,
    c: f64,
    label_id: u8,
    t_event: f64,
    n_steps: u64,
    min_r_earth: f64,
    min_r_moon: f64,
    max_r: f64,
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| a + step * i as f64).collect()
}

fn generate_basin_grid(
    c: f64,
    t_max: f64,
    res_x: usize,
    res_vx: usize,
    x_bounds: (f64, f64),
    vx_bounds: (f64, f64),
) -> Vec<GridRow> {
    let xs = linspace(x_bounds.0, x_bounds.1, res_x);
    let vxs = linspace(vx_bounds.0, vx_bounds.1, res_vx);
    let total = res_x * res_vx;

    (0..total)
        .into_par_iter()
        .map(|idx| {
            let i = idx / res_x;
            let j = idx % res_x;
            let (x0, vx0) = (xs[j], vxs[i]);
            let s = solve_single(x0, vx0, c, t_max);
            GridRow {
                x0,
                vx0,
                c,
                label_id: s.label,
                t_event: s.t_event,
                n_steps: s.n_steps,
                min_r_earth: s.min_r1,
                min_r_moon: s.min_r2,
                max_r: s.max_rc,
            }
        })
        .collect()
}

fn write_basin_csv(path: &str, rows: &[GridRow]) -> std::io::Result<()> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    writeln!(w, "x0,vx0,C,label_id,outcome,t_event,n_steps,min_r_earth,min_r_moon,max_r")?;
    for r in rows {
        writeln!(
            w,
            "{:?},{:?},{:?},{},{},{:?},{},{:?},{:?},{:?}",
            r.x0, r.vx0, r.c, r.label_id, OUTCOMES[r.label_id as usize],
            r.t_event, r.n_steps, r.min_r_earth, r.min_r_moon, r.max_r
        )?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct CsvRow {
    x0: f64,
    vx0: f64,
    c: f64,
    outcome: String,
    t_event: f64,
    n_steps: u64,
    min_r_earth: f64,
    min_r_moon: f64,
    max_r: f64,
}

fn read_basin_csv(path: &str) -> std::io::Result<Vec<CsvRow>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 {
            continue;
        }
        let p: Vec<&str> = line.split(',').collect();
        if p.len() != 10 {
            continue;
        }
        rows.push(CsvRow {
            x0: p[0].parse().unwrap(),
            vx0: p[1].parse().unwrap(),
            c: p[2].parse().unwrap(),
            outcome: p[4].to_string(),
            t_event: p[5].parse().unwrap(),
            n_steps: p[6].parse().unwrap(),
            min_r_earth: p[7].parse().unwrap(),
            min_r_moon: p[8].parse().unwrap(),
            max_r: p[9].parse().unwrap(),
        });
    }
    Ok(rows)
}

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * (self.next_u32() as f64) / ((u32::MAX as f64) + 1.0)
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (n - 1) as f64;
    let lo = (pos.floor() as usize).min(n - 1);
    let hi = (pos.ceil() as usize).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn sample_std(v: &[f64], mean: f64) -> f64 {
    if v.len() < 2 {
        return f64::NAN;
    }
    let ss: f64 = v.iter().map(|x| (x - mean) * (x - mean)).sum();
    (ss / (v.len() - 1) as f64).sqrt()
}

pub fn analyze_convergence(filepath: &str, max_samples: usize) {
    let df = read_basin_csv(filepath).expect("failed to read CSV");
    let unresolved: Vec<&CsvRow> = df.iter().filter(|r| r.outcome == "MAX_TIME").collect();

    let n = max_samples.min(unresolved.len());
    if n == 0 {
        println!("No unresolved (MAX_TIME) trajectories found.");
        return;
    }

    let mut rng = Lcg::new(42);
    let mut idx: Vec<usize> = (0..unresolved.len()).collect();
    for i in 0..n {
        let j = i + (rng.next_u32() as usize) % (unresolved.len() - i);
        idx.swap(i, j);
    }

    let x0: Vec<f64> = idx[..n].iter().map(|&i| unresolved[i].x0).collect();
    let vx0: Vec<f64> = idx[..n].iter().map(|&i| unresolved[i].vx0).collect();
    let c_val = unresolved[idx[0]].c;

    for t_max in [100.0, 300.0, 1000.0, 3000.0, 6000.0, 9000.0, 12000.0] {
        let sols = solve_batch(&x0, &vx0, c_val, t_max);

        let mut counts = [0usize; 5];
        for s in &sols {
            counts[s.label as usize] += 1;
        }

        println!("\nt_max = {:?}", t_max);
        let total: usize = counts.iter().sum();
        for (li, &count) in counts.iter().enumerate() {
            println!(
                "{:<20} {:>4} ({:.1}%)",
                OUTCOMES[li],
                count,
                100.0 * (count as f64) / total as f64
            );
        }
    }

    let sols = solve_batch(&x0, &vx0, c_val, 12000.0);
    let mut long_lived: Vec<GridRow> = Vec::new();
    for (i, s) in sols.iter().enumerate() {
        if s.label == 4 {
            long_lived.push(GridRow {
                x0: x0[i],
                vx0: vx0[i],
                c: c_val,
                label_id: 4,
                t_event: s.t_event,
                n_steps: s.n_steps,
                min_r_earth: s.min_r1,
                min_r_moon: s.min_r2,
                max_r: s.max_rc,
            });
        }
    }

    println!("\nRemaining MAX_TIME trajectories: {}", long_lived.len());
    if long_lived.is_empty() {
        return;
    }

    println!(
        "{:>12} {:>12} {:>15} {:>12} {:>9} {:>12} {:>12} {:>12}",
        "x0", "vx0", "outcome", "t_event", "n_steps", "min_r_earth", "min_r_moon", "max_r"
    );
    for r in &long_lived {
        println!(
            "{:>12.6} {:>12.6} {:>15} {:>12.4} {:>9} {:>12.6} {:>12.6} {:>12.6}",
            r.x0, r.vx0, "MAX_TIME", r.t_event, r.n_steps,
            r.min_r_earth, r.min_r_moon, r.max_r
        );
    }

    let col_names = ["min_r_earth", "min_r_moon", "max_r", "n_steps"];
    let col_data: Vec<Vec<f64>> = vec![
        long_lived.iter().map(|r| r.min_r_earth).collect(),
        long_lived.iter().map(|r| r.min_r_moon).collect(),
        long_lived.iter().map(|r| r.max_r).collect(),
        long_lived.iter().map(|r| r.n_steps as f64).collect(),
    ];

    let stat_labels = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
    let mut col_stats: Vec<[f64; 8]> = Vec::with_capacity(4);
    for v in &col_data {
        let mut sorted = v.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        col_stats.push([
            v.len() as f64,
            mean,
            sample_std(v, mean),
            sorted[0],
            quantile(&sorted, 0.25),
            quantile(&sorted, 0.50),
            quantile(&sorted, 0.75),
            sorted[sorted.len() - 1],
        ]);
    }

    println!(
        "\n{:>8} {:>14} {:>14} {:>14} {:>14}",
        "", col_names[0], col_names[1], col_names[2], col_names[3]
    );
    for (si, slabel) in stat_labels.iter().enumerate() {
        println!(
            "{:>8} {:>14.6} {:>14.6} {:>14.6} {:>14.6}",
            slabel, col_stats[0][si], col_stats[1][si], col_stats[2][si], col_stats[3][si]
        );
    }
}

pub fn pre_test_convergence(c: f64, t_max_list: &[f64], n_samples: usize) {
    println!("\nRunning Pre-Test for C={}", c);
    let mut rng = Lcg::new(42);
    let x0: Vec<f64> = (0..n_samples).map(|_| rng.uniform(-1.5, 1.5)).collect();
    let vx0: Vec<f64> = (0..n_samples).map(|_| rng.uniform(-3.0, 3.0)).collect();

    for &t in t_max_list {
        let sols = solve_batch(&x0, &vx0, c, t);
        let mut counts = [0usize; 5];
        for s in &sols {
            counts[s.label as usize] += 1;
        }
        let parts: Vec<String> = OUTCOMES
            .iter()
            .zip(counts.iter())
            .map(|(k, v)| format!("{}: {:>3}", k, v))
            .collect();
        println!("t_max = {:<6} | {}", format!("{:?}", t), parts.join(" | "));
    }
}

pub fn run_batch_generation(t_val: f64) {
    let c_values = [3.00, 3.05, 3.10, 3.14, 3.20];
    let grid_resolution = 300;

    fs::create_dir_all("files/data").unwrap();

    for &c in &c_values {
        let df = generate_basin_grid(
            c,
            t_val,
            grid_resolution,
            grid_resolution,
            (-1.5, 1.5),
            (-3.0, 3.0),
        );
        let output_path = format!("files/data/basin_C{:.2}.csv", c);
        write_basin_csv(&output_path, &df).unwrap();
        println!("Saved: {}", output_path);
    }
}

fn main() {
    // pre_test_convergence(3.00, &[500.0, 1000.0, 5000.0, 10000.0, 12000.0],1000)
    run_batch_generation(5000.0);
}