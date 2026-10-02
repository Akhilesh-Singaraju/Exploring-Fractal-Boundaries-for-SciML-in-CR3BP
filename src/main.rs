use ivp::prelude::*;
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

const RTOL: f64 = 1e-11;
const ATOL: f64 = 1e-13;

const MAX_STEP: f64 = 0.01;

struct Cr3bpSystem;

impl FirstOrderSystem for Cr3bpSystem {
    fn derivative(&self, _t: f64, y: &[f64], dydx: &mut [f64]) {
        let (x, yy, vx, vy) = (y[0], y[1], y[2], y[3]);
        let dx1 = x + MU;
        let dx2 = x - 1.0 + MU;
        let r1_sq = dx1 * dx1 + yy * yy;
        let r2_sq = dx2 * dx2 + yy * yy;

        let r1_3 = r1_sq * r1_sq.sqrt();
        let r2_3 = r2_sq * r2_sq.sqrt();

        let domega_dx = x - (1.0 - MU) * dx1 / r1_3 - MU * dx2 / r2_3;
        let domega_dy = yy - (1.0 - MU) * yy / r1_3 - MU * yy / r2_3;

        dydx[0] = vx;
        dydx[1] = vy;
        dydx[2] = domega_dx + 2.0 * vy;
        dydx[3] = domega_dy - 2.0 * vx;
    }

    fn events(&self, _t: f64, y: &[f64], out: &mut [f64]) {
        let (x, yy) = (y[0], y[1]);
        let r1 = ((x + MU) * (x + MU) + yy * yy).sqrt();
        let r2 = ((x - 1.0 + MU) * (x - 1.0 + MU) + yy * yy).sqrt();
        let rc = (x * x + yy * yy).sqrt();
        out[0] = r1 - R_EARTH;
        out[1] = r2 - R_MOON;
        out[2] = rc - R_ESCAPE;
    }

    fn n_events(&self) -> usize {
        3
    }

    fn event_config(&self, index: usize) -> EventConfig {
        let mut cfg = EventConfig::default();
        cfg.terminal();
        match index {
            0 | 1 => cfg.negative(),
            _ => cfg.positive(),
        }
        cfg
    }
}

#[inline]
fn classify(final_state: &[f64], valid: bool) -> u8 {
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

#[inline]
fn update_extremes(y: &[f64], min_r1: &mut f64, min_r2: &mut f64, max_rc: &mut f64) {
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


#[derive(Clone, Copy, Debug)]
pub struct Solution {
    pub label: u8,
    pub t_event: f64,
    pub n_steps: u64,
    pub min_r1: f64,
    pub min_r2: f64,
    pub max_rc: f64,
    pub jacobi_initial: f64,
    pub jacobi_final: f64,
    pub jacobi_error: f64,
}

pub fn solve_single(x0: f64, vx0: f64, jacobi_c: f64, t_max: f64) -> Solution {
    let r1_0 = (x0 + MU).abs();
    let r2_0 = (x0 - 1.0 + MU).abs();

    let omega0 = 0.5 * x0 * x0 + (1.0 - MU) / r1_0 + MU / r2_0;
    let vy2 = 2.0 * omega0 - jacobi_c - vx0 * vx0;

    if vy2 < 0.0 {
        return Solution {
            label: 0,
            t_event: 0.0,
            n_steps: 0,
            min_r1: r1_0,
            min_r2: r2_0,
            max_rc: x0.abs(),
            jacobi_initial: jacobi_c,
            jacobi_final: jacobi_c,
            jacobi_error: 0.0,
        };
    }

    let vy0 = vy2.sqrt();
    let y0 = [x0, 0.0, vx0, vy0];

    {
        let (x, yy) = (y0[0], y0[1]);
        let r1 = ((x + MU) * (x + MU) + yy * yy).sqrt();
        let r2 = ((x - 1.0 + MU) * (x - 1.0 + MU) + yy * yy).sqrt();
        let rc = (x * x + yy * yy).sqrt();
        if r1 <= R_EARTH || r2 <= R_MOON || rc >= R_ESCAPE {
            let label = classify(&y0, true);
            return Solution {
                label,
                t_event: 0.0,
                n_steps: 0,
                min_r1: r1,
                min_r2: r2,
                max_rc: rc,
                jacobi_initial: jacobi_c,
                jacobi_final: jacobi_c,
                jacobi_error: 0.0,
            };
        }
    }

    let max_steps = (((t_max / MAX_STEP).ceil() as usize) * 2).clamp(10_000, 20_000_000);

    let system = Cr3bpSystem;
    let sol = Ivp::first_order(&system, 0.0, t_max, &y0)
        .method(Method::DOP853)
        .rtol(RTOL)
        .atol(ATOL)
        .max_step(MAX_STEP)
        .max_steps(max_steps)
        .solve()
        .expect("DOP853 integration failed");

    const EVENT_LABELS: [u8; 3] = [1, 2, 3]; // Earth, Moon, Escape
    let mut earliest: Option<(usize, f64)> = None;
    for idx in 0..3 {
        if let Some(&te) = sol.t_events.get(idx).and_then(|v| v.first()) {
            match earliest {
                Some((_, t)) if te >= t => {}
                _ => earliest = Some((idx, te)),
            }
        }
    }

    let (label, t_event, final_state): (u8, f64, Vec<f64>) = match earliest {
        Some((idx, te)) => {
            let ys = sol
                .y_events
                .get(idx)
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or_else(|| y0.to_vec());
            (EVENT_LABELS[idx], te, ys)
        }
        None => {
            let fs = sol.y.last().cloned().unwrap_or_else(|| y0.to_vec());
            let ft = sol.t.last().copied().unwrap_or(0.0);
            (classify(&fs, true), ft, fs)
        }
    };

    let mut min_r1 = f64::INFINITY;
    let mut min_r2 = f64::INFINITY;
    let mut max_rc = 0.0f64;

    for (i, y) in sol.y.iter().enumerate() {
        if sol.t[i] <= t_event + 1e-12 {
            update_extremes(y, &mut min_r1, &mut min_r2, &mut max_rc);
        }
    }
    if !final_state.is_empty() {
        update_extremes(&final_state, &mut min_r1, &mut min_r2, &mut max_rc);
    }

    let (xf, yf, vxf, vyf) = (final_state[0], final_state[1], final_state[2], final_state[3]);
    let r1_f = ((xf + MU).powi(2) + yf * yf).sqrt();
    let r2_f = ((xf - 1.0 + MU).powi(2) + yf * yf).sqrt();
    let omega_f = 0.5 * (xf * xf + yf * yf) + (1.0 - MU) / r1_f + MU / r2_f;
    
    let jacobi_final = 2.0 * omega_f - (vxf * vxf + vyf * vyf);
    let jacobi_error = (jacobi_final - jacobi_c).abs();

    Solution {
        label,
        t_event,
        n_steps: sol.naccpt as u64,
        min_r1,
        min_r2,
        max_rc,
        jacobi_initial: jacobi_c,
        jacobi_final,
        jacobi_error,
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
    jacobi_initial: f64,
    jacobi_final: f64,
    jacobi_error: f64,
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
                jacobi_initial: s.jacobi_initial,
                jacobi_final: s.jacobi_final,
                jacobi_error: s.jacobi_error,
            }
        })
        .collect()
}

fn write_basin_csv(path: &str, rows: &[GridRow]) -> std::io::Result<()> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    writeln!(w, "x0,vx0,C,label_id,outcome,t_event,n_steps,min_r_earth,min_r_moon,max_r,jacobi_initial,jacobi_final,jacobi_error")?;
    for r in rows {
        writeln!(
            w,
            "{:.17e},{:.17e},{:.17e},{},{},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            r.x0, r.vx0, r.c, r.label_id, OUTCOMES[r.label_id as usize],
            r.t_event, r.n_steps, r.min_r_earth, r.min_r_moon, r.max_r,
            r.jacobi_initial, r.jacobi_final, r.jacobi_error
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
    jacobi_initial: f64,
    jacobi_final: f64,
    jacobi_error: f64,
}

fn read_basin_csv(path: &str) -> std::io::Result<Vec<CsvRow>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 { continue; }
        
        let p: Vec<&str> = line.split(',').collect();
        if p.len() != 13 {
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
            jacobi_initial: p[10].parse().unwrap(),
            jacobi_final: p[11].parse().unwrap(),
            jacobi_error: p[12].parse().unwrap(),
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
                jacobi_initial: s.jacobi_initial,
                jacobi_final: s.jacobi_final,
                jacobi_error: s.jacobi_error,
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
    // pre_test_convergence(3.00, &[500.0, 1000.0, 5000.0, 10000.0, 12000.0], 1000);
    run_batch_generation(5000.0);
}