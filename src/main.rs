use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ivp::dense::StepInterpolant;
use ivp::methods::DOP853;
use ivp::prelude::*;
use ivp::solout::{ControlFlag, SolOut};
use rayon::prelude::*;
use std::fs;
use std::io::{BufRead, BufReader, Write};

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
const N_EVENTS: usize = 3;

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
}

#[inline]
fn classify(y: &[f64]) -> u8 {
    let (r1, r2, rc) = radii(y[0], y[1]);

    if r1 <= R_EARTH {
        1
    } else if r2 <= R_MOON {
        2
    } else if rc >= R_ESCAPE {
        3
    } else {
        4
    }
}
#[inline]
fn radii(x: f64, y: f64) -> (f64, f64, f64) {
    let dx1 = x + MU;
    let dx2 = x - 1.0 + MU;

    let r1 = (dx1 * dx1 + y * y).sqrt();
    let r2 = (dx2 * dx2 + y * y).sqrt();
    let rc = (x * x + y * y).sqrt();

    (r1, r2, rc)
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

#[inline]
fn event_values(y: &[f64]) -> [f64; 3] {
    let (r1, r2, rc) = radii(y[0], y[1]);

    [r1 - R_EARTH, r2 - R_MOON, rc - R_ESCAPE]
}

#[inline]
fn event_value(index: usize, y: &[f64]) -> f64 {
    let (x, yy) = (y[0], y[1]);

    match index {
        0 => {
            let dx = x + MU;
            (dx * dx + yy * yy).sqrt() - R_EARTH
        }
        1 => {
            let dx = x - 1.0 + MU;
            (dx * dx + yy * yy).sqrt() - R_MOON
        }
        2 => (x * x + yy * yy).sqrt() - R_ESCAPE,
        _ => unreachable!("invalid event index"),
    }
}

#[inline]
fn event_crossed(index: usize, previous: f64, current: f64) -> bool {
    match index {
        0 | 1 => previous > 0.0 && current <= 0.0,

        2 => previous < 0.0 && current >= 0.0,

        _ => unreachable!("invalid event index"),
    }
}

fn refine_event(
    interpolant: &StepInterpolant<'_>,
    mut lo: f64,
    mut hi: f64,
    mut flo: f64,
    fhi: f64,
    index: usize,
) -> (f64, [f64; 4]) {
    let mut ym = [0.0; 4];

    if flo == 0.0 {
        interpolant.interpolate(lo, &mut ym);
        return (lo, ym);
    }

    if fhi == 0.0 {
        interpolant.interpolate(hi, &mut ym);
        return (hi, ym);
    }

    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);

        interpolant.interpolate(mid, &mut ym);
        let fm = event_value(index, &ym);

        if fm == 0.0 || (hi - lo).abs() <= 2.0e-12 * mid.abs().max(1.0) {
            return (mid, ym);
        }

        if flo.signum() == fm.signum() {
            lo = mid;
            flo = fm;
        } else {
            hi = mid;
        }
    }

    let t = 0.5 * (lo + hi);
    interpolant.interpolate(t, &mut ym);

    (t, ym)
}

struct BatchSolout {
    initialized: bool,
    prev_g: [f64; N_EVENTS],

    min_r1: f64,
    min_r2: f64,
    max_rc: f64,

    last_t: f64,
    last_y: [f64; 4],

    event: Option<(u8, f64, [f64; 4])>,
}

impl BatchSolout {
    fn new(y0: &[f64; 4]) -> Self {
        let mut out = Self {
            initialized: false,
            prev_g: [0.0; 3],
            min_r1: f64::INFINITY,
            min_r2: f64::INFINITY,
            max_rc: 0.0,
            last_t: 0.0,
            last_y: *y0,
            event: None,
        };

        out.update(y0);
        out
    }

    #[inline]
    fn update(&mut self, y: &[f64]) {
        let (r1, r2, rc) = radii(y[0], y[1]);

        self.min_r1 = self.min_r1.min(r1);
        self.min_r2 = self.min_r2.min(r2);
        self.max_rc = self.max_rc.max(rc);
    }
}

impl SolOut for BatchSolout {
    fn solout(
        &mut self,
        xold: f64,
        x: &mut f64,
        y: &mut [f64],
        interpolant: Option<&StepInterpolant<'_>>,
    ) -> ControlFlag {
        let curr_g = event_values(y);

        if !self.initialized {
            self.prev_g = curr_g;
            self.initialized = true;

            self.last_t = *x;
            self.last_y.copy_from_slice(y);

            return ControlFlag::Continue;
        }

        let mut earliest: Option<(usize, f64, [f64; 4])> = None;

        for index in 0..N_EVENTS {
            if !event_crossed(index, self.prev_g[index], curr_g[index]) {
                continue;
            }

            let (te, ye) = match interpolant {
                Some(ip) => refine_event(ip, xold, *x, self.prev_g[index], curr_g[index], index),

                None => {
                    let mut ys = [0.0; 4];
                    ys.copy_from_slice(y);
                    (*x, ys)
                }
            };

            let replace = match earliest {
                Some((_, previous_te, _)) => te < previous_te,
                None => true,
            };

            if replace {
                earliest = Some((index, te, ye));
            }
        }

        if let Some((index, te, ye)) = earliest {
            self.update(&ye);

            self.last_t = te;
            self.last_y = ye;

            self.event = Some(((index + 1) as u8, te, ye));

            return ControlFlag::Interrupt;
        }

        self.prev_g = curr_g;

        self.last_t = *x;
        self.last_y.copy_from_slice(y);

        self.update(y);

        ControlFlag::Continue
    }
}

pub fn solve_batch(xs: &[f64], vxs: &[f64], c: f64, t_max: f64) -> Vec<Solution> {
    xs.par_iter()
        .zip(vxs.par_iter())
        .map(|(&x, &vx)| solve_single(x, vx, c, t_max, MAX_STEP, RTOL, ATOL))
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
    pb: &ProgressBar,
) -> Vec<GridRow> {
    let xs = linspace(x_bounds.0, x_bounds.1, res_x);
    let vxs = linspace(vx_bounds.0, vx_bounds.1, res_vx);
    let total = res_x * res_vx;

    (0..total)
        .into_par_iter()
        .map(|idx| {
            let i = idx / res_x;
            let j = idx % res_x;

            let x0 = xs[j];
            let vx0 = vxs[i];

            let s = solve_single(
                x0,
                vx0,
                c,
                t_max,
                MAX_STEP,
                RTOL,
                ATOL,
            );

            pb.inc(1);

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
    let mut buf = Vec::with_capacity(rows.len() * 300);
    writeln!(
        buf,
        "x0,vx0,C,label_id,outcome,t_event,n_steps,min_r_earth,min_r_moon,max_r,jacobi_initial,jacobi_final,jacobi_error"
    )?;
    for r in rows {
        writeln!(
            buf,
            "{:.17e},{:.17e},{:.17e},{},{},{:.17e},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
            r.x0,
            r.vx0,
            r.c,
            r.label_id,
            OUTCOMES[r.label_id as usize],
            r.t_event,
            r.n_steps,
            r.min_r_earth,
            r.min_r_moon,
            r.max_r,
            r.jacobi_initial,
            r.jacobi_final,
            r.jacobi_error
        )?;
    }
    fs::write(path, buf)
}

fn read_basin_csv(path: &str) -> std::io::Result<Vec<GridRow>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    let mut rows = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;

        if i == 0 {
            continue;
        }

        let p: Vec<&str> = line.split(',').collect();

        if p.len() != 13 {
            continue;
        }

        rows.push(GridRow {
            x0: p[0].parse().unwrap(),
            vx0: p[1].parse().unwrap(),
            c: p[2].parse().unwrap(),
            label_id: p[3].parse().unwrap(),
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
    let unresolved: Vec<&GridRow> = df.iter().filter(|r| r.label_id == 4).collect();

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

    let convergence_times = [100.0, 300.0, 1000.0, 3000.0, 6000.0, 9000.0, 12000.0];

    let mut sols_12000: Option<Vec<Solution>> = None;

    for &t_max in &convergence_times {
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
                100.0 * count as f64 / total as f64
            );
        }

        if t_max == 12000.0 {
            sols_12000 = Some(sols);
        }
    }

    let sols = sols_12000.expect("12000.0 convergence result was not computed");
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
            r.x0, r.vx0, "MAX_TIME", r.t_event, r.n_steps, r.min_r_earth, r.min_r_moon, r.max_r
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

pub fn find_optimal_t_max(
    c_values: &[f64],
    absolute_max_t: f64,
    n_samples: usize,
    target_percentile: f64,
) {
    println!(
        "\nt_max Across {} C Values (Target: {:.1}%)",
        c_values.len(),
        target_percentile * 100.0
    );
    println!("Absolute max integration time: {:.1}", absolute_max_t);
    println!("Samples per C: {}", n_samples);

    let mut rng = Lcg::new(42);
    let x0: Vec<f64> = (0..n_samples).map(|_| rng.uniform(-1.5, 1.5)).collect();
    let vx0: Vec<f64> = (0..n_samples).map(|_| rng.uniform(-3.0, 3.0)).collect();

    for &c in c_values {
        print!("  C={:.4} ... ", c);

        let sols = solve_batch(&x0, &vx0, c, absolute_max_t);

        let mut t_events: Vec<f64> = sols
            .iter()
            .filter(|s| s.label == 1 || s.label == 2 || s.label == 3)
            .map(|s| s.t_event)
            .collect();

        let valid_count = t_events.len();
        let unresolved_count = sols.iter().filter(|s| s.label == 4).count();

        let optimal_t = if t_events.is_empty() {
            f64::INFINITY
        } else {
            t_events.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p_idx = ((valid_count as f64) * target_percentile) as usize;
            let p_idx = p_idx.clamp(0, valid_count.saturating_sub(1));
            t_events[p_idx]
        };

        println!(
            "valid={:>5}, unresolved={:>5}, t_{:.0}%={:>10.2}",
            valid_count,
            unresolved_count,
            target_percentile * 100.0,
            optimal_t
        );
    }
}

pub fn run_batch_generation(
    t_val: f64,
    c_values: &[f64],
    grid_resolutions: &[usize],
) {
    let multi = MultiProgress::new();

    let total_rows: u64 = c_values
        .iter()
        .map(|_| {
            grid_resolutions
                .iter()
                .map(|&res| (res * res) as u64)
                .sum::<u64>()
        })
        .sum();

    let main_pb = multi.add(ProgressBar::new(total_rows));

    main_pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "[Batch] [{bar:40.cyan/blue}] \
                 {pos}/{len} rows ({percent}%) {msg}"
            )
            .unwrap(),
    );

    fs::create_dir_all("files/data").unwrap();

    for &res in grid_resolutions {
        for &c in c_values {
            main_pb.set_message(format!("C={:.2} {}x{}", c, res, res));

            let df = generate_basin_grid(
                c,
                t_val,
                res,
                res,
                (-1.5, 1.5),
                (-3.0, 3.0),
                &main_pb,
            );

            let dir = format!("files/data/C{:.2}", c);
            fs::create_dir_all(&dir).unwrap();

            let output_path = format!("{}/{}x{}.csv", dir, res, res);
            write_basin_csv(&output_path, &df).unwrap();

            println!("Saved: {}", output_path);
        }
    }

    main_pb.finish_with_message("All rows completed!");
}

#[inline]
fn jacobi_constant(x: f64, y: f64, vx: f64, vy: f64) -> f64 {
    let (r1, r2, _) = radii(x, y);

    let omega = 0.5 * (x * x + y * y) + (1.0 - MU) / r1 + MU / r2;

    2.0 * omega - (vx * vx + vy * vy)
}

pub fn diagnose_single(x0: f64, vx0: f64, c: f64, t_max: f64) {
    println!("\n=== SINGLE TRAJECTORY DIAGNOSTIC ===");
    println!("x0={:.10}, vx0={:.10}, C={}, t_max={}", x0, vx0, c, t_max);

    let sol = solve_single(x0, vx0, c, t_max, MAX_STEP, RTOL, ATOL);

    println!(
        "Label:       {} ({})",
        OUTCOMES[sol.label as usize], sol.label
    );
    println!("t_event:     {:.17e}", sol.t_event);
    println!("n_steps:     {}", sol.n_steps);
    println!("Jacobi init: {:.17e}", sol.jacobi_initial);
    println!("Jacobi final:{:.17e}", sol.jacobi_final);
    println!("Jacobi error:{:.2e}", sol.jacobi_error);

    println!("min r_earth: {:.17e}", sol.min_r1);
    println!("min r_moon:  {:.17e}", sol.min_r2);
    println!("max r:       {:.17e}", sol.max_rc);

    println!("====================================\n");
}

pub fn validate_max_step(x0: f64, vx0: f64, c: f64, t_max: f64, candidate_steps: &[f64]) {
    println!("\n=== MAX_STEP VALIDATION ===");
    println!("x0={:.10}, vx0={:.10}, C={}, t_max={}", x0, vx0, c, t_max);

    let ref_sol = solve_single(x0, vx0, c, t_max, 1e-4, RTOL, ATOL);
    println!(
        "Reference (h=1e-4): label={}, t_event={:.15e}, jacobi_err={:.2e}",
        OUTCOMES[ref_sol.label as usize], ref_sol.t_event, ref_sol.jacobi_error
    );

    println!(
        "\n{:>12} {:>8} {:>15} {:>15} {:>12} {:>12}",
        "MAX_STEP", "Label", "t_event", "Δt vs ref", "Jacobi Err", "Agree?"
    );

    for &h in candidate_steps {
        let sol = solve_single(x0, vx0, c, t_max, h, RTOL, ATOL);
        let dt_diff = (sol.t_event - ref_sol.t_event).abs();
        let agree = sol.label == ref_sol.label;

        println!(
            "{:>12.1e} {:>8} {:>15.10e} {:>15.2e} {:>12.2e} {:>12}",
            h,
            OUTCOMES[sol.label as usize],
            sol.t_event,
            dt_diff,
            sol.jacobi_error,
            if agree { "✓" } else { "✗ MISMATCH" }
        );
    }
}

fn solve_single(
    x0: f64,
    vx0: f64,
    jacobi_c: f64,
    t_max: f64,
    max_step: f64,
    rtol: f64,
    atol: f64,
) -> Solution {
    let (r1_0, r2_0, rc_0) = radii(x0, 0.0);

    let omega0 = 0.5 * x0 * x0 + (1.0 - MU) / r1_0 + MU / r2_0;

    let vy2 = 2.0 * omega0 - jacobi_c - vx0 * vx0;

    if vy2 < 0.0 {
        return Solution {
            label: 0,
            t_event: 0.0,
            n_steps: 0,
            min_r1: r1_0,
            min_r2: r2_0,
            max_rc: rc_0,
            jacobi_initial: jacobi_c,
            jacobi_final: jacobi_c,
            jacobi_error: 0.0,
        };
    }

    let vy0 = vy2.sqrt();
    let y0 = [x0, 0.0, vx0, vy0];

    if r1_0 <= R_EARTH || r2_0 <= R_MOON || rc_0 >= R_ESCAPE {
        let label = classify(&y0);

        return Solution {
            label,
            t_event: 0.0,
            n_steps: 0,
            min_r1: r1_0,
            min_r2: r2_0,
            max_rc: rc_0,
            jacobi_initial: jacobi_c,
            jacobi_final: jacobi_c,
            jacobi_error: 0.0,
        };
    }

    let max_steps = (((t_max / max_step).ceil() as usize) * 2).clamp(10_000, 50_000_000);

    let system = Cr3bpSystem;

    let solver = DOP853 {
        max_step: Some(max_step),
        max_steps,
        dense_output: true,
        ..DOP853::default()
    };

    let mut out = BatchSolout::new(&y0);

    let result = solver
        .solve(
            &system,
            0.0,
            &y0,
            t_max,
            rtol.into(),
            atol.into(),
            Some(&mut out),
        )
        .expect("DOP853 integration failed");

    let (label, t_event, final_state) = match out.event {
        Some((label, te, ys)) => (label, te, ys),

        None => (4, out.last_t, out.last_y),
    };

    let jacobi_final = jacobi_constant(
        final_state[0],
        final_state[1],
        final_state[2],
        final_state[3],
    );

    Solution {
        label,
        t_event,
        n_steps: result.steps.accepted as u64,

        min_r1: out.min_r1,
        min_r2: out.min_r2,
        max_rc: out.max_rc,

        jacobi_initial: jacobi_c,
        jacobi_final,

        jacobi_error: (jacobi_final - jacobi_c).abs(),
    }
}

pub fn validate_tolerances(x0: f64, vx0: f64, c: f64, t_max: f64) {
    println!("\n=== TOLERANCE VALIDATION (MAX_STEP=0.01) ===");
    println!("x0={:.10}, vx0={:.10}, C={}", x0, vx0, c);

    let ref_sol = solve_single(x0, vx0, c, t_max, 0.01, 1e-11, 1e-13);
    let new_sol = solve_single(x0, vx0, c, t_max, 0.01, 1e-9, 1e-11);

    println!(
        "Old (1e-11/1e-13): label={}, t_event={:.15e}, jacobi_err={:.2e}, n_steps={}",
        OUTCOMES[ref_sol.label as usize], ref_sol.t_event, ref_sol.jacobi_error, ref_sol.n_steps
    );
    println!(
        "New (1e-9/1e-11):  label={}, t_event={:.15e}, jacobi_err={:.2e}, n_steps={}",
        OUTCOMES[new_sol.label as usize], new_sol.t_event, new_sol.jacobi_error, new_sol.n_steps
    );

    let agree = ref_sol.label == new_sol.label;
    let speedup = ref_sol.n_steps as f64 / new_sol.n_steps.max(1) as f64;

    println!(
        "Label match: {} | Speedup: {:.2}× | Δt: {:.2e} | ΔJacobi: {:.2e}",
        if agree { "✓" } else { "✗ MISMATCH" },
        speedup,
        (new_sol.t_event - ref_sol.t_event).abs(),
        (new_sol.jacobi_error - ref_sol.jacobi_error).abs()
    );
}

fn main() {
    println!("Rayon threads: {}", rayon::current_num_threads());
    // Finding Best t
    // let c_values = [3.00, 3.05, 3.10, 3.14, 3.20];
    // find_optimal_t_max(&c_values, 25000.0, 1000, 0.99);
    const T_OPTIM: f64 = 8100.0;

    // // Finding highest max_step
    // let test_cases = [
    //     (0.065217, -2.177258, 3.05),
    //     (0.5, -1.0, 3.14),
    //     (-0.8, 0.5, 3.00),
    //     (1.2, -2.5, 3.20),
    // ];

    // // let candidate_steps = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

    // // for &(x0, vx0, c) in &test_cases {
    // //     validate_max_step(x0, vx0, c, 8100.0, &candidate_steps);
    // // }

    // // Finding ATOL and RTOl values
    // for &(x0, vx0, c) in &test_cases {
    //     validate_tolerances(x0, vx0, c, 8100.0);
    // }

    // Running Test Batch
    run_batch_generation(T_OPTIM, &[3.05, 3.20], &[500]);
    analyze_convergence("files/data/C3.05/500x500.csv", 200);
    analyze_convergence("files/data/C3.20/500x500.csv", 200);

    // Final Run
    // run_batch_generation(t_optim);
}
