use differential_equations::derive::State;
use differential_equations::ivp::IVP;
use differential_equations::methods::ExplicitRungeKutta;
use differential_equations::ode::ODE;
use differential_equations::solout::{CrossingDirection, Event, EventConfig};
use differential_equations::status::Status;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::cell::RefCell;
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

const EVENT_TOL: f64 = 1e-8;

#[derive(State)]
pub struct StateVector<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub vx: T,
    pub vy: T,
    pub vz: T,
}

struct Cr3bpSystem;

const ONE_MINUS_MU: f64 = 1.0 - MU;

impl ODE<f64, StateVector<f64>> for Cr3bpSystem {
    #[inline(always)]
    #[inline(always)]
    fn diff(&self, _t: f64, sv: &StateVector<f64>, dsdt: &mut StateVector<f64>) {
        let dx1 = sv.x + MU;
        let dx2 = sv.x - ONE_MINUS_MU;
        let y2 = sv.y * sv.y;

        let r1_sq = dx1 * dx1 + y2;
        let r2_sq = dx2 * dx2 + y2;

        let r1_3_inv = 1.0 / (r1_sq * r1_sq.sqrt());
        let r2_3_inv = 1.0 / (r2_sq * r2_sq.sqrt());

        dsdt.x = sv.vx;
        dsdt.y = sv.vy;
        dsdt.z = 0.0;
        dsdt.vx = sv.x - ONE_MINUS_MU * dx1 * r1_3_inv - MU * dx2 * r1_3_inv + 2.0 * sv.vy;
        dsdt.vy = sv.y * (1.0 - ONE_MINUS_MU * r1_3_inv - MU * r2_3_inv) - 2.0 * sv.vx;
        dsdt.vz = 0.0;
    }
}

#[derive(Debug, Default)]
struct Tracker {
    min_r1: f64,
    min_r2: f64,
    max_rc: f64,
}

impl Tracker {
    #[inline]
    fn from_radii(r1: f64, r2: f64, rc: f64) -> Self {
        Self {
            min_r1: r1,
            min_r2: r2,
            max_rc: rc,
        }
    }

    #[inline]
    fn update_from_radii(&mut self, r1: f64, r2: f64, rc: f64) {
        if r1 < self.min_r1 {
            self.min_r1 = r1;
        }
        if r2 < self.min_r2 {
            self.min_r2 = r2;
        }
        if rc > self.max_rc {
            self.max_rc = rc;
        }
    }
}

struct Cr3bpEvent {
    tracker: RefCell<Tracker>,
}

impl Cr3bpEvent {
    fn from_radii(r1: f64, r2: f64, rc: f64) -> Self {
        Self {
            tracker: RefCell::new(Tracker::from_radii(r1, r2, rc)),
        }
    }
}

impl Event<f64, StateVector<f64>> for Cr3bpEvent {
    fn config(&self) -> EventConfig {
        EventConfig::new(CrossingDirection::Negative, Some(1))
    }

    fn event(&self, _t: f64, y: &StateVector<f64>) -> f64 {
        let (r1, r2, rc) = radii(y.x, y.y);
        self.tracker.borrow_mut().update_from_radii(r1, r2, rc);
        (r1 - R_EARTH).min(r2 - R_MOON).min(R_ESCAPE - rc)
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

#[inline]
fn classify_initial(sv: &StateVector<f64>) -> u8 {
    let (r1, r2, rc) = radii(sv.x, sv.y);

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
fn classify_event_state(sv: &StateVector<f64>) -> u8 {
    let (r1, r2, rc) = radii(sv.x, sv.y);

    if r1 <= R_EARTH + EVENT_TOL {
        return 1;
    }
    if r2 <= R_MOON + EVENT_TOL {
        return 2;
    }
    if rc >= R_ESCAPE - EVENT_TOL {
        return 3;
    }

    let d1 = (r1 - R_EARTH).abs();
    let d2 = (r2 - R_MOON).abs();
    let d3 = (rc - R_ESCAPE).abs();

    if d1 <= d2 && d1 <= d3 {
        1
    } else if d2 <= d3 {
        2
    } else {
        3
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

            let s = solve_single(x0, vx0, c, t_max, MAX_STEP, RTOL, ATOL);

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

pub fn run_batch_generation(t_val: f64, c_values: &[f64], grid_resolutions: &[usize]) {
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
            .template("[Batch] [{bar:40.cyan/blue}] {pos}/{len} ({percent}%) | ETA: {eta_precise} | {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏ "),
    );

    fs::create_dir_all("files/data").unwrap();

    for &res in grid_resolutions {
        for &c in c_values {
            main_pb.set_message(format!("C={:.2} {}x{}", c, res, res));

            let df = generate_basin_grid(c, t_val, res, res, (-1.5, 1.5), (-3.0, 3.0), &main_pb);

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

    let y0 = StateVector {
        x: x0,
        y: 0.0,
        z: 0.0,
        vx: vx0,
        vy: vy0,
        vz: 0.0,
    };

    if r1_0 <= R_EARTH || r2_0 <= R_MOON || rc_0 >= R_ESCAPE {
        let label = classify_initial(&y0);

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

    let y_fallback = StateVector {
        x: x0,
        y: 0.0,
        z: 0.0,
        vx: vx0,
        vy: vy0,
        vz: 0.0,
    };

    let max_steps =
        (((t_max / max_step).ceil() as u64).saturating_mul(2) + 1_000).min(4_000_000_000) as usize;

    let system = Cr3bpSystem;
    let event = Cr3bpEvent::from_radii(r1_0, r2_0, rc_0);

    let result = IVP::ode(&system, 0.0, t_max, y0)
        .t_eval(std::slice::from_ref(&t_max))
        .event(&event)
        .method(
            ExplicitRungeKutta::dop853()
                .h_max(max_step)
                .max_steps(max_steps)
                .rtol(rtol)
                .atol(atol),
        )
        .solve();

    match result {
        Ok(sol) => {
            let is_complete = matches!(sol.status, Status::Complete);

            let final_state = sol.y.last().unwrap_or(&y_fallback);

            let label = if is_complete {
                4
            } else {
                classify_event_state(final_state)
            };

            let t_event = if is_complete {
                t_max
            } else {
                sol.t.last().copied().unwrap_or(t_max)
            };

            let jacobi_final =
                jacobi_constant(final_state.x, final_state.y, final_state.vx, final_state.vy);

            let tracker = event.tracker.borrow();

            Solution {
                label,
                t_event,
                n_steps: sol.steps.accepted as u64,
                min_r1: tracker.min_r1,
                min_r2: tracker.min_r2,
                max_rc: tracker.max_rc,
                jacobi_initial: jacobi_c,
                jacobi_final,
                jacobi_error: (jacobi_final - jacobi_c).abs(),
            }
        }
        Err(e) => panic!("DOP853 integration failed: {:?}", e),
    }
}

fn main() {
    const T_OPTIM: f64 = 8100.0;

    let x0 = -0.5;
    let vx0 = 0.0;
    let c = 3.05;

    validate_tolerances(x0, vx0, c, T_OPTIM);

    validate_max_step(
        x0,
        vx0,
        c,
        T_OPTIM,
        &[1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 5e-1, 1.0],
    );

    // Test Batch
    // run_batch_generation(T_OPTIM, &[3.05, 3.20], &[500]);
    // analyze_convergence("files/data/C3.05/500x500.csv", 200);
    // analyze_convergence("files/data/C3.20/500x500.csv", 200);
}
