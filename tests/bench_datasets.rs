//! Dataset performance benchmarks for the SBPL fitter.
//!
//! Loads all ZTF photometry CSVs from a data directory and evaluates:
//!   • Fit success rate (n_ok / n_total)
//!   • Reduced chi² distribution (mean, median, std, p90)
//!   • Per-source timing and overall throughput
//!   • CPU thread-count scaling (sequential vs Rayon par_iter)
//!
//! Run:
//!   cargo test --release --test bench_datasets -- --ignored --nocapture
//!
//! Individual tests:
//!   cargo test --release --test bench_datasets sbpl_dataset_quality   -- --ignored --nocapture
//!   cargo test --release --test bench_datasets sbpl_dataset_throughput -- --ignored --nocapture
//!   cargo test --release --test bench_datasets sbpl_thread_scaling     -- --ignored --nocapture

use rayon::prelude::*;
use sbpl_pso::{find_csv_files, fit_sbpl_csv, PsoConfig};
use std::time::Instant;

// Default data directory (relative to sbpl-pso crate root).
// Override by setting SBPL_DATA_DIR environment variable.
const DEFAULT_DATA_DIR: &str = "../lc_fitting_comparison/data/photometry";

fn data_dir() -> String {
    std::env::var("SBPL_DATA_DIR").unwrap_or_else(|_| DEFAULT_DATA_DIR.to_string())
}

// ---------------------------------------------------------------------------
// Statistics helpers
// ---------------------------------------------------------------------------

fn percentile(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { f64::NAN } else { v.iter().sum::<f64>() / v.len() as f64 }
}

fn std_dev(v: &[f64]) -> f64 {
    if v.len() < 2 { return f64::NAN; }
    let m = mean(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Fit quality on the full dataset (sequential, for chi² statistics)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn sbpl_dataset_quality() {
    let dir = data_dir();
    let csvs = find_csv_files(&dir);

    if csvs.is_empty() {
        eprintln!("No CSV files found in {dir}");
        eprintln!("Set SBPL_DATA_DIR or copy photometry CSVs to {DEFAULT_DATA_DIR}");
        return;
    }

    eprintln!("\n{:=<80}", "");
    eprintln!("SBPL fit quality on {} sources from {}", csvs.len(), dir);
    eprintln!("{:=<80}", "");
    eprintln!(
        "{:<20}  {:>6}  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "source", "n_obs", "n_bands", "chi2", "alpha1", "alpha2", "beta", "tb_days"
    );
    eprintln!("{:-<80}", "");

    let config = PsoConfig::default();
    let mut chi2_vals = Vec::new();
    let mut n_ok = 0usize;
    let mut n_fail = 0usize;
    let t_start = Instant::now();

    for csv in &csvs {
        let source_name = std::path::Path::new(csv)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| csv.clone());

        match fit_sbpl_csv(csv, &config) {
            Ok(r) => {
                let chi2 = r.reduced_chi2.unwrap_or(f64::NAN);
                if chi2.is_finite() {
                    chi2_vals.push(chi2);
                    n_ok += 1;
                } else {
                    n_fail += 1;
                }
                eprintln!(
                    "{:<20}  {:>6}  {:>6}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>8.2}",
                    &source_name[..source_name.len().min(20)],
                    r.n_obs, r.n_bands, chi2,
                    r.alpha1.unwrap_or(f64::NAN),
                    r.alpha2.unwrap_or(f64::NAN),
                    r.beta.unwrap_or(f64::NAN),
                    r.tb.unwrap_or(f64::NAN),
                );
            }
            Err(e) => {
                n_fail += 1;
                eprintln!("{:<20}  FAIL: {}", &source_name[..source_name.len().min(20)], e);
            }
        }
    }

    let elapsed = t_start.elapsed();
    let n_total = csvs.len();

    eprintln!("\n{:=<80}", "");
    eprintln!("Chi² distribution ({} successful fits):", n_ok);
    eprintln!("  mean   = {:.4}", mean(&chi2_vals));
    eprintln!("  median = {:.4}", percentile(&chi2_vals, 50.0));
    eprintln!("  std    = {:.4}", std_dev(&chi2_vals));
    eprintln!("  p10    = {:.4}", percentile(&chi2_vals, 10.0));
    eprintln!("  p90    = {:.4}", percentile(&chi2_vals, 90.0));
    eprintln!("  min    = {:.4}", percentile(&chi2_vals, 0.0));
    eprintln!("  max    = {:.4}", percentile(&chi2_vals, 100.0));
    eprintln!();
    eprintln!("Results: {n_ok}/{n_total} sources fitted successfully ({n_fail} failed/skipped)");
    eprintln!(
        "Timing:  {:.2}s total, {:.1} ms/source, {:.0} sources/sec",
        elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1000.0 / n_total as f64,
        n_total as f64 / elapsed.as_secs_f64(),
    );
    eprintln!("{:=<80}", "");

    assert!(n_ok > 0, "No sources fitted successfully — check data path");
}

// ---------------------------------------------------------------------------
// Throughput: sequential vs parallel at the default PSO config
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn sbpl_dataset_throughput() {
    let dir = data_dir();
    let csvs = find_csv_files(&dir);

    if csvs.is_empty() {
        eprintln!("No CSV files found in {dir}");
        return;
    }

    let config = PsoConfig::default();

    eprintln!("\n{:=<80}", "");
    eprintln!("SBPL throughput: sequential vs parallel ({} sources)", csvs.len());
    eprintln!("{:=<80}", "");

    // Warmup
    let _ = fit_sbpl_csv(&csvs[0], &config);

    // Sequential
    let t0 = Instant::now();
    let seq_results: Vec<_> = csvs.iter()
        .map(|p| fit_sbpl_csv(p, &config).ok())
        .collect();
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let n_ok_seq = seq_results.iter().filter(|r| r.as_ref().map_or(false, |r| r.reduced_chi2.is_some())).count();

    // Parallel (rayon par_iter, default thread pool)
    let t0 = Instant::now();
    let par_results: Vec<_> = csvs.par_iter()
        .map(|p| fit_sbpl_csv(p, &config).ok())
        .collect();
    let par_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let n_ok_par = par_results.iter().filter(|r| r.as_ref().map_or(false, |r| r.reduced_chi2.is_some())).count();

    let n = csvs.len();

    eprintln!(
        "{:<12}  {:>8}  {:>10}  {:>12}  {:>12}  {:>8}",
        "backend", "n_src", "n_ok", "total_ms", "ms/source", "src/sec"
    );
    eprintln!("{:-<72}", "");
    eprintln!(
        "{:<12}  {:>8}  {:>10}  {:>12.1}  {:>12.2}  {:>8.0}",
        "CPU-seq", n, n_ok_seq, seq_ms, seq_ms / n as f64, n as f64 / (seq_ms / 1000.0)
    );
    eprintln!(
        "{:<12}  {:>8}  {:>10}  {:>12.1}  {:>12.2}  {:>8.0}",
        "CPU-par", n, n_ok_par, par_ms, par_ms / n as f64, n as f64 / (par_ms / 1000.0)
    );
    eprintln!(
        "\nSpeedup (par / seq): {:.1}x",
        seq_ms / par_ms.max(1.0)
    );
    eprintln!("{:=<80}", "");
}

// ---------------------------------------------------------------------------
// Thread-count scaling (par_iter across 1, 2, 4, 8 threads)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn sbpl_thread_scaling() {
    let dir = data_dir();
    let csvs = find_csv_files(&dir);

    if csvs.is_empty() {
        eprintln!("No CSV files found in {dir}");
        return;
    }

    let config = PsoConfig::default();
    let n = csvs.len();
    let thread_counts = [1, 2, 4, 8];

    eprintln!("\n{:=<80}", "");
    eprintln!("SBPL thread-count scaling ({} sources)", n);
    eprintln!("{:=<80}", "");
    eprintln!(
        "{:>9}  {:>12}  {:>10}  {:>10}  {:>10}  {:>8}  {:>10}  {:>10}",
        "threads", "total_srcs", "total_ms", "ms/src", "speedup",
        "n_ok", "chi2_mean", "chi2_med",
    );
    eprintln!("{}", "-".repeat(95));

    let mut baseline_ms = 0.0f64;

    // Warmup
    let _ = fit_sbpl_csv(&csvs[0], &config);

    for &n_threads in &thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .expect("Failed to build thread pool");

        let t0 = Instant::now();
        let results: Vec<Option<f64>> = pool.install(|| {
            csvs.par_iter()
                .map(|p| fit_sbpl_csv(p, &config).ok().and_then(|r| r.reduced_chi2))
                .collect()
        });
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let chi2_vals: Vec<f64> = results.iter().filter_map(|v| *v).collect();
        let n_ok = chi2_vals.len();

        if n_threads == thread_counts[0] {
            baseline_ms = total_ms;
        }
        let speedup = baseline_ms / total_ms;

        eprintln!(
            "{:>9}  {:>12}  {:>10.1}  {:>10.2}  {:>10.2}x  {:>8}  {:>10.4}  {:>10.4}",
            n_threads, n, total_ms, total_ms / n as f64, speedup,
            n_ok, mean(&chi2_vals), percentile(&chi2_vals, 50.0),
        );
    }

    eprintln!("{:=<80}", "");
}

// ---------------------------------------------------------------------------
// PSO config sensitivity: compare fewer vs more iterations
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn sbpl_pso_config_sensitivity() {
    let dir = data_dir();
    let csvs = find_csv_files(&dir);

    if csvs.is_empty() {
        eprintln!("No CSV files found in {dir}");
        return;
    }

    // Use a subset of 20 sources for the sensitivity study
    let subset: Vec<_> = csvs.iter().take(20).collect();

    let configs: &[(&str, PsoConfig)] = &[
        ("fast  (10p×50i×1r)", PsoConfig { n_particles: 10, n_iters: 50,  n_restarts: 1, seed: 42, ..Default::default() }),
        ("default(30p×200i×3r)", PsoConfig::default()),
        ("thorough(60p×400i×5r)", PsoConfig { n_particles: 60, n_iters: 400, n_restarts: 5, seed: 42, ..Default::default() }),
    ];

    eprintln!("\n{:=<80}", "");
    eprintln!("SBPL PSO config sensitivity ({} sources)", subset.len());
    eprintln!("{:=<80}", "");
    eprintln!(
        "{:<24}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "config", "total_ms", "ms/src", "chi2_mean", "chi2_med", "n_ok"
    );
    eprintln!("{:-<80}", "");

    for (label, cfg) in configs {
        let t0 = Instant::now();
        let results: Vec<Option<f64>> = subset.iter()
            .map(|p| fit_sbpl_csv(p, cfg).ok().and_then(|r| r.reduced_chi2))
            .collect();
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let chi2_vals: Vec<f64> = results.iter().filter_map(|v| *v).collect();
        let n_ok = chi2_vals.len();

        eprintln!(
            "{:<24}  {:>10.1}  {:>10.2}  {:>10.4}  {:>10.4}  {:>10}",
            label, total_ms, total_ms / subset.len() as f64,
            mean(&chi2_vals), percentile(&chi2_vals, 50.0), n_ok
        );
    }

    eprintln!("{:=<80}", "");
}
