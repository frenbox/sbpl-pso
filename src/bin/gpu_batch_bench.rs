//! Single-batch CUDA benchmark for the SBPL fitter.
//!
//! Loads every CSV in `data_dir`, packs it into one `GpuBatchData`, runs a
//! single `batch_fit` call on GPU 0, and reports wall time, peak GPU memory and
//! the reduced-χ² distribution. Pass `--cpu-compare` to run the same sources
//! through the CPU fitter and print a side-by-side comparison.
//!
//! Usage:
//!   gpu-batch-bench [data_dir] [--n-particles N] [--n-iters N] [--n-restarts N]
//!                   [--cpu-compare]
//!
//! Default data_dir: `../lc_fitting_comparison/data/photometry` (same as
//! `cpu-dispatch-bench`). The CSVs must be in the long format `load_csv`
//! accepts: `jd`/`mjd`, `magpsf`/`mag`, `sigmapsf`/`mag_err`, and `fid` or
//! `filter`.

use std::ffi::c_int;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use sbpl_pso::gpu::{load_sources_with, GpuBatchData, GpuContext, Stream};
use sbpl_pso::{fit_prepared, PsoConfig, SbplResult};

extern "C" {
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
}

fn mem_info() -> (usize, usize) {
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        cudaMemGetInfo(&mut free, &mut total);
    }
    (free, total)
}

fn mb(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn parse_usize_flag(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .and_then(|w| w[1].parse().ok())
}

fn chi2_stats(results: &[SbplResult]) -> (usize, f64, f64) {
    let mut v: Vec<f64> = results.iter().filter_map(|r| r.reduced_chi2).collect();
    if v.is_empty() {
        return (0, f64::NAN, f64::NAN);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let med = v[v.len() / 2];
    (v.len(), mean, med)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let data_dir = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--") && a.parse::<usize>().is_err())
        .map(|s| s.as_str())
        .unwrap_or("../lc_fitting_comparison/data/photometry");
    let cpu_compare = args.iter().any(|a| a == "--cpu-compare");

    let mut config = PsoConfig::default();
    if let Some(n) = parse_usize_flag(&args, "--n-particles") {
        config.n_particles = n;
    }
    if let Some(n) = parse_usize_flag(&args, "--n-iters") {
        config.n_iters = n;
    }
    if let Some(n) = parse_usize_flag(&args, "--n-restarts") {
        config.n_restarts = n;
    }

    eprintln!("Loading sources from {}...", data_dir);
    let t = Instant::now();
    let sources = load_sources_with(data_dir, &config);
    eprintln!(
        "Loaded {} sources in {:.2}s",
        sources.len(),
        t.elapsed().as_secs_f64()
    );

    if sources.is_empty() {
        eprintln!("No fittable sources found in {}", data_dir);
        std::process::exit(1);
    }

    let obs_counts: Vec<usize> = sources.iter().map(|s| s.prepared.n_obs).collect();
    let total_obs: usize = obs_counts.iter().sum();
    eprintln!(
        "Observations: total={}, min={}, max={}, mean={:.1}",
        total_obs,
        obs_counts.iter().min().unwrap(),
        obs_counts.iter().max().unwrap(),
        total_obs as f64 / sources.len() as f64
    );

    let stream = Stream::new_on_device(0).expect("create CUDA stream");
    let gpu = GpuContext::new(0, stream.as_ptr()).expect("Failed to init GPU 0");

    let (free_baseline, total_mem) = mem_info();
    eprintln!("\nGPU 0 total memory: {:.1} MB", mb(total_mem));
    eprintln!(
        "Baseline free:      {:.1} MB (in use by others: {:.1} MB)",
        mb(free_baseline),
        mb(total_mem - free_baseline)
    );

    let t_up = Instant::now();
    let batch = GpuBatchData::new(&gpu, &sources).expect("GPU upload failed");
    let upload_ms = t_up.elapsed().as_secs_f64() * 1000.0;
    let (free_after_upload, _) = mem_info();
    let upload_bytes = free_baseline.saturating_sub(free_after_upload);
    eprintln!(
        "\nGpuBatchData::new uploaded in {:.1}ms, allocated {:.1} MB",
        upload_ms,
        mb(upload_bytes)
    );

    let stop = Arc::new(AtomicBool::new(false));
    let min_free = Arc::new(AtomicUsize::new(usize::MAX));
    let poll = {
        let stop = Arc::clone(&stop);
        let min_free = Arc::clone(&min_free);
        thread::spawn(move || {
            unsafe {
                cudaSetDevice(0);
            }
            while !stop.load(Ordering::Relaxed) {
                let (free, _) = mem_info();
                let mut cur = min_free.load(Ordering::Relaxed);
                while free < cur {
                    match min_free.compare_exchange_weak(
                        cur,
                        free,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(c) => cur = c,
                    }
                }
                thread::sleep(Duration::from_millis(2));
            }
        })
    };

    eprintln!(
        "\nPSO config: n_particles={}, n_iters={}, n_restarts={}",
        config.n_particles, config.n_iters, config.n_restarts
    );
    eprintln!("Running batch fit on {} sources...", sources.len());

    let t_pso = Instant::now();
    let results = gpu.batch_fit(&batch, &config).expect("GPU fit failed");
    let gpu_s = t_pso.elapsed().as_secs_f64();

    stop.store(true, Ordering::Relaxed);
    poll.join().ok();

    let min_free_v = min_free.load(Ordering::Relaxed);
    let peak_used_total = total_mem.saturating_sub(min_free_v);
    let peak_used_incremental = free_baseline.saturating_sub(min_free_v);

    let (n_ok, mean, med) = chi2_stats(&results);

    eprintln!("\n=========== GPU batch summary ===========");
    eprintln!("Sources processed:    {}", results.len());
    eprintln!("Wall time:            {:.2}s", gpu_s);
    eprintln!(
        "ms/source:            {:.2}",
        gpu_s * 1000.0 / results.len() as f64
    );
    eprintln!("Obs upload bytes:     {:.1} MB", mb(upload_bytes));
    eprintln!("Peak GPU used (proc): {:.1} MB", mb(peak_used_incremental));
    eprintln!(
        "Peak GPU used (tot):  {:.1} MB / {:.1} MB",
        mb(peak_used_total),
        mb(total_mem)
    );
    eprintln!(
        "Reduced chi2:         n={}  mean={:.3}  median={:.3}",
        n_ok, mean, med
    );

    if cpu_compare {
        eprintln!("\nRunning the same sources on the CPU (rayon)...");
        let t_cpu = Instant::now();
        let cpu_results: Vec<SbplResult> = sources
            .par_iter()
            .map(|s| fit_prepared(&s.prepared, &config))
            .collect();
        let cpu_s = t_cpu.elapsed().as_secs_f64();
        let (cpu_n, cpu_mean, cpu_med) = chi2_stats(&cpu_results);

        eprintln!("\n=========== CPU vs GPU ===========");
        eprintln!(
            "{:>8}  {:>10}  {:>12}  {:>12}  {:>12}",
            "backend", "wall_s", "ms/source", "chi2_mean", "chi2_median"
        );
        eprintln!("{}", "-".repeat(62));
        eprintln!(
            "{:>8}  {:>10.2}  {:>12.2}  {:>12.3}  {:>12.3}",
            "cpu",
            cpu_s,
            cpu_s * 1000.0 / cpu_n.max(1) as f64,
            cpu_mean,
            cpu_med
        );
        eprintln!(
            "{:>8}  {:>10.2}  {:>12.2}  {:>12.3}  {:>12.3}",
            "gpu",
            gpu_s,
            gpu_s * 1000.0 / n_ok.max(1) as f64,
            mean,
            med
        );
        eprintln!("\nSpeedup: {:.2}x", cpu_s / gpu_s);

        // Per-source agreement: how often does the GPU land within 1% of the
        // CPU's reduced chi2 (or better)?
        let mut n_better = 0usize;
        let mut n_close = 0usize;
        let mut n_worse = 0usize;
        for (g, c) in results.iter().zip(cpu_results.iter()) {
            match (g.reduced_chi2, c.reduced_chi2) {
                (Some(gv), Some(cv)) => {
                    if gv <= cv {
                        n_better += 1;
                    } else if gv <= cv * 1.01 {
                        n_close += 1;
                    } else {
                        n_worse += 1;
                    }
                }
                _ => n_worse += 1,
            }
        }
        eprintln!(
            "Per-source chi2: gpu better/equal={}, within 1%={}, worse={}",
            n_better, n_close, n_worse
        );
    }
}
