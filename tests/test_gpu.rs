//! GPU backend tests. Compiled only with `--features cuda`; they need a real
//! NVIDIA device at runtime.
//!
//! Run:
//!   CUDA_HOME=/usr/local/cuda cargo test --release --features cuda --test test_gpu -- --nocapture
//!
//! There is no Metal equivalent here because Metal only builds on macOS; the
//! two backends share `gpu_common`, so what these tests pin down (batch
//! independence, CPU agreement, absolute-time round-trip) holds for both.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use sbpl_pso::gpu::{GpuBatchData, GpuContext, SourceData, Stream};
use sbpl_pso::{
    fit_prepared, prepare_source, sbpl_model, BandData, Preparation, PsoConfig, SbplResult,
};

const C_ANGSTROM_PER_SEC: f64 = 2.997_924_58e18;

// Simple xorshift64 PRNG — no extra crate dependencies in tests.
fn xorshift(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / ((1u64 << 53) as f64)
}

fn normal(rng: &mut u64) -> f64 {
    let u1 = xorshift(rng).max(1e-15);
    let u2 = xorshift(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generate a 3-band (g, r, i) synthetic SBPL source in flux space, on a
/// JD-scale time axis so the `t_ref` shift is actually exercised.
fn synthetic_source(
    n_per_band: usize,
    seed: u64,
    alpha1: f64,
    alpha2: f64,
    beta: f64,
    logd: f64,
    loga: f64,
    tb: f64,
    t0: f64,
) -> HashMap<String, BandData> {
    let mut rng = seed.max(1);
    let bands: &[(&str, f64)] = &[("g", 4770.0), ("r", 6231.0), ("i", 7625.0)];
    let t_start = t0 + 0.5;
    let t_end = t0 + tb * 10.0;
    let noise_frac = 0.05;

    let mut result: HashMap<String, BandData> = HashMap::new();
    for &(name, lambda) in bands {
        let nu_scaled = (C_ANGSTROM_PER_SEC / lambda) / 1e15;
        let mut bd = BandData::new();
        for i in 0..n_per_band {
            let frac = i as f64 / (n_per_band - 1).max(1) as f64;
            let t = t_start + frac * (t_end - t_start);
            let f = sbpl_model(t, nu_scaled, alpha1, alpha2, beta, logd, loga, tb, t0);
            if !f.is_finite() || f <= 0.0 {
                continue;
            }
            let err = (f * noise_frac).max(1e-12);
            bd.times.push(t);
            bd.fluxes.push(f + err * normal(&mut rng));
            bd.flux_errs.push(err);
        }
        result.insert(name.to_string(), bd);
    }
    result
}

/// A small batch of distinct synthetic sources, each with a different shape.
fn make_batch(config: &PsoConfig) -> Vec<SourceData> {
    let specs: &[(f64, f64, f64, f64, f64, f64, f64)] = &[
        // alpha1, alpha2, beta, logd, loga, tb, t0
        (1.5, -2.0, -0.6, -0.5, 2.0, 12.0, 2_458_200.0),
        (0.8, -1.2, -0.4, -1.0, 1.5, 25.0, 2_458_310.0),
        (2.2, -3.0, -0.9, -0.3, 2.5, 6.0, 2_458_005.0),
        (1.0, -1.8, -0.2, -0.8, 1.0, 40.0, 2_458_440.0),
        (1.7, -2.4, -0.7, -0.6, 2.2, 18.0, 2_458_150.0),
    ];

    specs
        .iter()
        .enumerate()
        .map(|(i, &(a1, a2, b, ld, la, tb, t0))| {
            let bands = synthetic_source(40, 1000 + i as u64, a1, a2, b, ld, la, tb, t0);
            match prepare_source(&bands, config) {
                Preparation::Ready(prepared) => SourceData {
                    name: format!("synthetic_{i}"),
                    prepared,
                },
                other => panic!("synthetic source {i} was not fittable: {other:?}"),
            }
        })
        .collect()
}

fn small_config() -> PsoConfig {
    PsoConfig {
        n_particles: 24,
        n_iters: 60,
        n_restarts: 2,
        ..PsoConfig::default()
    }
}

fn fit_on_gpu(sources: &[SourceData], config: &PsoConfig) -> Vec<SbplResult> {
    let stream = Stream::new_on_device(0).expect("create CUDA stream");
    let gpu = GpuContext::new(0, stream.as_ptr()).expect("init GPU 0");
    let batch = GpuBatchData::new(&gpu, sources).expect("upload batch");
    gpu.batch_fit(&batch, config).expect("batch fit")
}

#[test]
fn gpu_batch_fit_returns_one_finite_result_per_source() {
    let config = small_config();
    let sources = make_batch(&config);
    let results = fit_on_gpu(&sources, &config);

    assert_eq!(results.len(), sources.len());
    for (r, s) in results.iter().zip(sources.iter()) {
        let chi2 = r.reduced_chi2.expect("reduced_chi2 must be set");
        assert!(chi2.is_finite() && chi2 >= 0.0, "bad chi2 {chi2}");
        assert_eq!(r.n_obs, s.prepared.n_obs);
        assert_eq!(r.n_bands, s.prepared.n_bands);
        for name in ["alpha1", "alpha2", "beta", "logd", "loga", "tb", "t0"] {
            let v = match name {
                "alpha1" => r.alpha1,
                "alpha2" => r.alpha2,
                "beta" => r.beta,
                "logd" => r.logd,
                "loga" => r.loga,
                "tb" => r.tb,
                _ => r.t0,
            };
            assert!(v.is_some_and(f64::is_finite), "{name} is not finite");
        }
    }
}

/// `t0` must come back on the original (JD-scale) time axis, not in the shifted
/// `t − t_ref` space the kernel works in.
#[test]
fn gpu_t0_is_returned_in_absolute_time() {
    let config = small_config();
    let sources = make_batch(&config);
    let results = fit_on_gpu(&sources, &config);

    for (r, s) in results.iter().zip(sources.iter()) {
        let t0 = r.t0.unwrap();
        let lower = s.prepared.lower[6];
        let upper = s.prepared.upper[6];
        assert!(
            t0 >= lower - 1e-6 && t0 <= upper + 1e-6,
            "t0 {t0} outside absolute-time bounds [{lower}, {upper}] — \
             looks like the t_ref shift was not undone"
        );
    }
}

/// A source's fit must not depend on which other sources share its batch:
/// every source gets its own RNG stream keyed by index, and the kernel reads
/// only that source's observation slice.
#[test]
fn gpu_results_are_independent_of_batch_composition() {
    let config = small_config();
    let sources = make_batch(&config);
    let full = fit_on_gpu(&sources, &config);

    // Refit source 2 on its own; it should land on exactly the same answer as
    // it did inside the full batch.
    let solo_src = vec![SourceData {
        name: sources[2].name.clone(),
        prepared: sources[2].prepared.clone(),
    }];
    let solo = fit_on_gpu(&solo_src, &config);

    assert_eq!(solo.len(), 1);
    let (a, b) = (&full[2], &solo[0]);
    assert_eq!(
        a.reduced_chi2, b.reduced_chi2,
        "source 2 fit differently in a batch of {} than alone",
        sources.len()
    );
    assert_eq!(a.tb, b.tb);
    assert_eq!(a.t0, b.t0);
}

/// The GPU should land in the same basin as the CPU on clean synthetic data:
/// both minimise the identical objective, so neither should be systematically
/// worse.
#[test]
fn gpu_and_cpu_agree_on_synthetic_sources() {
    let config = small_config();
    let sources = make_batch(&config);

    let gpu = fit_on_gpu(&sources, &config);
    let cpu: Vec<SbplResult> = sources
        .iter()
        .map(|s| fit_prepared(&s.prepared, &config))
        .collect();

    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let gv = g.reduced_chi2.unwrap();
        let cv = c.reduced_chi2.unwrap();
        // Noise is 5% of flux, so a good fit sits near chi2 ~ 1. Allow a
        // generous factor either way: PSO is stochastic and the two paths use
        // different RNG streams.
        assert!(
            gv < 10.0 * cv.max(1.0),
            "source {i}: gpu chi2 {gv:.3} is far worse than cpu chi2 {cv:.3}"
        );
    }

    let gpu_med = median(gpu.iter().filter_map(|r| r.reduced_chi2).collect());
    let cpu_med = median(cpu.iter().filter_map(|r| r.reduced_chi2).collect());
    assert!(
        gpu_med < 5.0 * cpu_med.max(1.0),
        "gpu median chi2 {gpu_med:.3} vs cpu {cpu_med:.3}"
    );
}

fn median(mut v: Vec<f64>) -> f64 {
    assert!(!v.is_empty());
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}
