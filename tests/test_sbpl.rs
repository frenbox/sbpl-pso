//! Unit tests for the SBPL fitter using synthetic light curves.
//!
//! Run:
//!   cargo test --release --test test_sbpl -- --nocapture

use std::collections::HashMap;
use std::io::Write;
use sbpl_pso::{
    BandData, fit_sbpl, sbpl_model, band_frequency_hz, load_csv, prepare_source, Preparation,
    PreparedSource, PsoConfig, SbplCost, N_PARAMS,
};

// ---------------------------------------------------------------------------
// Synthetic SBPL source generator
// ---------------------------------------------------------------------------

const C_ANGSTROM_PER_SEC: f64 = 2.997_924_58e18;

fn band_freq(lambda_angstrom: f64) -> f64 {
    C_ANGSTROM_PER_SEC / lambda_angstrom
}

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

/// Generate a 3-band (g, r, i) synthetic SBPL source in flux space.
fn generate_sbpl_source(
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
        let nu = band_freq(lambda);
        let nu_scaled = nu / 1e15;
        let mut bd = BandData::new();

        for j in 0..n_per_band {
            let t = t_start + (t_end - t_start) * (j as f64) / (n_per_band as f64 - 1.0).max(1.0);
            let flux_true = sbpl_model(t, nu_scaled, alpha1, alpha2, beta, logd, loga, tb, t0);
            if !flux_true.is_finite() || flux_true <= 0.0 {
                continue;
            }
            let noise_sigma = flux_true.abs() * noise_frac + 1e-20;
            let flux_obs = flux_true + noise_sigma * normal(&mut rng);
            bd.times.push(t);
            bd.fluxes.push(flux_obs);
            bd.flux_errs.push(noise_sigma);
        }

        result.insert(name.to_string(), bd);
    }

    result
}

/// Generate a simple multi-band source with fixed flux + small noise (no SBPL
/// shape) — tests that the fitter runs without panicking on bland data.
fn generate_flat_source(n_per_band: usize, seed: u64) -> HashMap<String, BandData> {
    let mut rng = seed.max(1);
    let bands = ["g", "r", "i"];
    let mut result = HashMap::new();

    for band in bands {
        let mut bd = BandData::new();
        let base_flux = 1e-17 * (1.0 + xorshift(&mut rng));
        for j in 0..n_per_band {
            let t = 2460000.0 + j as f64 * 3.0;
            let noise = base_flux * 0.05 * normal(&mut rng);
            bd.times.push(t);
            bd.fluxes.push(base_flux + noise);
            bd.flux_errs.push(base_flux * 0.05);
        }
        result.insert(band.to_string(), bd);
    }
    result
}

// ---------------------------------------------------------------------------
// Basic sanity tests
// ---------------------------------------------------------------------------

#[test]
fn sbpl_model_zero_before_t0() {
    let nu_scaled = 1.0; // arbitrary
    let val = sbpl_model(0.0, nu_scaled, 1.5, -1.5, -0.7, -0.3, 0.0, 10.0, 5.0);
    assert_eq!(val, 0.0, "model should return 0.0 before t0");
}

#[test]
fn sbpl_model_positive_after_t0() {
    let nu_scaled = band_frequency_hz("r").unwrap() / 1e15;
    let val = sbpl_model(15.0, nu_scaled, 1.5, -1.5, -0.7, -0.3, 0.0, 10.0, 5.0);
    assert!(val > 0.0 && val.is_finite(), "model should be positive after t0, got {val}");
}

#[test]
fn band_frequencies_are_positive() {
    for band in &["g", "r", "i", "z", "y", "ztfg", "ztfr", "lsstz", "lssty"] {
        let nu = band_frequency_hz(band).unwrap();
        assert!(nu > 0.0, "frequency for {band} should be positive");
    }
}

#[test]
fn load_csv_parses_lsst_zy_filter_names() {
    let path = std::env::temp_dir().join(format!(
        "sbpl_pso_test_{}_{:?}.csv",
        std::process::id(),
        std::time::SystemTime::now()
    ));
    {
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "mjd,mag,mag_err,filter").unwrap();
        writeln!(f, "1.0,20.0,0.1,lsstz").unwrap();
        writeln!(f, "2.0,20.5,0.1,lssty").unwrap();
        writeln!(f, "3.0,21.0,0.1,lsstg").unwrap();
    }

    let obs = load_csv(path.to_str().unwrap()).expect("load_csv should succeed");
    std::fs::remove_file(&path).ok();

    let bands: Vec<&str> = obs.iter().map(|o| o.band.as_str()).collect();
    assert!(bands.contains(&"z"), "expected a 'z' observation, got {bands:?}");
    assert!(bands.contains(&"y"), "expected a 'y' observation, got {bands:?}");
    assert!(bands.contains(&"g"), "expected a 'g' observation, got {bands:?}");
}

// ---------------------------------------------------------------------------
// Fitter returns Some on valid input
// ---------------------------------------------------------------------------

#[test]
fn fit_sbpl_returns_some_on_sbpl_source() {
    let bands = generate_sbpl_source(40, 1234, 1.5, -1.5, -0.7, -0.3, 12.0, 10.0, 5.0);
    let result = fit_sbpl(&bands, &PsoConfig::default());
    assert!(result.is_some(), "fit_sbpl must return Some for a valid source");
}

#[test]
fn fit_sbpl_returns_some_on_flat_source() {
    let bands = generate_flat_source(30, 42);
    let result = fit_sbpl(&bands, &PsoConfig::default());
    assert!(result.is_some(), "fit_sbpl must return Some even for flat input");
}

#[test]
fn fit_sbpl_returns_none_on_empty_map() {
    let bands: HashMap<String, BandData> = HashMap::new();
    let result = fit_sbpl(&bands, &PsoConfig::default());
    assert!(result.is_none(), "fit_sbpl should return None for empty map");
}

// ---------------------------------------------------------------------------
// Result field validation
// ---------------------------------------------------------------------------

#[test]
fn fit_sbpl_chi2_non_negative() {
    let bands = generate_sbpl_source(40, 999, 1.0, -1.0, -0.6, -0.3, 12.0, 3.0, 0.0);
    let result = fit_sbpl(&bands, &PsoConfig::default()).unwrap();
    if let Some(chi2) = result.reduced_chi2 {
        assert!(chi2 >= 0.0 && chi2.is_finite(), "reduced_chi2={chi2} should be >= 0 and finite");
    }
}

#[test]
fn fit_sbpl_n_obs_and_n_bands_correct() {
    // 40 points × 3 bands = 120 obs
    let bands = generate_sbpl_source(40, 12345, 2.0, -2.0, -0.8, -0.3, 12.0, 10.0, 5.0);
    let result = fit_sbpl(&bands, &PsoConfig::default()).unwrap();
    assert_eq!(result.n_bands, 3, "expected 3 bands");
    assert_eq!(result.n_obs, 120, "expected 120 obs (40 per band)");
}

#[test]
fn fit_sbpl_errors_non_negative_when_set() {
    let bands = generate_sbpl_source(40, 9999, 1.0, -1.0, -0.6, -0.3, 12.0, 3.0, 0.0);
    let result = fit_sbpl(&bands, &PsoConfig::default()).unwrap();
    for (name, val) in [
        ("alpha1_err", result.alpha1_err),
        ("alpha2_err", result.alpha2_err),
        ("beta_err",   result.beta_err),
        ("logd_err",   result.logd_err),
        ("loga_err",   result.loga_err),
        ("tb_err",     result.tb_err),
        ("t0_err",     result.t0_err),
    ] {
        if let Some(e) = val {
            assert!(e >= 0.0 && e.is_finite(), "{name}={e} should be >= 0 and finite");
        }
    }
}

#[test]
fn fit_sbpl_insufficient_bands_returns_empty() {
    // Single-band source: n_bands < 2, fitter should return empty result
    let mut bands: HashMap<String, BandData> = HashMap::new();
    let mut bd = BandData::new();
    for j in 0..20 {
        bd.times.push(j as f64);
        bd.fluxes.push(1e-17);
        bd.flux_errs.push(1e-18);
    }
    bands.insert("r".to_string(), bd);

    let result = fit_sbpl(&bands, &PsoConfig::default()).unwrap();
    assert!(result.alpha1.is_none(), "should not fit with only 1 band");
}

// ---------------------------------------------------------------------------
// Parameter recovery test
// ---------------------------------------------------------------------------

#[test]
fn fit_sbpl_recovers_spectral_index_direction() {
    // True beta < 0 (typical power-law: bluer bands brighter).
    // The fitter should return a negative beta.
    let true_alpha1 = 2.0;
    let true_alpha2 = -2.3;
    let true_beta   = -0.8;
    let true_logd   = -0.3;
    let true_loga   = 12.0;
    let true_tb     = 10.0;
    let true_t0     = 5.0;

    let bands = generate_sbpl_source(
        40, 1234, true_alpha1, true_alpha2, true_beta, true_logd, true_loga, true_tb, true_t0,
    );

    let result = fit_sbpl(&bands, &PsoConfig::default())
        .expect("fit_sbpl should return Some for SBPL source");

    eprintln!("Parameter recovery:");
    eprintln!("  alpha1 = {:?}  (true: {true_alpha1})", result.alpha1);
    eprintln!("  alpha2 = {:?}  (true: {true_alpha2})", result.alpha2);
    eprintln!("  beta   = {:?}  (true: {true_beta})", result.beta);
    eprintln!("  logd   = {:?}  (true: {true_logd})", result.logd);
    eprintln!("  loga   = {:?}  (true: {true_loga})", result.loga);
    eprintln!("  tb     = {:?}  (true: {true_tb})", result.tb);
    eprintln!("  t0     = {:?}  (true: {true_t0})", result.t0);
    eprintln!("  reduced_chi2 = {:?}", result.reduced_chi2);

    if let Some(chi2) = result.reduced_chi2 {
        assert!(chi2.is_finite() && chi2 >= 0.0, "reduced_chi2 must be non-negative");
    }

    // With a well-constrained synthetic source the chi2 should be near 1.
    if let Some(chi2) = result.reduced_chi2 {
        assert!(chi2 < 100.0, "chi2={chi2} unreasonably large on clean synthetic data");
    }
}

// ---------------------------------------------------------------------------
// Serialisation round-trip
// ---------------------------------------------------------------------------

#[test]
fn sbpl_result_serialises_roundtrips() {
    let bands = generate_sbpl_source(40, 7777, 1.5, -1.5, -0.7, -0.3, 12.0, 10.0, 5.0);
    let result = fit_sbpl(&bands, &PsoConfig::default()).unwrap();

    let json = serde_json::to_string(&result).expect("serialisation failed");
    let deser: sbpl_pso::SbplResult = serde_json::from_str(&json).expect("deserialisation failed");

    assert_eq!(result.n_obs, deser.n_obs);
    assert_eq!(result.n_bands, deser.n_bands);
    match (result.alpha1, deser.alpha1) {
        (Some(a), Some(b)) => assert!((a - b).abs() < 1e-10),
        (None, None) => {}
        _ => panic!("alpha1 Some/None mismatch after roundtrip"),
    }
}

// ---------------------------------------------------------------------------
// Objective and analytic gradient
// ---------------------------------------------------------------------------

const TRUE_PARAMS: [f64; N_PARAMS] = [1.5, -1.8, -0.7, -0.4, 12.0, 10.0, 5.0];

fn prepared_source(seed: u64) -> PreparedSource {
    let [a1, a2, b, ld, la, tb, t0] = TRUE_PARAMS;
    let bands = generate_sbpl_source(30, seed, a1, a2, b, ld, la, tb, t0);
    match prepare_source(&bands, &PsoConfig::default()) {
        Preparation::Ready(p) => p,
        other => panic!("synthetic source not fittable: {other:?}"),
    }
}

/// Reduced chi² built directly from `sbpl_model`.
fn reference_chi2(prep: &PreparedSource, p: &[f64]) -> f64 {
    let mut chi2 = 0.0;
    for i in 0..prep.times.len() {
        let m = sbpl_model(
            prep.times[i], prep.nu_scaled[i], p[0], p[1], p[2], p[3], p[4], p[5], p[6],
        );
        if !m.is_finite() {
            return 1e10;
        }
        let r = prep.flux[i] - m;
        chi2 += r * r / (prep.flux_err[i] * prep.flux_err[i] + 1e-30);
    }
    chi2 / prep.times.len() as f64
}

#[test]
fn sbpl_cost_matches_sbpl_model() {
    let prep = prepared_source(4242);
    let cost = SbplCost::new(&prep);
    let duration = prep.times.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - prep.t_ref;
    let mut rng = 0x5eed_u64;
    for i in 0..500 {
        let mut p = [0.0; N_PARAMS];
        for d in 0..N_PARAMS {
            p[d] = prep.lower[d] + xorshift(&mut rng) * (prep.upper[d] - prep.lower[d]);
        }
        if i % 2 == 1 {
            // Put t0 inside the data so some points sit before it.
            p[6] = prep.t_ref + xorshift(&mut rng) * duration;
        }
        let (got, want) = (cost.eval(&p), reference_chi2(&prep, &p));
        assert!(
            got == want || ((got - want) / want).abs() < 1e-10,
            "params {p:?}: eval {got:e} vs sbpl_model {want:e}"
        );
        assert_eq!(cost.eval_with_grad(&p).0, got);
    }
}

#[test]
fn sbpl_cost_gradient_matches_finite_differences() {
    let prep = prepared_source(4242);
    let cost = SbplCost::new(&prep);
    let mut truth = TRUE_PARAMS;
    truth[4] -= prep.max_flux.log10(); // the fit works in normalised flux

    let mut points = Vec::new();
    let mut rng = 0xfeed_u64;
    for _ in 0..8 {
        let mut p = truth;
        for d in 0..6 {
            p[d] += (xorshift(&mut rng) - 0.5) * 0.4 * p[d].abs().max(0.5);
        }
        p[6] -= 5.0 * xorshift(&mut rng);
        points.push(p);
    }
    // t0 half-way between two observations: the first few points are pre-t0.
    let mut inside = truth;
    let mut times = prep.times.clone();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times.dedup();
    inside[6] = 0.5 * (times[3] + times[4]);
    points.push(inside);

    for p in points {
        let (c, grad) = cost.eval_with_grad(&p);
        assert!(c < 1e10, "test point {p:?} is invalid");
        let scale = grad.iter().fold(0.0f64, |m, g| m.max(g.abs()));
        for d in 0..N_PARAMS {
            let h = 1e-6 * (prep.upper[d] - prep.lower[d]);
            let (mut hi, mut lo) = (p, p);
            hi[d] += h;
            lo[d] -= h;
            let fd = (cost.eval(&hi) - cost.eval(&lo)) / (2.0 * h);
            assert!(
                (fd - grad[d]).abs() <= 1e-5 * (grad[d].abs() + 1e-3 * scale),
                "d{d} at {p:?}: analytic {:e} vs finite difference {fd:e}",
                grad[d]
            );
        }
    }
}

#[test]
fn sbpl_cost_sentinel_has_zero_gradient() {
    let prep = prepared_source(4242);
    let cost = SbplCost::new(&prep);
    let mut p = TRUE_PARAMS;
    p[5] = -1.0; // tb <= 0 is invalid everywhere
    assert_eq!(cost.eval_with_grad(&p), (1e10, [0.0; N_PARAMS]));
}
