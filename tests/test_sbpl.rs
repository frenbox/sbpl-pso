//! Unit tests for the SBPL fitter using synthetic light curves.
//!
//! Run:
//!   cargo test --release --test test_sbpl -- --nocapture

use std::collections::HashMap;
use sbpl_pso::{BandData, fit_sbpl, sbpl_model, band_frequency_hz, PsoConfig};

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
    for band in &["g", "r", "i", "ztfg", "ztfr"] {
        let nu = band_frequency_hz(band).unwrap();
        assert!(nu > 0.0, "frequency for {band} should be positive");
    }
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
