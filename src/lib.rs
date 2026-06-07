//! Smoothly Broken Power Law (SBPL) multi-band light-curve fitter.
//!
//! Model (afterglow / power-law transient physics):
//!
//!   F(t, ν) = 10^loga · (ν/1e15)^β · τ^α₁ · [0.5·(1 + τ^(1/D))]^((α₂−α₁)·D)
//!   where τ = (t−t₀)/tb,  D = 10^logd
//!
//! All bands with a known effective frequency are fit simultaneously.
//! During fitting, fluxes are normalised by the global max_flux so that
//! loga is in normalised space; the physical amplitude is recovered after:
//!   loga_phys = loga_fit + log10(max_flux) − 15·β
//!
//! Optimisation: deterministic multi-restart PSO for global exploration,
//! followed by L-BFGS polishing from the best PSO solution.

use std::collections::HashMap;

use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Physical constants & band frequencies
// ---------------------------------------------------------------------------

const C_ANGSTROM_PER_SEC: f64 = 2.997_924_58e18;

/// Effective central wavelength (Å) for supported filter names.
/// Accepts short names (g, r, i), ZTF names (ztfg, ztfr, ztfi), and LSST names.
pub fn band_wavelength(band: &str) -> Option<f64> {
    match band {
        "g" | "ztfg" | "lsstg" | "ZTF_g" => Some(4770.0),
        "r" | "ztfr" | "lsstr" | "ZTF_r" => Some(6231.0),
        "i" | "ztfi" | "lssti" | "ZTF_i" => Some(7625.0),
        "z" | "ztfz" => Some(9100.0),
        "y" => Some(9710.0),
        "u" | "lsstu" => Some(3540.0),
        _ => None,
    }
}

/// Effective frequency (Hz) for a band name, or `None` if unknown.
pub fn band_frequency_hz(band: &str) -> Option<f64> {
    let lambda = band_wavelength(band)?;
    if lambda <= 0.0 || !lambda.is_finite() {
        return None;
    }
    Some(C_ANGSTROM_PER_SEC / lambda)
}

// ---------------------------------------------------------------------------
// Magnitude → flux conversion (zeropoint = 23.9)
// ---------------------------------------------------------------------------

pub fn mag_to_flux(mag: f64, mag_err: f64) -> (f64, f64) {
    let flux = 10.0_f64.powf((23.9 - mag) / 2.5);
    let flux_err = flux * mag_err * std::f64::consts::LN_10 / 2.5;
    (flux, flux_err)
}

// ---------------------------------------------------------------------------
// CSV loading
// ---------------------------------------------------------------------------

/// One photometric observation loaded from a ZTF CSV.
#[derive(Debug, Clone)]
pub struct Obs {
    pub time: f64,    // Julian Date (days)
    pub flux: f64,
    pub flux_err: f64,
    pub band: String, // short name: "g", "r", "i"
}

/// Load a ZTF-style photometry CSV and convert magnitudes to flux.
///
/// Accepts columns: `jd` (or `mjd`), `magpsf` (or `mag`), `sigmapsf` (or
/// `mag_err`), and either `fid` (1=g, 2=r, 3=i) or a `filter` string column.
/// Rows with missing/non-finite values are silently skipped.
pub fn load_csv(path: &str) -> Result<Vec<Obs>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .from_path(path)
        .map_err(|e| format!("Cannot open {path}: {e}"))?;

    let headers = rdr
        .headers()
        .map_err(|e| format!("Cannot read headers: {e}"))?
        .clone();

    let col = |name: &str| headers.iter().position(|h| h == name);

    let time_col = col("jd").or_else(|| col("mjd"))
        .ok_or("CSV must have 'jd' or 'mjd' column")?;
    let mag_col = col("magpsf").or_else(|| col("mag"))
        .ok_or("CSV must have 'magpsf' or 'mag' column")?;
    let err_col = col("sigmapsf").or_else(|| col("mag_err"))
        .ok_or("CSV must have 'sigmapsf' or 'mag_err' column")?;
    let fid_col = col("fid");
    let filter_col = col("filter");

    let mut obs = Vec::new();

    for record in rdr.records() {
        let rec = record.map_err(|e| format!("Parse error: {e}"))?;

        let time: f64 = match rec.get(time_col).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let mag: f64 = match rec.get(mag_col).and_then(|s| s.trim().parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let mag_err: f64 = match rec.get(err_col).and_then(|s| s.trim().parse().ok()) {
            Some(v) if v > 0.0 => v,
            _ => continue,
        };

        let band = if let Some(fc) = fid_col {
            match rec.get(fc).and_then(|s| s.trim().parse::<u32>().ok()) {
                Some(1) => "g".to_string(),
                Some(2) => "r".to_string(),
                Some(3) => "i".to_string(),
                _ => continue,
            }
        } else if let Some(fc) = filter_col {
            match rec.get(fc).map(|s| s.trim()) {
                Some("g" | "ZTF_g" | "ztfg") => "g".to_string(),
                Some("r" | "ZTF_r" | "ztfr") => "r".to_string(),
                Some("i" | "ZTF_i" | "ztfi") => "i".to_string(),
                _ => continue,
            }
        } else {
            continue;
        };

        let (flux, flux_err) = mag_to_flux(mag, mag_err);
        if !flux.is_finite() || !flux_err.is_finite() || flux <= 0.0 {
            continue;
        }

        obs.push(Obs { time, flux, flux_err, band });
    }

    if obs.is_empty() {
        return Err(format!("No valid g/r/i observations in {path}"));
    }
    Ok(obs)
}

/// Group a flat `Vec<Obs>` into per-band maps for `fit_sbpl`.
pub fn obs_to_band_map(obs: &[Obs]) -> HashMap<String, BandData> {
    let mut map: HashMap<String, BandData> = HashMap::new();
    for o in obs {
        let e = map.entry(o.band.clone()).or_insert_with(BandData::new);
        e.times.push(o.time);
        e.fluxes.push(o.flux);
        e.flux_errs.push(o.flux_err);
    }
    map
}

// ---------------------------------------------------------------------------
// BandData
// ---------------------------------------------------------------------------

/// Flux-space time-series for one photometric band.
#[derive(Debug, Clone, Default)]
pub struct BandData {
    pub times: Vec<f64>,
    pub fluxes: Vec<f64>,
    pub flux_errs: Vec<f64>,
}

impl BandData {
    pub fn new() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// SBPL model
// ---------------------------------------------------------------------------

/// Evaluate SBPL flux at a single (t, nu_scaled) point.
/// `nu_scaled` = ν / 1e15 Hz.  Returns 0.0 before t0, NAN on domain errors.
pub fn sbpl_model(
    t: f64,
    nu_scaled: f64,
    alpha1: f64,
    alpha2: f64,
    beta: f64,
    logd: f64,
    loga: f64,
    tb: f64,
    t0: f64,
) -> f64 {
    let tau = t - t0;
    if tb <= 0.0 || nu_scaled <= 0.0 {
        return f64::NAN;
    }
    if tau < 0.0 {
        return 0.0;
    }
    let ratio = tau / tb;
    if ratio == 0.0 || !ratio.is_finite() {
        return f64::NAN;
    }
    let d = 10f64.powf(logd);
    let term1 = nu_scaled.powf(beta);
    let term2 = ratio.powf(alpha1);
    let inner = 0.5 * (1.0 + ratio.powf(1.0 / d));
    if !inner.is_finite() || inner <= 0.0 {
        return f64::NAN;
    }
    let term3 = inner.powf((alpha2 - alpha1) * d);
    let result = 10f64.powf(loga) * term1 * term2 * term3;
    if result.is_finite() { result } else { f64::NAN }
}

// ---------------------------------------------------------------------------
// Cost function (reduced chi²)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SbplObs {
    time: f64,
    nu_scaled: f64,
    flux: f64,
    flux_err: f64,
}

#[derive(Clone)]
struct SbplCost {
    observations: Vec<SbplObs>,
}

impl SbplCost {
    /// Evaluate reduced chi² for `p = [alpha1, alpha2, beta, logd, loga, tb, t0]`.
    fn eval(&self, p: &[f64]) -> f64 {
        let (alpha1, alpha2, beta, logd, loga, tb, t0) =
            (p[0], p[1], p[2], p[3], p[4], p[5], p[6]);

        let mut chi2 = 0.0;
        let mut n_valid = 0usize;
        for obs in &self.observations {
            let model = sbpl_model(obs.time, obs.nu_scaled, alpha1, alpha2, beta, logd, loga, tb, t0);
            if !model.is_finite() {
                return 1e10;
            }
            let residual = obs.flux - model;
            let err_sq = obs.flux_err * obs.flux_err + 1e-30;
            chi2 += residual * residual / err_sq;
            n_valid += 1;
        }
        if n_valid == 0 {
            return 1e10;
        }
        chi2 / n_valid as f64
    }
}

// ---------------------------------------------------------------------------
// Scaled wrapper for L-BFGS
//
// All 7 parameters are mapped to [0, 1] (unit-cube space) so L-BFGS sees a
// well-conditioned Hessian approximation — the same x_scale trick used by
// scipy.optimize.least_squares.
// ---------------------------------------------------------------------------

struct ScaledCost<'a> {
    inner: &'a SbplCost,
    lower: Vec<f64>,
    scale: Vec<f64>, // upper - lower
}

impl ScaledCost<'_> {
    fn unscale(&self, xs: &[f64]) -> Vec<f64> {
        xs.iter()
            .enumerate()
            .map(|(i, &v)| self.lower[i] + v.clamp(0.0, 1.0) * self.scale[i])
            .collect()
    }
}

impl CostFunction for ScaledCost<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, xs: &Self::Param) -> Result<Self::Output, ArgminError> {
        Ok(self.inner.eval(&self.unscale(xs)))
    }
}

impl Gradient for ScaledCost<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, xs: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let n = xs.len();
        let h = 1e-5;
        let mut grad = vec![0.0; n];
        for i in 0..n {
            let mut xp = xs.clone();
            let mut xm = xs.clone();
            xp[i] = (xs[i] + h).min(1.0);
            xm[i] = (xs[i] - h).max(0.0);
            let step = xp[i] - xm[i];
            if step > 0.0 {
                grad[i] = (self.inner.eval(&self.unscale(&xp))
                    - self.inner.eval(&self.unscale(&xm)))
                    / step;
            }
        }
        Ok(grad)
    }
}

// ---------------------------------------------------------------------------
// PSO (deterministic multi-restart)
// ---------------------------------------------------------------------------

fn pso_search(
    problem: &SbplCost,
    lower: &[f64],
    upper: &[f64],
    n_particles: usize,
    n_iters: usize,
    rng: &mut rand::rngs::SmallRng,
) -> (Vec<f64>, f64) {
    let n_dim = lower.len();
    let span: Vec<f64> = lower.iter().zip(upper).map(|(&lo, &hi)| hi - lo).collect();

    let mut pos: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| {
            lower.iter().zip(upper).map(|(&lo, &hi)| rng.random_range(lo..hi)).collect()
        })
        .collect();

    let mut vel: Vec<Vec<f64>> = (0..n_particles)
        .map(|_| span.iter().map(|&s| rng.random_range(-0.1 * s..0.1 * s)).collect())
        .collect();

    let mut pbest_pos = pos.clone();
    let mut pbest_cost: Vec<f64> = pos.iter().map(|p| problem.eval(p)).collect();

    let mut gbest_idx = 0usize;
    for i in 1..n_particles {
        if pbest_cost[i] < pbest_cost[gbest_idx] {
            gbest_idx = i;
        }
    }
    let mut gbest_pos = pbest_pos[gbest_idx].clone();
    let mut gbest_cost = pbest_cost[gbest_idx];

    // Clerc-Kennedy constriction coefficients
    let w = 0.7298;
    let c1 = 1.4962;
    let c2 = 1.4962;

    for _ in 0..n_iters {
        for i in 0..n_particles {
            for d in 0..n_dim {
                let r1: f64 = rng.random();
                let r2: f64 = rng.random();
                vel[i][d] = w * vel[i][d]
                    + c1 * r1 * (pbest_pos[i][d] - pos[i][d])
                    + c2 * r2 * (gbest_pos[d] - pos[i][d]);
                pos[i][d] = (pos[i][d] + vel[i][d]).clamp(lower[d], upper[d]);
            }
            let cost = problem.eval(&pos[i]);
            if cost < pbest_cost[i] {
                pbest_cost[i] = cost;
                pbest_pos[i] = pos[i].clone();
            }
            if cost < gbest_cost {
                gbest_cost = cost;
                gbest_pos = pos[i].clone();
            }
        }
    }

    (gbest_pos, gbest_cost)
}

// ---------------------------------------------------------------------------
// L-BFGS polishing
// ---------------------------------------------------------------------------

fn lbfgs_refine(
    problem: &SbplCost,
    start: Vec<f64>,
    start_cost: f64,
    lower: &[f64],
    upper: &[f64],
) -> (Vec<f64>, f64) {
    let scale: Vec<f64> = lower.iter().zip(upper).map(|(&lo, &hi)| hi - lo).collect();
    let xs0: Vec<f64> = start
        .iter()
        .enumerate()
        .map(|(i, &v)| ((v - lower[i]) / scale[i]).clamp(0.0, 1.0))
        .collect();

    let scaled = ScaledCost {
        inner: problem,
        lower: lower.to_vec(),
        scale: scale.clone(),
    };

    let linesearch = MoreThuenteLineSearch::new();
    let solver = match LBFGS::new(linesearch, 10).with_tolerance_grad(1e-7) {
        Ok(s) => s,
        Err(_) => return (start, start_cost),
    };

    let result = Executor::new(scaled, solver)
        .configure(|state| state.param(xs0).max_iters(200))
        .run();

    match result {
        Ok(res) => {
            let xs_best = res.state().best_param.clone().unwrap_or_default();
            if xs_best.is_empty() {
                return (start, start_cost);
            }
            let x_best: Vec<f64> = xs_best
                .iter()
                .enumerate()
                .map(|(i, &v)| lower[i] + v.clamp(0.0, 1.0) * scale[i])
                .collect();
            let final_cost = problem.eval(&x_best);
            if final_cost < start_cost && final_cost.is_finite() {
                (x_best, final_cost)
            } else {
                (start, start_cost)
            }
        }
        Err(_) => (start, start_cost),
    }
}

// ---------------------------------------------------------------------------
// PSO configuration
// ---------------------------------------------------------------------------

/// Tunable PSO parameters, including optional per-parameter bound overrides.
///
/// The most important overrides for physical models:
/// - `alpha1_min = 0.0` — enforce rising pre-break phase (F ∝ t^α₁, α₁ > 0)
/// - `alpha2_max = 0.0` — enforce declining post-break phase (α₂ < 0)
///
/// These two together constrain the break to be at the light-curve **peak**,
/// which is the physical interpretation for off-axis GRB afterglows.
///
/// `tb_lower` / `tb_upper` override the automatic data-driven t_b search bounds.
/// Useful when the real break timescale is known to fall outside the default
/// 1%–1000% of the observation span (e.g. a very early peak at < 1% of the span).
#[derive(Debug, Clone)]
pub struct PsoConfig {
    pub n_restarts: usize,
    pub n_particles: usize,
    pub n_iters: usize,
    pub seed: u64,
    /// Lower bound for α₁ (default −10; set to 0 to enforce rising pre-break).
    pub alpha1_min: f64,
    /// Upper bound for α₁ (default +10).
    pub alpha1_max: f64,
    /// Lower bound for α₂ (default −10).
    pub alpha2_min: f64,
    /// Upper bound for α₂ (default +10; set to 0 to enforce declining post-break).
    pub alpha2_max: f64,
    /// Override the lower bound for t_b (days).  `None` → use 1% of data span.
    pub tb_lower: Option<f64>,
    /// Override the upper bound for t_b (days).  `None` → use 10× data span (capped at 10 000 d).
    pub tb_upper: Option<f64>,
}

impl Default for PsoConfig {
    fn default() -> Self {
        PsoConfig {
            n_restarts: 3,
            n_particles: 30,
            n_iters: 200,
            seed: 1234,
            alpha1_min: -10.0,
            alpha1_max:  10.0,
            alpha2_min: -10.0,
            alpha2_max:  10.0,
            tb_lower: None,
            tb_upper: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

/// Fitted SBPL parameters and uncertainty estimates for one source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbplResult {
    /// Pre-break temporal power-law index.
    pub alpha1: Option<f64>,
    /// Post-break temporal power-law index.
    pub alpha2: Option<f64>,
    /// Spectral index (F_ν ∝ ν^β).
    pub beta: Option<f64>,
    /// log10 of the break-smoothness parameter D.
    pub logd: Option<f64>,
    /// Physical log10 flux amplitude (after rescaling from normalised-fit space).
    pub loga: Option<f64>,
    /// Break time tb (days).
    pub tb: Option<f64>,
    /// Time zero-point t0 (days).
    pub t0: Option<f64>,
    /// Uncertainties: std dev of physical parameters across PSO restarts.
    pub alpha1_err: Option<f64>,
    pub alpha2_err: Option<f64>,
    pub beta_err: Option<f64>,
    pub logd_err: Option<f64>,
    pub loga_err: Option<f64>,
    pub tb_err: Option<f64>,
    pub t0_err: Option<f64>,
    /// Reduced chi-squared of the best fit.
    pub reduced_chi2: Option<f64>,
    /// Total number of observations used across all bands.
    pub n_obs: usize,
    /// Number of bands used.
    pub n_bands: usize,
}

impl SbplResult {
    fn empty(n_obs: usize, n_bands: usize) -> Self {
        SbplResult {
            alpha1: None, alpha2: None, beta: None, logd: None, loga: None,
            tb: None, t0: None,
            alpha1_err: None, alpha2_err: None, beta_err: None, logd_err: None,
            loga_err: None, tb_err: None, t0_err: None,
            reduced_chi2: None, n_obs, n_bands,
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point: fit_sbpl
// ---------------------------------------------------------------------------

/// Fit the SBPL model to multi-band flux-space data.
///
/// `bands` is a map from band name → `BandData` (flux, not magnitudes).
/// Bands without a known effective frequency are silently skipped.
/// Returns `None` only when the input map is empty; returns an empty
/// `SbplResult` when there are fewer than 7 usable observations.
pub fn fit_sbpl(
    bands: &HashMap<String, BandData>,
    config: &PsoConfig,
) -> Option<SbplResult> {
    if bands.is_empty() {
        return None;
    }

    // --- Collect observations ------------------------------------------------
    let mut observations: Vec<SbplObs> = Vec::new();
    let mut n_bands = 0usize;
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;

    let mut band_names: Vec<&String> = bands.keys().collect();
    band_names.sort_unstable();

    for band_name in band_names {
        let Some(band_data) = bands.get(band_name) else { continue; };
        let nu = match band_frequency_hz(band_name) {
            Some(n) => n,
            None => continue,
        };
        if band_data.times.is_empty() {
            continue;
        }
        n_bands += 1;
        for i in 0..band_data.times.len() {
            let t = band_data.times[i];
            let flux = band_data.fluxes[i];
            let flux_err = band_data.flux_errs[i];
            if !t.is_finite() || !flux.is_finite() || !flux_err.is_finite() || flux_err <= 0.0 {
                continue;
            }
            t_min = t_min.min(t);
            t_max = t_max.max(t);
            observations.push(SbplObs { time: t, nu_scaled: nu / 1e15, flux, flux_err });
        }
    }

    let n_obs = observations.len();
    if n_obs < 7 || n_bands < 2 {
        return Some(SbplResult::empty(n_obs, n_bands));
    }

    let duration = t_max - t_min;
    if duration <= 0.0 {
        return Some(SbplResult::empty(n_obs, n_bands));
    }

    // --- Normalise fluxes by global max --------------------------------------
    let max_flux = observations
        .iter()
        .map(|o| o.flux)
        .filter(|f| f.is_finite() && *f > 0.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_flux = if max_flux > 0.0 { max_flux } else { 1.0 };
    for obs in &mut observations {
        obs.flux /= max_flux;
        obs.flux_err /= max_flux;
    }

    // --- Search bounds: [alpha1, alpha2, beta, logd, loga, tb, t0] ----------
    //
    // IMPORTANT UNIT NOTE
    // -------------------
    // The model computes τ = (t − t₀) / tb, so tb is a TIMESCALE (in the
    // same day-unit as t − t₀), NOT an absolute timestamp.  Bounds must be
    // in days, not MJD — otherwise τ ≪ 1 always and the break is never
    // reached (the model degenerates to a single power-law in tau^alpha1).
    //
    // tb range: 1% of the observation span up to 10× the span (floored at
    // 0.01 day and capped at 10 000 days to stay reasonable).
    //
    // t0 (absolute onset time) must be before the first observation; allow
    // it to go back by two observation spans so early-breaking sources like
    // kilonovae are handled correctly.
    let tb_lower = config.tb_lower.unwrap_or_else(|| (0.01f64 * duration).max(0.01));
    // Cap at least 500 days so sources with a break well beyond the
    // observation window (e.g. off-axis GRB afterglows peaking at ~150 d)
    // are still reachable.
    let tb_upper = config.tb_upper.unwrap_or_else(|| (10.0 * duration).max(500.0).min(10_000.0));

    let lower = vec![
        config.alpha1_min,       // alpha1 (default −10; set 0 for peak model)
        config.alpha2_min,       // alpha2 (default −10)
        -5.0,               // beta
        -3.0,               // logd
        -10.0,              // loga (normalised)
        tb_lower,           // tb — break timescale (days)
        t_min - 2.0 * duration, // t0 — onset, may precede data by 2× span
    ];
    let upper = vec![
        config.alpha1_max,  // alpha1 (default +10)
        config.alpha2_max,  // alpha2 (default +10; set 0 for peak model)
        5.0,     // beta
        0.0,     // logd
        10.0,    // loga (normalised)
        tb_upper, // tb — break timescale (days)
        t_min,   // t0 — onset must be at or before first observation
    ];

    let problem = SbplCost { observations };
    let mut rng = rand::rngs::SmallRng::seed_from_u64(config.seed);

    // --- Multi-restart PSO + L-BFGS ------------------------------------------
    let mut all_params: Vec<Vec<f64>> = Vec::new();
    let mut best_cost = f64::INFINITY;
    let mut best_params: Vec<f64> = lower.clone();

    for _ in 0..config.n_restarts {
        let (pso_best, pso_cost) =
            pso_search(&problem, &lower, &upper, config.n_particles, config.n_iters, &mut rng);
        let (refined, refined_cost) = lbfgs_refine(&problem, pso_best, pso_cost, &lower, &upper);
        all_params.push(refined.clone());
        if refined_cost < best_cost {
            best_cost = refined_cost;
            best_params = refined;
        }
    }

    if all_params.is_empty() {
        return Some(SbplResult::empty(n_obs, n_bands));
    }

    // --- Recover physical loga.
    //
    // During fitting, fluxes were normalised by max_flux and the model used
    // nu_scaled = nu / 1e15.  The cost function minimised:
    //   10^loga_fit · (nu/1e15)^β · time_terms  ≈  flux / max_flux
    //
    // Physical flux = max_flux · (normalised flux), so the physical loga for
    // the same (nu/1e15)^β formula is simply:
    //   loga_phys = loga_fit + log10(max_flux)
    //
    // No 15·β term: that correction would only be needed if the returned loga
    // were intended for use with raw nu^β (Hz), but sbpl_model and eval_sbpl
    // both use nu_scaled = nu/1e15 throughout.
    let log10_max_flux = max_flux.log10();
    for params in &mut all_params {
        params[4] += log10_max_flux;
    }
    best_params[4] += log10_max_flux;

    // --- Uncertainty: std dev of physical params across restarts -------------
    let n_params = 7;
    let n_r = all_params.len() as f64;
    let means: Vec<f64> = (0..n_params)
        .map(|i| all_params.iter().map(|p| p[i]).sum::<f64>() / n_r)
        .collect();
    let stds: Vec<f64> = (0..n_params)
        .map(|i| {
            let var = all_params.iter().map(|p| (p[i] - means[i]).powi(2)).sum::<f64>() / n_r;
            var.sqrt()
        })
        .collect();

    let f = |v: f64| if v.is_finite() { Some(v) } else { None };

    Some(SbplResult {
        alpha1: f(best_params[0]),
        alpha2: f(best_params[1]),
        beta:   f(best_params[2]),
        logd:   f(best_params[3]),
        loga:   f(best_params[4]),
        tb:     f(best_params[5]),
        t0:     f(best_params[6]),
        alpha1_err: f(stds[0]),
        alpha2_err: f(stds[1]),
        beta_err:   f(stds[2]),
        logd_err:   f(stds[3]),
        loga_err:   f(stds[4]),
        tb_err:     f(stds[5]),
        t0_err:     f(stds[6]),
        reduced_chi2: f(best_cost),
        n_obs,
        n_bands,
    })
}

// ---------------------------------------------------------------------------
// Convenience wrapper: load CSV → fit
// ---------------------------------------------------------------------------

/// Load a ZTF photometry CSV and fit SBPL with the given PSO config.
pub fn fit_sbpl_csv(path: &str, config: &PsoConfig) -> Result<SbplResult, String> {
    let obs = load_csv(path)?;
    let bands = obs_to_band_map(&obs);
    fit_sbpl(&bands, config).ok_or_else(|| "Empty band map".to_string())
}

// ---------------------------------------------------------------------------
// Utility: scan a directory and collect all CSV paths
// ---------------------------------------------------------------------------

pub fn find_csv_files(dir: &str) -> Vec<String> {
    let mut paths: Vec<String> = std::fs::read_dir(dir)
        .expect("Cannot read directory")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "csv"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    paths.sort();
    paths
}

// ---------------------------------------------------------------------------
// PyO3 Python bindings
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python_bindings {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    /// Fit an SBPL model to a ZTF photometry CSV.
    ///
    /// Returns a dict with keys:
    ///   params       – dict of fitted physical parameters (alpha1, alpha2, beta,
    ///                  logd, loga, tb, t0) and their *_err counterparts.
    ///   obs          – list of dicts [{time, flux, flux_err, band}, ...] in
    ///                  physical flux units (not normalised).
    ///   reduced_chi2 – float or None
    ///   n_obs        – int
    ///   n_bands      – int
    #[pyfunction]
    #[pyo3(signature = (
        csv_path,
        n_particles=30, n_iters=200, n_restarts=3, seed=1234,
        alpha1_min=-10.0f64, alpha1_max=10.0f64,
        alpha2_min=-10.0f64, alpha2_max=10.0f64,
        tb_lower=None, tb_upper=None,
    ))]
    fn fit(
        csv_path: &str,
        n_particles: usize,
        n_iters: usize,
        n_restarts: usize,
        seed: u64,
        alpha1_min: f64,
        alpha1_max: f64,
        alpha2_min: f64,
        alpha2_max: f64,
        tb_lower: Option<f64>,
        tb_upper: Option<f64>,
    ) -> PyResult<PyObject> {
        let obs = load_csv(csv_path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let bands = obs_to_band_map(&obs);
        let config = PsoConfig {
            n_particles, n_iters, n_restarts, seed,
            alpha1_min, alpha1_max, alpha2_min, alpha2_max,
            tb_lower, tb_upper,
        };

        let result = fit_sbpl(&bands, &config)
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Empty band map"))?;

        Python::with_gil(|py| {
            let dict = PyDict::new(py);

            // --- params dict -------------------------------------------------
            let params = PyDict::new(py);
            let scalars: &[(&str, Option<f64>)] = &[
                ("alpha1",     result.alpha1),
                ("alpha2",     result.alpha2),
                ("beta",       result.beta),
                ("logd",       result.logd),
                ("loga",       result.loga),
                ("tb",         result.tb),
                ("t0",         result.t0),
                ("alpha1_err", result.alpha1_err),
                ("alpha2_err", result.alpha2_err),
                ("beta_err",   result.beta_err),
                ("logd_err",   result.logd_err),
                ("loga_err",   result.loga_err),
                ("tb_err",     result.tb_err),
                ("t0_err",     result.t0_err),
            ];
            for &(name, val) in scalars {
                match val {
                    Some(v) => params.set_item(name, v)?,
                    None    => params.set_item(name, py.None())?,
                }
            }
            dict.set_item("params", params)?;

            // --- metadata ----------------------------------------------------
            match result.reduced_chi2 {
                Some(v) => dict.set_item("reduced_chi2", v)?,
                None    => dict.set_item("reduced_chi2", py.None())?,
            }
            dict.set_item("n_obs",   result.n_obs)?;
            dict.set_item("n_bands", result.n_bands)?;

            // --- observations (physical flux) ---------------------------------
            let obs_list = PyList::empty(py);
            for o in &obs {
                let d = PyDict::new(py);
                d.set_item("time",     o.time)?;
                d.set_item("flux",     o.flux)?;
                d.set_item("flux_err", o.flux_err)?;
                d.set_item("band",     o.band.as_str())?;
                obs_list.append(d)?;
            }
            dict.set_item("obs", obs_list)?;

            Ok(dict.into_any().unbind())
        })
    }

    /// Evaluate the SBPL model on a dense time grid for one photometric band.
    ///
    /// `params` must be the dict returned by `fit()["params"]` (physical units).
    /// Returns a list of flux values (same physical units as `fit()["obs"]`).
    #[pyfunction]
    fn eval_sbpl(params: &Bound<'_, PyDict>, t_dense: Vec<f64>, band: &str) -> PyResult<Vec<f64>> {
        let nu = band_frequency_hz(band).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown band '{band}'. Use 'g', 'r', 'i', 'ztfg', 'ztfr', 'ztfi', etc."
            ))
        })?;
        let nu_scaled = nu / 1e15;

        let get = |key: &str| -> PyResult<f64> {
            params
                .get_item(key)?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
                .extract()
        };

        let alpha1 = get("alpha1")?;
        let alpha2 = get("alpha2")?;
        let beta   = get("beta")?;
        let logd   = get("logd")?;
        let loga   = get("loga")?;
        let tb     = get("tb")?;
        let t0     = get("t0")?;

        Ok(t_dense
            .iter()
            .map(|&t| sbpl_model(t, nu_scaled, alpha1, alpha2, beta, logd, loga, tb, t0))
            .collect())
    }

    #[pymodule]
    fn sbpl_pso(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(fit, m)?)?;
        m.add_function(wrap_pyfunction!(eval_sbpl, m)?)?;
        Ok(())
    }
}
