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
use std::f64::consts::LN_10;

use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use rand::Rng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// NVIDIA GPU backend (batch PSO). Enabled with `--features cuda`.
#[cfg(feature = "cuda")]
pub mod gpu;

/// Apple Silicon GPU backend (batch PSO). Enabled with `--features metal`.
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod gpu_metal;

/// Host-side batch PSO driver shared by the CUDA and Metal backends. Also
/// built for unit tests, which drive it with a CPU stand-in for the kernel.
#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos"), test))]
pub(crate) mod gpu_common;

/// Number of fitted parameters: `[alpha1, alpha2, beta, logd, loga, tb, t0]`.
pub const N_PARAMS: usize = 7;

/// Fitted parameter names, in the order used by every parameter vector.
pub const PARAM_NAMES: [&str; N_PARAMS] =
    ["alpha1", "alpha2", "beta", "logd", "loga", "tb", "t0"];

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
        "z" | "ztfz" | "lsstz" => Some(9100.0),
        "y" | "lssty" => Some(9710.0),
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

/// Load a ZTF/LSST-style photometry CSV and convert magnitudes to flux.
///
/// Accepts columns: `jd` (or `mjd`), `magpsf` (or `mag`), `sigmapsf` (or
/// `mag_err`), and either `fid` (1=g, 2=r, 3=i; ZTF-only) or a `filter`
/// string column (g/r/i/z/y, plus ZTF/LSST-prefixed variants).
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
                Some("g" | "ZTF_g" | "ztfg" | "lsstg") => "g".to_string(),
                Some("r" | "ZTF_r" | "ztfr" | "lsstr") => "r".to_string(),
                Some("i" | "ZTF_i" | "ztfi" | "lssti") => "i".to_string(),
                Some("z" | "ztfz" | "lsstz") => "z".to_string(),
                Some("y" | "lssty") => "y".to_string(),
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
        return Err(format!("No valid g/r/i/z/y observations in {path}"));
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

/// One observation in the flattened fitting space (flux normalised by
/// `PreparedSource::max_flux`, frequency scaled by 1e15 Hz).
#[derive(Clone, Copy, Debug)]
pub struct SbplObs {
    pub time: f64,
    pub nu_scaled: f64,
    pub flux: f64,
    pub flux_err: f64,
}

/// Reduced-χ² objective for one prepared source.
///
/// This is the exact function the PSO minimises, on the CPU and on the GPU
/// (`cuda/sbpl.cu` and `metal/sbpl.metal` reimplement it device-side).
///
/// The model is [`sbpl_model`], with the same failure sentinels, evaluated in
/// log space: the per-particle constants are computed once per call rather
/// than once per point, `ln(nu)` once per observation, and the flux is a
/// single `exp` of a sum of logs. That is four transcendentals per point
/// instead of six `powf`, and it leaves behind the logs that the analytic
/// gradient in [`Self::eval_with_grad`] needs. Values agree with
/// [`sbpl_model`] to rounding.
#[derive(Clone, Debug)]
pub struct SbplCost {
    observations: Vec<SbplObs>,
    /// `ln(nu_scaled)` per observation.
    ln_nu: Vec<f64>,
    /// Some observation has `nu_scaled <= 0`, where [`sbpl_model`] is NaN for
    /// every parameter vector, so every evaluation is the failure sentinel.
    bad_nu: bool,
}

impl SbplCost {
    /// Build the objective for a [`PreparedSource`].
    pub fn new(prep: &PreparedSource) -> Self {
        let observations: Vec<SbplObs> = (0..prep.times.len())
            .map(|i| SbplObs {
                time: prep.times[i],
                nu_scaled: prep.nu_scaled[i],
                flux: prep.flux[i],
                flux_err: prep.flux_err[i],
            })
            .collect();
        SbplCost {
            ln_nu: observations.iter().map(|o| o.nu_scaled.ln()).collect(),
            bad_nu: observations.iter().any(|o| o.nu_scaled <= 0.0),
            observations,
        }
    }

    /// The flattened observations backing this objective.
    pub fn observations(&self) -> &[SbplObs] {
        &self.observations
    }

    /// Evaluate reduced chi² for `p = [alpha1, alpha2, beta, logd, loga, tb, t0]`.
    pub fn eval(&self, p: &[f64]) -> f64 {
        self.chi2::<false>(p).0
    }

    /// [`Self::eval`] together with its analytic gradient with respect to `p`,
    /// in one pass over the data. The gradient is zero wherever the cost is
    /// the `1e10` failure sentinel.
    pub fn eval_with_grad(&self, p: &[f64]) -> (f64, [f64; N_PARAMS]) {
        self.chi2::<true>(p)
    }

    fn chi2<const GRAD: bool>(&self, p: &[f64]) -> (f64, [f64; N_PARAMS]) {
        const FAIL: (f64, [f64; N_PARAMS]) = (1e10, [0.0; N_PARAMS]);
        let (alpha1, alpha2, beta, logd, loga, tb, t0) =
            (p[0], p[1], p[2], p[3], p[4], p[5], p[6]);
        if self.observations.is_empty() || self.bad_nu || tb <= 0.0 {
            return FAIL;
        }

        let d = 10f64.powf(logd);
        let inv_d = 1.0 / d;
        let k = (alpha2 - alpha1) * d;
        let ln_amp = loga * LN_10;

        let mut chi2 = 0.0;
        let mut grad = [0.0; N_PARAMS];
        for (obs, &ln_nu) in self.observations.iter().zip(&self.ln_nu) {
            let err_sq = obs.flux_err * obs.flux_err + 1e-30;
            let tau = obs.time - t0;
            if tau < 0.0 {
                // The model is 0 before t0, independent of every parameter.
                chi2 += obs.flux * obs.flux / err_sq;
                continue;
            }
            let ratio = tau / tb;
            if ratio == 0.0 || !ratio.is_finite() {
                return FAIL;
            }
            let ln_ratio = ratio.ln();
            let q = (ln_ratio * inv_d).exp(); // ratio^(1/D)
            let inner = 0.5 * (1.0 + q);
            if !inner.is_finite() || inner <= 0.0 {
                return FAIL;
            }
            let ln_inner = inner.ln();
            // ln F = ln(10^loga) + beta·ln(nu) + alpha1·ln(ratio) + k·ln(inner)
            let model = (ln_amp + beta * ln_nu + alpha1 * ln_ratio + k * ln_inner).exp();
            if !model.is_finite() {
                return FAIL;
            }
            let residual = obs.flux - model;
            chi2 += residual * residual / err_sq;

            if GRAD {
                // d chi2 / d p = (d chi2 / d ln F) · (d ln F / d p), per point.
                let w = -2.0 * residual * model / err_sq;
                // s = q / (1 + q) = D · d ln(inner) / d ln(ratio)
                let s = 0.5 * q / inner;
                let dlnf_dln_ratio = alpha1 + (alpha2 - alpha1) * s;
                grad[0] += w * (ln_ratio - d * ln_inner);
                grad[1] += w * d * ln_inner;
                grad[2] += w * ln_nu;
                grad[3] += w * LN_10 * (alpha2 - alpha1) * (d * ln_inner - s * ln_ratio);
                grad[4] += w * LN_10;
                grad[5] -= w * dlnf_dln_ratio / tb;
                grad[6] -= w * dlnf_dln_ratio / tau;
            }
        }

        let n = self.observations.len() as f64;
        if GRAD {
            for g in &mut grad {
                *g /= n;
            }
        }
        (chi2 / n, grad)
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
    /// Analytic gradient, chained through the unit-cube map. `unscale` clamps
    /// to the box, so the cost is flat (zero gradient) in any coordinate that
    /// has left it.
    fn gradient(&self, xs: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let (_, grad) = self.inner.eval_with_grad(&self.unscale(xs));
        Ok(xs
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if (0.0..=1.0).contains(&x) {
                    grad[i] * self.scale[i]
                } else {
                    0.0
                }
            })
            .collect())
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

/// Polish a PSO solution with bounded L-BFGS (unit-cube scaled).
///
/// Returns `(start, start_cost)` unchanged when the polish fails to improve —
/// so it is always safe to use the return value directly.
pub fn lbfgs_refine(
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
// Preparation: band map → flattened, normalised fitting problem
// ---------------------------------------------------------------------------

/// One source flattened into the form the optimisers consume: normalised
/// observations plus the search bounds and normalisation constants needed to
/// map fitted parameters back to physical units.
///
/// This is the hand-off point between the CPU and GPU paths — both consume a
/// `PreparedSource` and both produce results through [`assemble_result`], so
/// the two backends optimise exactly the same objective over exactly the same
/// bounds.
#[derive(Debug, Clone)]
pub struct PreparedSource {
    /// Observation times (days), raw — *not* shifted by [`Self::t_ref`].
    pub times: Vec<f64>,
    /// Per-observation effective frequency / 1e15 Hz.
    pub nu_scaled: Vec<f64>,
    /// Flux normalised by [`Self::max_flux`].
    pub flux: Vec<f64>,
    /// Flux error normalised by [`Self::max_flux`].
    pub flux_err: Vec<f64>,
    /// Lower search bound per parameter (see [`PARAM_NAMES`]).
    pub lower: [f64; N_PARAMS],
    /// Upper search bound per parameter.
    pub upper: [f64; N_PARAMS],
    /// Global max flux used to normalise; `loga` is shifted by its log10 on
    /// the way out (see [`assemble_result`]).
    pub max_flux: f64,
    /// Earliest observation time. The model depends only on `t − t0`, so GPU
    /// backends fit in `t − t_ref` space to keep fp32/fp64 conditioning sane
    /// for MJD/JD-scale timestamps, then shift `t0` back by `t_ref`.
    pub t_ref: f64,
    /// Number of usable observations.
    pub n_obs: usize,
    /// Number of bands with a known effective frequency.
    pub n_bands: usize,
    /// Stable RNG key derived from this source's observations (see
    /// [`content_hash`]). The GPU backends seed each source's PSO stream from
    /// it, so a light curve fits the same way whatever else shares its batch
    /// and whatever order the batch is in.
    pub seed_key: u64,
}

/// FNV-1a over the observation content: a stable identity for one prepared
/// source that does not depend on its position in a batch.
fn content_hash(times: &[f64], flux: &[f64], flux_err: &[f64]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x1000_0000_01b3;

    let mut h = OFFSET;
    let mut mix = |v: u64| {
        for byte in v.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(PRIME);
        }
    };
    mix(times.len() as u64);
    for i in 0..times.len() {
        // to_bits, not the float itself: NaN never reaches here (prepare_source
        // filters non-finite rows) and bit patterns hash exactly.
        mix(times[i].to_bits());
        mix(flux[i].to_bits());
        mix(flux_err[i].to_bits());
    }
    h
}

/// Outcome of [`prepare_source`].
#[derive(Debug, Clone)]
pub enum Preparation {
    /// The source has enough data to fit.
    Ready(PreparedSource),
    /// Too little data (< 7 observations, < 2 bands, or zero time span).
    /// `fit_sbpl` returns this empty result as-is.
    Unfittable(SbplResult),
    /// The band map was empty — `fit_sbpl` returns `None` for this.
    Empty,
}

/// Flatten and normalise a band map, and derive the PSO search bounds.
///
/// Split out of [`fit_sbpl`] so that the GPU backends can pack many sources
/// into one device batch without duplicating the preprocessing rules.
pub fn prepare_source(
    bands: &HashMap<String, BandData>,
    config: &PsoConfig,
) -> Preparation {
    if bands.is_empty() {
        return Preparation::Empty;
    }

    // --- Collect observations ------------------------------------------------
    let mut times: Vec<f64> = Vec::new();
    let mut nu_scaled: Vec<f64> = Vec::new();
    let mut flux: Vec<f64> = Vec::new();
    let mut flux_err: Vec<f64> = Vec::new();
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
            let f = band_data.fluxes[i];
            let fe = band_data.flux_errs[i];
            if !t.is_finite() || !f.is_finite() || !fe.is_finite() || fe <= 0.0 {
                continue;
            }
            t_min = t_min.min(t);
            t_max = t_max.max(t);
            times.push(t);
            nu_scaled.push(nu / 1e15);
            flux.push(f);
            flux_err.push(fe);
        }
    }

    let n_obs = times.len();
    if n_obs < 7 || n_bands < 2 {
        return Preparation::Unfittable(SbplResult::empty(n_obs, n_bands));
    }

    let duration = t_max - t_min;
    if duration <= 0.0 {
        return Preparation::Unfittable(SbplResult::empty(n_obs, n_bands));
    }

    // --- Normalise fluxes by global max --------------------------------------
    let max_flux = flux
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_flux = if max_flux > 0.0 { max_flux } else { 1.0 };
    for i in 0..n_obs {
        flux[i] /= max_flux;
        flux_err[i] /= max_flux;
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

    let lower = [
        config.alpha1_min,       // alpha1 (default −10; set 0 for peak model)
        config.alpha2_min,       // alpha2 (default −10)
        -5.0,               // beta
        -3.0,               // logd
        -10.0,              // loga (normalised)
        tb_lower,           // tb — break timescale (days)
        t_min - 2.0 * duration, // t0 — onset, may precede data by 2× span
    ];
    let upper = [
        config.alpha1_max,  // alpha1 (default +10)
        config.alpha2_max,  // alpha2 (default +10; set 0 for peak model)
        5.0,     // beta
        0.0,     // logd
        10.0,    // loga (normalised)
        tb_upper, // tb — break timescale (days)
        t_min,   // t0 — onset must be at or before first observation
    ];

    let seed_key = content_hash(&times, &flux, &flux_err);

    Preparation::Ready(PreparedSource {
        times,
        nu_scaled,
        flux,
        flux_err,
        lower,
        upper,
        max_flux,
        t_ref: t_min,
        n_obs,
        n_bands,
        seed_key,
    })
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
    match prepare_source(bands, config) {
        Preparation::Empty => None,
        Preparation::Unfittable(result) => Some(result),
        Preparation::Ready(prep) => Some(fit_prepared(&prep, config)),
    }
}

/// Run the multi-restart PSO + L-BFGS fit on an already-prepared source.
///
/// This is the CPU half of [`fit_sbpl`]; the GPU backends replace only the PSO
/// stage and then reuse [`lbfgs_refine`] and [`assemble_result`].
pub fn fit_prepared(prep: &PreparedSource, config: &PsoConfig) -> SbplResult {
    let problem = SbplCost::new(prep);
    let mut rng = rand::rngs::SmallRng::seed_from_u64(config.seed);

    // --- Multi-restart PSO + L-BFGS ------------------------------------------
    let mut restarts: Vec<(Vec<f64>, f64)> = Vec::with_capacity(config.n_restarts);

    for _ in 0..config.n_restarts {
        let (pso_best, pso_cost) = pso_search(
            &problem,
            &prep.lower,
            &prep.upper,
            config.n_particles,
            config.n_iters,
            &mut rng,
        );
        restarts.push(lbfgs_refine(
            &problem, pso_best, pso_cost, &prep.lower, &prep.upper,
        ));
    }

    assemble_result(prep, &restarts)
}

/// Turn the per-restart `(params, cost)` pairs into an [`SbplResult`].
///
/// Rescales `loga` out of normalised-flux space, picks the best restart, and
/// reports per-parameter uncertainties as the std dev across restarts. Both
/// the CPU and the GPU paths finish here, so their outputs are directly
/// comparable.
pub fn assemble_result(prep: &PreparedSource, restarts: &[(Vec<f64>, f64)]) -> SbplResult {
    let mut all_params: Vec<Vec<f64>> = Vec::with_capacity(restarts.len());
    let mut best_cost = f64::INFINITY;
    let mut best_params: Vec<f64> = prep.lower.to_vec();

    for (params, cost) in restarts {
        if params.len() != N_PARAMS {
            continue;
        }
        all_params.push(params.clone());
        if *cost < best_cost {
            best_cost = *cost;
            best_params = params.clone();
        }
    }

    if all_params.is_empty() {
        return SbplResult::empty(prep.n_obs, prep.n_bands);
    }

    let (n_obs, n_bands) = (prep.n_obs, prep.n_bands);
    let max_flux = prep.max_flux;

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
    let n_params = N_PARAMS;
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

    SbplResult {
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
    }
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
            Ok(result_to_dict(py, &result, Some(&obs))?.into_any().unbind())
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
                "Unknown band '{band}'. Use 'g', 'r', 'i', 'z', 'y', 'ztfg', 'ztfr', 'ztfi', \
                 'lsstg', 'lsstr', 'lssti', 'lsstz', 'lssty', etc."
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

    /// Which GPU backend this extension was compiled with, or `None`.
    const GPU_BACKEND: Option<&str> = if cfg!(feature = "cuda") {
        Some("cuda")
    } else if cfg!(all(feature = "metal", target_os = "macos")) {
        Some("metal")
    } else {
        None
    };

    /// Fit many ZTF/LSST photometry CSVs in one GPU batch.
    ///
    /// Accepts a single path or a list of paths, and takes the same tuning and
    /// bound keywords as `fit()`. Returns a list of dicts in the same shape
    /// `fit()` returns (`params`, `obs`, `reduced_chi2`, `n_obs`, `n_bands`),
    /// one per input path and in input order. A path that cannot be loaded or
    /// has too little data yields `{"error": "..."}` in its slot instead, so
    /// the output always lines up with the input.
    ///
    /// The backend is chosen at compile time:
    ///   * `--features cuda`  → NVIDIA GPUs
    ///   * `--features metal` → Apple Silicon GPUs (M1–M5)
    /// CUDA wins if both are somehow enabled.
    ///
    /// Batching is the point: one call with 500 light curves is far faster than
    /// 500 calls with one, because the GPU evaluates every source's swarm in a
    /// single dispatch per iteration.
    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[pyfunction]
    #[pyo3(signature = (
        csv_paths,
        n_particles=30, n_iters=200, n_restarts=3, seed=1234,
        alpha1_min=-10.0f64, alpha1_max=10.0f64,
        alpha2_min=-10.0f64, alpha2_max=10.0f64,
        tb_lower=None, tb_upper=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn fit_gpu(
        csv_paths: &Bound<'_, pyo3::types::PyAny>,
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
        // Both backends expose the same types and methods.
        #[cfg(feature = "cuda")]
        use crate::gpu as backend;
        #[cfg(all(feature = "metal", target_os = "macos", not(feature = "cuda")))]
        use crate::gpu_metal as backend;

        use backend::{GpuBatchData, GpuContext};

        let paths: Vec<String> = if let Ok(s) = csv_paths.extract::<String>() {
            vec![s]
        } else {
            csv_paths.extract::<Vec<String>>()?
        };

        let config = PsoConfig {
            n_particles, n_iters, n_restarts, seed,
            alpha1_min, alpha1_max, alpha2_min, alpha2_max,
            tb_lower, tb_upper,
        };

        // Preprocess on the host, keeping each source's original index so
        // failures can be reported in place. The parsed observations are kept
        // for the `obs` key, so each CSV is read exactly once.
        let mut sources = Vec::new();
        let mut slots: Vec<Result<usize, String>> = Vec::with_capacity(paths.len());
        let mut raw_obs: Vec<Option<Vec<Obs>>> = Vec::with_capacity(paths.len());
        for path in &paths {
            let obs = match load_csv(path) {
                Ok(o) => o,
                Err(e) => {
                    slots.push(Err(e));
                    raw_obs.push(None);
                    continue;
                }
            };
            let bands = obs_to_band_map(&obs);
            match prepare_source(&bands, &config) {
                Preparation::Ready(prepared) => {
                    let name = std::path::Path::new(path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("?")
                        .to_string();
                    slots.push(Ok(sources.len()));
                    raw_obs.push(Some(obs));
                    sources.push(backend::SourceData { name, prepared });
                }
                Preparation::Unfittable(r) => {
                    slots.push(Err(format!(
                        "too little data: {} obs across {} bands",
                        r.n_obs, r.n_bands
                    )));
                    raw_obs.push(None);
                }
                Preparation::Empty => {
                    slots.push(Err("no usable bands".to_string()));
                    raw_obs.push(None);
                }
            }
        }

        if sources.is_empty() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "No fittable sources (every path failed preprocessing)",
            ));
        }

        // CUDA: run on a dedicated stream rather than the legacy default one.
        // `_stream` outlives the context and the batch data below.
        #[cfg(feature = "cuda")]
        let _stream = backend::Stream::new_on_device(0)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        #[cfg(feature = "cuda")]
        let gpu = GpuContext::new(0, _stream.as_ptr())
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        #[cfg(all(feature = "metal", target_os = "macos", not(feature = "cuda")))]
        let gpu = GpuContext::new(0).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let batch = GpuBatchData::new(&gpu, &sources)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let results = gpu
            .batch_fit(&batch, &config)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        Python::with_gil(|py| {
            let out = PyList::empty(py);
            for (slot, obs) in slots.iter().zip(raw_obs.iter()) {
                match slot {
                    Err(e) => {
                        let d = PyDict::new(py);
                        d.set_item("error", e.as_str())?;
                        out.append(d)?;
                    }
                    Ok(i) => {
                        let d = result_to_dict(py, &results[*i], obs.as_deref())?;
                        out.append(d)?;
                    }
                }
            }
            Ok(out.into_any().unbind())
        })
    }

    /// Build the result dict returned by `fit()` and `fit_gpu()`, optionally
    /// with the physical-flux observations attached.
    fn result_to_dict<'py>(
        py: Python<'py>,
        result: &SbplResult,
        obs: Option<&[Obs]>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

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

        match result.reduced_chi2 {
            Some(v) => dict.set_item("reduced_chi2", v)?,
            None    => dict.set_item("reduced_chi2", py.None())?,
        }
        dict.set_item("n_obs",   result.n_obs)?;
        dict.set_item("n_bands", result.n_bands)?;

        if let Some(obs) = obs {
            let obs_list = PyList::empty(py);
            for o in obs {
                let d = PyDict::new(py);
                d.set_item("time",     o.time)?;
                d.set_item("flux",     o.flux)?;
                d.set_item("flux_err", o.flux_err)?;
                d.set_item("band",     o.band.as_str())?;
                obs_list.append(d)?;
            }
            dict.set_item("obs", obs_list)?;
        }

        Ok(dict)
    }

    #[pymodule]
    fn sbpl_pso(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(fit, m)?)?;
        m.add_function(wrap_pyfunction!(eval_sbpl, m)?)?;
        #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
        m.add_function(wrap_pyfunction!(fit_gpu, m)?)?;
        // Lets callers branch on the backend without a try/except import dance.
        m.add("GPU_BACKEND", GPU_BACKEND)?;
        Ok(())
    }
}
