//! Host-side batch PSO driver shared by the CUDA and Metal backends.
//!
//! Both GPUs run the *same* algorithm: the reduced-χ² cost of every particle of
//! every source is evaluated device-side, while the swarm dynamics (velocity,
//! position, personal/global bests) stay on the host. Only the cost evaluation
//! differs between backends, and that difference is hidden behind
//! [`BatchCostEvaluator`] — so this file is the single source of truth for the
//! optimiser itself and the two backends cannot drift apart.
//!
//! Differences from the CPU [`crate::fit_prepared`] path, both deliberate:
//!
//! * **Synchronous swarm.** The CPU PSO updates `gbest` in the middle of a
//!   sweep (a particle sees improvements made by earlier particles in the same
//!   iteration); the GPU evaluates a whole generation at once, so `gbest` is
//!   the one from the end of the previous iteration. Both are standard PSO
//!   variants and converge comparably.
//! * **Per-source RNG streams.** Each source gets its own `SmallRng` seeded
//!   from the config seed, the restart index and the source's own
//!   `seed_key` — a hash of its observations, not its position in the batch.
//!   A light curve therefore fits identically whether it is batched with 5
//!   others or 5000, and re-ordering or dropping sources changes nothing.
//!
//! The restarts do not run one after another: every restart's swarm of every
//! source is scored in the same dispatch (see [`batch_pso_restarts`]), and the
//! polish then runs once over all of them. Because each swarm has its own RNG
//! stream and the kernel scores every particle independently, this returns
//! exactly what sequential restarts would.
//!
//! Everything after the PSO — L-BFGS polish, `loga` rescaling, restart-spread
//! uncertainties — is the CPU code path, reused verbatim.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    assemble_result, lbfgs_refine, PreparedSource, PsoConfig, SbplCost, SbplResult, N_PARAMS,
};

/// Device-side evaluation of the batch cost function.
///
/// Implementors evaluate every particle of every source in one dispatch, for
/// the particles-per-source count they were built with (the `make_eval`
/// argument of [`batch_pso_restarts`] and [`batch_fit`]). `positions` is
/// `n_sources * particles_per_source * N_PARAMS` values in row-major order
/// (source-major, then particle, then parameter); `costs` is one value per
/// particle, in the same source/particle order.
///
/// Positions are in *shifted time space*: dimension 6 (`t0`) is relative to
/// each source's `t_ref` (see [`shifted_bounds`]).
pub(crate) trait BatchCostEvaluator {
    fn eval_batch(&self, positions: &[f64], costs: &mut [f64]) -> Result<(), String>;
}

/// PSO search bounds for one source, with `t0` shifted into `t − t_ref` space.
pub(crate) fn shifted_bounds(prep: &PreparedSource) -> ([f64; N_PARAMS], [f64; N_PARAMS]) {
    let mut lower = prep.lower;
    let mut upper = prep.upper;
    lower[6] -= prep.t_ref;
    upper[6] -= prep.t_ref;
    (lower, upper)
}

/// Seed of restart `r`: decorrelates restarts without making a source's
/// stream depend on the batch it landed in.
fn restart_seed(seed: u64, r: usize) -> u64 {
    seed.wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// One PSO pass over the whole batch. Returns the per-source best parameter
/// vector (in shifted time space) and its cost.
pub(crate) fn batch_pso<E: BatchCostEvaluator>(
    make_eval: impl FnOnce(usize) -> Result<E, String>,
    preps: &[&PreparedSource],
    n_particles: usize,
    n_iters: usize,
    seed: u64,
) -> Result<Vec<(Vec<f64>, f64)>, String> {
    batch_pso_restarts(make_eval, preps, n_particles, n_iters, &[seed])
}

/// `seeds.len()` independent PSO passes over the whole batch, run side by
/// side: one swarm per (source, seed), with every swarm scored in the same
/// dispatch each iteration.
///
/// Swarm `r` of a source is exactly the swarm a single [`batch_pso`] pass with
/// `seeds[r]` would run, so this returns what `seeds.len()` separate passes
/// would, with one dispatch and sync per iteration instead of `seeds.len()`.
///
/// The device sees `seeds.len() * n_particles` particles per source, laid out
/// source-major, then swarm, then particle; `make_eval` is called once with
/// that count to build the evaluator. Returns each swarm's best parameter
/// vector (in shifted time space) and cost, at index
/// `source * seeds.len() + r`.
pub(crate) fn batch_pso_restarts<E: BatchCostEvaluator>(
    make_eval: impl FnOnce(usize) -> Result<E, String>,
    preps: &[&PreparedSource],
    n_particles: usize,
    n_iters: usize,
    seeds: &[u64],
) -> Result<Vec<(Vec<f64>, f64)>, String> {
    let n_sources = preps.len();
    let swarms_per_source = seeds.len();
    if n_sources == 0 || swarms_per_source == 0 {
        return Ok(Vec::new());
    }
    let n_particles = n_particles.max(1);
    let eval = make_eval(swarms_per_source * n_particles)?;
    let n_swarms = n_sources * swarms_per_source;
    let total = n_swarms * n_particles;
    let dim = N_PARAMS;

    let bounds: Vec<([f64; N_PARAMS], [f64; N_PARAMS])> =
        preps.iter().map(|p| shifted_bounds(p)).collect();
    let bounds_of = |sw: usize| &bounds[sw / swarms_per_source];

    // --- Initialise the swarms (mirrors `pso_search`) -------------------------
    let mut positions = vec![0.0f64; total * dim];
    let mut velocities = vec![0.0f64; total * dim];
    // Keyed by source content, not by batch index: source 3 of 5 and the same
    // source fitted alone draw the identical stream.
    let mut rngs: Vec<SmallRng> = preps
        .iter()
        .flat_map(|p| {
            seeds
                .iter()
                .map(move |&seed| SmallRng::seed_from_u64(seed ^ p.seed_key))
        })
        .collect();

    for sw in 0..n_swarms {
        let (lower, upper) = bounds_of(sw);
        let rng = &mut rngs[sw];
        for p in 0..n_particles {
            let base = (sw * n_particles + p) * dim;
            for d in 0..dim {
                let span = upper[d] - lower[d];
                positions[base + d] = if span > 0.0 {
                    rng.random_range(lower[d]..upper[d])
                } else {
                    lower[d]
                };
                velocities[base + d] = if span > 0.0 {
                    rng.random_range(-0.1 * span..0.1 * span)
                } else {
                    0.0
                };
            }
        }
    }

    let mut costs = vec![f64::INFINITY; total];
    eval.eval_batch(&positions, &mut costs)?;

    let mut pbest_pos = positions.clone();
    let mut pbest_cost = costs.clone();

    let mut gbest_pos = vec![0.0f64; n_swarms * dim];
    let mut gbest_cost = vec![f64::INFINITY; n_swarms];
    for sw in 0..n_swarms {
        let mut best = 0usize;
        for p in 1..n_particles {
            if pbest_cost[sw * n_particles + p] < pbest_cost[sw * n_particles + best] {
                best = p;
            }
        }
        let src_base = (sw * n_particles + best) * dim;
        gbest_pos[sw * dim..sw * dim + dim]
            .copy_from_slice(&pbest_pos[src_base..src_base + dim]);
        gbest_cost[sw] = pbest_cost[sw * n_particles + best];
    }

    // Clerc-Kennedy constriction coefficients (identical to the CPU path).
    let w = 0.7298;
    let c1 = 1.4962;
    let c2 = 1.4962;

    for _ in 0..n_iters {
        // 1. Host: move every particle.
        for sw in 0..n_swarms {
            let (lower, upper) = bounds_of(sw);
            let rng = &mut rngs[sw];
            let gb = sw * dim;
            for p in 0..n_particles {
                let base = (sw * n_particles + p) * dim;
                for d in 0..dim {
                    let r1: f64 = rng.random();
                    let r2: f64 = rng.random();
                    let v = w * velocities[base + d]
                        + c1 * r1 * (pbest_pos[base + d] - positions[base + d])
                        + c2 * r2 * (gbest_pos[gb + d] - positions[base + d]);
                    velocities[base + d] = v;
                    positions[base + d] =
                        (positions[base + d] + v).clamp(lower[d], upper[d]);
                }
            }
        }

        // 2. Device: score the whole generation, every swarm, in one dispatch.
        eval.eval_batch(&positions, &mut costs)?;

        // 3. Host: personal and global bests.
        for sw in 0..n_swarms {
            let gb = sw * dim;
            for p in 0..n_particles {
                let idx = sw * n_particles + p;
                let cost = costs[idx];
                let base = idx * dim;
                if cost < pbest_cost[idx] {
                    pbest_cost[idx] = cost;
                    pbest_pos[base..base + dim]
                        .copy_from_slice(&positions[base..base + dim]);
                }
                if cost < gbest_cost[sw] {
                    gbest_cost[sw] = cost;
                    gbest_pos[gb..gb + dim].copy_from_slice(&positions[base..base + dim]);
                }
            }
        }
    }

    Ok((0..n_swarms)
        .map(|sw| {
            (
                gbest_pos[sw * dim..sw * dim + dim].to_vec(),
                gbest_cost[sw],
            )
        })
        .collect())
}

/// Full batch fit: all `config.n_restarts` PSO swarms of every source run side
/// by side on the GPU ([`batch_pso_restarts`]), then every restart is polished
/// on the CPU with L-BFGS and assembled into one [`SbplResult`] per source.
///
/// The polish is embarrassingly parallel across (source, restart) pairs and
/// runs as a single Rayon pass over all of them. One pass rather than one per
/// restart matters because a batch's longest light curve bounds each pass: its
/// restarts now polish concurrently instead of back to back.
pub(crate) fn batch_fit<E: BatchCostEvaluator>(
    make_eval: impl FnOnce(usize) -> Result<E, String>,
    preps: &[&PreparedSource],
    config: &PsoConfig,
) -> Result<Vec<SbplResult>, String> {
    let n_sources = preps.len();
    if n_sources == 0 {
        return Ok(Vec::new());
    }
    let n_restarts = config.n_restarts;

    // Objectives are in unshifted time space — the polish and the reported
    // reduced χ² use the same function the CPU path does.
    let problems: Vec<SbplCost> = preps.iter().map(|p| SbplCost::new(p)).collect();

    let seeds: Vec<u64> = (0..n_restarts)
        .map(|r| restart_seed(config.seed, r))
        .collect();
    let pso = batch_pso_restarts(
        make_eval,
        preps,
        config.n_particles,
        config.n_iters,
        &seeds,
    )?;

    // Jobs are restart-major, so one source's restarts sit far apart in the
    // job list and Rayon hands them to different threads.
    let polished: Vec<(Vec<f64>, f64)> = (0..n_restarts * n_sources)
        .into_par_iter()
        .map(|job| {
            let (r, s) = (job / n_sources, job % n_sources);
            let mut params = pso[s * n_restarts + r].0.clone();
            // Back to absolute time before touching the CPU objective.
            params[6] += preps[s].t_ref;
            let cost = problems[s].eval(&params);
            lbfgs_refine(&problems[s], params, cost, &preps[s].lower, &preps[s].upper)
        })
        .collect();

    Ok(preps
        .iter()
        .enumerate()
        .map(|(s, prep)| {
            let restarts: Vec<(Vec<f64>, f64)> = (0..n_restarts)
                .map(|r| polished[r * n_sources + s].clone())
                .collect();
            assemble_result(prep, &restarts)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use super::*;
    use crate::{band_frequency_hz, prepare_source, sbpl_model, BandData, Preparation};

    /// Stands in for a device kernel: scores each particle with the CPU
    /// objective and counts dispatches.
    struct CpuEvaluator<'a> {
        preps: &'a [&'a PreparedSource],
        problems: Vec<SbplCost>,
        particles_per_source: usize,
        dispatches: &'a Cell<usize>,
    }

    impl<'a> CpuEvaluator<'a> {
        fn new(
            preps: &'a [&'a PreparedSource],
            particles_per_source: usize,
            dispatches: &'a Cell<usize>,
        ) -> Self {
            Self {
                preps,
                problems: preps.iter().map(|p| SbplCost::new(p)).collect(),
                particles_per_source,
                dispatches,
            }
        }
    }

    impl BatchCostEvaluator for CpuEvaluator<'_> {
        fn eval_batch(&self, positions: &[f64], costs: &mut [f64]) -> Result<(), String> {
            assert_eq!(costs.len(), self.preps.len() * self.particles_per_source);
            assert_eq!(positions.len(), costs.len() * N_PARAMS);
            self.dispatches.set(self.dispatches.get() + 1);
            for (i, (cost, x)) in costs.iter_mut().zip(positions.chunks(N_PARAMS)).enumerate() {
                let s = i / self.particles_per_source;
                let mut p = x.to_vec();
                p[6] += self.preps[s].t_ref;
                *cost = self.problems[s].eval(&p);
            }
            Ok(())
        }
    }

    /// Two-band SBPL light curve on a JD-scale time axis, with deterministic
    /// multiplicative noise.
    fn synthetic(seed: u64, tb: f64, t0: f64) -> PreparedSource {
        let mut state = seed;
        let mut noise = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut bands = HashMap::new();
        for name in ["g", "r"] {
            let nu_scaled = band_frequency_hz(name).unwrap() / 1e15;
            let mut bd = BandData::new();
            for j in 0..15 {
                let t = t0 + 0.5 + j as f64 * tb / 3.0;
                let f = sbpl_model(t, nu_scaled, 1.5, -2.0, -0.6, -0.5, 2.0, tb, t0);
                bd.times.push(t);
                bd.fluxes.push(f * (1.0 + 0.1 * noise()));
                bd.flux_errs.push(0.05 * f);
            }
            bands.insert(name.to_string(), bd);
        }
        match prepare_source(&bands, &PsoConfig::default()) {
            Preparation::Ready(p) => p,
            other => panic!("synthetic source not fittable: {other:?}"),
        }
    }

    fn sources() -> Vec<PreparedSource> {
        vec![
            synthetic(1, 12.0, 2_458_200.0),
            synthetic(2, 30.0, 2_458_310.0),
            synthetic(3, 5.0, 2_458_005.0),
        ]
    }

    fn config() -> PsoConfig {
        PsoConfig {
            n_particles: 8,
            n_iters: 15,
            n_restarts: 3,
            ..PsoConfig::default()
        }
    }

    #[test]
    fn side_by_side_restarts_match_separate_passes() {
        let owned = sources();
        let preps: Vec<&PreparedSource> = owned.iter().collect();
        let config = config();
        let seeds: Vec<u64> = (0..config.n_restarts)
            .map(|r| restart_seed(config.seed, r))
            .collect();

        let dispatches = Cell::new(0);
        let together = batch_pso_restarts(
            |n| Ok(CpuEvaluator::new(&preps, n, &dispatches)),
            &preps,
            config.n_particles,
            config.n_iters,
            &seeds,
        )
        .unwrap();
        assert_eq!(dispatches.get(), config.n_iters + 1);

        for (r, &seed) in seeds.iter().enumerate() {
            let alone = batch_pso(
                |n| Ok(CpuEvaluator::new(&preps, n, &dispatches)),
                &preps,
                config.n_particles,
                config.n_iters,
                seed,
            )
            .unwrap();
            for s in 0..preps.len() {
                assert_eq!(
                    together[s * seeds.len() + r], alone[s],
                    "source {s}, restart {r}: side-by-side swarm diverged"
                );
            }
        }
    }

    /// `batch_fit` must give exactly what the old driver did: one PSO pass and
    /// one polish per restart, in turn.
    #[test]
    fn batch_fit_matches_sequential_restarts() {
        let owned = sources();
        let preps: Vec<&PreparedSource> = owned.iter().collect();
        let config = config();

        let dispatches = Cell::new(0);
        let fitted = batch_fit(
            |n| Ok(CpuEvaluator::new(&preps, n, &dispatches)),
            &preps,
            &config,
        )
        .unwrap();
        assert_eq!(dispatches.get(), config.n_iters + 1);

        let mut restarts: Vec<Vec<(Vec<f64>, f64)>> = vec![Vec::new(); preps.len()];
        for r in 0..config.n_restarts {
            let pso = batch_pso(
                |n| Ok(CpuEvaluator::new(&preps, n, &dispatches)),
                &preps,
                config.n_particles,
                config.n_iters,
                restart_seed(config.seed, r),
            )
            .unwrap();
            for (s, (mut params, _)) in pso.into_iter().enumerate() {
                params[6] += preps[s].t_ref;
                let problem = SbplCost::new(preps[s]);
                let cost = problem.eval(&params);
                restarts[s].push(lbfgs_refine(
                    &problem,
                    params,
                    cost,
                    &preps[s].lower,
                    &preps[s].upper,
                ));
            }
        }
        for (s, prep) in preps.iter().enumerate() {
            let expected = assemble_result(prep, &restarts[s]);
            assert_eq!(
                format!("{:?}", fitted[s]),
                format!("{expected:?}"),
                "source {s} fitted differently"
            );
        }
    }
}
