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
/// Implementors evaluate every particle of every source in one dispatch.
/// `positions` is `n_sources * n_particles * N_PARAMS` values in row-major
/// order (source-major, then particle, then parameter); `costs` is one value
/// per particle, in the same source/particle order.
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

/// One PSO pass over the whole batch. Returns the per-source best parameter
/// vector (in shifted time space) and its cost.
pub(crate) fn batch_pso<E: BatchCostEvaluator>(
    eval: &E,
    preps: &[&PreparedSource],
    n_particles: usize,
    n_iters: usize,
    seed: u64,
) -> Result<Vec<(Vec<f64>, f64)>, String> {
    let n_sources = preps.len();
    if n_sources == 0 {
        return Ok(Vec::new());
    }
    let n_particles = n_particles.max(1);
    let total = n_sources * n_particles;
    let dim = N_PARAMS;

    let bounds: Vec<([f64; N_PARAMS], [f64; N_PARAMS])> =
        preps.iter().map(|p| shifted_bounds(p)).collect();

    // --- Initialise the swarm (mirrors `pso_search`) --------------------------
    let mut positions = vec![0.0f64; total * dim];
    let mut velocities = vec![0.0f64; total * dim];
    // Keyed by source content, not by batch index: source 3 of 5 and the same
    // source fitted alone draw the identical stream.
    let mut rngs: Vec<SmallRng> = preps
        .iter()
        .map(|p| SmallRng::seed_from_u64(seed ^ p.seed_key))
        .collect();

    for s in 0..n_sources {
        let (lower, upper) = &bounds[s];
        let rng = &mut rngs[s];
        for p in 0..n_particles {
            let base = (s * n_particles + p) * dim;
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

    let mut gbest_pos = vec![0.0f64; n_sources * dim];
    let mut gbest_cost = vec![f64::INFINITY; n_sources];
    for s in 0..n_sources {
        let mut best = 0usize;
        for p in 1..n_particles {
            if pbest_cost[s * n_particles + p] < pbest_cost[s * n_particles + best] {
                best = p;
            }
        }
        let src_base = (s * n_particles + best) * dim;
        gbest_pos[s * dim..s * dim + dim]
            .copy_from_slice(&pbest_pos[src_base..src_base + dim]);
        gbest_cost[s] = pbest_cost[s * n_particles + best];
    }

    // Clerc-Kennedy constriction coefficients (identical to the CPU path).
    let w = 0.7298;
    let c1 = 1.4962;
    let c2 = 1.4962;

    for _ in 0..n_iters {
        // 1. Host: move every particle.
        for s in 0..n_sources {
            let (lower, upper) = &bounds[s];
            let rng = &mut rngs[s];
            let gb = s * dim;
            for p in 0..n_particles {
                let base = (s * n_particles + p) * dim;
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

        // 2. Device: score the whole generation in one dispatch.
        eval.eval_batch(&positions, &mut costs)?;

        // 3. Host: personal and global bests.
        for s in 0..n_sources {
            let gb = s * dim;
            for p in 0..n_particles {
                let idx = s * n_particles + p;
                let cost = costs[idx];
                let base = idx * dim;
                if cost < pbest_cost[idx] {
                    pbest_cost[idx] = cost;
                    pbest_pos[base..base + dim]
                        .copy_from_slice(&positions[base..base + dim]);
                }
                if cost < gbest_cost[s] {
                    gbest_cost[s] = cost;
                    gbest_pos[gb..gb + dim].copy_from_slice(&positions[base..base + dim]);
                }
            }
        }
    }

    Ok((0..n_sources)
        .map(|s| {
            (
                gbest_pos[s * dim..s * dim + dim].to_vec(),
                gbest_cost[s],
            )
        })
        .collect())
}

/// Full batch fit: `config.n_restarts` GPU PSO passes, each polished on the CPU
/// with L-BFGS, assembled into one [`SbplResult`] per source.
///
/// The L-BFGS polish is embarrassingly parallel across sources and cheap next
/// to the PSO, so it runs on the Rayon pool while the GPU is idle between
/// restarts.
pub(crate) fn batch_fit<E: BatchCostEvaluator>(
    eval: &E,
    preps: &[&PreparedSource],
    config: &PsoConfig,
) -> Result<Vec<SbplResult>, String> {
    let n_sources = preps.len();
    if n_sources == 0 {
        return Ok(Vec::new());
    }

    // Objectives are in unshifted time space — the polish and the reported
    // reduced χ² use the same function the CPU path does.
    let problems: Vec<SbplCost> = preps.iter().map(|p| SbplCost::new(p)).collect();

    let mut restarts: Vec<Vec<(Vec<f64>, f64)>> =
        (0..n_sources).map(|_| Vec::with_capacity(config.n_restarts)).collect();

    for r in 0..config.n_restarts {
        // Decorrelate restarts without making a source's stream depend on the
        // batch it landed in.
        let seed = config
            .seed
            .wrapping_add((r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let pso = batch_pso(eval, preps, config.n_particles, config.n_iters, seed)?;

        let polished: Vec<(Vec<f64>, f64)> = pso
            .into_par_iter()
            .zip(preps.par_iter())
            .zip(problems.par_iter())
            .map(|(((mut params, _cost), prep), problem)| {
                // Back to absolute time before touching the CPU objective.
                params[6] += prep.t_ref;
                let cost = problem.eval(&params);
                lbfgs_refine(problem, params, cost, &prep.lower, &prep.upper)
            })
            .collect();

        for (s, pr) in polished.into_iter().enumerate() {
            restarts[s].push(pr);
        }
    }

    Ok(preps
        .iter()
        .zip(restarts.iter())
        .map(|(prep, r)| assemble_result(prep, r))
        .collect())
}
