// CUDA kernel for batched SBPL PSO cost evaluation.
//
// One *warp* per (source, particle): the 32 lanes stride over that source's
// observations, evaluate the smoothly-broken power law, and warp-reduce the
// reduced chi-squared — a device-side transcription of `SbplCost::eval` in
// src/lib.rs, including its failure sentinels.
//
// A warp rather than a single thread because light-curve lengths vary by three
// orders of magnitude in real batches (7 to ~11 000 points here). With one
// thread per particle the longest source serialises the whole launch; striding
// a warp over the observations spreads that source across 32x more lanes and
// keeps the loads coalesced.
//
// Model:
//   F(t, nu) = 10^loga * nu_scaled^beta * tau^alpha1
//              * [0.5 * (1 + tau^(1/D))]^((alpha2 - alpha1) * D)
//   tau = (t - t0) / tb,  D = 10^logd
//
// Times are passed shifted by each source's t_ref (the earliest observation),
// with t0 shifted to match, so `t - t0` never loses precision to an
// MJD/JD-scale offset. Frequencies arrive pre-logged as log10(nu / 1e15).

#include <math.h>

#define N_PARAMS 7
#define WARP_SIZE 32
#define FULL_MASK 0xffffffffu

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Per-particle constants hoisted out of the observation loop.
struct SbplParams {
    double alpha1;
    double beta;
    double loga;
    double tb;
    double t0;
    double d;        // 10^logd
    double inv_d;    // 1 / d
    double k;        // (alpha2 - alpha1) * d
    bool   valid;    // false when tb <= 0 (whole particle is invalid)
};

__device__ inline SbplParams make_params(const double* p)
{
    SbplParams s;
    s.alpha1 = p[0];
    s.beta   = p[2];
    s.loga   = p[4];
    s.tb     = p[5];
    s.t0     = p[6];
    s.d      = exp10(p[3]);
    s.inv_d  = 1.0 / s.d;
    s.k      = (p[1] - p[0]) * s.d;
    s.valid  = (s.tb > 0.0);
    return s;
}

// SBPL flux at one (t, nu) point, given `log10_nu = log10(nu / 1e15)`.
// Returns 0 before t0 and NaN on domain errors — the same value and the same
// failure sentinels as `sbpl_model` in src/lib.rs, which the host relies on to
// reject a particle.
//
// The powers are evaluated as exp/log rather than pow(): the two ratio powers
// share a single log(ratio), and the amplitude folds 10^loga and nu^beta into
// one exp10. That is six transcendentals per point instead of nine, which
// matters because fp64 transcendentals are the whole cost of this kernel.
// The expression is algebraically identical; only the last ulp differs.
__device__ inline double sbpl_model(const SbplParams& s, double t, double log10_nu)
{
    if (!s.valid) {
        return nan("");
    }
    double tau = t - s.t0;
    if (tau < 0.0) {
        return 0.0;
    }
    double ratio = tau / s.tb;
    if (ratio == 0.0 || !isfinite(ratio)) {
        return nan("");
    }
    double log_ratio = log(ratio);
    double inner = 0.5 * (1.0 + exp(log_ratio * s.inv_d));   // 0.5 * (1 + ratio^(1/D))
    if (!isfinite(inner) || inner <= 0.0) {
        return nan("");
    }
    // 10^loga * (nu/1e15)^beta * ratio^alpha1 * inner^((alpha2-alpha1)*D)
    double amplitude = exp10(s.loga + s.beta * log10_nu);
    double result = amplitude * exp(log_ratio * s.alpha1) * exp(s.k * log(inner));
    return isfinite(result) ? result : nan("");
}

// ---------------------------------------------------------------------------
// Batch PSO cost kernel
// ---------------------------------------------------------------------------
//
// Thread grid: n_sources * n_particles warps, one cost per warp.
//
// Parameter layout per particle (7 doubles, row-major):
//   [alpha1, alpha2, beta, logd, loga, tb, t0]
//
// Observation data is concatenated across sources; source_offsets[src] is the
// start index and source_offsets[src + 1] the end.

extern "C" __global__ void batch_pso_cost_sbpl(
    const double* __restrict__ all_times,
    const double* __restrict__ all_log10_nu,
    const double* __restrict__ all_flux,
    const double* __restrict__ all_flux_err_sq,
    const int*    __restrict__ source_offsets,
    const double* __restrict__ positions,
    double*       __restrict__ costs,
    int n_sources,
    int n_particles)
{
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int particle = global_thread / WARP_SIZE;   // warp-uniform
    int lane = global_thread % WARP_SIZE;

    // The guard is warp-uniform, so every lane that reaches the shuffles below
    // is still active — required for the full-mask warp reduction.
    if (particle >= n_sources * n_particles) return;

    int src = particle / n_particles;

    SbplParams sp = make_params(positions + (long long)particle * N_PARAMS);

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];

    double chi2 = 0.0;
    int invalid = 0;

    for (int i = obs_start + lane; i < obs_end; i += WARP_SIZE) {
        double model = sbpl_model(sp, all_times[i], all_log10_nu[i]);
        if (!isfinite(model)) {
            invalid = 1;
            break;
        }
        double residual = all_flux[i] - model;
        double err_sq = all_flux_err_sq[i] + 1e-30;
        chi2 += residual * residual / err_sq;
    }

    // Warp reduction: sum the chi2 partials, OR the invalid flags. Any lane
    // hitting a non-finite model poisons the whole particle, matching the CPU's
    // early return.
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        chi2    += __shfl_down_sync(FULL_MASK, chi2, off);
        invalid |= __shfl_down_sync(FULL_MASK, invalid, off);
    }

    if (lane == 0) {
        int n_valid = obs_end - obs_start;
        costs[particle] = (invalid || n_valid <= 0) ? 1e10 : chi2 / (double)n_valid;
    }
}

// ---------------------------------------------------------------------------
// Host-side launch wrapper (callable from Rust via FFI)
// ---------------------------------------------------------------------------

extern "C" void launch_batch_pso_cost_sbpl(
    const double* all_times,
    const double* all_log10_nu,
    const double* all_flux,
    const double* all_flux_err_sq,
    const int*    source_offsets,
    const double* positions,
    double*       costs,
    int n_sources,
    int n_particles,
    int grid,
    int block,
    cudaStream_t stream)
{
    batch_pso_cost_sbpl<<<grid, block, 0, stream>>>(
        all_times, all_log10_nu, all_flux, all_flux_err_sq,
        source_offsets, positions, costs,
        n_sources, n_particles);
}
