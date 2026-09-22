// Metal Shading Language port of cuda/sbpl.cu — batched SBPL PSO cost
// evaluation for Apple Silicon GPUs (M1–M5).
//
// One SIMD-group (32 lanes) per (source, particle), exactly like the CUDA
// warp-per-particle layout: the lanes stride over that source's observations
// and reduce the reduced chi-squared with `simd_sum`.
//
// Apple GPUs have no fp64, so everything here is fp32:
//
//   * the per-band likelihood uses Kahan compensated summation, which recovers
//     most of the precision lost on long observation lists;
//   * a model value that overflows fp32 (|exponent| > ~88 in the exp terms)
//     becomes inf and the particle is rejected with the 1e10 sentinel, where
//     the fp64 CUDA path would have returned a finite but astronomically bad
//     chi-squared. Both outcomes discard the particle, and the CPU-side L-BFGS
//     polish that follows the PSO runs in fp64 regardless, so the fitted
//     parameters do not inherit the shader's precision.
//
// Buffer layout (must match gpu_metal.rs):
//   0  all_times        float*  (per-observation, all sources concatenated,
//                                shifted by each source's t_ref)
//   1  all_log10_nu     float*  (log10(nu / 1e15))
//   2  all_flux         float*  (normalised by the source's max flux)
//   3  all_flux_err_sq  float*
//   4  source_offsets   int*    (n_sources + 1)
//   5  positions        float*  (n_sources * n_particles * 7)
//   6  costs            float*  (n_sources * n_particles)
//   7  CostParams       struct  { int n_sources; int n_particles; }

#include <metal_stdlib>
using namespace metal;

constant int   N_PARAMS  = 7;
constant uint  SIMD_SIZE = 32u;

struct CostParams {
    int n_sources;
    int n_particles;
};

// Per-particle constants hoisted out of the observation loop.
struct SbplParams {
    float alpha1;
    float beta;
    float loga;
    float tb;
    float t0;
    float inv_d;   // 1 / 10^logd
    float k;       // (alpha2 - alpha1) * 10^logd
    bool  valid;   // false when tb <= 0
};

inline SbplParams make_params(device const float* p)
{
    SbplParams s;
    s.alpha1 = p[0];
    s.beta   = p[2];
    s.loga   = p[4];
    s.tb     = p[5];
    s.t0     = p[6];
    float d  = exp10(p[3]);
    s.inv_d  = 1.0f / d;
    s.k      = (p[1] - p[0]) * d;
    s.valid  = (s.tb > 0.0f);
    return s;
}

// SBPL flux at one (t, nu) point. Returns 0 before t0 and NaN on domain errors
// — the same sentinels `sbpl_model` in src/lib.rs uses, which the host relies
// on to reject a particle.
inline float sbpl_model(SbplParams s, float t, float log10_nu)
{
    if (!s.valid) {
        return NAN;
    }
    float tau = t - s.t0;
    if (tau < 0.0f) {
        return 0.0f;
    }
    float ratio = tau / s.tb;
    if (ratio == 0.0f || !isfinite(ratio)) {
        return NAN;
    }
    float log_ratio = log(ratio);
    float inner = 0.5f * (1.0f + exp(log_ratio * s.inv_d));   // 0.5 * (1 + ratio^(1/D))
    if (!isfinite(inner) || inner <= 0.0f) {
        return NAN;
    }
    // 10^loga * (nu/1e15)^beta * ratio^alpha1 * inner^((alpha2-alpha1)*D)
    float amplitude = exp10(s.loga + s.beta * log10_nu);
    float result = amplitude * exp(log_ratio * s.alpha1) * exp(s.k * log(inner));
    return isfinite(result) ? result : NAN;
}

// Kahan compensated add: accumulates `x` into `sum` while tracking the running
// rounding error in `comp`.
inline void kahan_add(thread float& sum, thread float& comp, float x)
{
    float y = x - comp;
    float t = sum + y;
    comp = (t - sum) - y;
    sum = t;
}

kernel void batch_pso_cost_sbpl(
    device const float* all_times       [[buffer(0)]],
    device const float* all_log10_nu    [[buffer(1)]],
    device const float* all_flux        [[buffer(2)]],
    device const float* all_flux_err_sq [[buffer(3)]],
    device const int*   source_offsets  [[buffer(4)]],
    device const float* positions       [[buffer(5)]],
    device float*       costs           [[buffer(6)]],
    constant CostParams& params         [[buffer(7)]],
    uint tid                            [[thread_position_in_grid]],
    uint lane                           [[thread_index_in_simdgroup]])
{
    uint particle = tid / SIMD_SIZE;   // SIMD-group-uniform
    int total = params.n_sources * params.n_particles;

    // The guard is SIMD-group-uniform (threadgroup size is a multiple of 32),
    // so every lane reaching simd_sum below is still active.
    if ((int)particle >= total) return;

    int src = (int)particle / params.n_particles;

    SbplParams sp = make_params(positions + particle * (uint)N_PARAMS);

    int obs_start = source_offsets[src];
    int obs_end   = source_offsets[src + 1];

    float chi2 = 0.0f, comp = 0.0f;
    float invalid = 0.0f;

    for (int i = obs_start + (int)lane; i < obs_end; i += (int)SIMD_SIZE) {
        float model = sbpl_model(sp, all_times[i], all_log10_nu[i]);
        if (!isfinite(model)) {
            invalid = 1.0f;
            break;
        }
        float residual = all_flux[i] - model;
        float err_sq = all_flux_err_sq[i] + 1e-30f;
        kahan_add(chi2, comp, residual * residual / err_sq);
    }

    // SIMD-group reduction: sum the chi2 partials, max the invalid flags. Any
    // lane hitting a non-finite model poisons the whole particle, matching the
    // CPU's early return.
    float total_chi2 = simd_sum(chi2);
    float any_invalid = simd_max(invalid);

    if (lane == 0) {
        int n_valid = obs_end - obs_start;
        costs[particle] = (any_invalid > 0.0f || n_valid <= 0)
            ? 1e10f
            : total_chi2 / (float)n_valid;
    }
}
