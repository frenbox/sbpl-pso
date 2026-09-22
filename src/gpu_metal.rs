//! Metal-accelerated batch PSO for the SBPL model (Apple Silicon, M1–M5).
//!
//! The macOS counterpart to [`crate::gpu`]: same batching, same host-side PSO
//! driver ([`crate::gpu_common`]), same CPU L-BFGS polish — only the cost
//! evaluation differs, running in `metal/sbpl.metal` instead of a CUDA kernel.
//!
//! Apple GPUs have no fp64, so the shader computes in fp32 with Kahan-corrected
//! reductions. The host API is f64-in / f64-out, and everything after the PSO
//! (polish, χ², uncertainties) runs in fp64 on the CPU, so the *reported*
//! parameters are not fp32 results — the shader only steers the search.
//!
//! The public surface deliberately matches [`crate::gpu`]:
//!   * [`SourceData`], [`GpuBatchData`], [`GpuContext`]
//!   * [`GpuContext::batch_fit`], [`GpuContext::batch_pso`]
//!   * [`prepare_csv`], [`load_sources`], [`load_sources_with`]
//!
//! The one difference is [`GpuContext::new`], which takes no CUDA stream.

use std::ffi::c_int;
use std::mem::size_of;

use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Library,
    MTLResourceOptions, MTLSize,
};

use crate::gpu_common::{self, BatchCostEvaluator};
use crate::{
    load_csv, obs_to_band_map, prepare_source, Preparation, PreparedSource, PsoConfig,
    SbplResult, N_PARAMS,
};

const KERNEL_SRC: &str = include_str!("../metal/sbpl.metal");
const KERNEL_FN: &str = "batch_pso_cost_sbpl";
/// Threads per threadgroup. Must be a multiple of the 32-lane SIMD-group width
/// so that the `particle >= total` guard in the shader stays SIMD-uniform.
const THREADGROUP: u64 = 256;
/// Must match `SIMD_SIZE` in `metal/sbpl.metal`.
const SIMD_SIZE: u64 = 32;

/// Inline scalar args passed to the kernel via `set_bytes`.
/// Layout must match `struct CostParams` in the .metal file.
#[repr(C)]
#[derive(Copy, Clone)]
struct CostParams {
    n_sources: c_int,
    n_particles: c_int,
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

/// Allocate a shared-storage Metal buffer and copy `data` into it.
/// `StorageModeShared` is the right choice on Apple Silicon — the buffer is
/// visible to both CPU and GPU through unified memory.
fn upload_bytes<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    let bytes = (data.len() * size_of::<T>()) as u64;
    if bytes == 0 {
        // Metal rejects zero-byte buffers; allocate one element so the binding
        // stays valid even when no data is present.
        return device.new_buffer(size_of::<T>() as u64, MTLResourceOptions::StorageModeShared);
    }
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        bytes,
        MTLResourceOptions::StorageModeShared,
    )
}

fn alloc_shared(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes.max(1) as u64, MTLResourceOptions::StorageModeShared)
}

/// Copy `src` into `buf`'s shared-memory backing store. Caller guarantees the
/// buffer is at least `src.len() * size_of::<T>()` bytes.
unsafe fn write_buffer<T: Copy>(buf: &Buffer, src: &[T]) {
    let dst = buf.contents() as *mut T;
    std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
}

/// Borrow `buf`'s shared-memory contents as a `&[T]` of `len` elements.
unsafe fn read_buffer<T: Copy>(buf: &Buffer, len: usize) -> &[T] {
    std::slice::from_raw_parts(buf.contents() as *const T, len)
}

// ---------------------------------------------------------------------------
// Batch data
// ---------------------------------------------------------------------------

/// One prepared light curve, ready for GPU packing.
pub struct SourceData {
    pub name: String,
    pub prepared: PreparedSource,
}

impl AsRef<SourceData> for SourceData {
    fn as_ref(&self) -> &SourceData {
        self
    }
}

/// Packed observation data resident on the GPU.
///
/// Mirrors [`crate::gpu::GpuBatchData`]. Observations are stored as f32 for the
/// shader; the per-source [`PreparedSource`] kept alongside stays f64 and is
/// what the host uses for bounds, polish and rescaling.
pub struct GpuBatchData {
    d_times: Buffer,
    d_log10_nu: Buffer,
    d_flux: Buffer,
    d_flux_err_sq: Buffer,
    d_offsets: Buffer,
    /// Per-source preparation, in device batch order.
    preps: Vec<PreparedSource>,
    /// Source names, in device batch order.
    names: Vec<String>,
    pub n_sources: usize,
    pub n_obs_total: usize,
}

impl GpuBatchData {
    /// Pack and upload prepared sources to the GPU.
    pub fn new<S: AsRef<SourceData>>(ctx: &GpuContext, sources: &[S]) -> Result<Self, String> {
        let n_sources = sources.len();
        let mut all_times: Vec<f32> = Vec::new();
        let mut all_log10_nu: Vec<f32> = Vec::new();
        let mut all_flux: Vec<f32> = Vec::new();
        let mut all_flux_err_sq: Vec<f32> = Vec::new();
        let mut offsets: Vec<c_int> = Vec::with_capacity(n_sources + 1);
        let mut preps = Vec::with_capacity(n_sources);
        let mut names = Vec::with_capacity(n_sources);

        offsets.push(0);
        for src in sources {
            let s = src.as_ref();
            let p = &s.prepared;
            for i in 0..p.times.len() {
                // Shift into t − t_ref space before the f32 narrowing: an
                // MJD/JD timestamp has no usable resolution in f32, a phase
                // does. The host shifts t0 to match.
                all_times.push((p.times[i] - p.t_ref) as f32);
                // The kernel wants log10(nu / 1e15) so the amplitude is one exp10.
                all_log10_nu.push(p.nu_scaled[i].log10() as f32);
                all_flux.push(p.flux[i] as f32);
                all_flux_err_sq.push((p.flux_err[i] * p.flux_err[i]) as f32);
            }
            offsets.push(all_times.len() as c_int);
            preps.push(p.clone());
            names.push(s.name.clone());
        }

        Ok(Self {
            d_times: upload_bytes(&ctx.device, &all_times),
            d_log10_nu: upload_bytes(&ctx.device, &all_log10_nu),
            d_flux: upload_bytes(&ctx.device, &all_flux),
            d_flux_err_sq: upload_bytes(&ctx.device, &all_flux_err_sq),
            d_offsets: upload_bytes(&ctx.device, &offsets),
            preps,
            names,
            n_sources,
            n_obs_total: all_times.len(),
        })
    }

    /// Source names in device batch order (results come back in this order).
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Per-source preparation in device batch order.
    pub fn preps(&self) -> &[PreparedSource] {
        &self.preps
    }
}

// ---------------------------------------------------------------------------
// GPU context
// ---------------------------------------------------------------------------

/// Holds the Metal device, command queue and compiled compute pipeline.
///
/// Constructing this compiles the shader (~100 ms on first call), so build one
/// and reuse it across batches.
pub struct GpuContext {
    device: Device,
    queue: CommandQueue,
    pipeline: ComputePipelineState,
    #[allow(dead_code)]
    library: Library,
}

// Safety: the underlying Objective-C objects (MTLDevice, MTLCommandQueue,
// MTLComputePipelineState, MTLLibrary) are documented thread-safe by Apple.
// Buffers are accessed only while no kernel is running (each dispatch is
// followed by wait_until_completed), so there is no host/device data race.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Construct a context on the system's default Metal device. `_device` is
    /// accepted for API parity with [`crate::gpu::GpuContext`] but ignored —
    /// Apple Silicon Macs have a single integrated GPU.
    pub fn new(_device: i32) -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "No Metal device found (this Mac has no GPU?)".to_string())?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        // Metal enables fast math by default, which lets the compiler assume no
        // NaNs or infinities. This kernel rejects a particle precisely by
        // returning NaN from the model and testing `isfinite`, so that
        // assumption would silently break particle rejection.
        opts.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(KERNEL_SRC, &opts)
            .map_err(|e| format!("Metal shader compile failed: {}", e))?;
        let function = library
            .get_function(KERNEL_FN, None)
            .map_err(|e| format!("Kernel function `{}` not found: {}", KERNEL_FN, e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("Pipeline state creation failed: {}", e))?;

        Ok(Self {
            device,
            queue,
            pipeline,
            library,
        })
    }

    /// No-op on Metal (single GPU on Apple Silicon). Present for API parity.
    pub fn set_device(&self) -> Result<(), String> {
        Ok(())
    }

    /// Always 0 on Apple Silicon — exists for API parity with the CUDA path.
    pub fn device_id(&self) -> i32 {
        0
    }

    /// Fit every source in the batch: `config.n_restarts` GPU PSO passes, each
    /// polished with L-BFGS on the CPU, assembled into one [`SbplResult`] per
    /// source (in [`GpuBatchData::names`] order).
    pub fn batch_fit(
        &self,
        data: &GpuBatchData,
        config: &PsoConfig,
    ) -> Result<Vec<SbplResult>, String> {
        let preps: Vec<&PreparedSource> = data.preps.iter().collect();
        let evaluator = MetalEvaluator::new(self, data, config.n_particles.max(1));
        gpu_common::batch_fit(&evaluator, &preps, config)
    }

    /// Single PSO pass over the batch, without the L-BFGS polish. Returns the
    /// per-source best parameter vector (absolute time, normalised `loga`) and
    /// its reduced χ².
    ///
    /// Mostly useful for benchmarking the device half on its own; prefer
    /// [`Self::batch_fit`].
    pub fn batch_pso(
        &self,
        data: &GpuBatchData,
        config: &PsoConfig,
        seed: u64,
    ) -> Result<Vec<(Vec<f64>, f64)>, String> {
        let preps: Vec<&PreparedSource> = data.preps.iter().collect();
        let evaluator = MetalEvaluator::new(self, data, config.n_particles.max(1));
        let mut out = gpu_common::batch_pso(
            &evaluator,
            &preps,
            config.n_particles,
            config.n_iters,
            seed,
        )?;
        for ((params, _), prep) in out.iter_mut().zip(preps.iter()) {
            params[6] += prep.t_ref;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Device-side cost evaluation
// ---------------------------------------------------------------------------

/// Holds the per-iteration device buffers (positions in, costs out) and the f32
/// staging arrays, and issues one dispatch per generation.
struct MetalEvaluator<'a> {
    ctx: &'a GpuContext,
    data: &'a GpuBatchData,
    d_positions: Buffer,
    d_costs: Buffer,
    /// f32 staging for the f64 host arrays. `RefCell` because
    /// [`BatchCostEvaluator::eval_batch`] takes `&self`.
    scratch: std::cell::RefCell<(Vec<f32>, Vec<f32>)>,
    total_particles: usize,
    n_particles: usize,
    threadgroup_count: MTLSize,
    threads_per_group: MTLSize,
}

impl<'a> MetalEvaluator<'a> {
    fn new(ctx: &'a GpuContext, data: &'a GpuBatchData, n_particles: usize) -> Self {
        let total_particles = data.n_sources * n_particles;
        // One SIMD-group (32 lanes) per particle, THREADGROUP threads per group.
        let n_threads = total_particles as u64 * SIMD_SIZE;
        let n_threadgroups = (n_threads + THREADGROUP - 1) / THREADGROUP;

        Self {
            ctx,
            data,
            d_positions: alloc_shared(
                &ctx.device,
                total_particles * N_PARAMS * size_of::<f32>(),
            ),
            d_costs: alloc_shared(&ctx.device, total_particles * size_of::<f32>()),
            scratch: std::cell::RefCell::new((
                vec![0.0f32; total_particles * N_PARAMS],
                vec![0.0f32; total_particles],
            )),
            total_particles,
            n_particles,
            threadgroup_count: MTLSize::new(n_threadgroups.max(1), 1, 1),
            threads_per_group: MTLSize::new(THREADGROUP, 1, 1),
        }
    }
}

impl BatchCostEvaluator for MetalEvaluator<'_> {
    fn eval_batch(&self, positions: &[f64], costs: &mut [f64]) -> Result<(), String> {
        let mut scratch = self.scratch.borrow_mut();
        let (pos_f32, costs_f32) = &mut *scratch;

        // 1. Narrow positions f64 → f32 and copy into the shared buffer.
        for (dst, &src) in pos_f32.iter_mut().zip(positions.iter()) {
            *dst = src as f32;
        }
        unsafe { write_buffer(&self.d_positions, pos_f32) };

        // 2. Dispatch the cost kernel.
        let cost_params = CostParams {
            n_sources: self.data.n_sources as c_int,
            n_particles: self.n_particles as c_int,
        };
        let cb = self.ctx.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.ctx.pipeline);
        enc.set_buffer(0, Some(&self.data.d_times), 0);
        enc.set_buffer(1, Some(&self.data.d_log10_nu), 0);
        enc.set_buffer(2, Some(&self.data.d_flux), 0);
        enc.set_buffer(3, Some(&self.data.d_flux_err_sq), 0);
        enc.set_buffer(4, Some(&self.data.d_offsets), 0);
        enc.set_buffer(5, Some(&self.d_positions), 0);
        enc.set_buffer(6, Some(&self.d_costs), 0);
        enc.set_bytes(
            7,
            size_of::<CostParams>() as u64,
            &cost_params as *const _ as *const _,
        );
        enc.dispatch_thread_groups(self.threadgroup_count, self.threads_per_group);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        // 3. Read costs back, widen f32 → f64.
        unsafe {
            costs_f32.copy_from_slice(read_buffer::<f32>(&self.d_costs, self.total_particles));
        }
        for (dst, &src) in costs.iter_mut().zip(costs_f32.iter()) {
            *dst = src as f64;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Convenience: load a directory of CSVs
// ---------------------------------------------------------------------------

/// Prepare one CSV for GPU fitting. Sources with too few observations or bands
/// are rejected with an explanatory message.
pub fn prepare_csv(path: &str, config: &PsoConfig) -> Result<SourceData, String> {
    let obs = load_csv(path)?;
    let bands = obs_to_band_map(&obs);
    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    match prepare_source(&bands, config) {
        Preparation::Ready(prepared) => Ok(SourceData { name, prepared }),
        Preparation::Unfittable(r) => Err(format!(
            "too little data: {} obs across {} bands",
            r.n_obs, r.n_bands
        )),
        Preparation::Empty => Err("no usable bands".to_string()),
    }
}

/// Load and prepare every CSV in a directory. Sources that cannot be fit are
/// skipped with a warning on stderr.
pub fn load_sources(data_dir: &str) -> Vec<SourceData> {
    load_sources_with(data_dir, &PsoConfig::default())
}

/// [`load_sources`] with an explicit config (bounds affect preparation).
pub fn load_sources_with(data_dir: &str, config: &PsoConfig) -> Vec<SourceData> {
    let paths = crate::find_csv_files(data_dir);
    let mut sources = Vec::with_capacity(paths.len());
    for path in paths {
        match prepare_csv(&path, config) {
            Ok(s) => sources.push(s),
            Err(e) => eprintln!("SKIP {}: {}", path, e),
        }
    }
    sources
}
