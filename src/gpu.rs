//! CUDA-accelerated batch PSO for the SBPL model.
//!
//! Many light curves are packed into one device-resident batch and fitted
//! simultaneously: the reduced-χ² cost of every particle of every source is
//! evaluated on the GPU (`cuda/sbpl.cu`), while the swarm dynamics and the
//! L-BFGS polish stay on the CPU (see [`crate::gpu_common`]).
//!
//! Typical use:
//!
//! ```no_run
//! use sbpl_pso::gpu::{load_sources, GpuBatchData, GpuContext, Stream};
//! use sbpl_pso::PsoConfig;
//!
//! let sources = load_sources("afterglow_photometry");
//! let stream = Stream::new_on_device(0).unwrap();
//! let gpu = GpuContext::new(0, stream.as_ptr()).unwrap();
//! let batch = GpuBatchData::new(&gpu, &sources).unwrap();
//! let results = gpu.batch_fit(&batch, &PsoConfig::default()).unwrap();
//! ```

use std::ffi::{c_int, c_void};
use std::mem::{size_of, size_of_val};
use std::ptr;

use crate::gpu_common::{self, BatchCostEvaluator};
use crate::{
    obs_to_band_map, load_csv, prepare_source, Preparation, PreparedSource, PsoConfig,
    SbplResult, N_PARAMS,
};

// ---------------------------------------------------------------------------
// CUDA FFI
// ---------------------------------------------------------------------------

type CudaResult = c_int;
/// Opaque CUDA stream handle. CUDA defines `cudaStream_t` as a pointer to an
/// incomplete struct, which is `*mut c_void` from the Rust side. A null
/// pointer means the legacy default stream.
pub type CudaStream = *mut c_void;

extern "C" {
    fn cudaSetDevice(device: c_int) -> CudaResult;
    fn cudaMalloc(devPtr: *mut *mut u8, size: usize) -> CudaResult;
    fn cudaFree(devPtr: *mut u8) -> CudaResult;
    fn cudaMemcpyAsync(
        dst: *mut u8,
        src: *const u8,
        count: usize,
        kind: c_int,
        stream: CudaStream,
    ) -> CudaResult;
    fn cudaStreamCreate(stream: *mut CudaStream) -> CudaResult;
    fn cudaStreamDestroy(stream: CudaStream) -> CudaResult;
    fn cudaStreamSynchronize(stream: CudaStream) -> CudaResult;
    fn cudaGetLastError() -> CudaResult;
    fn cudaGetErrorString(error: CudaResult) -> *const i8;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

extern "C" {
    fn launch_batch_pso_cost_sbpl(
        all_times: *const f64,
        all_log10_nu: *const f64,
        all_flux: *const f64,
        all_flux_err_sq: *const f64,
        source_offsets: *const c_int,
        positions: *const f64,
        costs: *mut f64,
        n_sources: c_int,
        n_particles: c_int,
        grid: c_int,
        block: c_int,
        stream: CudaStream,
    );
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

fn cuda_check(code: CudaResult) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = cudaGetErrorString(code);
            if ptr.is_null() {
                "unknown CUDA error".to_string()
            } else {
                std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(format!("CUDA error {}: {}", code, msg))
    }
}

struct DevBuf {
    ptr: *mut u8,
    #[allow(dead_code)]
    size: usize,
}

// Safety: CUDA device pointers can be used from any host thread provided
// cudaSetDevice is called first. GpuContext::set_device / batch_pso handle this.
unsafe impl Send for DevBuf {}

impl DevBuf {
    fn alloc(size: usize) -> Result<Self, String> {
        let mut ptr: *mut u8 = ptr::null_mut();
        cuda_check(unsafe { cudaMalloc(&mut ptr, size.max(1)) })?;
        Ok(Self { ptr, size })
    }

    fn upload<T>(data: &[T], stream: CudaStream) -> Result<Self, String> {
        let bytes = size_of_val(data);
        let buf = Self::alloc(bytes)?;
        if bytes > 0 {
            cuda_check(unsafe {
                cudaMemcpyAsync(
                    buf.ptr,
                    data.as_ptr() as *const u8,
                    bytes,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    stream,
                )
            })?;
        }
        Ok(buf)
    }

    fn download_into<T>(&self, host: &mut [T], stream: CudaStream) -> Result<(), String> {
        let bytes = size_of_val(host);
        cuda_check(unsafe {
            cudaMemcpyAsync(
                host.as_mut_ptr() as *mut u8,
                self.ptr,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        })?;
        // Callers pair this with cudaStreamSynchronize on the same stream
        // before reading `host`, so we don't sync here.
        Ok(())
    }

    fn upload_from<T>(&self, data: &[T], stream: CudaStream) -> Result<(), String> {
        let bytes = size_of_val(data);
        cuda_check(unsafe {
            cudaMemcpyAsync(
                self.ptr,
                data.as_ptr() as *const u8,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        })
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        unsafe {
            cudaFree(self.ptr);
        }
    }
}

/// RAII wrapper around a CUDA stream. Calls `cudaStreamCreate` on construction
/// and `cudaStreamDestroy` on drop. Pass `stream.as_ptr()` into
/// [`GpuContext::new`], or share the raw pointer with another library that
/// expects `cudaStream_t` (e.g. ONNX Runtime's `with_compute_stream`).
///
/// The owner must ensure the [`Stream`] outlives every `GpuContext` and any
/// foreign session told to use it. Field-declaration order in the owning struct
/// controls drop order.
pub struct Stream {
    raw: CudaStream,
}

// Safety: CUDA stream handles are usable from any host thread; concurrent
// submissions to a stream are serialized internally by the CUDA driver.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Create a new CUDA stream on the currently-bound device.
    pub fn new() -> Result<Self, String> {
        let mut raw: CudaStream = ptr::null_mut();
        cuda_check(unsafe { cudaStreamCreate(&mut raw) })?;
        Ok(Self { raw })
    }

    /// Create a stream on the given device, binding the calling thread to it
    /// first.
    pub fn new_on_device(device: i32) -> Result<Self, String> {
        cuda_check(unsafe { cudaSetDevice(device) })?;
        Self::new()
    }

    /// Raw `cudaStream_t` pointer, for FFI consumers.
    pub fn as_ptr(&self) -> CudaStream {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                cudaStreamDestroy(self.raw);
            }
        }
    }
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
/// Holds one device copy of every source's observations plus the per-source
/// [`PreparedSource`] metadata the host needs to bound the search and rescale
/// the results. Build it once and reuse it across restarts and configs.
pub struct GpuBatchData {
    d_times: DevBuf,
    d_log10_nu: DevBuf,
    d_flux: DevBuf,
    d_flux_err_sq: DevBuf,
    d_offsets: DevBuf,
    /// Per-source preparation, in device batch order.
    preps: Vec<PreparedSource>,
    /// Source names, in device batch order.
    names: Vec<String>,
    pub n_sources: usize,
    pub n_obs_total: usize,
}

impl GpuBatchData {
    /// Pack and upload prepared sources to the GPU. Uploads run async on
    /// `ctx`'s stream, so they are ordered before any kernel later submitted to
    /// the same stream.
    pub fn new<S: AsRef<SourceData>>(ctx: &GpuContext, sources: &[S]) -> Result<Self, String> {
        ctx.set_device()?;

        let n_sources = sources.len();
        let mut all_times: Vec<f64> = Vec::new();
        let mut all_log10_nu: Vec<f64> = Vec::new();
        let mut all_flux: Vec<f64> = Vec::new();
        let mut all_flux_err_sq: Vec<f64> = Vec::new();
        let mut offsets: Vec<c_int> = Vec::with_capacity(n_sources + 1);
        let mut preps = Vec::with_capacity(n_sources);
        let mut names = Vec::with_capacity(n_sources);

        offsets.push(0);
        for src in sources {
            let s = src.as_ref();
            let p = &s.prepared;
            for i in 0..p.times.len() {
                // Shift into t − t_ref space; the host shifts t0 to match.
                all_times.push(p.times[i] - p.t_ref);
                // The kernel wants log10(nu / 1e15) so the amplitude is one exp10.
                all_log10_nu.push(p.nu_scaled[i].log10());
                all_flux.push(p.flux[i]);
                all_flux_err_sq.push(p.flux_err[i] * p.flux_err[i]);
            }
            offsets.push(all_times.len() as c_int);
            preps.push(p.clone());
            names.push(s.name.clone());
        }

        let stream = ctx.stream();
        let batch = Self {
            d_times: DevBuf::upload(&all_times, stream)?,
            d_log10_nu: DevBuf::upload(&all_log10_nu, stream)?,
            d_flux: DevBuf::upload(&all_flux, stream)?,
            d_flux_err_sq: DevBuf::upload(&all_flux_err_sq, stream)?,
            d_offsets: DevBuf::upload(&offsets, stream)?,
            preps,
            names,
            n_sources,
            n_obs_total: all_times.len(),
        };
        // Block until the uploads finish so the staging Vecs can be dropped.
        // Ordering is on the same stream, so this cost lands here at upload
        // time rather than on the per-iteration hot path.
        cuda_check(unsafe { cudaStreamSynchronize(stream) })?;
        Ok(batch)
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

pub struct GpuContext {
    device: i32,
    /// Stream on which all CUDA work for this context is submitted. May be null
    /// (default stream). Lifetime is the caller's responsibility — see
    /// [`Stream`].
    stream: CudaStream,
}

// Safety: GpuContext is a device ID plus a CUDA stream pointer. Each method
// calls cudaSetDevice before any CUDA work, and stream submissions are
// thread-safe, so it is safe to Send/Sync across threads.
unsafe impl Send for GpuContext {}
unsafe impl Sync for GpuContext {}

impl GpuContext {
    /// Construct a context bound to `device`, submitting work on `stream`.
    /// Pass `std::ptr::null_mut()` for the legacy default stream.
    ///
    /// # Safety
    /// `stream` must be a valid `cudaStream_t` belonging to `device` (or null),
    /// and must outlive this `GpuContext` and any [`GpuBatchData`] built from
    /// it.
    pub fn new(device: i32, stream: CudaStream) -> Result<Self, String> {
        cuda_check(unsafe { cudaSetDevice(device) })?;
        Ok(Self { device, stream })
    }

    /// Bind the calling thread to this context's GPU device. Call this before
    /// [`GpuBatchData::new`] when using multi-GPU.
    pub fn set_device(&self) -> Result<(), String> {
        cuda_check(unsafe { cudaSetDevice(self.device) })
    }

    /// Device index for this context.
    pub fn device_id(&self) -> i32 {
        self.device
    }

    /// Raw `cudaStream_t` used by this context.
    pub fn stream(&self) -> CudaStream {
        self.stream
    }

    /// Fit every source in the batch: the `config.n_restarts` PSO swarms of
    /// every source run side by side on the GPU, then each restart is polished
    /// with L-BFGS on the CPU and assembled into one [`SbplResult`] per source
    /// (in [`GpuBatchData::names`] order).
    pub fn batch_fit(
        &self,
        data: &GpuBatchData,
        config: &PsoConfig,
    ) -> Result<Vec<SbplResult>, String> {
        self.set_device()?;
        let preps: Vec<&PreparedSource> = data.preps.iter().collect();
        gpu_common::batch_fit(|n| CudaEvaluator::new(self, data, n), &preps, config)
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
        self.set_device()?;
        let preps: Vec<&PreparedSource> = data.preps.iter().collect();
        let mut out = gpu_common::batch_pso(
            |n| CudaEvaluator::new(self, data, n),
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

/// Holds the per-iteration device buffers (positions in, costs out) and issues
/// one kernel launch per generation.
struct CudaEvaluator<'a> {
    ctx: &'a GpuContext,
    data: &'a GpuBatchData,
    d_positions: DevBuf,
    d_costs: DevBuf,
    /// Particles per source as the kernel sees them: every restart's swarm.
    n_particles: usize,
    grid: c_int,
    block: c_int,
}

const BLOCK: c_int = 256;
/// Must match `WARP_SIZE` in `cuda/sbpl.cu`.
const WARP_SIZE: c_int = 32;

impl<'a> CudaEvaluator<'a> {
    fn new(
        ctx: &'a GpuContext,
        data: &'a GpuBatchData,
        n_particles: usize,
    ) -> Result<Self, String> {
        let total = data.n_sources * n_particles;
        // One warp per particle (see cuda/sbpl.cu), and the kernel's thread
        // index is a 32-bit int.
        let threads = total
            .checked_mul(WARP_SIZE as usize)
            .filter(|&t| t <= c_int::MAX as usize)
            .ok_or_else(|| {
                format!(
                    "{} sources x {} particles is too many for one kernel launch; \
                     split the batch",
                    data.n_sources, n_particles
                )
            })? as c_int;
        let grid = (threads + BLOCK - 1) / BLOCK;
        Ok(Self {
            ctx,
            data,
            d_positions: DevBuf::alloc(total * N_PARAMS * size_of::<f64>())?,
            d_costs: DevBuf::alloc(total * size_of::<f64>())?,
            n_particles,
            grid: grid.max(1),
            block: BLOCK,
        })
    }
}

impl BatchCostEvaluator for CudaEvaluator<'_> {
    fn eval_batch(&self, positions: &[f64], costs: &mut [f64]) -> Result<(), String> {
        // The copies below trust these lengths, and the device buffers were
        // sized for exactly this many particles.
        let total = self.data.n_sources * self.n_particles;
        if positions.len() != total * N_PARAMS || costs.len() != total {
            return Err(format!(
                "eval_batch: evaluator holds {total} particles, got {} positions and {} costs",
                positions.len() / N_PARAMS,
                costs.len()
            ));
        }
        let stream = self.ctx.stream();
        self.d_positions.upload_from(positions, stream)?;
        unsafe {
            launch_batch_pso_cost_sbpl(
                self.data.d_times.ptr as _,
                self.data.d_log10_nu.ptr as _,
                self.data.d_flux.ptr as _,
                self.data.d_flux_err_sq.ptr as _,
                self.data.d_offsets.ptr as _,
                self.d_positions.ptr as _,
                self.d_costs.ptr as _,
                self.data.n_sources as c_int,
                self.n_particles as c_int,
                self.grid,
                self.block,
                stream,
            );
            cuda_check(cudaGetLastError())?;
        }
        self.d_costs.download_into(costs, stream)?;
        cuda_check(unsafe { cudaStreamSynchronize(stream) })
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
