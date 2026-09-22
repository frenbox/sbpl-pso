# sbpl-pso

Smoothly Broken Power Law (SBPL) multi-band light-curve fitter using
deterministic PSO + L-BFGS, structured to mirror `villar-pso`.

## Model

```
F(t, ν) = 10^loga · (ν/1e15)^β · τ^α₁ · [0.5·(1 + τ^(1/D))]^((α₂−α₁)·D)
```
where `τ = (t−t₀)/tb`, `D = 10^logd`.  All bands with a known effective
frequency (g, r, i, z, u, ZTF/LSST variants) are fit simultaneously.

## Backends

| Backend | Feature flag | Requirement | What it accelerates |
|---------|--------------|-------------|---------------------|
| CPU | *(none)* | — | Rayon across sources, one source per thread |
| CUDA | `cuda` | `nvcc` + CUDA toolkit, NVIDIA GPU | Batch PSO: every source's swarm scored in one kernel launch per iteration |
| Metal | `metal` | macOS on Apple Silicon (M1–M5) | Same batch PSO, via `metal/sbpl.metal` |

All three produce `SbplResult` values through the same code path, so their
outputs are directly comparable — see [GPU backends](#gpu-backends) below.

## Build & install the Python extension

```bash
cd sbpl-pso

maturin develop --release --features python            # CPU only
CUDA_HOME=/usr/local/cuda \
  maturin develop --release --features "python,cuda"   # CPU + NVIDIA GPU
maturin develop --release --features "python,metal"    # CPU + Apple Silicon GPU
```

Verify:
```bash
python -c "import sbpl_pso; print(sbpl_pso.GPU_BACKEND, dir(sbpl_pso))"
# CPU-only build → None ['GPU_BACKEND', 'eval_sbpl', 'fit', ...]
# CUDA build     → cuda ['GPU_BACKEND', 'eval_sbpl', 'fit', 'fit_gpu', ...]
```

## Python API

### `sbpl_pso.fit(csv_path) → dict`

Fits a ZTF photometry CSV and returns:

| Key | Type | Description |
|-----|------|-------------|
| `params` | dict | Fitted physical parameters: `alpha1`, `alpha2`, `beta`, `logd`, `loga`, `tb`, `t0` and `*_err` uncertainties |
| `obs` | list of dicts | Original observations: `{time, flux, flux_err, band}` in physical flux units |
| `reduced_chi2` | float\|None | Reduced χ² of the best fit |
| `n_obs` | int | Total observations used |
| `n_bands` | int | Number of bands used |

### `sbpl_pso.fit_gpu(csv_paths, **kwargs) → list[dict]`

*Requires a `cuda` or `metal` build.* Fits many light curves in **one** GPU
batch and returns a list of `fit()`-shaped dicts, one per input path, in input
order. Accepts the same tuning and bound keywords as `fit()`. A path that fails
to load or has too little data gets `{"error": "..."}` in its slot, so the
output always lines up with the input.

Batching is the whole point — one call with 500 light curves is far faster than
500 calls with one, because a single kernel launch scores every source's swarm.

```python
import glob, sbpl_pso

paths = sorted(glob.glob("photometry/*.csv"))
results = sbpl_pso.fit_gpu(paths, n_particles=30, n_iters=200)

for path, res in zip(paths, results):
    if "error" in res:
        print(f"{path}: {res['error']}")
        continue
    print(path, res["params"]["tb"], res["reduced_chi2"])
```

`sbpl_pso.GPU_BACKEND` is `"cuda"`, `"metal"` or `None`, so callers can pick a
path without a try/except import dance:

```python
fit_many = sbpl_pso.fit_gpu if sbpl_pso.GPU_BACKEND else \
           (lambda paths, **kw: [sbpl_pso.fit(p, **kw) for p in paths])
```

### `sbpl_pso.eval_sbpl(params, t_dense, band) → list[float]`

Evaluate the fitted model on a dense time grid for one band.

```python
import sbpl_pso, numpy as np

res    = sbpl_pso.fit("ZTF17aabuqoz.csv")
t      = np.linspace(2458000, 2458200, 500)
flux_r = sbpl_pso.eval_sbpl(res["params"], t.tolist(), "r")
```

## Plotting scripts

The Python drivers live in [`scripts/`](scripts/). Their default input and
output paths are anchored to the repo root, so they behave the same whether you
run them from the repo root or from inside `scripts/`.

```bash
# Install Python deps once
pip install numpy pandas matplotlib
```

| Script | What it does |
|--------|--------------|
| `scripts/fit_sbpl.py` | Fit one CSV and save a magnitude (or flux) plot |
| `scripts/fit_afterglow_batch.py` | Batch fit `afterglow_photometry/` → `afterglow_sbpl_plots/` |
| `scripts/fit_afterglow_batch_v2.py` | Same, with shape-adaptive α bounds and a retry on high χ² |
| `scripts/fit_afterglow_parquet.py` | Batch fit a long-format LSST parquet, one fit per `event_id` |
| `scripts/fit_at2017gfo_sbpl.py` | Fit the GRB 170817A off-axis afterglow to AT2017gfo PS1 data |

```bash
# Plot in magnitudes (default)
python scripts/fit_sbpl.py ../lc_fitting_comparison/data/photometry/ZTF17aabuqoz.csv

# Plot in flux space
python scripts/fit_sbpl.py ../lc_fitting_comparison/data/photometry/ZTF17aabuqoz.csv --flux

# Custom output dir
python scripts/fit_sbpl.py data.csv --output-dir my_results/

# Batch runs (defaults resolve to <repo>/afterglow_photometry etc.)
python scripts/fit_afterglow_batch_v2.py
python scripts/fit_afterglow_parquet.py --max-events 5
```

## Running tests & benchmarks

```bash
# Unit tests (no data required, ~0.3 s)
cargo test --release --test test_sbpl -- --nocapture

# Dataset quality / chi² distribution
# Set SBPL_DATA_DIR or use default ../lc_fitting_comparison/data/photometry
cargo test --release --test bench_datasets sbpl_dataset_quality \
    -- --ignored --nocapture

# CPU throughput (sequential vs parallel)
cargo test --release --test bench_datasets sbpl_dataset_throughput \
    -- --ignored --nocapture

# Thread-count scaling (1, 2, 4, 8 threads)
cargo test --release --test bench_datasets sbpl_thread_scaling \
    -- --ignored --nocapture

# PSO config sensitivity (fast / default / thorough)
cargo test --release --test bench_datasets sbpl_pso_config_sensitivity \
    -- --ignored --nocapture

# CPU dispatch binary (par_iter vs par_chunks, 2–10 threads)
cargo run --release --bin cpu-dispatch-bench \
    -- ../lc_fitting_comparison/data/photometry
```

## GPU backends

### How the work is split

The GPU evaluates the cost function; the CPU runs everything else.

```
per PSO iteration:
  CPU   move every particle of every source      (cheap, O(particles x 7))
  GPU   score every particle of every source     (one launch, the expensive part)
  CPU   update personal / global bests
after each restart:
  CPU   L-BFGS polish per source, in parallel (fp64)
  CPU   loga rescaling + restart-spread uncertainties
```

`src/gpu_common.rs` holds that driver and is shared by both backends — only the
cost evaluation differs, behind one trait. The final numbers therefore come out
of the same `lbfgs_refine` / `assemble_result` code the CPU path uses: the GPU
steers the search, it does not produce the reported parameters.

Two deliberate differences from the CPU fitter:

* **Synchronous swarm.** The CPU updates `gbest` mid-sweep; the GPU scores a
  whole generation at once, so `gbest` is last iteration's. Both are standard
  PSO variants.
* **Per-source RNG keyed by content.** Each source's stream is seeded from its
  own observation hash, not its batch index — so a light curve fits identically
  whether it is batched with 5 sources or 5000, and dropping or re-ordering
  sources changes nothing. `tests/test_gpu.rs` pins this down.

### Kernel design

One **warp** (CUDA) / **SIMD-group** (Metal) per particle, striding over that
source's observations and reducing with `__shfl_down_sync` / `simd_sum`. One
thread per particle is the obvious layout and it is 5x slower here: real batches
span three orders of magnitude in light-curve length (7 to ~11 000 points), and
a single thread walking the longest source serialises the entire launch.

Times are shifted by each source's first observation before upload, and `t0` is
shifted to match, so `t − t0` never loses precision to a JD-scale offset — this
is what makes the fp32 Metal path viable at all. Frequencies arrive pre-logged
so the amplitude is a single `exp10`.

**Metal runs in fp32** (Apple GPUs have no fp64), with Kahan-compensated
reductions. The CPU polish that follows is fp64 regardless, so fitted parameters
do not inherit the shader's precision.

### Measured throughput

483 ZTF light curves (233 688 observations), `n_particles=30, n_iters=200,
n_restarts=3`, RTX 5080 vs 24-thread Ryzen CPU:

| Backend | Wall time | ms/source | median reduced χ² |
|---------|-----------|-----------|-------------------|
| CPU (rayon, 24 threads) | 50.4 s | 104.4 | 2.503 |
| CUDA (RTX 5080) | 20.1 s | 41.7 | 2.368 |

Fit quality is equivalent — the GPU matches or beats the CPU's χ² on 333 of 483
sources and the median is marginally better.

The kernel is fp64 and therefore bound by fp64 throughput, which consumer
GeForce cards run at 1/64 of fp32. A datacenter card (A100/H100, 1/2 rate)
should widen the gap considerably; conversely, a GPU with weak fp64 and a fast
CPU may not be worth it. Measure with `--cpu-compare` before committing.

### GPU benchmarks & tests

```bash
# CUDA: one batch of everything in a directory, vs the CPU
CUDA_HOME=/usr/local/cuda cargo build --release --features cuda --bin gpu-batch-bench
./target/release/gpu-batch-bench ../lc_fitting_comparison/data/photometry --cpu-compare

# Apple Silicon equivalent
cargo run --release --features metal --bin metal-batch-bench -- photometry/ --cpu-compare

# GPU correctness tests (need a real device)
CUDA_HOME=/usr/local/cuda cargo test --release --features cuda --test test_gpu -- --nocapture
```

Both benches accept `--n-particles N`, `--n-iters N` and `--n-restarts N`.

### Rust API

```rust
use sbpl_pso::gpu::{load_sources_with, GpuBatchData, GpuContext, Stream};
use sbpl_pso::PsoConfig;

let config = PsoConfig::default();
let sources = load_sources_with("photometry/", &config);   // skips unfittable CSVs

let stream = Stream::new_on_device(0)?;   // omit for Metal: GpuContext::new(0)
let gpu = GpuContext::new(0, stream.as_ptr())?;
let batch = GpuBatchData::new(&gpu, &sources)?;            // upload once
let results = gpu.batch_fit(&batch, &config)?;             // reuse across configs
```

`gpu_metal` exposes the same names, so a `#[cfg]` alias is all it takes to
support both. `Stream` is a plain `cudaStream_t` wrapper — pass `as_ptr()` to
another library (e.g. ONNX Runtime) to share the stream.

## Parameters

| Parameter | Description | Typical range |
|-----------|-------------|---------------|
| `alpha1`  | Pre-break temporal power-law index | −5 … +5 |
| `alpha2`  | Post-break temporal power-law index | −5 … +5 |
| `beta`    | Spectral index (F_ν ∝ ν^β) | −3 … +1 |
| `logd`    | log₁₀ of break-smoothness D | −3 … 0 |
| `loga`    | log₁₀ of physical flux amplitude | depends on source |
| `tb`      | Break time (days) | — |
| `t0`      | Time zero-point (days, same frame as JD in CSV) | — |

## Project structure

```
sbpl-pso/
├── Cargo.toml
├── build.rs                  # compiles the CUDA kernel when --features cuda
├── cuda/sbpl.cu              # CUDA cost kernel (fp64)
├── metal/sbpl.metal          # Metal cost kernel (fp32 + Kahan)
├── src/
│   ├── lib.rs                # model, preprocessing, CPU PSO + L-BFGS, PyO3
│   ├── gpu_common.rs         # host-side batch PSO driver, shared by both GPUs
│   ├── gpu.rs                # CUDA backend
│   ├── gpu_metal.rs          # Metal backend
│   └── bin/
│       ├── cpu_dispatch_bench.rs
│       ├── gpu_batch_bench.rs
│       └── metal_batch_bench.rs
├── scripts/                  # Python drivers (paths anchored to the repo root)
│   ├── fit_sbpl.py
│   ├── fit_afterglow_batch.py
│   ├── fit_afterglow_batch_v2.py
│   ├── fit_afterglow_parquet.py
│   └── fit_at2017gfo_sbpl.py
└── tests/
    ├── test_sbpl.rs
    ├── test_gpu.rs           # --features cuda
    └── bench_datasets.rs
```
