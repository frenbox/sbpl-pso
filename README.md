# sbpl-pso

Smoothly Broken Power Law (SBPL) multi-band light-curve fitter using
deterministic PSO + L-BFGS, structured to mirror `villar-pso`.

## Model

```
F(t, ν) = 10^loga · (ν/1e15)^β · τ^α₁ · [0.5·(1 + τ^(1/D))]^((α₂−α₁)·D)
```
where `τ = (t−t₀)/tb`, `D = 10^logd`.  All bands with a known effective
frequency (g, r, i, z, u, ZTF/LSST variants) are fit simultaneously.

## Build & install the Python extension

```bash
cd sbpl-pso
maturin develop --release --features python
```

Verify:
```bash
python -c "import sbpl_pso; print(dir(sbpl_pso))"
# → ['eval_sbpl', 'fit', ...]
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

### `sbpl_pso.eval_sbpl(params, t_dense, band) → list[float]`

Evaluate the fitted model on a dense time grid for one band.

```python
import sbpl_pso, numpy as np

res    = sbpl_pso.fit("ZTF17aabuqoz.csv")
t      = np.linspace(2458000, 2458200, 500)
flux_r = sbpl_pso.eval_sbpl(res["params"], t.tolist(), "r")
```

## Plotting script

Fits one CSV and saves a magnitude (or flux) plot:

```bash
# Install Python deps once
pip install numpy pandas matplotlib

# Plot in magnitudes (default)
python fit_sbpl.py ../lc_fitting_comparison/data/photometry/ZTF17aabuqoz.csv

# Plot in flux space
python fit_sbpl.py ../lc_fitting_comparison/data/photometry/ZTF17aabuqoz.csv --flux

# Custom output dir
python fit_sbpl.py data.csv --output-dir my_results/
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
