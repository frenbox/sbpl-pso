"""
Fit an SBPL model to a ZTF photometry CSV and plot the result.

Usage:
    python scripts/fit_sbpl.py <photometry.csv>
    python scripts/fit_sbpl.py <photometry.csv> --output-dir results/
    python scripts/fit_sbpl.py <photometry.csv> --flux   # plot in flux instead of mag

The CSV must have columns: jd (or mjd), magpsf (or mag), sigmapsf (or
mag_err), and either fid (1=g, 2=r, 3=i) or a filter string column.

Build the Rust extension first:
    cd sbpl-pso
    maturin develop --release --features python
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sbpl_pso

# Default paths below are resolved against the repo root rather than the current
# working directory, so this script works the same from anywhere:
#     python scripts/fit_sbpl.py ...
#     cd scripts && python fit_sbpl.py ...
REPO_ROOT = Path(__file__).resolve().parent.parent

ZP = 23.9  # AB zero-point used by sbpl_pso

BAND_COLOR = {"g": "#2ca02c", "r": "#d62728", "i": "#ff7f0e", "z": "#9467bd", "y": "#8c564b"}
BAND_LABEL = {"g": "g-band", "r": "r-band", "i": "i-band", "z": "z-band", "y": "y-band"}


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    out = np.full_like(flux, np.nan, dtype=float)
    pos = flux > 0
    out[pos] = ZP - 2.5 * np.log10(flux[pos])
    return out


def mag_err_from_flux(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, 2.5 / np.log(10.0) * flux_err / flux, np.nan)


def fit_and_plot(csv_path: str, output_dir: str, plot_flux: bool = False) -> str:
    name = Path(csv_path).stem
    print(f"\n=== {name} ===")
    print(f"  running sbpl_pso.fit() ...")

    res = sbpl_pso.fit(csv_path)
    p = res["params"]
    chi2 = res["reduced_chi2"]
    n_obs = res["n_obs"]
    n_bands = res["n_bands"]

    print(f"  n_obs={n_obs}  n_bands={n_bands}  reduced_chi2={chi2:.4f}")
    print(f"  alpha1={p['alpha1']:.3f}  alpha2={p['alpha2']:.3f}  beta={p['beta']:.3f}")
    print(f"  logd={p['logd']:.3f}  loga={p['loga']:.3f}")
    print(f"  tb={p['tb']:.2f} days  t0={p['t0']:.2f}")

    obs = pd.DataFrame(res["obs"])
    bands_present = sorted(obs["band"].unique())

    t_min = obs["time"].min()
    t_max = obs["time"].max()
    t_dense = np.linspace(t_min - 0.5, t_max + 2.0, 600)

    fig, ax = plt.subplots(figsize=(10, 6))

    for band in bands_present:
        color = BAND_COLOR.get(band, "gray")
        sub = obs[obs["band"] == band]
        fluxes = sub["flux"].values
        flux_errs = sub["flux_err"].values
        times = sub["time"].values

        # Try to evaluate model curve; skip if band unknown
        try:
            curve_flux = np.array(sbpl_pso.eval_sbpl(p, t_dense.tolist(), band))
        except Exception:
            curve_flux = None

        if plot_flux:
            ax.errorbar(times, fluxes, yerr=flux_errs,
                        fmt="o", ms=5, color=color, alpha=0.85,
                        label=f"{BAND_LABEL.get(band, band)} data", zorder=3)
            if curve_flux is not None:
                mask = curve_flux > 0
                ax.plot(t_dense[mask], curve_flux[mask], color=color, lw=2,
                        label=f"{BAND_LABEL.get(band, band)} SBPL fit", zorder=4)
        else:
            mags = flux_to_mag(fluxes)
            merrs = mag_err_from_flux(fluxes, flux_errs)
            valid = np.isfinite(mags)
            ax.errorbar(times[valid], mags[valid], yerr=merrs[valid],
                        fmt="o", ms=5, color=color, alpha=0.85,
                        label=f"{BAND_LABEL.get(band, band)} data", zorder=3)
            if curve_flux is not None:
                curve_mag = flux_to_mag(curve_flux)
                valid_c = np.isfinite(curve_mag)
                ax.plot(t_dense[valid_c], curve_mag[valid_c], color=color, lw=2,
                        label=f"{BAND_LABEL.get(band, band)} SBPL fit", zorder=4)

    # Axis labels and magnitude inversion (once, after all bands are plotted)
    if plot_flux:
        ax.set_ylabel("Flux (physical units)", fontsize=12)
    else:
        ax.set_ylabel("Apparent magnitude (AB, bright at top)", fontsize=12)
        ax.invert_yaxis()

    ax.set_xlabel("Time (JD)", fontsize=12)
    chi2_str = f"{chi2:.3f}" if chi2 is not None else "n/a"
    ax.set_title(
        f"{name} — SBPL fit (PSO + L-BFGS)\n"
        f"reduced χ² = {chi2_str}  |  "
        f"α₁={p['alpha1']:.2f}  α₂={p['alpha2']:.2f}  "
        f"β={p['beta']:.2f}  tb={p['tb']:.1f}d",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_flux" if plot_flux else "_mag"
    out_path = os.path.join(output_dir, f"{name}_sbpl{suffix}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="ZTF photometry CSV to fit")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "sbpl_results"),
                        help="Directory to save the plot (default: <repo>/sbpl_results)")
    parser.add_argument("--flux", action="store_true",
                        help="Plot in physical flux instead of magnitudes")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        raise SystemExit(1)

    fit_and_plot(args.csv, output_dir=args.output_dir, plot_flux=args.flux)


if __name__ == "__main__":
    main()
