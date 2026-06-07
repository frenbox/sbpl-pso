"""
Batch SBPL fit for all CSVs in afterglow_photometry/.

For each sample_XXXX.csv:
  - Reads time_days, g_ps1_mag, r_ps1_mag
  - Adds a small synthetic uncertainty (MAG_ERR = 0.1 mag)
  - Subsamples to at most MAX_PTS_PER_BAND points per band
  - Runs sbpl_pso.fit()
  - Saves a plot to afterglow_sbpl_plots/

At the end writes a summary CSV: afterglow_sbpl_plots/fit_summary.csv

Usage:
    python fit_afterglow_batch.py
    python fit_afterglow_batch.py --input-dir afterglow_photometry --output-dir afterglow_sbpl_plots
    python fit_afterglow_batch.py --max-pts 40 --n-particles 50 --n-iters 300
"""

from __future__ import annotations

import argparse
import os
import tempfile
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sbpl_pso

# ── Defaults ──────────────────────────────────────────────────────────────────
MAG_ERR         = 0.10          # synthetic photometric uncertainty (mag)
MAX_PTS_PER_BAND = 60           # subsample long light curves for speed
MAG_VALID_MAX    = 40.0         # skip bands where all points are fainter than this
ZP               = 23.9         # AB zero-point used by sbpl_pso

BAND_COLOR = {"g": "#2ca02c", "r": "#d62728"}
BAND_LABEL = {"g": "g-PS1",   "r": "r-PS1"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    out = np.full_like(flux, np.nan, dtype=float)
    pos = flux > 0
    out[pos] = ZP - 2.5 * np.log10(flux[pos])
    return out


def mag_err_from_flux(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, 2.5 / np.log(10.0) * flux_err / flux, np.nan)


def thin(arr: np.ndarray, max_pts: int) -> np.ndarray:
    """Return indices that uniformly subsample arr to at most max_pts points."""
    n = len(arr)
    if n <= max_pts:
        return np.arange(n)
    return np.round(np.linspace(0, n - 1, max_pts)).astype(int)


def make_fit_csv(df_raw: pd.DataFrame, max_pts: int, mag_err: float) -> tuple[str, list[str]]:
    """Convert wide afterglow CSV to long sbpl_pso format; return (tmp_path, bands_used)."""
    rows = []
    for band, col in [("g", "g_ps1_mag"), ("r", "r_ps1_mag")]:
        if col not in df_raw.columns:
            continue
        mag = df_raw[col].values
        valid = mag < MAG_VALID_MAX
        if valid.sum() < 4:
            continue
        t   = df_raw["time_days"].values[valid]
        m   = mag[valid]
        idx = thin(t, max_pts)
        for i in idx:
            rows.append({"mjd": t[i], "mag": m[i], "mag_err": mag_err, "filter": band})

    if not rows:
        raise ValueError("No valid photometry after filtering (all bands too faint?)")

    long_df = pd.DataFrame(rows)
    bands   = sorted(long_df["filter"].unique().tolist())
    with tempfile.NamedTemporaryFile(
        prefix="ag_fit_", suffix=".csv", mode="w", delete=False
    ) as tmp:
        long_df.to_csv(tmp.name, index=False)
        return tmp.name, bands


def fit_and_plot(
    csv_path: str,
    output_dir: str,
    max_pts: int,
    mag_err: float,
    n_particles: int,
    n_iters: int,
    n_restarts: int,
) -> dict:
    name    = Path(csv_path).stem
    df_raw  = pd.read_csv(csv_path, comment="#")
    tmp_csv, bands = make_fit_csv(df_raw, max_pts, mag_err)

    try:
        res = sbpl_pso.fit(
            tmp_csv,
            n_particles=n_particles,
            n_iters=n_iters,
            n_restarts=n_restarts,
        )
    finally:
        os.unlink(tmp_csv)

    p    = res["params"]
    chi2 = res["reduced_chi2"]
    obs  = pd.DataFrame(res["obs"])

    t_min   = obs["time"].min()
    t_max   = obs["time"].max()
    t_dense = np.linspace(max(t_min * 0.8, 1e-3), t_max * 1.2, 600)

    fig, ax = plt.subplots(figsize=(10, 6))

    for band in bands:
        color = BAND_COLOR.get(band, "gray")
        sub   = obs[obs["band"] == band]
        if sub.empty:
            continue

        mags  = flux_to_mag(sub["flux"].values)
        merrs = mag_err_from_flux(sub["flux"].values, sub["flux_err"].values)
        valid = np.isfinite(mags)

        ax.errorbar(
            sub["time"].values[valid], mags[valid], yerr=merrs[valid],
            fmt="o", ms=4, color=color, alpha=0.75,
            label=f"{BAND_LABEL.get(band, band)} data", zorder=3,
        )

        try:
            curve_flux = np.array(sbpl_pso.eval_sbpl(p, t_dense.tolist(), band))
            curve_mag  = flux_to_mag(curve_flux)
            valid_c    = np.isfinite(curve_mag)
            ax.plot(t_dense[valid_c], curve_mag[valid_c],
                    color=color, lw=2,
                    label=f"{BAND_LABEL.get(band, band)} SBPL fit", zorder=4)
        except Exception:
            pass

    ax.invert_yaxis()
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Apparent magnitude (AB, bright at top)", fontsize=12)
    chi2_str = f"{chi2:.3f}" if chi2 is not None else "n/a"
    ax.set_title(
        f"{name}  —  SBPL fit (PSO + L-BFGS)\n"
        f"χ²_red = {chi2_str}  |  "
        f"α₁ = {p['alpha1']:.2f}  α₂ = {p['alpha2']:.2f}  "
        f"β = {p['beta']:.2f}  t_b = {p['tb']:.2f} d",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{name}_sbpl.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "name":     name,
        "n_obs":    res["n_obs"],
        "n_bands":  res["n_bands"],
        "alpha1":   p["alpha1"],
        "alpha1_err": p.get("alpha1_err", np.nan),
        "alpha2":   p["alpha2"],
        "alpha2_err": p.get("alpha2_err", np.nan),
        "beta":     p["beta"],
        "beta_err": p.get("beta_err", np.nan),
        "tb":       p["tb"],
        "t0":       p["t0"],
        "logd":     p["logd"],
        "loga":     p["loga"],
        "reduced_chi2": chi2,
        "status":   "ok",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",   default="afterglow_photometry")
    parser.add_argument("--output-dir",  default="afterglow_sbpl_plots")
    parser.add_argument("--max-pts",     type=int,   default=MAX_PTS_PER_BAND,
                        help="Max data points per band (subsampled if longer)")
    parser.add_argument("--mag-err",     type=float, default=MAG_ERR,
                        help="Synthetic magnitude uncertainty to assign")
    parser.add_argument("--n-particles", type=int,   default=30)
    parser.add_argument("--n-iters",     type=int,   default=200)
    parser.add_argument("--n-restarts",  type=int,   default=3)
    args = parser.parse_args()

    csvs = sorted(Path(args.input_dir).glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(csvs)} files — saving plots to {args.output_dir}/\n")

    results = []
    for i, csv in enumerate(csvs, 1):
        name = csv.stem
        print(f"[{i:3d}/{len(csvs)}] {name} ...", end="", flush=True)
        try:
            row = fit_and_plot(
                str(csv),
                output_dir=args.output_dir,
                max_pts=args.max_pts,
                mag_err=args.mag_err,
                n_particles=args.n_particles,
                n_iters=args.n_iters,
                n_restarts=args.n_restarts,
            )
            print(f"  χ²_red={row['reduced_chi2']:.3f}  α₁={row['alpha1']:.2f}"
                  f"  α₂={row['alpha2']:.2f}  t_b={row['tb']:.2f}d")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results.append({"name": name, "status": "failed", "error": str(exc)})
            continue
        results.append(row)

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(args.output_dir, "fit_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    n_ok   = (summary_df["status"] == "ok").sum()
    n_fail = (summary_df["status"] != "ok").sum()
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
