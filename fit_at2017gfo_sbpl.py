"""
Fit the GRB 170817A off-axis afterglow SBPL to AT2017gfo PS1 r/g data.

Physical picture
----------------
The SBPL break is the **light-curve peak** (rise-to-decline turnover).
For GW170817's off-axis structured jet the peak occurs at t_b ≈ 150–160
days post-merger, when the relativistic beaming cone opens to the observer.

  F(t) ∝ (t/t_b)^α₁ · [0.5·(1+(t/t_b)^(1/D))]^((α₂−α₁)·D)

  α₁ > 0  (F rises as t^~0.8 before the break)
  α₂ < 0  (F declines as t^~-2 after the break)
  t_b = break timescale ≈ peak time (155 d for GW170817)

The PS1 r/g data (first ~10 d) is kilonova-dominated; the rising
afterglow is buried under it.  Fitting with α₁ > 0 and α₂ < 0
constrained, the PSO can still extrapolate t_b into the ~150 d range
and give the physically correct curve shape.

Usage:
    python fit_at2017gfo_sbpl.py
    python fit_at2017gfo_sbpl.py --dat /path/to/AT2017gfo.dat
"""

from __future__ import annotations

import argparse
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time

import sbpl_pso

# ── Constants ─────────────────────────────────────────────────────────────────
ZP           = 23.9
MERGER_MJD   = 57982.529          # GW170817 / GRB170817A merger (MJD)
EXPECTED_TB  = 155.0              # days post-merger: expected afterglow peak
PS1_TO_SHORT = {"ps1::r": "r", "ps1::g": "g"}
BAND_COLOR   = {"g": "#2ca02c", "r": "#d62728"}
BAND_LABEL   = {"g": "ps1-g",   "r": "ps1-r"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_dat(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep=r"\s+", header=None,
        names=["datetime", "band", "mag", "mag_err"], comment="#",
    )
    df = df[df["band"].isin(PS1_TO_SHORT)].copy()
    if df.empty:
        raise ValueError(f"No ps1::r / ps1::g rows in {path}")
    df["filter"] = df["band"].map(PS1_TO_SHORT)
    mjd = Time(df["datetime"].tolist(), format="isot", scale="utc").mjd
    # Work in days post-merger so t₀ is near 0 and t_b is in physical day units
    df["t_pm"] = mjd - MERGER_MJD
    return df


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    out = np.full_like(flux, np.nan, dtype=float)
    pos = flux > 0
    out[pos] = ZP - 2.5 * np.log10(flux[pos])
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dat",
        default="../lc_fitting_comparison/AT2017gfo_GRB170817A_corrected.dat",
    )
    parser.add_argument("--output-dir", default="sbpl_results")
    args = parser.parse_args()

    print(f"Reading {args.dat} ...")
    df = read_dat(args.dat)
    print(f"  ps1::r = {(df['filter']=='r').sum()} pts, "
          f"ps1::g = {(df['filter']=='g').sum()} pts")
    print(f"  time range: {df['t_pm'].min():.2f} – {df['t_pm'].max():.2f} days post-merger")

    # Write temp CSV in days-post-merger frame.
    # t_min ≈ 0.7 d, t_max ≈ 9.7 d  →  Rust bounds search t_b in [0.01, 500] d
    tmp_df = df[["t_pm", "mag", "mag_err", "filter"]].rename(columns={"t_pm": "mjd"})
    with tempfile.NamedTemporaryFile(
        prefix="at2017gfo_pm_", suffix=".csv", mode="w", delete=False
    ) as tmp:
        tmp_df.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    try:
        print("Running sbpl_pso.fit() with physical priors (α₁>0, α₂<0) ...")
        res = sbpl_pso.fit(
            csv_path,
            n_particles=150,
            n_iters=800,
            n_restarts=12,
            # Enforce peak-at-break: rising before, declining after
            alpha1_min=0.0,    # α₁ ∈ [0, 10]  — rising pre-peak
            alpha2_max=0.0,    # α₂ ∈ [-10, 0] — declining post-peak
        )
    finally:
        os.unlink(csv_path)

    p    = res["params"]
    chi2 = res["reduced_chi2"]
    tb   = p["tb"]    # days post-merger
    t0   = p["t0"]    # days post-merger (should be ≤ 0.7)

    print(f"\n  n_obs={res['n_obs']}  n_bands={res['n_bands']}  χ²_red={chi2:.3f}")
    print(f"  t₀   = {t0:.3f} d  (merger = day 0; first obs ≈ day 0.7)")
    print(f"  t_b  = {tb:.1f} d  (break = peak; GW170817 expected ≈ {EXPECTED_TB:.0f} d)")
    print(f"  α₁   = {p['alpha1']:.3f} ± {p['alpha1_err']:.3f}  (rising; theory ≈ +0.8)")
    print(f"  α₂   = {p['alpha2']:.3f} ± {p['alpha2_err']:.3f}  (declining; theory ≈ −2)")
    print(f"  β    = {p['beta']:.3f} ± {p['beta_err']:.3f}")
    print(f"  logD = {p['logd']:.3f} ± {p['logd_err']:.3f}")

    # ── Dense grid: merger → 20 d ─────────────────────────────────────────────
    obs     = pd.DataFrame(res["obs"])
    t_dense = np.linspace(max(0.1, t0 + 0.01), 10.0, 400)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    for band in ["r", "g"]:
        color = BAND_COLOR[band]
        sub   = obs[obs["band"] == band]
        if sub.empty:
            continue

        mags  = flux_to_mag(sub["flux"].values)
        flux_vals = sub["flux"].values
        merrs = np.where(flux_vals > 0,
                         2.5 / np.log(10) * sub["flux_err"].values / flux_vals,
                         np.nan)
        valid = np.isfinite(mags)
        ax.errorbar(
            sub["time"].values[valid], mags[valid], yerr=merrs[valid],
            fmt="o", ms=6, color=color, alpha=0.85, zorder=4,
            label=f"{BAND_LABEL[band]} data",
        )

        curve_flux = np.array(sbpl_pso.eval_sbpl(p, t_dense.tolist(), band))
        curve_mag  = flux_to_mag(curve_flux)
        valid_c    = np.isfinite(curve_mag)
        ax.plot(t_dense[valid_c], curve_mag[valid_c],
                color=color, lw=2, zorder=3,
                label=f"{BAND_LABEL[band]} SBPL (extrapolated)")

    # Shade the observed data window
    ax.axvspan(0, obs["time"].max() + 0.3, alpha=0.06, color="steelblue",
               label="PS1 data window")

    ax.invert_yaxis()
    ax.set_xlabel("Days post-merger", fontsize=12)
    ax.set_ylabel("Apparent magnitude (AB, bright at top)", fontsize=12)
    chi2_str = f"{chi2:.2f}" if chi2 is not None else "n/a"
    ax.set_title(
        "GW170817 / GRB170817A — SBPL (PS1 r/g)\n"
        f"χ²_red = {chi2_str}  |  "
        f"α₁ = {p['alpha1']:.2f}  α₂ = {p['alpha2']:.2f}  "
        f"β = {p['beta']:.2f}  t_b = {tb:.0f} d",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "AT2017gfo_afterglow_sbpl.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
