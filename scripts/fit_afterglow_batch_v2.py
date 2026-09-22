"""
Adaptive-prior SBPL batch fit for afterglow_photometry/.

Improvement over v1: each light curve is classified by shape before fitting,
and the PSO alpha bounds are set to match the expected physical behaviour:

  Shape          | α₁ constraint  | α₂ constraint  | Rationale
  ---------------+----------------+----------------+----------------------------
  clear_peak     | ≥ 0            | ≤ 0            | rising → peak → fading
  early_peak     | try both (*)   | ≤ 0            | peak near start → try peak
                 |                |                | model AND double-decay
  monotone_fade  | ≤ 0            | ≤ 0            | broken power-law fading
  monotone_rise  | ≥ 0            | ≥ 0            | broken power-law rising
  late_peak      | ≥ 0            | ≥ 0            | mostly rising data

(*) early_peak: fit twice (peaked model + double-decay), keep the lower χ².

All alpha bounds are widened to ±15 to accommodate steep rise/fall slopes
(previous ±10 limit caused α to stick at the boundary for many sources).

For fits that still have χ²_red > CHI2_RETRY, a fallback run is attempted
with unconstrained alphas (±15) and 2× more PSO particles.

Usage:
    python scripts/fit_afterglow_batch_v2.py
    python scripts/fit_afterglow_batch_v2.py --input-dir afterglow_photometry \\
                                     --output-dir afterglow_sbpl_plots_v2
    python scripts/fit_afterglow_batch_v2.py --n-particles 60 --n-iters 400 --n-restarts 5
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

# Default paths below are resolved against the repo root rather than the current
# working directory, so this script works the same from anywhere:
#     python scripts/fit_afterglow_batch_v2.py ...
#     cd scripts && python fit_afterglow_batch_v2.py ...
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Defaults ──────────────────────────────────────────────────────────────────
MAG_ERR          = 0.10
MAX_PTS_PER_BAND = 60
MAG_VALID_MAX    = 40.0
ZP               = 23.9
CHI2_RETRY       = 10.0   # retry with looser constraints if chi2 > this

ALPHA_MAX        = 15.0   # extended from ±10 to catch steeper slopes
ALPHA_MIN        = -15.0

BAND_COLOR = {"g": "#2ca02c", "r": "#d62728", "i": "#ff7f0e", "z": "#9467bd", "y": "#8c564b"}
BAND_LABEL = {"g": "g-PS1",   "r": "r-PS1",   "i": "i-PS1",   "z": "z-PS1",   "y": "y-PS1"}

# Preference order for picking a reference band (shape classification, etc.)
BAND_PREFERENCE = ("g", "r", "i", "z", "y")


# ── Light-curve shape classifier ──────────────────────────────────────────────

def reference_mag_column(df_raw: pd.DataFrame) -> str | None:
    """First available `{band}_ps1_mag` column, in BAND_PREFERENCE order."""
    return next(
        (f"{band}_ps1_mag" for band in BAND_PREFERENCE if f"{band}_ps1_mag" in df_raw.columns),
        None,
    )


def classify_shape(df_raw: pd.DataFrame) -> tuple[str, float | None]:
    """
    Return (shape, peak_t_days) where shape is one of:
      clear_peak / early_peak / late_peak / monotone_fade / monotone_rise / too_faint
    peak_t_days is the estimated break time in days (None if too faint).
    """
    ref_col = reference_mag_column(df_raw)
    if ref_col is None:
        return "too_faint", None

    g = df_raw[ref_col].values
    t = df_raw["time_days"].values
    valid = g < MAG_VALID_MAX
    if valid.sum() < 4:
        return "too_faint", None
    g, t = g[valid], t[valid]

    peak_t = t[np.argmin(g)]
    span   = t[-1] - t[0]
    diff   = np.diff(g)
    frac_rising = (diff < 0).sum() / len(diff)

    if frac_rising < 0.10:
        return "monotone_fade", None
    if frac_rising > 0.90:
        return "monotone_rise", None
    if peak_t < t[0] + 0.15 * span:
        return "early_peak", peak_t
    if peak_t > t[0] + 0.85 * span:
        return "late_peak", peak_t
    return "clear_peak", peak_t


def fit_kwargs(shape: str, peak_t: float | None, t_min: float, t_max: float) -> list[dict]:
    """
    Return a list of kwarg dicts for sbpl_pso.fit(), one per strategy to try.
    Multiple entries → the best chi2 is kept.

    Now includes tb_lower/tb_upper so the PSO search range brackets the real break.
    """
    duration = t_max - t_min

    # t_b bounds: default Rust formula (reference only)
    tb_lo_default = max(0.01 * duration, 0.01)
    tb_hi_default = min(max(10.0 * duration, 500.0), 10_000.0)

    # If we know the peak time, set t_b to search [peak/10, peak*10] ∩ [0.01, 10000]
    if peak_t is not None and peak_t > 0:
        tb_lo_peak = max(peak_t / 10.0, 0.001)
        tb_hi_peak = min(peak_t * 10.0, 10_000.0)
    else:
        tb_lo_peak = tb_lo_default
        tb_hi_peak = tb_hi_default

    if shape == "clear_peak":
        return [dict(alpha1_min=0.0,       alpha1_max=ALPHA_MAX,
                     alpha2_min=ALPHA_MIN,  alpha2_max=0.0,
                     tb_lower=tb_lo_peak,   tb_upper=tb_hi_peak)]

    if shape == "early_peak":
        # Two strategies: peaked model (α₁>0, α₂<0) and double-decay (both α<0)
        return [
            dict(alpha1_min=0.0,       alpha1_max=ALPHA_MAX,
                 alpha2_min=ALPHA_MIN,  alpha2_max=0.0,
                 tb_lower=tb_lo_peak,   tb_upper=tb_hi_peak),
            dict(alpha1_min=ALPHA_MIN,  alpha1_max=0.0,
                 alpha2_min=ALPHA_MIN,  alpha2_max=0.0,
                 tb_lower=tb_lo_default, tb_upper=tb_hi_default),
        ]

    if shape == "monotone_fade":
        return [dict(alpha1_min=ALPHA_MIN, alpha1_max=0.0,
                     alpha2_min=ALPHA_MIN, alpha2_max=0.0,
                     tb_lower=tb_lo_default, tb_upper=tb_hi_default)]

    if shape in ("monotone_rise", "late_peak"):
        return [dict(alpha1_min=0.0,  alpha1_max=ALPHA_MAX,
                     alpha2_min=0.0,  alpha2_max=ALPHA_MAX,
                     tb_lower=tb_lo_peak if peak_t else tb_lo_default,
                     tb_upper=tb_hi_peak if peak_t else tb_hi_default)]

    # Fallback: unconstrained
    return [dict(alpha1_min=ALPHA_MIN, alpha1_max=ALPHA_MAX,
                 alpha2_min=ALPHA_MIN, alpha2_max=ALPHA_MAX,
                 tb_lower=tb_lo_default, tb_upper=tb_hi_default)]


# ── Data helpers ──────────────────────────────────────────────────────────────

def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    out = np.full_like(flux, np.nan, dtype=float)
    pos = flux > 0
    out[pos] = ZP - 2.5 * np.log10(flux[pos])
    return out


def mag_err_from_flux(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(flux > 0, 2.5 / np.log(10.0) * flux_err / flux, np.nan)


def thin(arr: np.ndarray, max_pts: int) -> np.ndarray:
    n = len(arr)
    if n <= max_pts:
        return np.arange(n)
    return np.round(np.linspace(0, n - 1, max_pts)).astype(int)


def make_fit_csv(df_raw: pd.DataFrame, max_pts: int, mag_err: float) -> tuple[str, list[str]]:
    rows = []
    for band in BAND_PREFERENCE:
        col = f"{band}_ps1_mag"
        if col not in df_raw.columns:
            continue
        mag   = df_raw[col].values
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


# ── Single run ────────────────────────────────────────────────────────────────

def run_fit(tmp_csv: str, n_particles: int, n_iters: int, n_restarts: int,
            **alpha_kw) -> dict:
    """Run sbpl_pso.fit() with the given alpha bounds; return the result dict."""
    return sbpl_pso.fit(
        tmp_csv,
        n_particles=n_particles,
        n_iters=n_iters,
        n_restarts=n_restarts,
        **alpha_kw,
    )


def best_of(results: list[dict]) -> dict:
    """Return the result with the lowest reduced chi2."""
    def chi2(r):
        v = r.get("reduced_chi2")
        return v if (v is not None and np.isfinite(v)) else 1e18
    return min(results, key=chi2)


# ── Fit + plot ────────────────────────────────────────────────────────────────

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
    shape, peak_t = classify_shape(df_raw)
    tmp_csv, bands = make_fit_csv(df_raw, max_pts, mag_err)

    # Determine time range for adaptive t_b bounds
    ref_col = reference_mag_column(df_raw)
    t_vals = df_raw["time_days"].values
    g_vals = df_raw[ref_col].values
    valid  = g_vals < MAG_VALID_MAX
    t_min  = t_vals[valid].min() if valid.any() else t_vals.min()
    t_max  = t_vals[valid].max() if valid.any() else t_vals.max()

    try:
        # --- Primary fit(s): shape-adaptive constraints ----------------------
        strategies = fit_kwargs(shape, peak_t, t_min, t_max)
        candidates = []
        for kw in strategies:
            try:
                candidates.append(run_fit(tmp_csv, n_particles, n_iters, n_restarts, **kw))
            except Exception:
                pass

        if not candidates:
            raise RuntimeError("All primary strategies failed")
        res = best_of(candidates)

        # --- Retry pass if chi2 is still poor --------------------------------
        chi2_primary = res.get("reduced_chi2") or 1e18
        if chi2_primary > CHI2_RETRY:
            duration = t_max - t_min
            fallback_kw = dict(
                alpha1_min=ALPHA_MIN, alpha1_max=ALPHA_MAX,
                alpha2_min=ALPHA_MIN, alpha2_max=ALPHA_MAX,
                tb_lower=max(0.001 * duration, 0.001),   # 0.1% of span
                tb_upper=min(max(10.0 * duration, 500.0), 10_000.0),
            )
            try:
                res_fallback = run_fit(tmp_csv, n_particles * 2, n_iters, n_restarts + 2,
                                       **fallback_kw)
                chi2_fallback = res_fallback.get("reduced_chi2") or 1e18
                if chi2_fallback < chi2_primary:
                    res = res_fallback
                    shape = shape + "+fallback"
            except Exception:
                pass

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
        f"{name}  [{shape}]  —  SBPL fit (PSO + L-BFGS)\n"
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
        "name":       name,
        "shape":      shape,
        "n_obs":      res["n_obs"],
        "n_bands":    res["n_bands"],
        "alpha1":     p["alpha1"],
        "alpha1_err": p.get("alpha1_err", np.nan),
        "alpha2":     p["alpha2"],
        "alpha2_err": p.get("alpha2_err", np.nan),
        "beta":       p["beta"],
        "beta_err":   p.get("beta_err", np.nan),
        "tb":         p["tb"],
        "t0":         p["t0"],
        "logd":       p["logd"],
        "loga":       p["loga"],
        "reduced_chi2": chi2,
        "status":     "ok",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",   default=str(REPO_ROOT / "afterglow_photometry"))
    parser.add_argument("--output-dir",  default=str(REPO_ROOT / "afterglow_sbpl_plots_v2"))
    parser.add_argument("--max-pts",     type=int,   default=MAX_PTS_PER_BAND)
    parser.add_argument("--mag-err",     type=float, default=MAG_ERR)
    parser.add_argument("--n-particles", type=int,   default=50)
    parser.add_argument("--n-iters",     type=int,   default=300)
    parser.add_argument("--n-restarts",  type=int,   default=5)
    args = parser.parse_args()

    csvs = sorted(Path(args.input_dir).glob("*.csv"))
    if not csvs:
        print(f"No CSV files found in {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Found {len(csvs)} files — adaptive-prior fit → {args.output_dir}/\n")

    results = []
    for i, csv in enumerate(csvs, 1):
        name = csv.stem
        # Quick classify for label before fitting
        df_raw        = pd.read_csv(str(csv), comment="#")
        shape, peak_t = classify_shape(df_raw)
        print(f"[{i:3d}/{len(csvs)}] {name} [{shape:15s}] ...", end="", flush=True)
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
            chi2 = row["reduced_chi2"]
            flag = "  ✓" if (chi2 is not None and chi2 < 5) else "  !"
            print(f"  χ²={chi2:.3f}  α₁={row['alpha1']:.2f}  α₂={row['alpha2']:.2f}"
                  f"  tb={row['tb']:.2f}d{flag}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            results.append({"name": name, "shape": shape, "status": "failed",
                            "error": str(exc)})
            continue
        results.append(row)

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(args.output_dir, "fit_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    ok = summary_df[summary_df["status"] == "ok"].copy()
    ok["reduced_chi2"] = pd.to_numeric(ok["reduced_chi2"], errors="coerce")
    bins   = [0, 0.5, 1, 2, 5, 10, 50, 1e9]
    labels = ["<0.5", "0.5-1", "1-2", "2-5", "5-10", "10-50", ">50"]
    ok["bin"] = pd.cut(ok["reduced_chi2"], bins=bins, labels=labels)

    n_ok   = (summary_df["status"] == "ok").sum()
    n_fail = (summary_df["status"] != "ok").sum()
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed.")
    print(f"Summary saved to: {summary_path}")
    print("\n=== χ²_red distribution ===")
    print(ok["bin"].value_counts().sort_index().to_string())
    print(f"\nMedian χ²_red = {ok['reduced_chi2'].median():.3f}")
    print(f"Mean   χ²_red = {ok['reduced_chi2'].mean():.3f}")


if __name__ == "__main__":
    main()
