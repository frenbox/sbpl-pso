"""
Adaptive-prior SBPL batch fit directly from a long-format LSST afterglow parquet file
(e.g. afterglow_example_Jim/afterglow_models_sed_all_simulated_observations.parquet).

Unlike fit_afterglow_batch.py / fit_afterglow_batch_v2.py, which pivot a *wide* PS1-style
CSV (one column per band) into the long format sbpl_pso.fit() expects, this script's
input is already long-format per row: event_id, band, time (days), magnitude,
e_magnitude, detected. Each distinct event_id is treated as one light curve and fit
independently.

Bands used: g, r, i, z, y (LSST filter names lsstg/lsstr/lssti/lsstz/lssty are mapped to
their short form). lsstu is intentionally excluded to match the ztf(g,r,i) / lsst(g,r,i,z,y)
band scope used elsewhere in this repo.

Uses the same shape-adaptive alpha-bound strategy and retry-on-high-chi2 fallback as
fit_afterglow_batch_v2.py, and the real per-point e_magnitude column as the magnitude
error (instead of an assumed constant).

Usage:
    python scripts/fit_afterglow_parquet.py
    python scripts/fit_afterglow_parquet.py --parquet-path afterglow_example_Jim/afterglow_models_sed_all_simulated_observations.parquet \\
                                     --output-dir afterglow_sbpl_plots_parquet
    python scripts/fit_afterglow_parquet.py --max-events 5   # smoke test before a full run
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
#     python scripts/fit_afterglow_parquet.py ...
#     cd scripts && python fit_afterglow_parquet.py ...
REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Defaults ──────────────────────────────────────────────────────────────────
MAX_PTS_PER_BAND = 60
MAG_ERR_FLOOR    = 0.02   # floor applied to e_magnitude to avoid zero/negative errors
ZP               = 23.9
CHI2_RETRY       = 10.0

ALPHA_MAX        = 15.0
ALPHA_MIN        = -15.0

MIN_OBS          = 7      # matches the Rust fit_sbpl cutoff (n_obs < 7 -> empty result)
MIN_BANDS        = 2      # matches the Rust fit_sbpl cutoff (n_bands < 2 -> empty result)

# LSST filter name -> short band code. lsstu is deliberately excluded.
BAND_MAP = {"lsstg": "g", "lsstr": "r", "lssti": "i", "lsstz": "z", "lssty": "y"}
BAND_PREFERENCE = ("g", "r", "i", "z", "y")

BAND_COLOR = {"g": "#2ca02c", "r": "#d62728", "i": "#ff7f0e", "z": "#9467bd", "y": "#8c564b"}
BAND_LABEL = {"g": "g-LSST",  "r": "r-LSST",  "i": "i-LSST",  "z": "z-LSST",  "y": "y-LSST"}


# ── Light-curve shape classifier (mirrors fit_afterglow_batch_v2.py) ──────────

def classify_shape(t: np.ndarray, mag: np.ndarray) -> tuple[str, float | None]:
    """
    Return (shape, peak_t_days) for a single reference band's (time, mag) arrays.
    shape is one of: clear_peak / early_peak / late_peak / monotone_fade /
    monotone_rise / too_faint.
    """
    if len(t) < 4:
        return "too_faint", None

    order = np.argsort(t)
    t, mag = t[order], mag[order]

    peak_t = t[np.argmin(mag)]
    span   = t[-1] - t[0]
    if span <= 0:
        return "too_faint", None
    diff   = np.diff(mag)
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
    """Return a list of kwarg dicts for sbpl_pso.fit(), one per strategy to try."""
    duration = t_max - t_min

    tb_lo_default = max(0.01 * duration, 0.01)
    tb_hi_default = min(max(10.0 * duration, 500.0), 10_000.0)

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


def load_events(parquet_path: str, survey: str | None) -> pd.DataFrame:
    """Load the parquet, restrict to detected rows in known LSST bands."""
    df = pd.read_parquet(parquet_path)
    if survey is not None and "survey" in df.columns:
        df = df[df["survey"] == survey]
    df = df[df["detected"] == 1].copy()
    df["band_short"] = df["band"].map(BAND_MAP)
    df = df.dropna(subset=["band_short", "magnitude", "time (days)"])
    df["mag_err"] = df["e_magnitude"].clip(lower=MAG_ERR_FLOOR)
    return df


def make_fit_csv(event_df: pd.DataFrame, max_pts: int) -> tuple[str, list[str]]:
    """Write a per-event long CSV (mjd,mag,mag_err,filter) for sbpl_pso.fit()."""
    rows = []
    for band in BAND_PREFERENCE:
        sub = event_df[event_df["band_short"] == band].sort_values("time (days)")
        if sub.empty:
            continue
        idx = thin(sub["time (days)"].values, max_pts)
        for i in idx:
            r = sub.iloc[i]
            rows.append({
                "mjd": r["time (days)"], "mag": r["magnitude"],
                "mag_err": r["mag_err"], "filter": band,
            })

    if not rows:
        raise ValueError("No valid photometry after filtering")

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
    return sbpl_pso.fit(
        tmp_csv,
        n_particles=n_particles,
        n_iters=n_iters,
        n_restarts=n_restarts,
        **alpha_kw,
    )


def best_of(results: list[dict]) -> dict:
    def chi2(r):
        v = r.get("reduced_chi2")
        return v if (v is not None and np.isfinite(v)) else 1e18
    return min(results, key=chi2)


# ── Fit + plot ────────────────────────────────────────────────────────────────

def fit_and_plot(
    event_id: str,
    event_df: pd.DataFrame,
    output_dir: str,
    max_pts: int,
    n_particles: int,
    n_iters: int,
    n_restarts: int,
) -> dict:
    band_counts = event_df["band_short"].value_counts()
    ref_band = max(BAND_PREFERENCE, key=lambda b: band_counts.get(b, 0))
    ref_sub  = event_df[event_df["band_short"] == ref_band]
    shape, peak_t = classify_shape(ref_sub["time (days)"].values, ref_sub["magnitude"].values)

    tmp_csv, bands = make_fit_csv(event_df, max_pts)

    t_min = event_df["time (days)"].min()
    t_max = event_df["time (days)"].max()

    try:
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

        chi2_primary = res.get("reduced_chi2") or 1e18
        if chi2_primary > CHI2_RETRY:
            duration = t_max - t_min
            fallback_kw = dict(
                alpha1_min=ALPHA_MIN, alpha1_max=ALPHA_MAX,
                alpha2_min=ALPHA_MIN, alpha2_max=ALPHA_MAX,
                tb_lower=max(0.001 * duration, 0.001),
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

    t_min_obs = obs["time"].min()
    t_max_obs = obs["time"].max()
    t_dense   = np.linspace(max(t_min_obs * 0.8, 1e-3), t_max_obs * 1.2, 600)

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
        f"{event_id}  [{shape}]  —  SBPL fit (PSO + L-BFGS)\n"
        f"χ²_red = {chi2_str}  |  "
        f"α₁ = {p['alpha1']:.2f}  α₂ = {p['alpha2']:.2f}  "
        f"β = {p['beta']:.2f}  t_b = {p['tb']:.2f} d",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{event_id}_sbpl.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "event_id":   event_id,
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
    parser.add_argument("--parquet-path", default=str(
        REPO_ROOT / "afterglow_example_Jim"
                  / "afterglow_models_sed_all_simulated_observations.parquet"))
    parser.add_argument("--output-dir",  default=str(REPO_ROOT / "afterglow_sbpl_plots_parquet"))
    parser.add_argument("--survey",      default="lsst",
                        help="Restrict to this survey value (set to '' to disable filtering)")
    parser.add_argument("--min-obs",     type=int, default=MIN_OBS)
    parser.add_argument("--min-bands",   type=int, default=MIN_BANDS)
    parser.add_argument("--max-events",  type=int, default=None,
                        help="Fit only the first N events (smoke-testing)")
    parser.add_argument("--max-pts",     type=int,   default=MAX_PTS_PER_BAND)
    parser.add_argument("--n-particles", type=int,   default=50)
    parser.add_argument("--n-iters",     type=int,   default=300)
    parser.add_argument("--n-restarts",  type=int,   default=5)
    args = parser.parse_args()

    survey = args.survey or None
    df = load_events(args.parquet_path, survey)
    if df.empty:
        print(f"No detected g/r/i/z/y observations found in {args.parquet_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    event_ids = sorted(df["event_id"].unique())
    n_skipped_data = 0
    if args.max_events is not None:
        event_ids = event_ids[: args.max_events]

    print(f"Found {len(event_ids)} events — fitting → {args.output_dir}/\n")

    results = []
    for i, event_id in enumerate(event_ids, 1):
        event_df = df[df["event_id"] == event_id]
        n_obs   = len(event_df)
        n_bands = event_df["band_short"].nunique()
        print(f"[{i:4d}/{len(event_ids)}] {event_id} "
              f"(n_obs={n_obs}, n_bands={n_bands}) ...", end="", flush=True)

        if n_obs < args.min_obs or n_bands < args.min_bands:
            print(f"  SKIPPED (below min_obs={args.min_obs}/min_bands={args.min_bands})")
            n_skipped_data += 1
            continue

        try:
            row = fit_and_plot(
                event_id, event_df,
                output_dir=args.output_dir,
                max_pts=args.max_pts,
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
            results.append({"event_id": event_id, "status": "failed", "error": str(exc)})
            continue
        results.append(row)

    summary_df = pd.DataFrame(results)
    summary_path = os.path.join(args.output_dir, "fit_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    n_ok   = (summary_df["status"] == "ok").sum() if not summary_df.empty else 0
    n_fail = (summary_df["status"] != "ok").sum() if not summary_df.empty else 0
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed, {n_skipped_data} skipped "
          f"(insufficient data) out of {len(event_ids)} events considered.")
    print(f"Summary saved to: {summary_path}")

    if n_ok:
        ok = summary_df[summary_df["status"] == "ok"].copy()
        ok["reduced_chi2"] = pd.to_numeric(ok["reduced_chi2"], errors="coerce")
        bins   = [0, 0.5, 1, 2, 5, 10, 50, 1e9]
        labels = ["<0.5", "0.5-1", "1-2", "2-5", "5-10", "10-50", ">50"]
        ok["bin"] = pd.cut(ok["reduced_chi2"], bins=bins, labels=labels)
        print("\n=== χ²_red distribution ===")
        print(ok["bin"].value_counts().sort_index().to_string())
        print(f"\nMedian χ²_red = {ok['reduced_chi2'].median():.3f}")
        print(f"Mean   χ²_red = {ok['reduced_chi2'].mean():.3f}")


if __name__ == "__main__":
    main()
