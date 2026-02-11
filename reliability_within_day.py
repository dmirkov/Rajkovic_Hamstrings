#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Within-day reliability analysis for Rajkovic isometrics (seated position).

- Uses trial-level data: 3 consecutive trials per ID × position × angle × channel
- Position: Seated only (Polozaj=1)
- Angles: 120° and 150° (subjects were NOT measured at 90°)
- Separate analyses for: left_uni (Uni_1), right_uni (Uni_2), left_bi (Bi_1), right_bi (Bi_2)
- ICC(3,1): two-way mixed, single measure, consistency (standard for within-day reliability)
- Also reports: Mean(SD), CV%, SEM
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False

# ----------------------------
# Config
# ----------------------------

POS_MAP = {1: "Seated", 2: "Prone", 3: "Supine"}
ANG_MAP = {1: 90, 2: 120, 3: 150}

# Channel labels -> user-facing names
CHANNEL_NAMES = {
    "Uni_1": "left_uni",
    "Uni_2": "right_uni",
    "Bi_1": "left_bi",
    "Bi_2": "right_bi",
}

# KPI variables for reliability (all numeric outcomes from pipeline)
KPI_COLS = [
    "PeakF_N",
    "t_PeakF_s",
    "F_endRise_N",
    "t_endRise_s",
    "RFDmax_Nps",
    "t_RFDmax_s",
    "RFD_50ms_Nps",
    "RFD_100ms_Nps",
    "RFD_150ms_Nps",
    "RFD_200ms_Nps",
    "RFD_250ms_Nps",
    "J_to_RFDmax_Ns",
    "J_to_endRise_Ns",
    "PlateauDur_s",
    "F_plateau_mean_N",
    "rmse_F_N",
    "rmse_RFD_Nps",
    "Score_geom",
]
# Score_bi_combined only applies to BI channels; will skip for UNI


def load_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filter: Seated, angles 120 and 150 (exclude 90°)."""
    out = df.copy()
    if "Position" not in out.columns and "Polozaj" in out.columns:
        out["Position"] = out["Polozaj"].map(POS_MAP)
    if "Angle_deg" not in out.columns and "Ugao" in out.columns:
        out["Angle_deg"] = out["Ugao"].map(ANG_MAP)

    out = out[
        (out["Polozaj"] == 1) &  # Seated only
        (out["Ugao"].isin([2, 3]))  # 120° and 150° (exclude 90°)
    ].copy()
    return out


def compute_icc_manual(data: pd.DataFrame, targets: str, raters: str, ratings: str) -> dict:
    """
    ICC(3,1) via ANOVA when pingouin is not available.
    Returns dict with ICC, CI95%, pval, F, df1, df2.
    """
    d = data[[targets, raters, ratings]].dropna()
    if d.empty or d[ratings].nunique() < 2:
        return {"ICC": np.nan, "CI95%": (np.nan, np.nan), "pval": np.nan, "F": np.nan, "df1": np.nan, "df2": np.nan}

    n = d[targets].nunique()
    k = d[raters].nunique()
    if n < 2 or k < 2:
        return {"ICC": np.nan, "CI95%": (np.nan, np.nan), "pval": np.nan, "F": np.nan, "df1": np.nan, "df2": np.nan}

    # Two-way ANOVA: ratings ~ targets + raters
    from scipy import stats

    piv = d.pivot(index=targets, columns=raters, values=ratings)
    piv = piv.dropna(how="any")
    if len(piv) < 2:
        return {"ICC": np.nan, "CI95%": (np.nan, np.nan), "pval": np.nan, "F": np.nan, "df1": np.nan, "df2": np.nan}

    n, k = piv.shape
    grand_mean = piv.values.mean()
    ss_between = n * ((piv.mean(axis=1) - grand_mean) ** 2).sum()
    ss_within = ((piv.values - piv.mean(axis=1).values[:, None]) ** 2).sum()
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    ms_residual = ms_within  # simplified
    F = ms_between / ms_residual if ms_residual > 0 else np.nan
    df1, df2 = n - 1, n * (k - 1)
    pval = 1 - stats.f.cdf(F, df1, df2) if np.isfinite(F) else np.nan

    # ICC(3,1) = (MS_between - MS_within) / (MS_between + (k-1)*MS_within)
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within) if (ms_between + (k - 1) * ms_within) > 0 else np.nan
    icc = max(0, min(1, icc)) if np.isfinite(icc) else np.nan

    # Simple CI via F limits (Kistemaker et al.)
    if np.isfinite(F) and F > 0:
        F_lo = F / stats.f.ppf(0.975, df1, df2)
        F_hi = F * stats.f.ppf(0.975, df2, df1)
        icc_lo = (F_lo - 1) / (F_lo + k - 1) if (F_lo + k - 1) > 0 else 0
        icc_hi = (F_hi - 1) / (F_hi + k - 1) if (F_hi + k - 1) > 0 else 1
        icc_lo = max(0, icc_lo)
        icc_hi = min(1, icc_hi)
        ci95 = (icc_lo, icc_hi)
    else:
        ci95 = (np.nan, np.nan)

    return {"ICC": icc, "CI95%": ci95, "pval": pval, "F": F, "df1": df1, "df2": df2}


def compute_reliability(df_sub: pd.DataFrame, var: str) -> dict:
    """
    Compute ICC(3,1) and auxiliary stats for one variable.
    df_sub must have ID, Pokusaj, and var columns.
    """
    d = df_sub[["ID", "Pokusaj", var]].dropna()
    d = d[d[var].notna() & np.isfinite(d[var])]

    # Require exactly 3 trials per subject
    counts = d.groupby("ID").size()
    valid_ids = counts[counts == 3].index.tolist()
    d = d[d["ID"].isin(valid_ids)]

    if len(d) < 6 or d["ID"].nunique() < 2:
        return {
            "ICC_3_1": np.nan,
            "ICC_CI95_lo": np.nan,
            "ICC_CI95_hi": np.nan,
            "pval": np.nan,
            "N_subjects": int(d["ID"].nunique()),
            "Mean": np.nan,
            "SD": np.nan,
            "CV_pct": np.nan,
            "SEM": np.nan,
        }

    if HAS_PINGOUIN:
        try:
            icc_df = pg.intraclass_corr(data=d, targets="ID", raters="Pokusaj", ratings=var)
            row = icc_df[icc_df["Type"] == "ICC3"].iloc[0]
            icc_val = row["ICC"]
            ci = row["CI95%"]
            ci_lo = float(ci[0]) if hasattr(ci, "__getitem__") and len(ci) >= 2 else np.nan
            ci_hi = float(ci[1]) if hasattr(ci, "__getitem__") and len(ci) >= 2 else np.nan
            pval = row["pval"]
        except Exception:
            res = compute_icc_manual(d, "ID", "Pokusaj", var)
            icc_val = res["ICC"]
            ci_lo, ci_hi = res["CI95%"]
            pval = res["pval"]
    else:
        res = compute_icc_manual(d, "ID", "Pokusaj", var)
        icc_val = res["ICC"]
        ci_lo, ci_hi = res["CI95%"]
        pval = res["pval"]

    vals = d[var].values
    mean_val = float(np.mean(vals))
    sd_val = float(np.std(vals, ddof=1))
    cv_pct = 100 * sd_val / mean_val if mean_val != 0 else np.nan
    sem = sd_val * np.sqrt(1 - icc_val) if np.isfinite(icc_val) and icc_val < 1 else np.nan

    return {
        "ICC_3_1": icc_val,
        "ICC_CI95_lo": ci_lo,
        "ICC_CI95_hi": ci_hi,
        "pval": pval,
        "N_subjects": int(d["ID"].nunique()),
        "Mean": mean_val,
        "SD": sd_val,
        "CV_pct": cv_pct,
        "SEM": sem,
    }


def run_reliability(input_path: Path, output_path: Path):
    """Main reliability pipeline."""
    df = pd.read_excel(input_path, sheet_name="All")
    df = load_and_filter(df)

    # Check available KPI columns
    kpi_cols = [c for c in KPI_COLS if c in df.columns]
    # Skip Score_bi_combined for UNI (it's NaN there)
    if "Score_bi_combined" in df.columns:
        kpi_cols = [c for c in kpi_cols if c != "Score_bi_combined"]

    all_results = []

    for ch_label, ch_name in CHANNEL_NAMES.items():
        df_ch = df[df["ChannelLabel"] == ch_label].copy()
        if df_ch.empty:
            continue

        # Score_bi_combined: only for Bi channels
        vars_to_use = list(kpi_cols)
        if ch_label.startswith("Uni") and "Score_bi_combined" in df.columns:
            vars_to_use = [v for v in vars_to_use if v != "Score_bi_combined"]
        elif ch_label.startswith("Bi") and "Score_bi_combined" in df.columns and "Score_bi_combined" not in vars_to_use:
            vars_to_use = vars_to_use + ["Score_bi_combined"]

        for ang_deg in [120, 150]:
            df_ang = df_ch[df_ch["Angle_deg"] == ang_deg]
            if df_ang.empty:
                continue

            for var in vars_to_use:
                if var not in df_ang.columns:
                    continue
                res = compute_reliability(df_ang, var)
                all_results.append({
                    "Channel": ch_name,
                    "Angle_deg": ang_deg,
                    "Variable": var,
                    **res,
                })

    results_df = pd.DataFrame(all_results)

    # Format for output
    def fmt_icc(row):
        if pd.isna(row["ICC_3_1"]):
            return "—"
        ci = f"[{row['ICC_CI95_lo']:.3f}, {row['ICC_CI95_hi']:.3f}]"
        return f"{row['ICC_3_1']:.3f} {ci}"

    results_df["ICC_formatted"] = results_df.apply(fmt_icc, axis=1)

    # Save to Excel
    with pd.ExcelWriter(output_path, engine="openpyxl") as w:
        results_df.to_excel(w, sheet_name="Reliability", index=False)

        # Summary sheet: pivot for easier reading
        for ch_name in CHANNEL_NAMES.values():
            sub = results_df[results_df["Channel"] == ch_name]
            if sub.empty:
                continue
            sub.to_excel(w, sheet_name=ch_name[:31], index=False)  # Excel sheet name limit

    print(f"Saved: {output_path}")
    print(f"  Channels: {list(CHANNEL_NAMES.values())}")
    print(f"  Angles: 120°, 150° (seated only, 90° excluded)")
    print(f"  Variables: {len(kpi_cols)}")
    print(f"  Total ICC estimates: {len(results_df)}")


def main():
    ap = argparse.ArgumentParser(description="Within-day reliability (ICC) for seated position")
    ap.add_argument("--input", default="Rajkovic_Isometrics_Results_v2.xlsx", help="Input Excel with All sheet")
    ap.add_argument("--output", default="Reliability_WithinDay_Seated.xlsx", help="Output Excel")
    args = ap.parse_args()

    if not HAS_PINGOUIN:
        print("Note: 'pingouin' not installed. Using manual ICC implementation.")
        print("  Install with: pip install pingouin")

    run_reliability(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
