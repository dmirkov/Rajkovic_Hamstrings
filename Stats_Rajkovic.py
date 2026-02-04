# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 20:48:09 2026

@author: dmirk
"""

# bd_stats_pipeline.py
# -----------------------------------------
# BD statistics pipeline (paper-ready):
# 1) Mean over 3 trials per ID×Position×Angle×ChannelLabel
# 2) BI_sum = Bi_1 + Bi_2; UNI_sum = Uni_1 + Uni_2
# 3) BD = BI_sum / UNI_sum
# 4) Repeated-measures (ID as subject) Type-III ANOVA: F, p, partial eta^2
# 5) Pairwise comparisons (paired t-tests) with Bonferroni + Cohen's d (paired)
# 6) Plot (grayscale, single legend, dashed line at 1.0, no top/right spines)
# -----------------------------------------

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


# ----------------------------
# Configuration (edit if needed)
# ----------------------------

POS_MAP = {1: "Seated", 2: "Prone", 3: "Supine"}
ANG_MAP = {1: 90, 2: 120, 3: 150}

# KPI columns we use for BD outcomes
KPI_COLS = {
    "Fmax": "PeakF_N",
    "RFDmax": "RFDmax_Nps",
    "RFD50ms": "RFD_50ms_Nps",
    "RFD200ms": "RFD_200ms_Nps",
}

# Expected channel labels
BI_LABELS = ("Bi_1", "Bi_2")
UNI_LABELS = ("Uni_1", "Uni_2")


# ----------------------------
# Utilities
# ----------------------------

def p_format(p: float) -> str:
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d for paired samples: mean(diff)/sd(diff)."""
    d = x - y
    sd = np.std(d, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(np.mean(d) / sd)


def infer_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tries to standardize columns:
    - ID
    - Position (Seated/Prone/Supine)
    - Angle_deg (90/120/150)
    - ChannelLabel (Bi_1, Bi_2, Uni_1, Uni_2)
    - Trial (optional)
    """
    out = df.copy()

    # ID
    if "ID" not in out.columns:
        raise ValueError("Missing column 'ID'.")

    # Position
    if "Position" not in out.columns:
        if "Polozaj" in out.columns:
            out["Position"] = out["Polozaj"].map(POS_MAP)
        else:
            raise ValueError("Missing 'Position' or 'Polozaj' column.")
    # Angle
    if "Angle_deg" not in out.columns:
        if "Ugao" in out.columns:
            vals = pd.Series(out["Ugao"].dropna().unique())
            if set(vals.astype(int).tolist()).issubset({1, 2, 3}):
                out["Angle_deg"] = out["Ugao"].map(ANG_MAP)
            else:
                out["Angle_deg"] = out["Ugao"]
        else:
            raise ValueError("Missing 'Angle_deg' or 'Ugao' column.")

    # Channel label
    if "ChannelLabel" not in out.columns:
        raise ValueError("Missing column 'ChannelLabel' (expect Bi_1/Bi_2/Uni_1/Uni_2).")

    # Trial column optional (not strictly needed for mean)
    if "Trial" not in out.columns:
        if "Pokusaj" in out.columns:
            out["Trial"] = out["Pokusaj"]
        elif "trial" in out.columns:
            out["Trial"] = out["trial"]
        else:
            out["Trial"] = np.nan  # ok

    return out


def compute_bd_from_trials(df_trials: pd.DataFrame) -> pd.DataFrame:
    """
    From trial-level data -> BD by subject:
    1) mean across trials per ID×Position×Angle×ChannelLabel
    2) pivot to wide, compute BI_sum, UNI_sum, BD for each KPI
    Returns df with columns:
      ID, Position, Angle_deg,
      BD_Fmax, BD_RFDmax, BD_RFD50ms, BD_RFD200ms,
      BI_sum_*, UNI_sum_*
    """
    df = infer_columns(df_trials)

    # sanity: ensure KPI columns exist
    missing = [col for col in KPI_COLS.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing KPI columns in data: {missing}")

    # mean over trials
    means = (
        df.groupby(["ID", "Position", "Angle_deg", "ChannelLabel"], as_index=False)[list(KPI_COLS.values())]
        .mean()
    )

    # pivot wide
    wide = means.pivot_table(
        index=["ID", "Position", "Angle_deg"],
        columns="ChannelLabel",
        values=list(KPI_COLS.values()),
        aggfunc="first",
    )

    wide.columns = [f"{kpi}_{ch}" for kpi, ch in wide.columns]
    wide = wide.reset_index()

    # check required labels
    present_labels = set(df["ChannelLabel"].unique().tolist())
    for lab in BI_LABELS + UNI_LABELS:
        if lab not in present_labels:
            raise ValueError(f"ChannelLabel '{lab}' not found in dataset. Found: {sorted(present_labels)}")

    # compute sums + BD ratios
    out = wide[["ID", "Position", "Angle_deg"]].copy()

    for nice, kpi_col in KPI_COLS.items():
        bi_sum = wide[f"{kpi_col}_{BI_LABELS[0]}"] + wide[f"{kpi_col}_{BI_LABELS[1]}"]
        uni_sum = wide[f"{kpi_col}_{UNI_LABELS[0]}"] + wide[f"{kpi_col}_{UNI_LABELS[1]}"]

        out[f"BI_sum_{nice}"] = bi_sum
        out[f"UNI_sum_{nice}"] = uni_sum

        # avoid divide-by-zero
        out[f"BD_{nice}"] = np.where(uni_sum > 0, bi_sum / uni_sum, np.nan)

    # drop rows where BD missing across all
    bd_cols = [c for c in out.columns if c.startswith("BD_")]
    out = out.dropna(subset=bd_cols, how="all")

    return out


def rm_anova_type3(df_bd: pd.DataFrame, dv: str) -> dict:
    """
    Repeated-measures ANOVA implemented as:
      OLS: DV ~ C(ID) + C(Position)*C(Angle)
    Returns:
      dict with Position, Angle, Interaction stats including F, p, df1, df2, eta_p2
    """
    d = df_bd.copy()
    d["ID"] = d["ID"].astype("category")
    d["Position"] = pd.Categorical(d["Position"], categories=["Seated", "Prone", "Supine"])
    d["Angle"] = pd.Categorical(d["Angle_deg"], categories=[90, 120, 150])

    model = smf.ols(f"{dv} ~ C(ID) + C(Position)*C(Angle)", data=d).fit()
    aov = anova_lm(model, typ=3)

    ss_res = aov.loc["Residual", "sum_sq"]
    df_res = aov.loc["Residual", "df"]

    def get_term(term):
        ss = aov.loc[term, "sum_sq"]
        eta_p2 = float(ss / (ss + ss_res)) if (ss + ss_res) > 0 else np.nan
        return {
            "F": float(aov.loc[term, "F"]),
            "p": float(aov.loc[term, "PR(>F)"]),
            "df1": int(aov.loc[term, "df"]),
            "df2": int(df_res),
            "eta_p2": eta_p2,
        }

    out = {
        "Position": get_term("C(Position)"),
        "Angle": get_term("C(Angle)"),
        "Position×Angle": get_term("C(Position):C(Angle)"),
        "_anova_table": aov,   # for saving
    }
    return out


def pairwise_position_within_angle(df_bd: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Paired t-tests: Position comparisons within each Angle, Bonferroni(3) + Cohen's d."""
    results = []
    pairs = [("Seated", "Prone"), ("Seated", "Supine"), ("Prone", "Supine")]
    m = len(pairs)

    for ang in sorted(df_bd["Angle_deg"].dropna().unique()):
        sub = df_bd[df_bd["Angle_deg"] == ang][["ID", "Position", dv]].dropna()
        wide = sub.pivot(index="ID", columns="Position", values=dv)

        for a, b in pairs:
            if a not in wide.columns or b not in wide.columns:
                continue
            w = wide[[a, b]].dropna()
            if len(w) < 3:
                continue

            t, p = stats.ttest_rel(w[a].values, w[b].values)
            p_b = min(p * m, 1.0)
            d = cohen_d_paired(w[a].values, w[b].values)

            results.append({
                "DV": dv,
                "Angle_deg": int(ang),
                "Comparison": f"{a} vs {b}",
                "N": int(len(w)),
                "Mean_A": float(np.mean(w[a])),
                "Mean_B": float(np.mean(w[b])),
                "Mean_Diff(A-B)": float(np.mean(w[a] - w[b])),
                "t": float(t),
                "p_raw": float(p),
                "p_bonf": float(p_b),
                "Cohen_d_paired": float(d),
                "Significant_Bonf": bool(p_b < 0.05),
            })

    return pd.DataFrame(results)


def pairwise_angle_within_position(df_bd: pd.DataFrame, dv: str) -> pd.DataFrame:
    """Paired t-tests: Angle comparisons within each Position, Bonferroni(3) + Cohen's d."""
    results = []
    pairs = [(90, 120), (90, 150), (120, 150)]
    m = len(pairs)

    for pos in ["Seated", "Prone", "Supine"]:
        sub = df_bd[df_bd["Position"] == pos][["ID", "Angle_deg", dv]].dropna()
        wide = sub.pivot(index="ID", columns="Angle_deg", values=dv)

        for a, b in pairs:
            if a not in wide.columns or b not in wide.columns:
                continue
            w = wide[[a, b]].dropna()
            if len(w) < 3:
                continue

            t, p = stats.ttest_rel(w[a].values, w[b].values)
            p_b = min(p * m, 1.0)
            d = cohen_d_paired(w[a].values, w[b].values)

            results.append({
                "DV": dv,
                "Position": pos,
                "Comparison": f"{a}° vs {b}°",
                "N": int(len(w)),
                "Mean_A": float(np.mean(w[a])),
                "Mean_B": float(np.mean(w[b])),
                "Mean_Diff(A-B)": float(np.mean(w[a] - w[b])),
                "t": float(t),
                "p_raw": float(p),
                "p_bonf": float(p_b),
                "Cohen_d_paired": float(d),
                "Significant_Bonf": bool(p_b < 0.05),
            })

    return pd.DataFrame(results)


def plot_bd_2x2(df_bd: pd.DataFrame, out_png: Path,
               font_scale: float = 1.0,
               colors=None):
    """
    2x2 plot with one legend, grayscale, dashed line at 1.0, no top/right spines.
    Uses mean ± SD by Position×Angle.
    """
    if colors is None:
        # grayscale as in your figure: 120 light, 150 mid, 90 dark
        colors = {120: "#d9d9d9", 150: "#9e9e9e", 90: "#4d4d4d"}

    # font bump
    base = 14 * font_scale
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": base + 2,
        "axes.labelsize": base,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "legend.title_fontsize": base - 1,
    })

    positions = ["Seated", "Prone", "Supine"]
    angles = [120, 150, 90]  # legend order

    dvs = [
        ("BD_Fmax", "Fmax (N)"),
        ("BD_RFDmax", "RFDmax (N/s)"),
        ("BD_RFD50ms", "RFD50ms (N/s)"),
        ("BD_RFD200ms", "RFD200ms (N/s)"),
    ]

    # prepare mean/sd
    def summarise(col):
        g = df_bd.groupby(["Position", "Angle_deg"])[col].agg(["mean", "std"]).reset_index()
        return g

    summaries = {col: summarise(col) for col, _ in dvs}

    # Make 4 separate figs, then stitch into 2x2 with one legend
    from PIL import Image

    plot_paths = []
    for col, title in dvs:
        summ = summaries[col].copy()
        fig = plt.figure(figsize=(7.2, 4.8))
        ax = plt.gca()

        x = np.arange(len(positions))
        width = 0.22

        for i, ang in enumerate(angles):
            off = (i - 1) * width
            for j, pos in enumerate(positions):
                r = summ[(summ["Position"] == pos) & (summ["Angle_deg"] == ang)]
                if r.empty:
                    continue
                y = float(r["mean"].values[0])
                e = float(r["std"].values[0])
                ax.bar(x[j] + off, y, width, yerr=e, capsize=3,
                       color=colors[ang], edgecolor="none", linewidth=0)

        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.set_xlabel("Position")
        ax.set_ylabel("Magnitude of Deficit")
        ax.set_title(title)
        ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.8, alpha=0.5)
        ax.xaxis.grid(False)

        # y-lim
        max_val = float((summ["mean"] + summ["std"]).max())
        ax.set_ylim(0, max(1.05, max_val + 0.08))

        fig.tight_layout()
        p = out_png.parent / f"_tmp_{col}.png"
        fig.savefig(p, dpi=220)
        plt.close(fig)
        plot_paths.append(p)

    # Legend figure
    fig = plt.figure(figsize=(4.8, 0.9))
    ax = plt.gca()
    ax.axis("off")
    patches = [mpatches.Patch(color=colors[a], label=f"{a}°") for a in angles]
    ax.legend(handles=patches, title="Angle", loc="center", ncol=3, frameon=False)
    legend_path = out_png.parent / "_tmp_legend.png"
    fig.tight_layout(pad=0.2)
    fig.savefig(legend_path, dpi=220)
    plt.close(fig)

    # Stitch
    imgs = [Image.open(p) for p in plot_paths]
    leg = Image.open(legend_path)

    w = max(im.size[0] for im in imgs)
    h = max(im.size[1] for im in imgs)
    imgs = [im.resize((w, h)) for im in imgs]

    pad = 12
    canvas = Image.new("RGB", (w * 2 + pad, leg.size[1] + pad + h * 2 + pad), (255, 255, 255))
    canvas.paste(leg, ((canvas.size[0] - leg.size[0]) // 2, 0))
    y0 = leg.size[1] + pad
    canvas.paste(imgs[0], (0, y0))
    canvas.paste(imgs[1], (w + pad, y0))
    canvas.paste(imgs[2], (0, y0 + h + pad))
    canvas.paste(imgs[3], (w + pad, y0 + h + pad))
    canvas.save(out_png, quality=95)

    # cleanup tmp
    for p in plot_paths + [legend_path]:
        try:
            p.unlink()
        except Exception:
            pass


# ----------------------------
# Main pipeline
# ----------------------------

def run_pipeline(input_path: Path, sheet: str, out_xlsx: Path, out_png: Path, font_scale: float):
    # Load
    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path, sheet_name=sheet)
    else:
        df = pd.read_csv(input_path)

    # Compute BD by subject
    bd = compute_bd_from_trials(df)

    # ANOVA + pairwise for each DV
    anova_rows = []
    pair_pos_rows = []
    pair_ang_rows = []

    for nice in KPI_COLS.keys():
        dv = f"BD_{nice}"

        a = rm_anova_type3(bd, dv)
        for effect in ["Position", "Angle", "Position×Angle"]:
            d = a[effect]
            anova_rows.append({
                "DV": nice,
                "Effect": effect,
                "F": round(d["F"], 2),
                "df1": d["df1"],
                "df2": d["df2"],
                "p": p_format(d["p"]),
                "eta_p2": round(d["eta_p2"], 3),
            })

        # pairwise position within angle
        pp = pairwise_position_within_angle(bd, dv)
        if not pp.empty:
            pp.insert(0, "Outcome", nice)
            pair_pos_rows.append(pp)

        # pairwise angle within position
        pa = pairwise_angle_within_position(bd, dv)
        if not pa.empty:
            pa.insert(0, "Outcome", nice)
            pair_ang_rows.append(pa)

    anova_df = pd.DataFrame(anova_rows)
    pair_pos_df = pd.concat(pair_pos_rows, ignore_index=True) if pair_pos_rows else pd.DataFrame()
    pair_ang_df = pd.concat(pair_ang_rows, ignore_index=True) if pair_ang_rows else pd.DataFrame()

    # Summary mean(SD)
    summary_rows = []
    for nice in KPI_COLS.keys():
        col = f"BD_{nice}"
        g = bd.groupby(["Position", "Angle_deg"])[col].agg(["mean", "std", "count"]).reset_index()
        g["Outcome"] = nice
        summary_rows.append(g.rename(columns={"mean": "Mean", "std": "SD", "count": "N"}))
    bd_meansd = pd.concat(summary_rows, ignore_index=True)

    # Save Excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        bd.sort_values(["ID", "Position", "Angle_deg"]).to_excel(w, sheet_name="BD_by_subject", index=False)
        bd_meansd.to_excel(w, sheet_name="BD_MeanSD", index=False)
        anova_df.to_excel(w, sheet_name="ANOVA_TypeIII", index=False)
        if not pair_pos_df.empty:
            pair_pos_df.to_excel(w, sheet_name="Pairwise_Pos_in_Angle", index=False)
        if not pair_ang_df.empty:
            pair_ang_df.to_excel(w, sheet_name="Pairwise_Ang_in_Pos", index=False)

    # Plot
    plot_bd_2x2(bd, out_png, font_scale=font_scale)

    print(f"Saved Excel: {out_xlsx}")
    print(f"Saved Plot:  {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input .xlsx (trial-level) or .csv")
    ap.add_argument("--sheet", default="Paper_data", help="Excel sheet name (if .xlsx)")
    ap.add_argument("--out_xlsx", default="BD_stats_outputs.xlsx", help="Output Excel path")
    ap.add_argument("--out_png", default="BD_plots.png", help="Output plot (PNG) path")
    ap.add_argument("--font_scale", type=float, default=1.10, help="Global font scaling for plot (e.g., 1.1)")
    args = ap.parse_args()

    run_pipeline(
        input_path=Path(args.input),
        sheet=args.sheet,
        out_xlsx=Path(args.out_xlsx),
        out_png=Path(args.out_png),
        font_scale=args.font_scale,
    )


if __name__ == "__main__":
    main()
