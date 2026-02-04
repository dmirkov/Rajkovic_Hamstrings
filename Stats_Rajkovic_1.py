# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 20:56:01 2026

@author: dmirk
"""

# bd_stats_pipeline_select.py
# -----------------------------------------
# Paper-ready BD stats pipeline with interactive file+sheet selection from a folder.
# BD = (mean(BI_1)+mean(BI_2)) / (mean(UNI_1)+mean(UNI_2))
# Means are across trials per ID×Position×Angle×ChannelLabel.
# Repeated-measures Type-III ANOVA: DV ~ C(ID) + C(Position)*C(Angle)
# Pairwise (paired t-tests) + Bonferroni + Cohen's d (paired)
# Plot: grayscale, one legend, dashed line at 1.0, no top/right spines
# -----------------------------------------

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statsmodels.stats.anova import anova_lm
import statsmodels.formula.api as smf


# ----------------------------
# Config
# ----------------------------
POS_MAP = {1: "Seated", 2: "Prone", 3: "Supine"}
ANG_MAP = {1: 90, 2: 120, 3: 150}

KPI_COLS = {
    "Fmax": "PeakF_N",
    "RFDmax": "RFDmax_Nps",
    "RFD50ms": "RFD_50ms_Nps",
    "RFD200ms": "RFD_200ms_Nps",
}

BI_LABELS = ("Bi_1", "Bi_2")
UNI_LABELS = ("Uni_1", "Uni_2")

DEFAULT_SHEET = "Paper_data"


# ----------------------------
# Helpers
# ----------------------------
def p_format(p: float) -> str:
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".lstrip("0")


def cohen_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = x - y
    sd = np.std(d, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(np.mean(d) / sd)


def choose_from_folder(folder: Path) -> Path:
    exts = (".xlsx", ".xls", ".csv")
    files = sorted([p for p in folder.glob("*") if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No .xlsx/.xls/.csv files found in: {folder}")

    print("\nFiles found:")
    for i, f in enumerate(files, start=1):
        print(f"  [{i}] {f.name}")

    while True:
        s = input("Select file number: ").strip()
        if s.isdigit() and 1 <= int(s) <= len(files):
            return files[int(s) - 1]
        print("Invalid selection. Try again.")


def choose_excel_sheet(xlsx_path: Path, default_sheet: str = DEFAULT_SHEET) -> str:
    xl = pd.ExcelFile(xlsx_path)
    sheets = xl.sheet_names
    print("\nSheets found:")
    for i, sh in enumerate(sheets, start=1):
        tag = " (default)" if sh == default_sheet else ""
        print(f"  [{i}] {sh}{tag}")

    # if default exists, allow Enter to pick it
    prompt = f"Select sheet number (Enter for '{default_sheet}' if present): "
    while True:
        s = input(prompt).strip()
        if s == "" and default_sheet in sheets:
            return default_sheet
        if s.isdigit() and 1 <= int(s) <= len(sheets):
            return sheets[int(s) - 1]
        print("Invalid selection. Try again.")


def infer_columns(df: pd.DataFrame) -> pd.DataFrame:
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

    # Angle_deg
    if "Angle_deg" not in out.columns:
        if "Ugao" in out.columns:
            vals = pd.Series(out["Ugao"].dropna().unique())
            if set(vals.astype(int).tolist()).issubset({1, 2, 3}):
                out["Angle_deg"] = out["Ugao"].map(ANG_MAP)
            else:
                out["Angle_deg"] = out["Ugao"]
        else:
            raise ValueError("Missing 'Angle_deg' or 'Ugao' column.")

    # ChannelLabel
    if "ChannelLabel" not in out.columns:
        raise ValueError("Missing column 'ChannelLabel' (expect Bi_1/Bi_2/Uni_1/Uni_2).")

    return out


def compute_bd_from_trials(df_trials: pd.DataFrame) -> pd.DataFrame:
    df = infer_columns(df_trials)

    missing = [col for col in KPI_COLS.values() if col not in df.columns]
    if missing:
        raise ValueError(f"Missing KPI columns in data: {missing}")

    means = (
        df.groupby(["ID", "Position", "Angle_deg", "ChannelLabel"], as_index=False)[list(KPI_COLS.values())]
        .mean()
    )

    wide = means.pivot_table(
        index=["ID", "Position", "Angle_deg"],
        columns="ChannelLabel",
        values=list(KPI_COLS.values()),
        aggfunc="first",
    )
    wide.columns = [f"{kpi}_{ch}" for kpi, ch in wide.columns]
    wide = wide.reset_index()

    present_labels = set(df["ChannelLabel"].unique().tolist())
    for lab in BI_LABELS + UNI_LABELS:
        if lab not in present_labels:
            raise ValueError(f"ChannelLabel '{lab}' not found. Found: {sorted(present_labels)}")

    out = wide[["ID", "Position", "Angle_deg"]].copy()
    for nice, kpi_col in KPI_COLS.items():
        bi_sum = wide[f"{kpi_col}_{BI_LABELS[0]}"] + wide[f"{kpi_col}_{BI_LABELS[1]}"]
        uni_sum = wide[f"{kpi_col}_{UNI_LABELS[0]}"] + wide[f"{kpi_col}_{UNI_LABELS[1]}"]

        out[f"BI_sum_{nice}"] = bi_sum
        out[f"UNI_sum_{nice}"] = uni_sum
        out[f"BD_{nice}"] = np.where(uni_sum > 0, bi_sum / uni_sum, np.nan)

    bd_cols = [c for c in out.columns if c.startswith("BD_")]
    out = out.dropna(subset=bd_cols, how="all")
    return out


def rm_anova_type3(df_bd: pd.DataFrame, dv: str) -> dict:
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

    return {
        "Position": get_term("C(Position)"),
        "Angle": get_term("C(Angle)"),
        "Position×Angle": get_term("C(Position):C(Angle)"),
        "_anova_table": aov,
    }


def pairwise_position_within_angle(df_bd: pd.DataFrame, dv: str) -> pd.DataFrame:
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


def plot_bd_2x2(df_bd: pd.DataFrame, out_png: Path, font_scale: float = 1.15):
    colors = {120: "#d9d9d9", 150: "#9e9e9e", 90: "#4d4d4d"}  # 120 light, 150 mid, 90 dark
    positions = ["Seated", "Prone", "Supine"]
    angles = [120, 150, 90]

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

    from PIL import Image

    dvs = [
        ("BD_Fmax", "Fmax (N)"),
        ("BD_RFDmax", "RFDmax (N/s)"),
        ("BD_RFD50ms", "RFD50ms (N/s)"),
        ("BD_RFD200ms", "RFD200ms (N/s)"),
    ]

    def summarise(col):
        return df_bd.groupby(["Position", "Angle_deg"])[col].agg(["mean", "std"]).reset_index()

    plot_paths = []
    for col, title in dvs:
        summ = summarise(col)

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

    # cleanup
    for p in plot_paths + [legend_path]:
        try:
            p.unlink()
        except Exception:
            pass


def run_pipeline(input_file: Path | None,
                 input_folder: Path,
                 sheet: str | None,
                 out_xlsx: Path,
                 out_png: Path,
                 font_scale: float):
    # choose file if not provided
    if input_file is None:
        input_file = choose_from_folder(input_folder)

    # load
    if input_file.suffix.lower() in [".xlsx", ".xls"]:
        if sheet is None:
            sheet = choose_excel_sheet(input_file, default_sheet=DEFAULT_SHEET)
        df = pd.read_excel(input_file, sheet_name=sheet)
    elif input_file.suffix.lower() == ".csv":
        df = pd.read_csv(input_file)
    else:
        raise ValueError("Unsupported file type. Use .xlsx/.xls/.csv")

    # compute BD
    bd = compute_bd_from_trials(df)

    # stats
    anova_rows = []
    pair_rows = []

    for nice in KPI_COLS.keys():
        dv = f"BD_{nice}"
        a = rm_anova_type3(bd, dv)
        for eff in ["Position", "Angle", "Position×Angle"]:
            d = a[eff]
            anova_rows.append({
                "Outcome": nice,
                "Effect": eff,
                "F": round(d["F"], 2),
                "df1": d["df1"],
                "df2": d["df2"],
                "p": p_format(d["p"]),
                "eta_p2": round(d["eta_p2"], 3),
            })

        # pairwise positions within each angle (Bonferroni)
        pp = pairwise_position_within_angle(bd, dv)
        if not pp.empty:
            pp.insert(0, "Outcome", nice)
            pair_rows.append(pp)

    anova_df = pd.DataFrame(anova_rows)
    pair_df = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()

    # summary mean±SD per condition
    summary_rows = []
    for nice in KPI_COLS.keys():
        col = f"BD_{nice}"
        g = bd.groupby(["Position", "Angle_deg"])[col].agg(["mean", "std", "count"]).reset_index()
        g["Outcome"] = nice
        summary_rows.append(g.rename(columns={"mean": "Mean", "std": "SD", "count": "N"}))
    meansd = pd.concat(summary_rows, ignore_index=True)

    # save excel
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        bd.sort_values(["ID", "Position", "Angle_deg"]).to_excel(w, sheet_name="BD_by_subject", index=False)
        meansd.to_excel(w, sheet_name="BD_MeanSD", index=False)
        anova_df.to_excel(w, sheet_name="ANOVA_TypeIII", index=False)
        if not pair_df.empty:
            pair_df.to_excel(w, sheet_name="Pairwise_Pos_in_Angle", index=False)

    # plot
    plot_bd_2x2(bd, out_png, font_scale=font_scale)

    print("\nDONE")
    print(f"Input:   {input_file}")
    if input_file.suffix.lower() in [".xlsx", ".xls"]:
        print(f"Sheet:   {sheet}")
    print(f"Excel:   {out_xlsx}")
    print(f"Plot:    {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder", default=".", help="Folder with data files (.xlsx/.csv). Default=current folder.")
    ap.add_argument("--input_file", default=None, help="Optional: exact file name/path. If omitted, you will pick from folder.")
    ap.add_argument("--sheet", default=None, help="Optional: sheet name. If omitted (Excel), you will pick a sheet.")
    ap.add_argument("--out_xlsx", default="BD_stats_outputs.xlsx", help="Output Excel file")
    ap.add_argument("--out_png", default="BD_plots.png", help="Output plot PNG file")
    ap.add_argument("--font_scale", type=float, default=1.15, help="Plot font scale (e.g., 1.15)")
    args = ap.parse_args()

    folder = Path(args.input_folder).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    input_file = Path(args.input_file).expanduser().resolve() if args.input_file else None

    run_pipeline(
        input_file=input_file,
        input_folder=folder,
        sheet=args.sheet,
        out_xlsx=Path(args.out_xlsx),
        out_png=Path(args.out_png),
        font_scale=args.font_scale
    )


if __name__ == "__main__":
    main()
