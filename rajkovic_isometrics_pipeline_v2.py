#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rajković obrada podataka ponovo — Isometric hamstrings (UNI + BI) KPI pipeline

What this script does:
- Reads all .tsv files in an input folder (format: ID_Pol_Strana_Polozaj_Ugao_Pokusaj.tsv)
- Detects case from filename (StranaCode):
    - 0 => BI (bilateral): processes BOTH force channels (col2 -> Bi_1, col3 -> Bi_2)
    - 1 => UNI_1: selects the dominant channel (col2 or col3) by signal activity
    - 2 => UNI_2: selects the dominant channel (col2 or col3) by signal activity
- Filtering (Matlab-equivalent): 2nd order Butterworth LPF @15 Hz, filtfilt, fs=1000 Hz
- KPI calculation: robust point recognition (baseline quiet window, sustained onset, early RFDmax, plateau start/end using rolling RMS of RFD)
- Timed RFDs: RFD_50/100/150/200/250 ms from onset, computed as (F(t)-F0)/t
- Exports one Excel with:
    - All: all processed trials (1 row per processed channel)
    - Uni_1, Uni_2: best trial per (ID, Polozaj, Ugao) using Score_geom = sqrt(PeakF * RFDmax)
    - Bi_1, Bi_2: best bilateral trial per (ID, Polozaj, Ugao), where "best" is selected per attempt using
          Score_bi_combined = sqrt( (PeakF_1 + PeakF_2) * (RFDmax_1 + RFDmax_2) )
      and then BOTH channel rows (Bi_1 and Bi_2) for that chosen attempt are kept.
"""

import argparse
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ------------------------- Helpers -------------------------

def parse_filename(fname: str):
    """
    Expected: ID_Pol_Strana_Polozaj_Ugao_Pokusaj.tsv
    Example: 101_1_0_1_2_1.tsv
    """
    base = os.path.basename(fname)
    m = re.match(r"(\d+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)\.tsv$", base)
    if not m:
        raise ValueError(f"Unexpected filename format: {base}")
    return dict(
        FileName=base,
        ID=int(m.group(1)),
        Pol=int(m.group(2)),
        StranaCode=int(m.group(3)),
        Polozaj=int(m.group(4)),
        Ugao=int(m.group(5)),
        Pokusaj=int(m.group(6)),
    )

def lowpass_filtfilt(x, fs=1000.0, cutoff=15.0, order=2):
    b, a = butter(order, cutoff / (fs / 2.0), btype="low")
    return filtfilt(b, a, x, axis=0)

def rolling_rms(x, win):
    if win <= 1:
        return np.abs(x)
    x2 = x * x
    kernel = np.ones(win) / win
    y = np.convolve(x2, kernel, mode="same")
    return np.sqrt(y)

def first_sustained(cond, k):
    """First index where cond is True for k consecutive samples."""
    if k <= 1:
        idx = np.argmax(cond)
        return int(idx) if cond[idx] else None
    c = np.convolve(cond.astype(int), np.ones(k, dtype=int), mode="valid")
    idx = np.where(c == k)[0]
    return int(idx[0]) if idx.size else None

def last_sustained(cond, k):
    """Last index where cond is True for k consecutive samples (returns start index)."""
    if k <= 1:
        idx = np.where(cond)[0]
        return int(idx[-1]) if idx.size else None
    c = np.convolve(cond.astype(int), np.ones(k, dtype=int), mode="valid")
    idx = np.where(c == k)[0]
    return int(idx[-1]) if idx.size else None

def best_quiet_window(x, fs=1000, search_s=2.0, win_s=0.4):
    """
    Finds the window (in the first search_s seconds) with minimal std,
    returns ((start,end), std).
    """
    n = int(min(len(x), search_s * fs))
    win = int(win_s * fs)
    if win < 20 or n < win:
        return (0, min(len(x), win)), float(np.std(x[: min(len(x), win)]))
    best = (0, win)
    best_std = np.inf
    for start in range(0, n - win):
        seg = x[start : start + win]
        s = float(np.std(seg))
        if s < best_std:
            best_std = s
            best = (start, start + win)
    return best, best_std

# ------------------------- KPI Core -------------------------

def compute_kpis(force_raw, fs=1000.0, cutoff=15.0, order=2):
    qc = []

    # Filter (Matlab-equivalent)
    f_f = lowpass_filtfilt(force_raw, fs=fs, cutoff=cutoff, order=order)

    # Baseline (quiet window)
    (b0, b1), _ = best_quiet_window(f_f, fs=int(fs))
    baseline_med = float(np.median(f_f[b0:b1]))
    baseline_std = float(np.std(f_f[b0:b1]))
    f = f_f - baseline_med

    # Sign normalization
    n_search = int(min(len(f), 5 * fs))
    peak_idx = int(np.argmax(np.abs(f[:n_search])))
    main_sign = 1.0 if f[peak_idx] >= 0 else -1.0
    f = main_sign * f

    # Onset detection (sustained)
    thr_on = max(5 * baseline_std, 10.0)  # N
    k_on = int(0.03 * fs)  # 30 ms
    onset = first_sustained(f > thr_on, k_on)
    if onset is None:
        qc.append("No onset detected")
        onset = 0

    # RFD from force
    rfd = np.gradient(f, 1 / fs)

    # RFDmax in early window after onset
    w_max = int(min(len(rfd) - onset, 1.5 * fs))
    idx_rfdmax = None
    if w_max > 5:
        seg = rfd[onset : onset + w_max]
        idx_local = int(np.argmax(seg))
        idx_rfdmax = onset + idx_local
        rfdmax = float(seg[idx_local])
    else:
        rfdmax = np.nan
        qc.append("RFDmax window too short")

    # Plateau detection using rolling RMS of RFD
    win_rms = int(0.025 * fs)  # 25 ms
    rfd_rms = rolling_rms(rfd, win_rms)
    noise_rms = float(np.median(rfd_rms[b0:b1]))
    thr_plateau = max(3 * noise_rms, 0.05 * (rfdmax if np.isfinite(rfdmax) else 0.0))
    k_pl = int(0.05 * fs)  # 50 ms sustained

    # Plateau start: after onset + 100 ms
    search_start = onset + int(0.10 * fs)
    pl_start = None
    if search_start < len(rfd_rms):
        pl_rel = first_sustained(rfd_rms[search_start:] < thr_plateau, k_pl)
        pl_start = (search_start + pl_rel) if pl_rel is not None else None
    if pl_start is None:
        qc.append("No plateau start")

    # Release (RFDmin) after plateau start
    idx_rfdmin = None
    start_rel = (pl_start + int(0.4 * fs)) if pl_start is not None else (onset + int(0.8 * fs))
    if start_rel < len(rfd):
        seg = rfd[start_rel:]
        if len(seg) > 10:
            idx_rfdmin = start_rel + int(np.argmin(seg))

    # Plateau end: last sustained low-RMS before release
    pl_end = None
    if pl_start is not None:
        end_search = idx_rfdmin if idx_rfdmin is not None else (len(rfd_rms) - 1)
        if end_search > pl_start + k_pl:
            last_rel = last_sustained(rfd_rms[pl_start:end_search] < thr_plateau, k_pl)
            if last_rel is not None:
                pl_end = pl_start + last_rel + k_pl
    if pl_end is None:
        qc.append("No plateau end")

    # Peak force before plateau end (fallback: full signal)
    peak_end = int(pl_end) if pl_end is not None else len(f)
    if onset < peak_end:
        seg = f[onset:peak_end]
        idx_peak = onset + int(np.argmax(seg))
        peakF = float(np.max(seg))
    else:
        idx_peak = int(np.argmax(f))
        peakF = float(np.max(f))

    # End of rise
    if pl_start is not None and pl_start < len(f):
        F_endRise = float(f[pl_start])
        t_endRise = float((pl_start - onset) / fs)
    else:
        F_endRise = np.nan
        t_endRise = np.nan

    # Times
    t_peak = float((idx_peak - onset) / fs)
    t_rfdmax = float((idx_rfdmax - onset) / fs) if idx_rfdmax is not None else np.nan

    # Timed RFDs (use F0 = mean of first 20 ms after onset)
    f0 = float(np.mean(f[onset : onset + int(0.02 * fs)])) if (onset + int(0.02 * fs) < len(f)) else float(f[onset])
    timed = {}
    for ms in (50, 100, 150, 200, 250):
        idx = onset + int(round(ms / 1000 * fs))
        timed[f"RFD_{ms}ms_Nps"] = float((f[idx] - f0) / (ms / 1000)) if idx < len(f) else np.nan

    # Integrals
    def trapz_seg(x, i0, i1):
        if i0 is None or i1 is None:
            return np.nan
        i0 = int(max(0, i0))
        i1 = int(min(len(x) - 1, i1))
        if i1 <= i0:
            return np.nan
        return float(np.trapezoid(x[i0 : i1 + 1], dx=1 / fs))

    J_to_RFDmax = trapz_seg(f, onset, idx_rfdmax)
    J_to_endRise = trapz_seg(f, onset, pl_start)

    # Plateau metrics
    if pl_start is not None and pl_end is not None and pl_end > pl_start:
        segF = f[pl_start:pl_end]
        segR = rfd[pl_start:pl_end]
        F_pl_mean = float(np.mean(segF))
        rmse_F = float(np.sqrt(np.mean((segF - F_pl_mean) ** 2)))
        rmse_RFD = float(np.sqrt(np.mean((segR - 0.0) ** 2)))
        plateau_dur = float((pl_end - pl_start) / fs)
    else:
        F_pl_mean = np.nan
        rmse_F = np.nan
        rmse_RFD = np.nan
        plateau_dur = np.nan

    qc_flag = 1 if qc else 0
    qc_note = "; ".join(qc) if qc else ""

    out = dict(
        PeakF_N=peakF,
        t_PeakF_s=t_peak,
        F_endRise_N=F_endRise,
        t_endRise_s=t_endRise,
        RFDmax_Nps=rfdmax,
        t_RFDmax_s=t_rfdmax,
        J_to_RFDmax_Ns=J_to_RFDmax,
        J_to_endRise_Ns=J_to_endRise,
        PlateauDur_s=plateau_dur,
        F_plateau_mean_N=F_pl_mean,
        rmse_F_N=rmse_F,
        rmse_RFD_Nps=rmse_RFD,
        QC_flag=qc_flag,
        QC_note=qc_note,
    )
    out.update(timed)
    return out

# ------------------------- Pipeline -------------------------

def process_folder(folder: str, fs=1000.0, cutoff=15.0, order=2, uni_ratio=1.3):
    rows = []
    for path in sorted(Path(folder).glob("*.tsv")):
        meta = parse_filename(path.name)

        data = pd.read_csv(path, sep="\t", header=None, dtype=float).values
        if data.shape[1] < 3:
            raise ValueError(f"{path.name}: expected 3 columns (time, col2, col3). Got {data.shape[1]}.")

        f1 = data[:, 1]
        f2 = data[:, 2]

        if meta["StranaCode"] == 0:
            # BI: always process both channels
            for col, lbl, f in ((2, "Bi_1", f1), (3, "Bi_2", f2)):
                k = compute_kpis(f, fs=fs, cutoff=cutoff, order=order)
                row = dict(
                    **meta,
                    Case="BI",
                    ChannelLabel=lbl,
                    ChannelCol=col,
                    fs_Hz=fs,
                    LPF_Hz=cutoff,
                    LPF_order=order,
                )
                peak = k["PeakF_N"]
                rfdmax = k["RFDmax_Nps"]
                row["Score_geom"] = float(np.sqrt(peak * rfdmax)) if (np.isfinite(peak) and np.isfinite(rfdmax) and peak > 0 and rfdmax > 0) else np.nan
                row.update(k)
                rows.append(row)

        else:
            # UNI: compute both and choose the dominant
            k1 = compute_kpis(f1, fs=fs, cutoff=cutoff, order=order)
            k2 = compute_kpis(f2, fs=fs, cutoff=cutoff, order=order)
            peak1, peak2 = k1["PeakF_N"], k2["PeakF_N"]

            chosen = 1
            if np.isfinite(peak1) and np.isfinite(peak2):
                if peak2 > uni_ratio * peak1:
                    chosen = 2
                elif peak1 > uni_ratio * peak2:
                    chosen = 1
                else:
                    # fallback
                    chosen = 1 if np.mean(f1) >= np.mean(f2) else 2
            elif np.isfinite(peak2) and not np.isfinite(peak1):
                chosen = 2

            col = 2 if chosen == 1 else 3
            k = k1 if chosen == 1 else k2
            row = dict(
                **meta,
                Case="UNI",
                ChannelLabel=f"Uni_{meta['StranaCode']}",
                ChannelCol=col,
                fs_Hz=fs,
                LPF_Hz=cutoff,
                LPF_order=order,
            )
            peak = k["PeakF_N"]
            rfdmax = k["RFDmax_Nps"]
            row["Score_geom"] = float(np.sqrt(peak * rfdmax)) if (np.isfinite(peak) and np.isfinite(rfdmax) and peak > 0 and rfdmax > 0) else np.nan
            row.update(k)
            rows.append(row)

    all_df = pd.DataFrame(rows)
    return all_df

def build_best_sheets(all_df: pd.DataFrame):
    # UNI best per (ID, Polozaj, Ugao, side)
    uni = all_df[all_df["Case"] == "UNI"].copy()
    uni_best = (uni.sort_values("Score_geom", ascending=False)
                .groupby(["ID", "Polozaj", "Ugao", "ChannelLabel"], as_index=False)
                .head(1))
    uni_1 = uni_best[uni_best["ChannelLabel"] == "Uni_1"].copy()
    uni_2 = uni_best[uni_best["ChannelLabel"] == "Uni_2"].copy()

    # BI combined selection per attempt using sums of both channels
    bi = all_df[all_df["Case"] == "BI"].copy()
    comb = (bi.groupby(["ID", "Polozaj", "Ugao", "Pokusaj"], as_index=False)
              .agg(PeakF_sum=("PeakF_N", "sum"),
                   RFDmax_sum=("RFDmax_Nps", "sum")))
    comb["Score_bi_combined"] = np.sqrt(comb["PeakF_sum"] * comb["RFDmax_sum"])

    comb_best = (comb.sort_values("Score_bi_combined", ascending=False)
                   .groupby(["ID", "Polozaj", "Ugao"], as_index=False)
                   .head(1))

    bi = bi.merge(comb[["ID", "Polozaj", "Ugao", "Pokusaj", "Score_bi_combined"]],
                  on=["ID", "Polozaj", "Ugao", "Pokusaj"], how="left")

    bi_best = bi.merge(comb_best[["ID", "Polozaj", "Ugao", "Pokusaj"]],
                       on=["ID", "Polozaj", "Ugao", "Pokusaj"], how="inner")

    bi_1 = bi_best[bi_best["ChannelLabel"] == "Bi_1"].copy()
    bi_2 = bi_best[bi_best["ChannelLabel"] == "Bi_2"].copy()

    # also return bi with combined scores (for All sheet)
    return uni_1, uni_2, bi_1, bi_2, bi

def write_excel(all_df: pd.DataFrame, out_path: str):
    uni_1, uni_2, bi_1, bi_2, bi_scored = build_best_sheets(all_df)

    # Add Score_bi_combined to BI rows in All
    all_out = all_df.copy()
    if "Score_bi_combined" not in all_out.columns:
        all_out = all_out.merge(
            bi_scored[["FileName", "ChannelLabel", "Score_bi_combined"]],
            on=["FileName", "ChannelLabel"],
            how="left"
        )

    # Column ordering
    meta_cols = ["FileName","ID","Pol","StranaCode","Polozaj","Ugao","Pokusaj","Case","ChannelLabel","ChannelCol",
                 "fs_Hz","LPF_Hz","LPF_order","QC_flag","QC_note"]
    preferred_kpis = [
        "PeakF_N","t_PeakF_s","F_endRise_N","t_endRise_s","RFDmax_Nps","t_RFDmax_s",
        "RFD_50ms_Nps","RFD_100ms_Nps","RFD_150ms_Nps","RFD_200ms_Nps","RFD_250ms_Nps",
        "J_to_RFDmax_Ns","J_to_endRise_Ns","PlateauDur_s","F_plateau_mean_N","rmse_F_N","rmse_RFD_Nps",
        "Score_geom","Score_bi_combined"
    ]
    cols = list(all_out.columns)
    kpi_cols = [c for c in cols if c not in meta_cols]
    ordered_kpis = [c for c in preferred_kpis if c in cols]
    rest = [c for c in kpi_cols if c not in ordered_kpis]
    all_out = all_out[meta_cols + ordered_kpis + rest]

    def align(df):
        for c in all_out.columns:
            if c not in df.columns:
                df[c] = np.nan
        return df[all_out.columns]

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_out.to_excel(writer, sheet_name="All", index=False)
        align(uni_1).to_excel(writer, sheet_name="Uni_1", index=False)
        align(uni_2).to_excel(writer, sheet_name="Uni_2", index=False)
        align(bi_1).to_excel(writer, sheet_name="Bi_1", index=False)
        align(bi_2).to_excel(writer, sheet_name="Bi_2", index=False)

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="Data", help="Folder with TSV files (default: Data)")
    ap.add_argument("--output", default="Rajkovic_Isometrics_Results_v2.xlsx", help="Output Excel file path")
    ap.add_argument("--fs", type=float, default=1000.0)
    ap.add_argument("--cutoff", type=float, default=15.0)
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--uni_ratio", type=float, default=1.3)
    args = ap.parse_args()

    all_df = process_folder(args.input, fs=args.fs, cutoff=args.cutoff, order=args.order, uni_ratio=args.uni_ratio)
    write_excel(all_df, args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()


