# -*- coding: utf-8 -*-
"""
Rajkovic isometrics pipeline (Hamstrings)
- Reads all .tsv from input folder
- Detects case from filename: BI (Strana=0) or UNI_1/UNI_2 (Strana=1/2)
- Chooses channel(s):
    BI  -> processes both force channels (col2 and col3 in MATLAB => Python cols 1 and 2)
    UNI -> picks dominant channel (robust activity metric), with QC flags
- Filters force like MATLAB: 2nd order Butterworth low-pass @15 Hz, filtfilt, fs=1000
- KPI logic: robust baseline quiet window, sustained onset, RFD from force, plateau via rolling RMS RFD
- Exports:
    Sheet "All" -> all processed trials (one row per processed channel)
    Sheets "Uni_1", "Uni_2", "Bi_1", "Bi_2" -> best trial per ID×Položaj×Ugao within that channel label
      Best trial criterion: Score_geom = sqrt(PeakF_N * RFDmax_Nps)
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


# -----------------------------
# Config / helpers
# -----------------------------

@dataclass
class Meta:
    filename: str
    ID: int
    Pol: int
    StranaCode: int  # 0=BI, 1=UNI_1, 2=UNI_2
    Polozaj: int
    Ugao: int
    Pokusaj: int


def parse_filename(name: str) -> Optional[Meta]:
    """
    Expected pattern: ID_Pol_Strana_Polozaj_Ugao_Pokusaj.tsv
    Example: 101_1_0_1_2_1.tsv
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) >= 6 and all(p.isdigit() for p in parts[:6]):
        ID, Pol, Strana, Polozaj, Ugao, Pokusaj = map(int, parts[:6])
        return Meta(name, ID, Pol, Strana, Polozaj, Ugao, Pokusaj)

    # Fallback regex if underscores not perfectly present
    m = re.match(r"^\s*(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)[^\d]+(\d+)\s*$", stem)
    if m:
        ID, Pol, Strana, Polozaj, Ugao, Pokusaj = map(int, m.groups())
        return Meta(name, ID, Pol, Strana, Polozaj, Ugao, Pokusaj)
    return None


def butter_lowpass(cutoff_hz: float, fs_hz: float, order: int = 2):
    nyq = fs_hz / 2.0
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low")
    return b, a


def rolling_rms(x: np.ndarray, win: int) -> np.ndarray:
    """Rolling RMS using convolution (centered-ish by padding at ends)."""
    win = max(int(win), 1)
    kernel = np.ones(win, dtype=float) / float(win)
    x2 = x.astype(float) ** 2
    ma = np.convolve(x2, kernel, mode="same")
    return np.sqrt(ma)


def first_sustained(mask: np.ndarray, k: int) -> Optional[int]:
    """First index where mask is True for k consecutive samples."""
    k = max(int(k), 1)
    if mask.size < k:
        return None
    run = np.convolve(mask.astype(int), np.ones(k, dtype=int), mode="valid")
    idx = np.where(run == k)[0]
    if idx.size == 0:
        return None
    return int(idx[0])


def best_quiet_window(x: np.ndarray, search_end: int, win: int) -> Tuple[int, int]:
    """Find window with minimal std in first search_end samples."""
    n = min(len(x), int(search_end))
    win = min(int(win), n)
    if win < 10:
        return 0, n
    best_i = 0
    best_std = np.inf
    for i in range(0, n - win + 1, max(1, win // 10)):
        s = float(np.std(x[i:i + win]))
        if s < best_std:
            best_std = s
            best_i = i
    return best_i, best_i + win


def activity_metric(force: np.ndarray, base_med: float) -> float:
    """Robust 'activity' score: p99 - baseline_median (positive)."""
    p99 = float(np.percentile(force, 99))
    return max(0.0, p99 - float(base_med))


# -----------------------------
# KPI extraction (single force channel)
# -----------------------------

def compute_kpis_from_force(
    raw_force: np.ndarray,
    fs: float = 1000.0,
    cutoff: float = 15.0,
    order: int = 2,
    onset_thr_min_N: float = 10.0,
    onset_sigma: float = 5.0,
    onset_sustain_s: float = 0.030,
    rfd_early_window_s: float = 1.50,
    rfd_rms_win_s: float = 0.025,
    plateau_sustain_s: float = 0.050,
    plateau_search_start_after_rfdmax_s: float = 0.05,
    plateau_thr_frac_of_rfdmax: float = 0.10,
    plateau_thr_sigma: float = 3.0,
    release_min_after_onset_s: float = 0.50,
) -> Dict[str, float | int | str]:
    """
    Returns a dict with KPI values + QC flags.
    The signal is filtered like MATLAB and processed robustly.
    """
    out: Dict[str, float | int | str] = {}

    n = len(raw_force)
    if n < int(0.5 * fs):
        out["QC_flag"] = 1
        out["QC_note"] = "Signal too short"
        return out

    # Filter
    b, a = butter_lowpass(cutoff, fs, order=order)
    f = filtfilt(b, a, raw_force.astype(float))

    # Quiet window for baseline (search first 2.5s or full length)
    q_end = min(n, int(2.5 * fs))
    q_win = min(int(0.5 * fs), q_end)  # 0.5s window
    q0, q1 = best_quiet_window(f, search_end=q_end, win=q_win)
    base_med = float(np.median(f[q0:q1]))
    base_std = float(np.std(f[q0:q1]))

    # Baseline subtract
    f0 = f - base_med

    # Determine contraction sign (make contraction positive)
    # Use upper tail of absolute signal after baseline
    tail = f0
    if np.allclose(tail, 0):
        main_sign = 1.0
    else:
        # sign of median of top 1% absolute samples
        abs_tail = np.abs(tail)
        thr = np.percentile(abs_tail, 99)
        sel = tail[abs_tail >= thr]
        main_sign = float(np.sign(np.median(sel))) if sel.size else 1.0
        if main_sign == 0:
            main_sign = 1.0

    fp = main_sign * f0  # positive contraction

    # Onset detection on force (sustained)
    thr_on = max(onset_thr_min_N, onset_sigma * base_std)
    k_on = int(round(onset_sustain_s * fs))
    onset_idx = first_sustained(fp > thr_on, k_on)

    if onset_idx is None:
        out["QC_flag"] = 1
        out["QC_note"] = f"Onset not found (thr={thr_on:.2f}N)"
        return out

    # Compute RFD from force
    dt = 1.0 / fs
    rfd = np.gradient(fp, dt)

    # RFDmax in early window after onset
    early_end = min(n, onset_idx + int(round(rfd_early_window_s * fs)))
    if early_end <= onset_idx + 5:
        out["QC_flag"] = 1
        out["QC_note"] = "Early window too short"
        return out

    rfd_early = rfd[onset_idx:early_end]
    rfdmax = float(np.max(rfd_early))
    i_rfdmax_rel = int(np.argmax(rfd_early))
    rfdmax_idx = onset_idx + i_rfdmax_rel

    # Rolling RMS of RFD for plateau detection
    w_rms = int(round(rfd_rms_win_s * fs))
    rfd_rms = rolling_rms(rfd, w_rms)

    # Plateau threshold: max(noise-based, frac of RFDmax)
    noise_thr = plateau_thr_sigma * float(np.std(rfd[q0:q1]))
    frac_thr = plateau_thr_frac_of_rfdmax * max(rfdmax, 1e-6)
    thr_plat = max(noise_thr, frac_thr)

    # Plateau start search starts after RFDmax + small delay
    start_search = min(n - 1, rfdmax_idx + int(round(plateau_search_start_after_rfdmax_s * fs)))
    k_plat = int(round(plateau_sustain_s * fs))

    ps_rel = first_sustained(rfd_rms[start_search:] < thr_plat, k_plat)
    if ps_rel is None:
        plateau_start = None
    else:
        plateau_start = start_search + ps_rel

    # Release valley (RFD minimum) after onset + 0.5s
    rel_start = min(n - 1, onset_idx + int(round(release_min_after_onset_s * fs)))
    if rel_start < n - 10:
        rfd_tail = rfd[rel_start:]
        rfdmin = float(np.min(rfd_tail))
        rfdmin_idx = rel_start + int(np.argmin(rfd_tail))
    else:
        rfdmin = float(np.min(rfd))
        rfdmin_idx = int(np.argmin(rfd))

    # Plateau end: first sustained rise of rfd_rms above thr_plat between plateau_start and rfdmin_idx
    plateau_end = None
    if plateau_start is not None and rfdmin_idx > plateau_start + k_plat:
        search_seg = rfd_rms[plateau_start:rfdmin_idx]
        pe_rel = first_sustained(search_seg > thr_plat, k_plat)
        if pe_rel is None:
            plateau_end = rfdmin_idx
        else:
            plateau_end = plateau_start + pe_rel
            if plateau_end <= plateau_start:
                plateau_end = rfdmin_idx

    # If plateau wasn't found, fallback windows
    if plateau_start is None:
        # fallback: use endRise at 0.5s after onset (or at RFDmax)
        plateau_start = min(n - 1, onset_idx + int(round(0.50 * fs)))
        out["QC_flag"] = 1
        out["QC_note"] = "Plateau start fallback (not detected robustly)"
    else:
        out["QC_flag"] = 0
        out["QC_note"] = ""

    if plateau_end is None:
        plateau_end = min(n - 1, onset_idx + int(round(3.0 * fs)))  # fallback end
        out["QC_flag"] = 1
        out["QC_note"] = (out["QC_note"] + " | Plateau end fallback").strip(" |")

    plateau_start = int(plateau_start)
    plateau_end = int(max(plateau_end, plateau_start + 1))

    # Peak force in [onset, plateau_end]
    seg_end = min(n - 1, plateau_end)
    if seg_end <= onset_idx + 5:
        seg_end = min(n - 1, onset_idx + int(round(1.0 * fs)))
    seg = fp[onset_idx:seg_end + 1]
    peakf = float(np.max(seg))
    peak_idx = onset_idx + int(np.argmax(seg))

    # End of rise force = force at plateau_start
    f_endrise = float(fp[plateau_start])

    # Times (relative to onset)
    t_peak = (peak_idx - onset_idx) / fs
    t_endrise = (plateau_start - onset_idx) / fs
    t_rfdmax = (rfdmax_idx - onset_idx) / fs

    # Plateau stats
    plat_seg = fp[plateau_start:plateau_end]
    plat_rfd_seg = rfd[plateau_start:plateau_end]
    f_plat_mean = float(np.mean(plat_seg))
    rmse_f = float(np.sqrt(np.mean((plat_seg - f_plat_mean) ** 2)))
    rmse_rfd = float(np.sqrt(np.mean((plat_rfd_seg - 0.0) ** 2)))
    plateau_dur = (plateau_end - plateau_start) / fs

    # Impulses (N*s)
    J_to_rfdmax = float(np.trapezoid(fp[onset_idx:rfdmax_idx + 1], dx=dt))
    J_to_endrise = float(np.trapezoid(fp[onset_idx:plateau_start + 1], dx=dt))

    # Timed RFDs: (F(t) - F0) / t, with F0 = mean first 20 ms after onset
    f0_ref_end = min(n, onset_idx + int(round(0.020 * fs)))
    f0_ref = float(np.mean(fp[onset_idx:f0_ref_end])) if f0_ref_end > onset_idx else float(fp[onset_idx])

    timed = {}
    for ms, tsec in [(50, 0.05), (100, 0.10), (150, 0.15), (200, 0.20), (250, 0.25)]:
        idx = onset_idx + int(round(tsec * fs))
        if idx < n:
            timed[f"RFD_{ms}ms_Nps"] = float((fp[idx] - f0_ref) / tsec)
        else:
            timed[f"RFD_{ms}ms_Nps"] = np.nan

    # Score for best trial selection
    score_geom = math.sqrt(max(peakf, 0.0) * max(rfdmax, 0.0))

    # Export
    out.update({
        "PeakF_N": peakf,
        "t_PeakF_s": t_peak,
        "F_endRise_N": f_endrise,
        "t_endRise_s": t_endrise,
        "RFDmax_Nps": rfdmax,
        "t_RFDmax_s": t_rfdmax,
        "J_to_RFDmax_Ns": J_to_rfdmax,
        "J_to_endRise_Ns": J_to_endrise,
        "PlateauDur_s": plateau_dur,
        "F_plateau_mean_N": f_plat_mean,
        "rmse_F_N": rmse_f,
        "rmse_RFD_Nps": rmse_rfd,
        "Score_geom": score_geom,
        "OnsetThr_N": thr_on,
        "BaselineStd_N": base_std,
        "MainSign": int(np.sign(main_sign)),
    })
    out.update(timed)
    return out


# -----------------------------
# File processing
# -----------------------------

def read_tsv_force_channels(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads TSV and returns (force_col2, force_col3) in MATLAB sense:
    - MATLAB col2 => Python column index 1
    - MATLAB col3 => Python column index 2
    """
    df = pd.read_csv(path, sep="\t", header=None, engine="python")
    if df.shape[1] < 3:
        raise ValueError(f"TSV has {df.shape[1]} columns, expected >= 3 (time + 2 forces).")
    f2 = df.iloc[:, 1].to_numpy(dtype=float)
    f3 = df.iloc[:, 2].to_numpy(dtype=float)
    return f2, f3


def decide_uni_channel(
    f2_raw: np.ndarray,
    f3_raw: np.ndarray,
    fs: float,
    cutoff: float,
    order: int,
    uni_ratio: float = 1.3,
) -> Tuple[int, Dict[str, float | int | str]]:
    """
    Returns chosen column (2 or 3 in MATLAB sense) and QC notes.
    Uses robust activity metric after filtering and baseline correction (quiet window).
    """
    qc: Dict[str, float | int | str] = {"QC_uni_flag": 0, "QC_uni_note": ""}

    # Filter both quickly for channel decision
    b, a = butter_lowpass(cutoff, fs, order=order)
    f2 = filtfilt(b, a, f2_raw.astype(float))
    f3 = filtfilt(b, a, f3_raw.astype(float))

    # Baseline from quiet window (shared logic)
    n = len(f2)
    q_end = min(n, int(2.5 * fs))
    q_win = min(int(0.5 * fs), q_end)
    q0, q1 = best_quiet_window(f2, search_end=q_end, win=q_win)  # use f2 for window placement (usually same)
    base2 = float(np.median(f2[q0:q1]))
    base3 = float(np.median(f3[q0:q1]))

    a2 = activity_metric(f2, base2)
    a3 = activity_metric(f3, base3)

    # Ratio-based decision
    if a3 == 0 and a2 == 0:
        # fallback mean-based
        m2 = float(np.mean(f2))
        m3 = float(np.mean(f3))
        chosen = 2 if m2 >= m3 else 3
        qc["QC_uni_flag"] = 1
        qc["QC_uni_note"] = "UNI channel ambiguous (both low). Fallback to mean()."
        return chosen, qc

    if a2 >= uni_ratio * (a3 + 1e-9):
        return 2, qc
    if a3 >= uni_ratio * (a2 + 1e-9):
        return 3, qc

    # If close, pick the bigger activity but flag ambiguity
    chosen = 2 if a2 >= a3 else 3
    qc["QC_uni_flag"] = 1
    qc["QC_uni_note"] = f"UNI channel close (a2={a2:.2f}, a3={a3:.2f}). Picked higher activity."
    return chosen, qc


def case_and_channels(meta: Meta) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Returns (CaseName, list of (ChannelLabel, ChannelCol))
    ChannelCol is MATLAB sense (2 or 3).
    """
    if meta.StranaCode == 0:
        return "BI", [("Bi_1", 2), ("Bi_2", 3)]
    elif meta.StranaCode == 1:
        return "UNI", [("Uni_1", -1)]  # -1 means decide dynamically
    elif meta.StranaCode == 2:
        return "UNI", [("Uni_2", -1)]
    else:
        # Unknown -> treat as UNI and decide
        return "UNI", [(f"Uni_{meta.StranaCode}", -1)]


# -----------------------------
# Main
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    base = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    # >>> KEY CHANGE: defaults instead of required <<<
    p.add_argument("--input", default=str(base / "Data"),
                   help="Folder sa TSV fajlovima (default: ./Data pored skripta).")
    p.add_argument("--output", default=str(base / "Rajkovic_Isometrics_Results.xlsx"),
                   help="Izlazni Excel fajl (default: pored skripta).")

    p.add_argument("--fs", type=float, default=1000.0, help="Sampling rate (Hz). Default=1000.")
    p.add_argument("--cutoff", type=float, default=15.0, help="Low-pass cutoff (Hz). Default=15.")
    p.add_argument("--uni_ratio", type=float, default=1.3, help="UNI channel dominance ratio. Default=1.3.")
    return p


def main():
    args = build_argparser().parse_args()

    in_dir = Path(args.input)
    out_xlsx = Path(args.output)

    if not in_dir.exists() or not in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {in_dir}")

    files = sorted(in_dir.glob("*.tsv"))
    if not files:
        raise SystemExit(f"No .tsv files found in: {in_dir}")

    rows: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    for fpath in files:
        meta = parse_filename(fpath.name)
        if meta is None:
            errors.append({"FileName": fpath.name, "Error": "Filename parse failed"})
            continue

        try:
            f2_raw, f3_raw = read_tsv_force_channels(fpath)
        except Exception as e:
            errors.append({"FileName": fpath.name, "Error": f"Read TSV failed: {e}"})
            continue

        case_name, channels = case_and_channels(meta)

        # If UNI: decide channel once per file
        uni_choice = None
        uni_qc = {}
        if case_name == "UNI":
            try:
                uni_choice, uni_qc = decide_uni_channel(
                    f2_raw, f3_raw,
                    fs=args.fs, cutoff=args.cutoff, order=2,
                    uni_ratio=args.uni_ratio
                )
            except Exception as e:
                uni_choice = 2
                uni_qc = {"QC_uni_flag": 1, "QC_uni_note": f"UNI choose failed: {e}. Fallback col2."}

        for (ch_label, ch_col) in channels:
            # Determine actual column for UNI
            actual_col = ch_col
            if ch_col == -1:
                actual_col = int(uni_choice)

            raw_force = f2_raw if actual_col == 2 else f3_raw

            try:
                kpi = compute_kpis_from_force(
                    raw_force,
                    fs=args.fs,
                    cutoff=args.cutoff,
                    order=2
                )
            except Exception as e:
                errors.append({"FileName": fpath.name, "Error": f"KPI calc failed ({ch_label}, col{actual_col}): {e}"})
                continue

            row = {
                "FileName": meta.filename,
                "ID": meta.ID,
                "Pol": meta.Pol,
                "StranaCode": meta.StranaCode,
                "Polozaj": meta.Polozaj,
                "Ugao": meta.Ugao,
                "Pokusaj": meta.Pokusaj,
                "Case": case_name,
                "ChannelLabel": ch_label,
                "ChannelCol": actual_col,
                "fs_Hz": args.fs,
                "LPF_Hz": args.cutoff,
                "LPF_order": 2,
            }

            # Merge UNI QC info if relevant
            if case_name == "UNI":
                row.update(uni_qc)

            # Merge KPI outputs
            row.update(kpi)

            # Simple BI QC: flag if one BI channel is extremely small vs other (done later in All)
            rows.append(row)

    if not rows:
        raise SystemExit("No valid rows produced. Check errors and file naming/format.")

    df_all = pd.DataFrame(rows)

    # BI cross-check QC (optional): compare Bi_1 vs Bi_2 within same file
    # If one channel peak is much smaller, note it (does not change which channels we export).
    if "PeakF_N" in df_all.columns:
        bi = df_all[df_all["ChannelLabel"].isin(["Bi_1", "Bi_2"])].copy()
        if not bi.empty:
            for fname, g in bi.groupby("FileName"):
                if set(g["ChannelLabel"]) == {"Bi_1", "Bi_2"}:
                    p1 = float(g.loc[g["ChannelLabel"] == "Bi_1", "PeakF_N"].values[0])
                    p2 = float(g.loc[g["ChannelLabel"] == "Bi_2", "PeakF_N"].values[0])
                    mx = max(p1, p2)
                    mn = min(p1, p2)
                    if mx > 0 and mn / mx < 0.33:
                        idxs = df_all.index[df_all["FileName"] == fname].tolist()
                        for idx in idxs:
                            note = str(df_all.at[idx, "QC_note"]) if "QC_note" in df_all.columns else ""
                            add = "BI channels strongly unbalanced"
                            df_all.at[idx, "QC_note"] = (note + " | " + add).strip(" | ")
                            df_all.at[idx, "QC_flag"] = int(df_all.at[idx, "QC_flag"]) if "QC_flag" in df_all.columns else 1

    # Create best-trial sheets per channel label, grouped by ID×Polozaj×Ugao
    best_sheets: Dict[str, pd.DataFrame] = {}
    group_cols = ["ID", "Polozaj", "Ugao"]

    for label in ["Uni_1", "Uni_2", "Bi_1", "Bi_2"]:
        d = df_all[df_all["ChannelLabel"] == label].copy()
        if d.empty:
            best_sheets[label] = d
            continue

        # pick max Score_geom within each group
        d["_rank"] = d["Score_geom"]
        d = d.sort_values(group_cols + ["_rank", "PeakF_N", "RFDmax_Nps"], ascending=[True, True, True, False, False, False])

        # idx of best per group
        idx_best = d.groupby(group_cols, as_index=False)["Score_geom"].idxmax()["Score_geom"]
        idx_best = idx_best.dropna().astype(int)
        d_best = d.loc[idx_best].drop(columns=["_rank"], errors="ignore").reset_index(drop=True)


        best_sheets[label] = d_best

    # Write Excel
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="All", index=False)

        for label, d_best in best_sheets.items():
            d_best.to_excel(writer, sheet_name=label, index=False)

        # Optional: errors sheet
        if errors:
            pd.DataFrame(errors).to_excel(writer, sheet_name="Errors", index=False)

    print(f"Done. Saved: {out_xlsx}")
    if errors:
        print(f"Warnings: {len(errors)} file-level issues. See 'Errors' sheet.")


if __name__ == "__main__":
    main()
