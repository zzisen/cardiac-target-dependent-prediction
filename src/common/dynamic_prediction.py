"""Small public-data utilities shared by the Q-T empirical CM analyses."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

COMMON_HZ = [1.0, 3.1622776601683795, 10.0, 31.622776601683793]


def spectrum_from_row(row, freqs, real_pattern, imag_pattern):
    real = np.asarray([float(row[real_pattern.format(f=f)]) for f in freqs])
    imag = np.asarray([float(row[imag_pattern.format(f=f)]) for f in freqs])
    return np.r_[real, imag]


def _find_col(columns, prefix, component, f):
    # All candidate label lookup is exact within the supplied published grid.
    suffix = "re" if component == "re" else "im"
    target = float(f)
    options = []
    for col in columns:
        m = re.match(rf"^{re.escape(prefix)}{suffix}_(\d+(?:\.\d+)?)$", str(col))
        if m:
            options.append((abs(float(m.group(1)) - target), str(col), float(m.group(1))))
    if not options:
        raise KeyError((prefix, component, f))
    err, col, found = min(options)
    if err > max(1e-5, target * 5e-4):
        raise KeyError(f"No exact grid match for {prefix} {component} {f}; nearest {found}")
    return col


def load_human(path: Path):
    frame = pd.read_csv(path)
    freqs = [0.1778, 0.3162, 0.5623, 1.0, 1.778, 3.162, 5.623, 10.0, 17.78, 31.62, 56.23, 100.0]
    re_cols = [f"CM_f{f:g}Hz_real_MPa" for f in freqs]
    im_cols = [f"CM_f{f:g}Hz_imag_MPa" for f in freqs]
    # Read by fuzzy exact decimal label because source headers retain selected digits.
    def get_spec(row):
        r = [float(row[next(c for c in frame.columns if c.startswith("CM_f") and "_real_" in c and abs(float(c.split("f",1)[1].split("Hz",1)[0])-f)<2e-4)]) for f in freqs]
        i = [float(row[next(c for c in frame.columns if c.startswith("CM_f") and "_imag_" in c and abs(float(c.split("f",1)[1].split("Hz",1)[0])-f)<2e-4)]) for f in freqs]
        return np.r_[r, i]
    frame["_spectrum"] = [get_spec(row) for _, row in frame.iterrows()]
    samples = []
    for prep, sub in frame.groupby("preparation_id", sort=True):
        orig = {str(row["condition"]): row for _, row in sub.iterrows()}
        baseline = orig["baseline"]
        base_stress = float(baseline["observed_steady_stress_kPa"])
        for target in ("ATP0.1", "ATP1"):
            candidate = "ATP1" if target == "ATP0.1" else "ATP0.1"
            yrow = orig[target]
            crow = orig[candidate]
            samples.append({"unit_id": str(prep), "cluster_id": str(prep), "group": str(baseline["group"]),
                            "target": target, "candidate_condition": candidate, "freqs": np.asarray(freqs, float),
                            "baseline": baseline["_spectrum"], "baseline_stress": base_stress,
                            "candidate": crow["_spectrum"], "y": yrow["_spectrum"],
                            "source_muscle_id": baseline["source_muscle_id"]})
    return samples


def load_awinda(path: Path):
    frame = pd.read_csv(path)
    def colmap(prefix):
        return {float(str(c)[len(prefix):]): str(c) for c in frame.columns if str(c).startswith(prefix)}
    base_re = colmap("base_re_"); base_im = colmap("base_im_")
    base_freqs = sorted(base_re)
    samples = []
    for _, row in frame.iterrows():
        base = np.r_[[float(row[base_re[f]]) for f in base_freqs],
                     [float(row[base_im[f]]) for f in base_freqs]]
        for target, label in (("ATP0.1", "0.1"), ("ATP1", "1")):
            candidate_label = "1" if target == "ATP0.1" else "0.1"
            target_re = colmap(f"target_{label}_re_"); target_im = colmap(f"target_{label}_im_")
            cand_re = colmap(f"target_{candidate_label}_re_"); cand_im = colmap(f"target_{candidate_label}_im_")
            target_spec = np.r_[[float(row[target_re[f]]) for f in base_freqs],
                                [float(row[target_im[f]]) for f in base_freqs]]
            cand_spec = np.r_[[float(row[cand_re[f]]) for f in base_freqs],
                              [float(row[cand_im[f]]) for f in base_freqs]]
            samples.append({"unit_id": f"{row['AnimalID']}|{row['Mutation_x']}|{row['drug_state']}|{target}",
                            "cluster_id": str(row["AnimalID"]), "group": str(row["group"]),
                            "target": target, "candidate_condition": f"ATP{candidate_label}",
                            "freqs": np.asarray(base_freqs, float), "baseline": base,
                            "baseline_stress": float(row["stress_kPa"]), "candidate": cand_spec,
                            "y": target_spec, "n_fibres": int(row["n_fibres_total"])})
    return samples


def nearest_frequency_index(freqs, nominal_hz, tolerance=0.05):
    freqs = np.asarray(freqs, float)
    idx = int(np.argmin(np.abs(freqs - nominal_hz)))
    rel = abs(freqs[idx] - nominal_hz) / nominal_hz
    if rel > tolerance:
        raise ValueError(f"{nominal_hz} Hz is not represented within {tolerance:.1%}; closest is {freqs[idx]} Hz")
    return idx


def panel_feature_indices(freqs, panel_hz):
    n = len(freqs)
    inds = [nearest_frequency_index(freqs, f) for f in panel_hz]
    return np.asarray(inds + [n + i for i in inds], dtype=int)


def _zscore(train, test):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (train - mu) / sd, (test - mu) / sd


def predict_fold(samples, y, train_idx, test_idx, panel_hz, alpha=1.0, n_pcs=2):
    """Fit fixed-group-effects ridge; PCA and scaling are learned inside each fold."""
    base = np.vstack([s["baseline"] for s in samples])
    stress = np.asarray([s["baseline_stress"] for s in samples], float)[:, None]
    candidate = np.vstack([s["candidate"] for s in samples])
    group = np.asarray([s["group"] for s in samples], object)
    freqs = samples[0]["freqs"]
    btr = base[train_idx]
    bmu = btr.mean(axis=0)
    u, singular, vt = np.linalg.svd(btr - bmu, full_matrices=False)
    k = min(n_pcs, max(1, len(train_idx) - 1), vt.shape[0])
    loadings = vt[:k].T
    ptrain = (btr - bmu) @ loadings
    ptest = (base[test_idx] - bmu) @ loadings
    strn, stes = _zscore(stress[train_idx], stress[test_idx])
    xtr_parts = [ptrain, strn]
    xte_parts = [ptest, stes]
    if panel_hz is not None:
        fi = panel_feature_indices(freqs, panel_hz) if len(panel_hz) else np.asarray([], int)
        ctrain, ctest = _zscore(candidate[train_idx][:, fi], candidate[test_idx][:, fi])
        xtr_parts.append(ctrain)
        xte_parts.append(ctest)
    xtr = np.column_stack(xtr_parts)
    xte = np.column_stack(xte_parts)
    yall = np.asarray(y, float)
    ymu = yall[train_idx].mean(axis=0)
    ysd = yall[train_idx].std(axis=0)
    ysd = np.where(ysd > 1e-12, ysd, 1.0)
    yz = (yall[train_idx] - ymu) / ysd
    levels = sorted(set(group[train_idx].tolist()))
    gtr = np.column_stack([(group[train_idx] == lvl).astype(float) for lvl in levels])
    gte = np.column_stack([(group[test_idx] == lvl).astype(float) for lvl in levels])
    # Fixed effects: center X and Y within group, fit penalized slopes, then restore group means.
    xres = xtr.copy(); yres = yz.copy()
    for j, lvl in enumerate(levels):
        trmask = group[train_idx] == lvl
        xres[trmask] -= xtr[trmask].mean(axis=0)
        yres[trmask] -= yz[trmask].mean(axis=0)
    if xres.shape[1] > len(train_idx):
        beta = xres.T @ np.linalg.solve(xres @ xres.T + alpha * np.eye(len(train_idx)), yres)
    else:
        beta = np.linalg.solve(xres.T @ xres + alpha * np.eye(xres.shape[1]), xres.T @ yres)
    means = {}
    for lvl in levels:
        trmask = group[train_idx] == lvl
        means[lvl] = yz[trmask].mean(axis=0) - xtr[trmask].mean(axis=0) @ beta
    global_intercept = yz.mean(axis=0) - xtr.mean(axis=0) @ beta
    predz = np.vstack([(means.get(group[idx], global_intercept) + xte[j] @ beta)
                       for j, idx in enumerate(test_idx)])
    return predz * ysd + ymu


def spectrum_metrics(y, pred, freqs):
    y = np.asarray(y, float); pred = np.asarray(pred, float)
    n = len(freqs)
    e = pred - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    denom = float(np.sqrt(np.mean(y ** 2)))
    rmse_re = float(np.sqrt(np.mean(e[:n] ** 2)))
    rmse_im = float(np.sqrt(np.mean(e[n:] ** 2)))
    corr = float(np.corrcoef(y, pred)[0, 1]) if np.std(y) > 0 and np.std(pred) > 0 else float("nan")
    lf = np.log(np.asarray(freqs, float))
    if len(lf) > 1:
        w = np.empty(len(lf), float)
        w[0] = (lf[1] - lf[0]) / 2
        w[-1] = (lf[-1] - lf[-2]) / 2
        if len(lf) > 2:
            w[1:-1] = (lf[2:] - lf[:-2]) / 2
        w = w / w.sum()
    else:
        w = np.ones(len(lf))
    wboth = np.r_[w / 2, w / 2]
    log_rmse = float(np.sqrt(np.sum(wboth * e**2)))
    keep = np.asarray([abs(f - 1.0) / 1.0 > 0.05 for f in freqs])
    keep2 = np.r_[keep, keep]
    omit_1 = float(np.sqrt(np.mean(e[keep2]**2)) / np.sqrt(np.mean(y[keep2]**2)) * 100)
    return {"whole_spectrum_rmse": rmse, "whole_spectrum_nrmse_pct": float(rmse / denom * 100) if denom else float("nan"),
            "re_rmse": rmse_re, "im_rmse": rmse_im, "spectral_shape_correlation": corr,
            "log_frequency_weighted_rmse": log_rmse, "omit_1hz_nrmse_pct": omit_1}
