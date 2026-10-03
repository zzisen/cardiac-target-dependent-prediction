#!/usr/bin/env python
"""Mouse-aware Awinda target × frequency utility surface and nested target rule.

Run from any working directory:
  python run_yz_landscape.py build
  python run_yz_landscape.py score

The build phase writes all Y leave-one-target-out curves and a Z nested rule
freeze. The score phase verifies the immutable freeze hash before evaluating
outer held-out target utility.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

ROOT = REPO
HERE = ROOT / "analysis/03_awinda/landscape"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
INPUT = ROOT / "analysis/03_awinda/continuum/results/U_ALL_FREQUENCY_HELDOUT.csv"
TARGETS = np.array([0.05, 0.10, 0.25, 0.50, 1.00, 2.50], dtype=float)
BOOT_SEED = 20261003
N_BOOT = 10_000
NEAR_OPTIMAL_PP = 0.5
TOP_N = 24
MODEL_ORDER = ["Y0_frequency_only", "Y1_target_only", "Y2_additive", "Y3_interaction"]
MODEL_FEATURES = {
    "Y0_frequency_only": ["1", "xf", "xf2"],
    "Y1_target_only": ["1", "xt", "xt2"],
    "Y2_additive": ["1", "xt", "xt2", "xf", "xf2"],
    "Y3_interaction": ["1", "xt", "xt2", "xf", "xf2", "xt_xf"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spearman(x, y) -> float:
    if np.ptp(np.asarray(y, float)) <= 1e-12 or np.ptp(np.asarray(x, float)) <= 1e-12:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def kendall(x, y) -> float:
    if np.ptp(np.asarray(y, float)) <= 1e-12 or np.ptp(np.asarray(x, float)) <= 1e-12:
        return float("nan")
    return float(kendalltau(x, y).statistic)


def grid_band(frequencies: np.ndarray, f: float) -> str:
    i = int(np.argmin(np.abs(frequencies - f)))
    return ("low", "mid", "high")[min(2, i // max(1, int(np.ceil(len(frequencies) / 3))))]


def rank_bands(n: int) -> list[np.ndarray]:
    return list(np.array_split(np.arange(n), 3))


def coordinate_values(targets: np.ndarray, frequencies: np.ndarray):
    lt = np.log10(targets)
    lf = np.log10(frequencies)
    xt = (lt - lt.mean()) / lt.std(ddof=0)
    xf = (lf - lf.mean()) / lf.std(ddof=0)
    return xt, xf


def feature_matrix(xt_value: float, xf_values: np.ndarray, names: list[str]) -> np.ndarray:
    cols = {
        "1": np.ones_like(xf_values),
        "xt": np.full_like(xf_values, xt_value),
        "xt2": np.full_like(xf_values, xt_value**2),
        "xf": xf_values,
        "xf2": xf_values**2,
        "xt_xf": xt_value * xf_values,
    }
    return np.column_stack([cols[n] for n in names])


def fit_model(train_targets, curve_map, frequencies, model):
    xt_all, xf = coordinate_values(TARGETS, frequencies)
    tindex = {float(t): i for i, t in enumerate(TARGETS)}
    names = MODEL_FEATURES[model]
    xx, yy = [], []
    for t in train_targets:
        i = tindex[float(t)]
        xx.append(feature_matrix(float(xt_all[i]), xf, names))
        yy.append(curve_map[float(t)])
    X = np.vstack(xx)
    y = np.concatenate(yy)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return coef, names


def predict_model(target, frequencies, coef, names):
    xt_all, xf = coordinate_values(TARGETS, frequencies)
    i = int(np.where(TARGETS == float(target))[0][0])
    X = feature_matrix(float(xt_all[i]), xf, names)
    return X @ np.asarray(coef, float)


def top_set(curve, n=TOP_N):
    # Frequencies are in ascending order; stable sorting makes tie handling explicit.
    return set(np.argsort(-np.asarray(curve), kind="stable")[:n].tolist())


def top_jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def curve_features(t, freq, y, mice_matrix, bootstrap_indices):
    i_best = int(np.argmax(y))
    peak = float(y[i_best])
    out = {
        "target_ATP_mM": float(t),
        "best_frequency_Hz": float(freq[i_best]),
        "best_frequency_band": grid_band(freq, float(freq[i_best])),
        "peak_gain_pp": peak,
        "peak_minus_median_pp": float(peak - np.median(y)),
        "peak_minus_mean_pp": float(peak - np.mean(y)),
        "spearman_gain_log_frequency": spearman(np.log10(freq), y),
        "kendall_gain_log_frequency": kendall(np.log10(freq), y),
        "n_local_maxima": int(sum(y[j] >= y[j - 1] and y[j] >= y[j + 1] and
                                  (y[j] > y[j - 1] or y[j] > y[j + 1])
                                  for j in range(1, len(y) - 1))),
        "integrated_gain_over_log_frequency_pp": float(np.trapezoid(y, x=np.log10(freq)) /
                                                        (np.log10(freq[-1]) - np.log10(freq[0]))),
    }
    for cutoff in (0.25, 0.5, 1.0):
        near = (peak - y) <= cutoff
        out[f"near_optimal_n_within_{cutoff:g}pp"] = int(near.sum())
        out[f"near_optimal_min_Hz_within_{cutoff:g}pp"] = float(freq[near].min())
        out[f"near_optimal_max_Hz_within_{cutoff:g}pp"] = float(freq[near].max())
        # Connected plateau containing the peak; no unimodality assumption.
        left = right = i_best
        while left > 0 and near[left - 1]:
            left -= 1
        while right < len(freq) - 1 and near[right + 1]:
            right += 1
        out[f"peak_plateau_n_within_{cutoff:g}pp"] = int(right - left + 1)
        out[f"peak_plateau_width_Hz_within_{cutoff:g}pp"] = float(freq[right] - freq[left])
        out[f"peak_plateau_ratio_within_{cutoff:g}pp"] = float(freq[right] / freq[left])
    bands = rank_bands(len(freq))
    for name, idx in zip(("low", "mid", "high"), bands):
        out[f"mean_gain_{name}_band_pp"] = float(np.mean(y[idx]))
    out["low_minus_high_band_pp"] = out["mean_gain_low_band_pp"] - out["mean_gain_high_band_pp"]
    boot = mice_matrix[bootstrap_indices].mean(axis=1)
    iboot = np.argmax(boot, axis=1)
    band_names = np.asarray([grid_band(freq, float(x)) for x in freq])
    counts = np.bincount(iboot, minlength=len(freq))
    out["bootstrap_best_frequency_95pct_set"] = ";".join(
        f"{freq[j]:.4f}" for j in np.flatnonzero(counts / len(iboot) >= 0.025)
    )
    best_band = np.asarray([band_names[j] for j in iboot])
    for b in ("low", "mid", "high"):
        out[f"bootstrap_best_band_probability_{b}"] = float(np.mean(best_band == b))
    return out


def load_mouse_curves():
    raw = pd.read_csv(INPUT)
    required = {"target_ATP_mM", "heldout_mouse", "group", "frequency_Hz", "gain_pp",
                "baseline_only_nrmse_pct", "augmented_nrmse_pct"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"U output missing columns: {sorted(missing)}")
    targets = np.sort(raw.target_ATP_mM.unique().astype(float))
    frequencies = np.sort(raw.frequency_Hz.unique().astype(float))
    mice = sorted(raw.heldout_mouse.astype(str).unique())
    if not np.array_equal(targets, TARGETS) or len(frequencies) != 72 or len(mice) != 10:
        raise ValueError("U frozen targets, 72-frequency grid, or ten-mouse set changed")
    mouse = (raw.groupby(["target_ATP_mM", "heldout_mouse", "frequency_Hz"], as_index=False)
             .agg(gain_pp=("gain_pp", "mean"),
                  baseline_only_nrmse_pct=("baseline_only_nrmse_pct", "mean"),
                  augmented_nrmse_pct=("augmented_nrmse_pct", "mean"),
                  n_state_rows=("group", "size")))
    mouse.to_csv(OUT / "Y_MOUSE_LEVEL_UTILITY.csv", index=False)
    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(mice), size=(N_BOOT, len(mice)))
    wide = {}
    rows = []
    for t in TARGETS:
        sub = mouse[mouse.target_ATP_mM.eq(t)]
        mat = (sub.pivot(index="heldout_mouse", columns="frequency_Hz", values="gain_pp")
               .reindex(index=mice, columns=frequencies).to_numpy(float))
        if not np.isfinite(mat).all():
            raise ValueError(f"Incomplete mouse × frequency curve for target {t}")
        wide[float(t)] = mat.mean(axis=0)
        bmeans = mat[boot_idx].mean(axis=1)
        point = mat.mean(axis=0)
        lo, hi = np.quantile(bmeans, [0.025, 0.975], axis=0)
        for j, f in enumerate(frequencies):
            per_mouse = mat[:, j]
            rows.append({"target_ATP_mM": float(t), "frequency_Hz": float(f),
                         "frequency_band": grid_band(frequencies, float(f)),
                         "mean_gain_pp": float(point[j]), "median_gain_pp": float(np.median(per_mouse)),
                         "bootstrap_95pct_low_pp": float(lo[j]), "bootstrap_95pct_high_pp": float(hi[j]),
                         "mean_baseline_only_nrmse_pct": float(sub[sub.frequency_Hz.eq(f)].baseline_only_nrmse_pct.mean()),
                         "mean_augmented_nrmse_pct": float(sub[sub.frequency_Hz.eq(f)].augmented_nrmse_pct.mean()),
                         "improved_mice": int((per_mouse > 0).sum()), "n_mice": len(mice),
                         "rank_within_target": int(rankdata(-point, method="min")[j]),
                         "near_optimal_within_0.5pp": bool(point.max() - point[j] <= NEAR_OPTIMAL_PP)})
    landscape = pd.DataFrame(rows)
    landscape.to_csv(OUT / "Y_FULL_UTILITY_LANDSCAPE.csv", index=False)
    feat = []
    for t in TARGETS:
        sub = landscape[landscape.target_ATP_mM.eq(t)].sort_values("frequency_Hz")
        mat = (mouse[mouse.target_ATP_mM.eq(t)]
               .pivot(index="heldout_mouse", columns="frequency_Hz", values="gain_pp")
               .reindex(index=mice, columns=frequencies).to_numpy(float))
        feat.append(curve_features(float(t), frequencies, sub.mean_gain_pp.to_numpy(float), mat, boot_idx))
    features = pd.DataFrame(feat)
    features.to_csv(OUT / "Y_LANDSCAPE_FEATURES.csv", index=False)
    return targets, frequencies, mice, wide, landscape


def run_y_loto(targets, frequencies, curves):
    pred_rows = []
    summary_rows = []
    for held in targets:
        train = [float(t) for t in targets if float(t) != float(held)]
        obs = curves[float(held)]
        for model in MODEL_ORDER:
            coef, names = fit_model(train, curves, frequencies, model)
            pred = predict_model(float(held), frequencies, coef, names)
            i_true = int(np.argmax(obs)); i_pred = int(np.argmax(pred))
            ts, ps = top_set(obs), top_set(pred)
            row = {"heldout_target_ATP_mM": float(held), "model": model,
                   "outer_training_targets_mM": ";".join(f"{x:g}" for x in train),
                   "rmse_pp": float(np.sqrt(np.mean((obs - pred) ** 2))),
                   "mae_pp": float(np.mean(np.abs(obs - pred))),
                   "spearman_rank_correlation": spearman(obs, pred),
                   "kendall_rank_correlation": kendall(obs, pred),
                   "observed_best_frequency_Hz": float(frequencies[i_true]),
                   "predicted_best_frequency_Hz": float(frequencies[i_pred]),
                   "best_frequency_error_log10_Hz": float(abs(np.log10(frequencies[i_pred]) - np.log10(frequencies[i_true]))),
                   "observed_best_band": grid_band(frequencies, float(frequencies[i_true])),
                   "predicted_best_band": grid_band(frequencies, float(frequencies[i_pred])),
                   "best_band_hit": grid_band(frequencies, float(frequencies[i_true])) == grid_band(frequencies, float(frequencies[i_pred])),
                   "top24_set_jaccard": top_jaccard(ts, ps),
                   "predicted_best_gain_observed_pp": float(obs[i_pred]),
                   "oracle_gain_observed_pp": float(obs[i_true]),
                   "regret_pp": float(obs[i_true] - obs[i_pred]),
                   "oracle_retention_fraction": float(obs[i_pred] / obs[i_true]) if obs[i_true] > 0 else np.nan,
                   "coefficient_vector": ";".join(f"{x:.12g}" for x in coef)}
            summary_rows.append(row)
            for j, f in enumerate(frequencies):
                pred_rows.append({"heldout_target_ATP_mM": float(held), "model": model,
                                  "frequency_Hz": float(f), "observed_gain_pp": float(obs[j]),
                                  "predicted_gain_pp": float(pred[j]), "prediction_error_pp": float(pred[j] - obs[j]),
                                  "outer_training_targets_mM": row["outer_training_targets_mM"]})
    p = pd.DataFrame(pred_rows)
    s = pd.DataFrame(summary_rows)
    p.to_csv(OUT / "Y_LOTO_TARGET_PREDICTION.csv", index=False)
    s.to_csv(OUT / "Y_LOTO_TARGET_SUMMARY.csv", index=False)
    comparison = (s.groupby("model", as_index=False)
                  .agg(n_outer_targets=("heldout_target_ATP_mM", "nunique"),
                       mean_rmse_pp=("rmse_pp", "mean"), median_rmse_pp=("rmse_pp", "median"),
                       mean_mae_pp=("mae_pp", "mean"), mean_spearman=("spearman_rank_correlation", "mean"),
                       mean_kendall=("kendall_rank_correlation", "mean"),
                       best_band_hit_fraction=("best_band_hit", "mean"),
                       mean_best_frequency_error_log10=("best_frequency_error_log10_Hz", "mean"),
                       mean_regret_pp=("regret_pp", "mean"),
                       mean_oracle_retention=("oracle_retention_fraction", "mean")))
    comparison.to_csv(OUT / "Y_MODEL_COMPARISON.csv", index=False)
    return s


def pava_increasing(y):
    """Equal-weight pool-adjacent-violators; returns a nondecreasing fit."""
    blocks = [[float(v), 1, i, i] for i, v in enumerate(y)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0]:
            i += 1
            continue
        w = blocks[i][1] + blocks[i + 1][1]
        m = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / w
        blocks[i:i + 2] = [[m, w, blocks[i][2], blocks[i + 1][3]]]
        i = max(0, i - 1)
    fit = np.empty(len(y), float)
    for mean, _, a, b in blocks:
        fit[a:b + 1] = mean
    return fit


def build_z_freeze(targets, frequencies, curves):
    folds = []
    rule_compare = []
    for held in targets:
        outer_train = [float(t) for t in targets if float(t) != float(held)]
        inner_errors = {}
        for model in MODEL_ORDER:
            errs = []
            for inner in outer_train:
                inner_train = [float(t) for t in outer_train if float(t) != float(inner)]
                coef, names = fit_model(inner_train, curves, frequencies, model)
                pred = predict_model(float(inner), frequencies, coef, names)
                errs.append(float(np.sqrt(np.mean((curves[float(inner)] - pred) ** 2))))
            inner_errors[model] = float(np.mean(errs))
        # Tie-break toward the lower-complexity prespecified family.
        chosen = min(MODEL_ORDER, key=lambda m: (inner_errors[m], MODEL_ORDER.index(m)))
        coef, names = fit_model(outer_train, curves, frequencies, chosen)
        predicted_curve = predict_model(float(held), frequencies, coef, names)
        zidx = int(np.argmax(predicted_curve))
        # A flat target-only curve has no frequency ranking; use frozen conventional fallback.
        conventional_idx = int(np.argmin(np.abs(frequencies - 1.0)))
        if np.ptp(predicted_curve) <= 1e-12:
            zidx = conventional_idx
        train_opt_t = np.array(outer_train, float)
        train_opt_f = np.array([frequencies[int(np.argmax(curves[t]))] for t in outer_train], float)
        nearest_t = float(outer_train[int(np.argmin(np.abs(np.log10(train_opt_t) - np.log10(float(held)))) )])
        nearest_f = float(frequencies[int(np.argmax(curves[nearest_t]))])
        global_curve = np.mean(np.vstack([curves[t] for t in outer_train]), axis=0)
        global_f = float(frequencies[int(np.argmax(global_curve))])
        monotone_f = pava_increasing(np.log10(train_opt_f[np.argsort(train_opt_t)]))
        sorted_t = np.sort(train_opt_t)
        log_pred_monotone = float(np.interp(np.log10(float(held)), np.log10(sorted_t), monotone_f))
        monotone_hz = float(frequencies[int(np.argmin(np.abs(np.log10(frequencies) - log_pred_monotone)))])
        y0coef, y0names = fit_model(outer_train, curves, frequencies, "Y0_frequency_only")
        y0curve = predict_model(float(held), frequencies, y0coef, y0names)
        y0_f = float(frequencies[int(np.argmax(y0curve))])
        y1coef, y1names = fit_model(outer_train, curves, frequencies, "Y1_target_only")
        y1curve = predict_model(float(held), frequencies, y1coef, y1names)
        y1_f = (float(frequencies[int(np.argmax(y1curve))]) if np.ptp(y1curve) > 1e-12
                else float(frequencies[conventional_idx]))
        panels = {
            "target_conditioned_nested_selected": float(frequencies[zidx]),
            "fixed_conventional_1Hz": float(frequencies[conventional_idx]),
            "random_uniform_candidate_grid": None,
            "nearest_training_target": nearest_f,
            "global_average_best": global_f,
            "frequency_only_Y0": y0_f,
            "target_only_Y1_conventional_tie_break": y1_f,
            "monotonic_isotonic_best_frequency_rule": monotone_hz,
        }
        fold = {"outer_heldout_target_ATP_mM": float(held),
                "outer_training_targets_mM": outer_train,
                "inner_LOTO_RMSE_by_model_pp": inner_errors,
                "selected_model": chosen,
                "feature_names": names,
                "coefficients": [float(x) for x in coef],
                "predicted_utility_curve_pp": [float(x) for x in predicted_curve],
                "predicted_best_frequency_Hz": float(frequencies[zidx]),
                "recommendations_Hz": panels,
                "nearest_training_target_mM": nearest_t,
                "frequency_grid_sha256": hashlib.sha256(np.asarray(frequencies, dtype="<f8").tobytes()).hexdigest()}
        folds.append(fold)
        for model, e in inner_errors.items():
            rule_compare.append({"outer_heldout_target_ATP_mM": float(held), "model": model,
                                 "inner_LOTO_mean_RMSE_pp": e, "selected_for_fold": model == chosen})
    version = {
        "title": "Nested target-conditioned frequency design rule",
        "status": "FROZEN_BEFORE_Z_HELDOUT_UTILITY_SCORING",
        "version": "YAB-Z-1.0",
        "primary_design": "one baseline complex-modulus Re/Im pair at a single frequency (two scalar budget)",
        "target_grid_mM": [float(x) for x in targets],
        "frequency_grid_Hz": [float(x) for x in frequencies],
        "frequency_grid_sha256": hashlib.sha256(np.asarray(frequencies, dtype="<f8").tobytes()).hexdigest(),
        "selection": "For each outer held-out target, compare Y0-Y3 by mean inner leave-one-target-out curve RMSE over the other five targets; tie-break lower complexity; refit chosen model on all five outer-training targets. No outer utility is used for selection.",
        "model_families": MODEL_FEATURES,
        "coordinate_transform": "log10 ATP and log10 Hz standardized by mean/SD of the frozen full design grids; grid constants use no outcomes.",
        "optimizer": "numpy.linalg.lstsq(rcond=None), ordinary least squares, no regularization or tunable hyperparameter",
        "random_comparator_seed": 20261002,
        "single_frequency_only_reason": "Two scalar Re/Im observations at one frequency are the unit scored by U; multi-frequency utility is not additive and was not inferred from single-frequency scores.",
        "code_sha256": sha256(Path(__file__).resolve()),
        "fold_rules": folds,
    }
    path = OUT / "Z_DESIGN_RULE_FREEZE.json"
    path.write_text(json.dumps(version, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(path)
    (OUT / "Z_DESIGN_RULE_SHA256.txt").write_text(f"{digest}  Z_DESIGN_RULE_FREEZE.json\n", encoding="ascii")
    pd.DataFrame(rule_compare).to_csv(OUT / "Z_NESTED_MODEL_SELECTION.csv", index=False)
    return version, digest


def score_z(frequencies, curves):
    freeze_path = OUT / "Z_DESIGN_RULE_FREEZE.json"
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    recorded = (OUT / "Z_DESIGN_RULE_SHA256.txt").read_text(encoding="ascii").split()[0]
    actual = sha256(freeze_path)
    if recorded != actual or frozen.get("status") != "FROZEN_BEFORE_Z_HELDOUT_UTILITY_SCORING":
        raise RuntimeError("Z freeze hash/status check failed")
    if frozen["frequency_grid_sha256"] != hashlib.sha256(np.asarray(frequencies, dtype="<f8").tobytes()).hexdigest():
        raise RuntimeError("Z frequency-grid hash mismatch")
    rng = np.random.default_rng(20261002)
    rows = []
    for fold in frozen["fold_rules"]:
        t = float(fold["outer_heldout_target_ATP_mM"])
        obs = curves[t]
        oracle_idx = int(np.argmax(obs))
        recommendations = fold["recommendations_Hz"]
        recs = {k: v for k, v in recommendations.items() if v is not None}
        recs["target_specific_oracle_reference"] = float(frequencies[oracle_idx])
        # Exact finite-grid random expectation; seed controls reproducible Monte-Carlo check.
        draws = rng.integers(0, len(frequencies), size=20_000)
        random_util = obs[draws]
        recs["random_uniform_candidate_grid"] = float("nan")
        for method, f in recs.items():
            if not np.isfinite(f):
                rows.append({"heldout_target_ATP_mM": t, "design_rule": method,
                             "selected_frequency_Hz": "uniform_random_over_72",
                             "selected_frequency_band": "distribution", "observed_gain_pp": float(np.mean(obs)),
                             "random_gain_95pct_low_pp": float(np.quantile(random_util, 0.025)),
                             "random_gain_95pct_high_pp": float(np.quantile(random_util, 0.975)),
                             "oracle_gain_pp": float(obs[oracle_idx]), "regret_pp": float(obs[oracle_idx]-np.mean(obs)),
                             "oracle_retention_fraction": float(np.mean(obs)/obs[oracle_idx]) if obs[oracle_idx] > 0 else np.nan,
                             "rank_position": float(np.mean(rankdata(-obs, method="average"))),
                             "best_band_hit": np.nan,
                             "outer_target_outcome_used_for_selection": False,
                             "frozen_rule_sha256": actual})
                continue
            idx = int(np.argmin(np.abs(frequencies - float(f))))
            gain = float(obs[idx])
            rows.append({"heldout_target_ATP_mM": t, "design_rule": method,
                         "selected_frequency_Hz": float(frequencies[idx]),
                         "selected_frequency_band": grid_band(frequencies, float(frequencies[idx])),
                         "observed_gain_pp": gain, "random_gain_95pct_low_pp": np.nan,
                         "random_gain_95pct_high_pp": np.nan, "oracle_gain_pp": float(obs[oracle_idx]),
                         "regret_pp": float(obs[oracle_idx] - gain),
                         "oracle_retention_fraction": float(gain / obs[oracle_idx]) if obs[oracle_idx] > 0 else np.nan,
                         "rank_position": int(rankdata(-obs, method="min")[idx]),
                         "best_band_hit": grid_band(frequencies, float(frequencies[idx])) == grid_band(frequencies, float(frequencies[oracle_idx])),
                         "outer_target_outcome_used_for_selection": False,
                         "frozen_rule_sha256": actual})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "Z_HELDOUT_TARGET_DESIGN_RESULTS.csv", index=False)
    summary = (result.groupby("design_rule", as_index=False)
               .agg(n_targets=("heldout_target_ATP_mM", "nunique"), mean_regret_pp=("regret_pp", "mean"),
                    mean_oracle_retention=("oracle_retention_fraction", "mean"),
                    median_oracle_retention=("oracle_retention_fraction", "median"),
                    physical_band_hit_fraction=("best_band_hit", "mean"), mean_rank_position=("rank_position", "mean"),
                    mean_observed_gain_pp=("observed_gain_pp", "mean")))
    summary.to_csv(OUT / "Z_HELDOUT_TARGET_DESIGN_SUMMARY.csv", index=False)
    return result, summary, actual


def plot_landscape(landscape):
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for t, sub in landscape.groupby("target_ATP_mM", sort=True):
        sub = sub.sort_values("frequency_Hz")
        ax.plot(sub.frequency_Hz, sub.mean_gain_pp, lw=1.8, marker="o", ms=2.7, label=f"{t:g} mM")
        ax.fill_between(sub.frequency_Hz, sub.bootstrap_95pct_low_pp, sub.bootstrap_95pct_high_pp, alpha=0.08)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Candidate baseline frequency (Hz; log scale)")
    ax.set_ylabel("Held-out NRMSE gain (percentage points)")
    ax.set_title("Awinda frequency-domain predictive utility by future MgATP target")
    ax.legend(title="Future MgATP", ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "Y_target_frequency_utility_landscape.png", dpi=220)
    plt.close(fig)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"build", "score"}:
        raise SystemExit("usage: run_yz_landscape.py build|score")
    if sys.argv[1] == "build":
        targets, frequencies, mice, curves, landscape = load_mouse_curves()
        run_y_loto(targets, frequencies, curves)
        _, digest = build_z_freeze(targets, frequencies, curves)
        (HERE / "figures").mkdir(exist_ok=True)
        plot_landscape(landscape)
        print(f"Y curves={len(targets)*len(frequencies)}; outer predictions={len(targets)*len(frequencies)*len(MODEL_ORDER)}; Z freeze sha256={digest}")
    else:
        raw = pd.read_csv(INPUT)
        targets = np.sort(raw.target_ATP_mM.unique().astype(float))
        frequencies = np.sort(raw.frequency_Hz.unique().astype(float))
        mouse = (raw.groupby(["target_ATP_mM", "heldout_mouse", "frequency_Hz"], as_index=False)
                 .agg(gain_pp=("gain_pp", "mean")))
        curves = {float(t): (mouse[mouse.target_ATP_mM.eq(t)]
                             .pivot(index="heldout_mouse", columns="frequency_Hz", values="gain_pp")
                             .reindex(columns=frequencies).mean(axis=0).to_numpy(float)) for t in targets}
        _, summary, digest = score_z(frequencies, curves)
        print(f"Z scored; frozen rule sha256={digest};\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
