#!/usr/bin/env python3
"""Preparation-level retrospective human condition-holdout benchmark.

Fits the public human atrial cross-bridge model separately to each trabecula's
baseline observations, then to baseline plus the ATP0.1 steady-stress scalar.
ATP1 and Pi10 stress and complex-modulus observations are scored only after fit.
The result is descriptive and does not use patient-held-out folds because the
public table has no patient linkage field.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))


import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = REPO
CLOSURE = ROOT / "analysis/08_boundaries/preparation_support"
sys.path.insert(0, str(HERE))
import public_human_model_evaluator as human  # noqa: E402

OBS_PATH = CLOSURE / "input/B_HUMAN_CONDITION_OBSERVATIONS.csv"
PREP_PATH = CLOSURE / "input/B_HUMAN_PREPARATION_MAP.csv"
PARSE_AUDIT_PATH = CLOSURE / "input/B_HUMAN_PARSE_MACHINE_AUDIT.json"
REF_PATH = CLOSURE / "input/ND_xb_fit.mat"
OUT = CLOSURE / "results"
FREQ_COUNT = 11  # exact convention in the public human model evaluator
N_STARTS = 32
RNG_SEED = 20261002
TEMPERATURE_K = 295.0
SUPPORT_CUTOFF = 3.841458820694124
TRAIN_CONDITIONS = {
    "baseline": (5.0, 1.0),
    "ATP0.1": (0.1, 1.0),
    "ATP1": (1.0, 1.0),
    "Pi0": (5.0, 1e-6),
    "Pi10": (5.0, 10.0),
}
TARGETS = ("ATP1", "Pi10")
DESIGNS = ("baseline_only", "baseline_plus_ATP0.1_stress")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def starts_for(lo: np.ndarray, hi: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = [np.zeros(len(lo))]
    for i in range(len(lo)):
        up = np.zeros(len(lo)); up[i] = 0.8 * hi[i]
        dn = np.zeros(len(lo)); dn[i] = 0.8 * lo[i]
        starts.extend([up, dn])
    while len(starts) < N_STARTS:
        starts.append(rng.uniform(lo, hi))
    return [np.clip(x, lo, hi) for x in starts[:N_STARTS]]


def make_rows(obs_df: pd.DataFrame, freqs: np.ndarray) -> dict[str, dict]:
    by_prep: dict[str, dict] = {}
    for prep_id, prep_group in obs_df.groupby("preparation_id", sort=True):
        by_prep[prep_id] = {"group": str(prep_group["group"].iloc[0]), "conditions": {}}
        for _, row in prep_group.iterrows():
            condition = str(row["condition"])
            re = np.array([row[f"CM_f{f:g}Hz_real_MPa"] for f in freqs], float)
            im = np.array([row[f"CM_f{f:g}Hz_imag_MPa"] for f in freqs], float)
            by_prep[prep_id]["conditions"][condition] = {
                "met": np.asarray(TRAIN_CONDITIONS[condition], float),
                "Y": re + 1j * im,
                "F0": float(row["observed_steady_stress_kPa"]),
            }
    return by_prep


def model_data_view(prep: dict) -> dict:
    data = {"Baseline": prep["conditions"]["baseline"]}
    data.update({target: prep["conditions"][target] for target in TARGETS})
    return data


def scales_by_group(obs_df: pd.DataFrame, freqs: np.ndarray) -> dict:
    scales = {}
    floors_used = []
    for group in sorted(obs_df["group"].unique()):
        sub = obs_df[(obs_df["group"] == group) & obs_df["condition"].isin(["baseline", "ATP0.1"])]
        scales[group] = {}
        for condition in ("baseline", "ATP0.1"):
            cond = sub[sub["condition"] == condition].sort_values("preparation_id")
            s_re = np.asarray([cond[f"CM_f{f:g}Hz_real_MPa"].std(ddof=1) for f in freqs], float)
            s_im = np.asarray([cond[f"CM_f{f:g}Hz_imag_MPa"].std(ddof=1) for f in freqs], float)
            s_f = float(cond["observed_steady_stress_kPa"].std(ddof=1))
            floor_re = s_re < 1e-8; floor_im = s_im < 1e-8
            if np.any(floor_re) or np.any(floor_im) or s_f < 1e-8:
                floors_used.append({"group": group, "condition": condition,
                                    "real_indices": np.flatnonzero(floor_re).tolist(),
                                    "imag_indices": np.flatnonzero(floor_im).tolist(),
                                    "stress_floor": bool(s_f < 1e-8)})
            scales[group][condition] = {
                "re": np.maximum(s_re, 1e-8),
                "im": np.maximum(s_im, 1e-8),
                "stress": max(s_f, 1e-8),
            }
    return scales, floors_used


def fit_residuals(z: np.ndarray, p_ref: np.ndarray, prep: dict, scales: dict,
                  freqs: np.ndarray, augmented: bool) -> np.ndarray:
    p = p_ref.copy()
    p[human.FREE_IDX] = p_ref[human.FREE_IDX] * np.exp(np.asarray(z, float))
    chunks = []
    item = prep["conditions"]["baseline"]
    scl = scales["baseline"]
    pred_y, pred_f = human.model(p, item["met"], freqs, TEMPERATURE_K)
    chunks.extend([
        (pred_y.real - item["Y"].real[:len(freqs)]) / scl["re"],
        (pred_y.imag - item["Y"].imag[:len(freqs)]) / scl["im"],
        np.array([(pred_f - item["F0"]) / scl["stress"]]),
    ])
    if augmented:
        # The predeclared added assay is exactly one ATP0.1 steady-stress
        # scalar. The ATP0.1 CM spectrum is not part of this fit.
        item = prep["conditions"]["ATP0.1"]
        _, pred_f = human.model(p, item["met"], np.array([1.0]), TEMPERATURE_K)
        chunks.append(np.array([(pred_f - item["F0"]) / scales["ATP0.1"]["stress"]]))
    return np.concatenate(chunks)


def predicted_observation_vector(z: np.ndarray, p_ref: np.ndarray, prep: dict,
                                 scales: dict, freqs: np.ndarray,
                                 augmented: bool) -> np.ndarray:
    """Fitted-value vector; its derivative matches the training residual Jacobian."""
    p = p_ref.copy()
    p[human.FREE_IDX] = p_ref[human.FREE_IDX] * np.exp(np.asarray(z, float))
    chunks = []
    item = prep["conditions"]["baseline"]
    scl = scales["baseline"]
    pred_y, pred_f = human.model(p, item["met"], freqs, TEMPERATURE_K)
    chunks.extend([pred_y.real / scl["re"], pred_y.imag / scl["im"], np.array([pred_f / scl["stress"]])])
    if augmented:
        item = prep["conditions"]["ATP0.1"]
        _, pred_f = human.model(p, item["met"], np.array([1.0]), TEMPERATURE_K)
        chunks.append(np.array([pred_f / scales["ATP0.1"]["stress"]]))
    return np.concatenate(chunks)


def fit_one(p_ref: np.ndarray, prep: dict, group_scales: dict,
            freqs: np.ndarray, prep_seed: int) -> dict:
    lo = np.log(human.LB / p_ref[human.FREE_IDX])
    hi = np.log(human.UB / p_ref[human.FREE_IDX])
    starts = starts_for(lo, hi, RNG_SEED + prep_seed)
    records = {}
    for design, augmented in (("baseline_only", False), ("baseline_plus_ATP0.1_stress", True)):
        residual = lambda z, use_augmented=augmented: fit_residuals(z, p_ref, prep, group_scales, freqs, use_augmented)
        solutions = []
        for start_index, x0 in enumerate(starts):
            opt = least_squares(
                residual, x0=x0, bounds=(lo, hi), max_nfev=8000,
                xtol=1e-10, ftol=1e-10, gtol=1e-10,
            )
            obj = float(np.dot(opt.fun, opt.fun))
            solutions.append({"start_index": start_index, "z": opt.x.copy(), "objective": obj,
                              "success": bool(opt.success), "status": int(opt.status),
                              "nfev": int(opt.nfev), "message": str(opt.message)})
        best = min(solutions, key=lambda x: x["objective"])
        p_fit = p_ref.copy()
        p_fit[human.FREE_IDX] = p_ref[human.FREE_IDX] * np.exp(best["z"])
        active = int(np.sum(np.isclose(best["z"], lo, atol=1e-5) | np.isclose(best["z"], hi, atol=1e-5)))
        records[design] = {
            "z": best["z"], "params": p_fit, "objective": best["objective"],
            "start_index": best["start_index"], "best_success": best["success"],
            "best_status": best["status"], "best_message": best["message"],
            "best_nfev": best["nfev"], "active_bounds": active,
            "successful_starts": int(sum(x["success"] for x in solutions)),
            "n_starts": len(solutions),
            "start_objective_min": float(min(x["objective"] for x in solutions)),
            "start_objective_median": float(np.median([x["objective"] for x in solutions])),
            "start_objective_max": float(max(x["objective"] for x in solutions)),
            "all_start_statuses": [x["status"] for x in solutions],
            "residual": residual,
            "augmented": augmented,
        }
    for fit in records.values():
        fit_residual = lambda p, f=fit: fit_residuals(
            np.log(p[human.FREE_IDX] / p_ref[human.FREE_IDX]),
            p_ref, prep, group_scales, freqs, f["augmented"],
        )
        fit["J"] = human.jac_log(fit_residual, fit["params"])
    return records


def target_support(fit: dict, p_ref: np.ndarray, prep: dict, scales: dict,
                   freqs: np.ndarray, target: str) -> dict:
    p_fit = fit["params"]
    J = fit["J"]
    q0, gradient = human.grad_q(p_fit, model_data_view(prep), target, TEMPERATURE_K)
    lo = np.log(human.LB / p_fit[human.FREE_IDX])
    hi = np.log(human.UB / p_fit[human.FREE_IDX])
    H = J.T @ J
    H = 0.5 * (H + H.T)
    rank = int(np.linalg.matrix_rank(J))
    pinv = np.linalg.pinv(H, rcond=1e-12)
    variance = float(gradient @ pinv @ gradient)
    result = {"width": None, "min": None, "max": None}
    for side, sign in (("min", 1.0), ("max", -1.0)):
        unconstrained = np.zeros(len(lo))
        if variance > 0 and np.isfinite(variance):
            unconstrained = -sign * np.sqrt(SUPPORT_CUTOFF / variance) * (pinv @ gradient)
        in_box = bool(np.all(unconstrained >= lo - 1e-10) and np.all(unconstrained <= hi + 1e-10))
        if rank == len(lo) and in_box and variance > 0:
            x = unconstrained
            success = True
            solver = "closed_form_full_rank"
            attempts = 0
        else:
            # The local support is a convex ellipsoid intersected with the
            # published box. Two feasible starts check numerical stability.
            x_projected = np.clip(unconstrained, lo, hi)
            radius = float(x_projected @ H @ x_projected)
            if radius > SUPPORT_CUTOFF and radius > 0:
                x_projected *= 0.95 * np.sqrt(SUPPORT_CUTOFF / radius)
            starts = [np.zeros(len(lo)), np.clip(x_projected, lo, hi)]
            candidates = []
            for x0 in starts:
                opt = minimize(
                    lambda x: sign * float(gradient @ x), x0=x0,
                    jac=lambda x: sign * gradient, method="SLSQP",
                    bounds=list(zip(lo, hi)),
                    constraints=[{
                        "type": "ineq",
                        "fun": lambda x: SUPPORT_CUTOFF - float(x @ H @ x),
                        "jac": lambda x: -2.0 * (H @ x),
                    }],
                    options={"maxiter": 2500, "ftol": 1e-10, "disp": False},
                )
                delta_s = float(opt.x @ H @ opt.x)
                feasible = delta_s <= SUPPORT_CUTOFF + 1e-6 and np.all(opt.x >= lo - 1e-8) and np.all(opt.x <= hi + 1e-8)
                if feasible:
                    candidates.append((float(gradient @ opt.x), opt))
            if not candidates:
                x = np.zeros(len(lo)); success = False
                solver = "feasible_origin_fallback"; attempts = len(starts)
            else:
                selected = min(candidates, key=lambda a: a[0]) if side == "min" else max(candidates, key=lambda a: a[0])
                opt = selected[1]; x = opt.x
                success = bool(opt.success); solver = "two_start_SLSQP"; attempts = len(starts)
        q = float(q0 + gradient @ x)
        delta_s = float(x @ H @ x)
        active = int(np.sum(np.isclose(x, lo, atol=1e-5) | np.isclose(x, hi, atol=1e-5)))
        result[side] = {"q": q, "deltaS": delta_s,
                        "feasible": bool(delta_s <= SUPPORT_CUTOFF + 1e-6),
                        "success": success, "solver": solver, "attempts": attempts,
                        "active_bounds": active, "delta_z": x.tolist()}
    result["width"] = result["max"]["q"] - result["min"]["q"]
    result["method"] = "first-order local support from weighted residual Jacobian within published parameter bounds; convex ellipsoid-box optimization; ΔS cutoff 3.841458820694124; not a calibrated interval"
    result["jacobian_rank"] = rank
    result["n_residual_components"] = int(J.shape[0])
    return result


def main() -> None:
    threadpool_limits(limits=1)
    OUT.mkdir(parents=True, exist_ok=True)
    obs_df = pd.read_csv(OBS_PATH)
    prep_df = pd.read_csv(PREP_PATH)
    parse_audit = json.loads(PARSE_AUDIT_PATH.read_text(encoding="utf-8"))
    freqs_all = np.asarray(parse_audit["frequency_hz"], float)
    freqs = freqs_all[:FREQ_COUNT]
    if len(obs_df) != 100 or prep_df.shape[0] != 20:
        raise ValueError("Unexpected source row count or preparation table shape.")
    ref_mat = loadmat(REF_PATH, squeeze_me=False, struct_as_record=False)
    p_ref = human.vec(ref_mat["x_p"], float)
    if p_ref.size != 14:
        raise ValueError(f"Expected 14 public model slots, found {p_ref.size}.")
    if np.any(p_ref[human.FREE_IDX] < human.LB) or np.any(p_ref[human.FREE_IDX] > human.UB):
        raise ValueError("Public x_p reference lies outside the published human bounds.")
    prep_data = make_rows(obs_df, freqs_all)
    scales, scale_floors = scales_by_group(obs_df, freqs)

    result_rows = []
    fit_summary_rows = []
    diagnostic = {"seed": RNG_SEED, "n_starts": N_STARTS, "preparations": {}}
    for ordinal, prep_id in enumerate(sorted(prep_data), start=1):
        prep = prep_data[prep_id]
        group_scales = scales[prep["group"]]
        fits = fit_one(p_ref, prep, group_scales, freqs, int(prep_id.split("-")[-1]))
        diagnostic["preparations"][prep_id] = {}
        supports = {}
        for design in DESIGNS:
            fit = fits[design]
            residual = fit["residual"](fit["z"])
            p_fit = fit["params"]
            base_obs = prep["conditions"]["baseline"]
            base_y_pred, base_f_pred = human.model(p_fit, base_obs["met"], freqs, TEMPERATURE_K)
            cm_rmse_train = float(np.sqrt(np.mean(np.abs(base_y_pred - base_obs["Y"][:FREQ_COUNT]) ** 2)))
            base_stress_resid = float(base_f_pred - base_obs["F0"])
            added_stress_resid = ""
            if fit["augmented"]:
                a_obs = prep["conditions"]["ATP0.1"]
                _, a_pred = human.model(p_fit, a_obs["met"], np.array([1.0]), TEMPERATURE_K)
                added_stress_resid = float(a_pred - a_obs["F0"])
            fit_summary_rows.append({
                "preparation_id": prep_id,
                "group": prep["group"],
                "design": design,
                "n_training_observations": int(residual.size),
                "best_objective": fit["objective"],
                "best_start_index": fit["start_index"],
                "best_solver_success": fit["best_success"],
                "best_solver_status": fit["best_status"],
                "best_solver_message": fit["best_message"],
                "best_nfev": fit["best_nfev"],
                "active_parameter_bounds": fit["active_bounds"],
                "successful_starts_of_32": fit["successful_starts"],
                "start_objective_min": fit["start_objective_min"],
                "start_objective_median": fit["start_objective_median"],
                "start_objective_max": fit["start_objective_max"],
                "baseline_CM_RMSE_MPa": cm_rmse_train,
                "baseline_stress_signed_residual_kPa": base_stress_resid,
                "ATP0.1_stress_signed_residual_kPa": added_stress_resid,
            })
            diagnostic["preparations"][prep_id][design] = {
                "best_objective": fit["objective"], "best_start_index": fit["start_index"],
                "all_start_statuses": fit["all_start_statuses"], "active_bounds": fit["active_bounds"],
            }
            supports[design] = {}
            for target in TARGETS:
                supports[design][target] = target_support(fit, p_ref, prep, group_scales, freqs, target)

        for target in TARGETS:
            target_obs = prep["conditions"][target]
            baseline_obs = prep["conditions"]["baseline"]
            observed_q = 100.0 * (target_obs["F0"] / baseline_obs["F0"] - 1.0)
            for design in DESIGNS:
                fit = fits[design]
                p_fit = fit["params"]
                q_pred = human.qresp(p_fit, model_data_view(prep), target, TEMPERATURE_K)
                pred_y, pred_f_target = human.model(p_fit, target_obs["met"], freqs, TEMPERATURE_K)
                _, pred_f_base = human.model(p_fit, baseline_obs["met"], freqs, TEMPERATURE_K)
                cm_error = pred_y - target_obs["Y"][:FREQ_COUNT]
                cm_rmse = float(np.sqrt(np.mean(np.abs(cm_error) ** 2)))
                observed_cm_rms = float(np.sqrt(np.mean(np.abs(target_obs["Y"][:FREQ_COUNT]) ** 2)))
                support = supports[design][target]
                low, high = float(support["min"]["q"]), float(support["max"]["q"])
                result_rows.append({
                    "patient_id": "",
                    "patient_id_status": "not available; patient-preparation linkage absent",
                    "preparation_id": prep_id,
                    "group": prep["group"],
                    "target": target,
                    "design": design,
                    "observed_baseline_stress_kPa": baseline_obs["F0"],
                    "observed_target_stress_kPa": target_obs["F0"],
                    "observed_target_stress_response_pct": observed_q,
                    "predicted_baseline_stress_kPa": pred_f_base,
                    "predicted_target_stress_kPa": pred_f_target,
                    "predicted_target_stress_response_pct": q_pred,
                    "signed_error_pred_minus_observed_pct": q_pred - observed_q,
                    "absolute_error_pct": abs(q_pred - observed_q),
                    "support_low_pct": low,
                    "support_high_pct": high,
                    "support_width_percentage_points": high - low,
                    "observed_inside_nominal_support": bool(low <= observed_q <= high),
                    "support_type": "local linear nominal support; not a calibrated interval",
                    "training_fit_objective": fits[design]["objective"],
                    "solver_success": fits[design]["best_success"],
                    "solver_status": fits[design]["best_status"],
                    "active_parameter_bounds": fits[design]["active_bounds"],
                    "successful_multistarts_of_32": fits[design]["successful_starts"],
                    "CM_target_RMSE_MPa": cm_rmse,
                    "CM_target_NRMSE_pct_of_observed_RMS": 100.0 * cm_rmse / observed_cm_rms if observed_cm_rms else "",
                    "n_CM_frequencies_scored": FREQ_COUNT,
                    "model_frequency_range_hz": f"{freqs[0]:g}-{freqs[-1]:g}",
                })

        # Persist one preparation at a time so interruption never discards
        # completed fits or their held-out predictions.
        pd.DataFrame(result_rows).to_csv(OUT / "B_HUMAN_HELDOUT_RESULTS.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(fit_summary_rows).to_csv(OUT / "B_HUMAN_FIT_AUDIT.csv", index=False, encoding="utf-8-sig")
        print(f"completed {ordinal}/20: {prep_id}", flush=True)

    heldout = pd.DataFrame(result_rows)
    fit_summary = pd.DataFrame(fit_summary_rows)
    heldout.to_csv(OUT / "B_HUMAN_HELDOUT_RESULTS.csv", index=False, encoding="utf-8-sig")
    fit_summary.to_csv(OUT / "B_HUMAN_FIT_AUDIT.csv", index=False, encoding="utf-8-sig")

    # Descriptive summaries only; do not resample preparations as independent patients.
    summary = []
    for group in ["all", "non-diabetic", "diabetic"]:
        for target in TARGETS:
            for design in DESIGNS:
                sub = heldout[(heldout["target"] == target) & (heldout["design"] == design)]
                if group != "all": sub = sub[sub["group"] == group]
                summary.append({
                    "group": group,
                    "target": target,
                    "design": design,
                    "n_preparations": int(sub["preparation_id"].nunique()),
                    "n_unique_patients_known": 0,
                    "MAE_stress_response_percentage_points": float(sub["absolute_error_pct"].mean()),
                    "median_absolute_error_percentage_points": float(sub["absolute_error_pct"].median()),
                    "RMSE_stress_response_percentage_points": float(np.sqrt(np.mean(sub["signed_error_pred_minus_observed_pct"] ** 2))),
                    "mean_signed_error_pred_minus_observed_pp": float(sub["signed_error_pred_minus_observed_pct"].mean()),
                    "nominal_support_inclusion_count": int(sub["observed_inside_nominal_support"].sum()),
                    "nominal_support_inclusion_fraction": float(sub["observed_inside_nominal_support"].mean()),
                    "median_support_width_pp": float(sub["support_width_percentage_points"].median()),
                    "mean_CM_target_RMSE_MPa": float(sub["CM_target_RMSE_MPa"].mean()),
                    "mean_CM_target_NRMSE_pct": float(sub["CM_target_NRMSE_pct_of_observed_RMS"].mean()),
                })
    pd.DataFrame(summary).to_csv(OUT / "B_HUMAN_PATIENT_AWARE_SUMMARY.csv", index=False, encoding="utf-8-sig")
    (OUT / "B_HUMAN_FIT_DIAGNOSTIC.json").write_text(json.dumps({
        "classification": "RETROSPECTIVE_PREPARATION_LEVEL_CONDITION_HOLDOUT",
        "input_sha256": parse_audit["source_sha256"],
        "model_fit_sha256": sha256(REF_PATH),
        "model_code_source": "JuliaMusgrave/AtrialModel_2025_Human; pinned source snapshot b22e5bef970adb95b7dc413979d8cd1c2ca482b3",
        "n_preparations": int(prep_df.shape[0]),
        "patient_ids_available": False,
        "patient_cluster_bootstrap": "not performed; donor linkage is unavailable",
        "n_starts_per_prep_design": N_STARTS,
        "frequency_count_used": FREQ_COUNT,
        "frequencies_used_hz": freqs.tolist(),
        "frequency_omitted_by_public_model_convention_hz": float(freqs_all[-1]),
        "fit_temperature_K": TEMPERATURE_K,
        "fitting_scales": "group- and condition-specific sample SD across 10 preparations; diagonal descriptive weights, not measurement-level SEs",
        "support_definition": "first-order local predictive support from the weighted residual Jacobian within published human parameter bounds and ΔS cutoff 3.841458820694124; no coverage claim",
        "added_measurement": "one steady-stress scalar at ATP0.1 mM ATP and 1 mM Pi",
        "held_out_targets": {"ATP1": [1.0, 1.0], "Pi10": [5.0, 10.0]},
        "scale_floors_used": scale_floors,
        "fit_summaries": int(fit_summary.shape[0]),
        "heldout_rows": int(heldout.shape[0]),
        "heldout_results_sha256": sha256(OUT / "B_HUMAN_HELDOUT_RESULTS.csv"),
        "fit_audit_sha256": sha256(OUT / "B_HUMAN_FIT_AUDIT.csv"),
    }, indent=2), encoding="utf-8")

    starts_statuses = [s for prep in diagnostic["preparations"].values() for d in prep.values() for s in d["all_start_statuses"]]
    diagnostic["summary"] = {
        "total_starts": len(starts_statuses),
        "successful_starts": int(sum(s > 0 for s in starts_statuses)),
        "non_success_starts": int(sum(s <= 0 for s in starts_statuses)),
    }
    (OUT / "B_HUMAN_MULTISTART_STATUS.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(json.dumps({
        "n_preparations": int(prep_df.shape[0]),
        "heldout_rows": int(heldout.shape[0]),
        "fit_rows": int(fit_summary.shape[0]),
        "multistart": diagnostic["summary"],
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
