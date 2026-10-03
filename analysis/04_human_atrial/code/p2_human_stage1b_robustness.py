#!/usr/bin/env python3
"""
P2 Human Atrial Pilot Stage 1B:
1) trust-radius sensitivity of the frozen Stage-1 local information screen;
2) nonlinear re-evaluation of the original Stage-1 endpoints.

This imports the existing frozen Stage-1 runner and does not alter its science.
"""
from __future__ import annotations
import csv, json, importlib.util, math
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.optimize import minimize

CHI2 = 3.841458820694124
RADII = [0.25, 0.5, 1.0]

def load_stage1_module(pilot_root: Path):
    path = pilot_root / "code" / "p2_human_local_information_pilot.py"
    spec = importlib.util.spec_from_file_location("stage1_human", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def support_with_radius(mod, J, q0, g, p, radius):
    ref = p[mod.FREE_IDX]
    lo = np.maximum(np.log(mod.LB / ref), -radius)
    hi = np.minimum(np.log(mod.UB / ref),  radius)
    bounds = list(zip(lo, hi))

    def ds(x):
        return float(np.dot(J @ x, J @ x))

    cons = {"type":"ineq", "fun":lambda x: CHI2 - ds(x)}
    seeds = [np.zeros(len(ref))]
    gn = np.linalg.norm(g)
    if gn:
        u = g / gn
        for scale in (0.05, 0.15, 0.3, 0.6, 1.0):
            for sign in (-1, 1):
                x = np.clip(sign * scale * u, lo, hi)
                val = ds(x)
                if val > CHI2 and val > 0:
                    x *= 0.95 * math.sqrt(CHI2 / val)
                seeds.append(np.clip(x, lo, hi))

    _, _, vh = np.linalg.svd(J, full_matrices=False)
    for v in vh[-min(4, len(vh)):]:
        for sign in (-1, 1):
            x = np.clip(sign * radius * v, lo, hi)
            val = ds(x)
            if val > CHI2 and val > 0:
                x *= 0.95 * math.sqrt(CHI2 / val)
            seeds.append(np.clip(x, lo, hi))

    out = {}
    for name, sgn in (("min", 1), ("max", -1)):
        best = None
        for x0 in seeds:
            r = minimize(lambda x: sgn * float(g @ x), x0,
                         method="SLSQP", bounds=bounds, constraints=[cons],
                         options={"ftol":1e-12, "maxiter":4000})
            d = ds(r.x)
            feasible = d <= CHI2 + 1e-6
            if feasible:
                q = q0 + float(g @ r.x)
                cand = {
                    "q_linear_pp": q,
                    "deltaS_linear": d,
                    "success": bool(r.success),
                    "message": str(r.message),
                    "active_trust_or_box_bounds": int(np.sum(
                        np.isclose(r.x, lo, atol=1e-6) |
                        np.isclose(r.x, hi, atol=1e-6))),
                    "delta_z": r.x.tolist(),
                }
                if best is None or (q < best["q_linear_pp"] if name=="min"
                                     else q > best["q_linear_pp"]):
                    best = cand
        if best is None:
            raise RuntimeError(f"No feasible endpoint for radius={radius}, {name}")
        out[name] = best
    out["width_pp"] = out["max"]["q_linear_pp"] - out["min"]["q_linear_pp"]
    return out

def nonlinear_deltaS(mod, p_ref, p_new, data, freqs11, augmented):
    href = mod.obs(p_ref, data, freqs11, augmented, 295.0)
    hnew = mod.obs(p_new, data, freqs11, augmented, 295.0)
    d = np.asarray(hnew) - np.asarray(href)
    return float(np.dot(d, d))

def apply_delta(mod, p_ref, delta_z):
    p = p_ref.copy()
    p[mod.FREE_IDX] = p_ref[mod.FREE_IDX] * np.exp(np.asarray(delta_z, float))
    return p

def main():
    pilot_root = Path(__file__).resolve().parent.parent
    out = pilot_root / "results_stage1b"
    out.mkdir(parents=True, exist_ok=True)

    mod = load_stage1_module(pilot_root)

    # Reuse exactly the Stage-1 source locator and source data.
    p1_root = REPO / "data" / "third_party" / "human_model"
    fit, datafile, _ = mod.find_sources(p1_root)
    p = mod.load_ref(fit)
    freqs, data = mod.load_data(datafile)
    f11 = freqs[:11]

    Jb = mod.jac_log(lambda x: mod.obs(x, data, f11, False, 295.0), p)
    Ja = mod.jac_log(lambda x: mod.obs(x, data, f11, True, 295.0), p)

    rows = []
    detail = {}
    for target in ("ATP1", "Pi10"):
        q0, g = mod.grad_q(p, data, target, 295.0)
        detail[target] = {}
        for radius in RADII:
            rb = support_with_radius(mod, Jb, q0, g, p, radius)
            ra = support_with_radius(mod, Ja, q0, g, p, radius)
            red = 100.0 * (1.0 - ra["width_pp"] / rb["width_pp"])
            rows.append({
                "target": target,
                "trust_radius_log": radius,
                "baseline_low_pp": rb["min"]["q_linear_pp"],
                "baseline_high_pp": rb["max"]["q_linear_pp"],
                "baseline_width_pp": rb["width_pp"],
                "augmented_low_pp": ra["min"]["q_linear_pp"],
                "augmented_high_pp": ra["max"]["q_linear_pp"],
                "augmented_width_pp": ra["width_pp"],
                "width_reduction_percent": red,
                "baseline_min_active_bounds": rb["min"]["active_trust_or_box_bounds"],
                "baseline_max_active_bounds": rb["max"]["active_trust_or_box_bounds"],
                "augmented_min_active_bounds": ra["min"]["active_trust_or_box_bounds"],
                "augmented_max_active_bounds": ra["max"]["active_trust_or_box_bounds"],
            })
            detail[target][str(radius)] = {"baseline": rb, "augmented": ra}

    csv_path = out / "trust_radius_sensitivity.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "trust_radius_endpoints.json").write_text(
        json.dumps(detail, indent=2), encoding="utf-8")

    # Re-evaluate the original Stage-1 endpoints through the full nonlinear model.
    stage1_ep = json.loads((pilot_root / "results" / "local_information_endpoints.json").read_text(encoding="utf-8"))
    reeval = []
    for target in ("ATP1", "Pi10"):
        for design,aug in (("baseline",False),("augmented",True)):
            for side in ("min","max"):
                ep = stage1_ep[target][design][side]
                pnew = apply_delta(mod, p, ep["delta_z"])
                q_actual = mod.qresp(pnew, data, target, 295.0)
                ds_actual = nonlinear_deltaS(mod, p, pnew, data, f11, aug)
                reeval.append({
                    "target":target,
                    "design":design,
                    "side":side,
                    "q_linear_pp":ep["q"],
                    "q_nonlinear_pp":q_actual,
                    "q_error_pp":q_actual-ep["q"],
                    "deltaS_linear":ep["deltaS"],
                    "deltaS_nonlinear":ds_actual,
                    "deltaS_ratio_nonlinear_to_linear":ds_actual/ep["deltaS"] if ep["deltaS"] else np.nan,
                    "original_success":ep["success"],
                    "original_active_bounds":ep["active_bounds"],
                    "max_abs_delta_z":float(np.max(np.abs(ep["delta_z"]))),
                })

    re_path = out / "stage1_original_endpoint_nonlinear_reevaluation.csv"
    with re_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(reeval[0].keys()))
        w.writeheader(); w.writerows(reeval)

    # Machine-readable decision support only; final GO/STOP is made after review.
    summary = {
        "classification":"HUMAN_PILOT_STAGE1B_ROBUSTNESS",
        "trust_radii_log":RADII,
        "trust_radius_results":rows,
        "original_endpoint_nonlinear_reevaluation":reeval,
        "decision_rule":{
            "GO_STAGE2":"ATP1 reduction remains clearly larger than Pi10 at radii 0.25 and 0.5, with no qualitative reversal on nonlinear endpoint checks.",
            "STOP_OR_RETHINK":"Target contrast collapses under modest trust radii or nonlinear checks reverse the ranking."
        }
    }
    (out / "stage1b_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "RUN_STATUS.md").write_text(
        "# Human Pilot Stage 1B — Run Status\n\n**Execution: PASS**\n\n"
        "Upload the complete `results_stage1b` folder/ZIP to ChatGPT for the GO/STOP decision.\n",
        encoding="utf-8")

    print(json.dumps({
        "status":"PASS",
        "rows":rows,
        "results_dir":str(out)
    }, indent=2))

if __name__ == "__main__":
    main()
