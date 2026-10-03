#!/usr/bin/env python3
"""
P2 Human Atrial Pilot Stage 2:
bounded nonlinear working-likelihood support analysis.

Requires the already-installed Stage-1 runner in the same pilot folder and reads the
public P1 human source read-only.
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
from scipy.optimize import least_squares, minimize

CHI2 = 3.841458820694124
RADII = [0.25, 0.50]
RNG_SEED = 20260913
N_MLE_STARTS = 40
N_ENDPOINT_RANDOM_STARTS = 48

def load_stage1(pilot_root):
    path = pilot_root / "code" / "p2_human_local_information_pilot.py"
    spec = importlib.util.spec_from_file_location("stage1_human", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def domain_bounds(mod, p_ref, radius):
    ref = p_ref[mod.FREE_IDX]
    lo_pub = np.log(mod.LB / ref)
    hi_pub = np.log(mod.UB / ref)
    lo = np.maximum(lo_pub, -float(radius))
    hi = np.minimum(hi_pub, +float(radius))
    return lo, hi

def apply_z(mod, p_ref, z):
    p = p_ref.copy()
    p[mod.FREE_IDX] = p_ref[mod.FREE_IDX] * np.exp(np.asarray(z, float))
    return p

def residuals(mod, p_ref, z, data, freqs11, augmented):
    p = apply_z(mod, p_ref, z)
    b = data["Baseline"]
    Y, F = mod.model(p, b["met"], freqs11, 295.0)
    r = [
        (Y.real - b["Y"][:11].real) / b["se_re"][:11],
        (Y.imag - b["Y"][:11].imag) / b["se_im"][:11],
        np.array([(F - b["F0"]) / b["se_F0"]]),
    ]
    if augmented:
        a = data["ATP0.1"]
        _, Fa = mod.model(p, a["met"], [1.0], 295.0)
        r.append(np.array([(Fa - a["F0"]) / a["se_F0"]]))
    return np.concatenate(r)

def objective(mod, p_ref, z, data, freqs11, augmented):
    r = residuals(mod, p_ref, z, data, freqs11, augmented)
    return float(np.dot(r, r))

def q_value(mod, p_ref, z, data, target):
    return float(mod.qresp(apply_z(mod, p_ref, z), data, target, 295.0))

def deterministic_starts(lo, hi, n, rng):
    starts = [np.zeros(len(lo))]
    # axis starts
    for i in range(len(lo)):
        a = np.zeros(len(lo)); a[i] = 0.8 * hi[i]
        b = np.zeros(len(lo)); b[i] = 0.8 * lo[i]
        starts.extend([a, b])
    # reproducible uniform starts
    while len(starts) < n:
        starts.append(rng.uniform(lo, hi))
    return starts[:n]

def fit_mle(mod, p_ref, data, freqs11, augmented, lo, hi, rng):
    starts = deterministic_starts(lo, hi, N_MLE_STARTS, rng)
    records = []
    best = None
    for k, x0 in enumerate(starts):
        res = least_squares(
            lambda z: residuals(mod, p_ref, z, data, freqs11, augmented),
            x0=x0, bounds=(lo, hi),
            max_nfev=8000, xtol=1e-11, ftol=1e-11, gtol=1e-11
        )
        s = float(np.dot(res.fun, res.fun))
        rec = {
            "start_index": k,
            "objective": s,
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(res.nfev),
            "z": res.x.tolist(),
        }
        records.append(rec)
        if best is None or s < best["objective"]:
            best = rec
    return best, records

def feasible_boundary_seed(mod, p_ref, z_mle, direction, data, freqs11, augmented,
                           cutoff, lo, hi):
    d = np.asarray(direction, float)
    norm = np.linalg.norm(d)
    if norm == 0:
        return z_mle.copy()
    d = d / norm

    # maximum alpha before hitting coordinate bounds
    alpha_max = np.inf
    for i, di in enumerate(d):
        if di > 0:
            alpha_max = min(alpha_max, (hi[i] - z_mle[i]) / di)
        elif di < 0:
            alpha_max = min(alpha_max, (lo[i] - z_mle[i]) / di)
    if not np.isfinite(alpha_max) or alpha_max <= 0:
        return z_mle.copy()

    # find furthest point along direction still satisfying objective cutoff
    low, high = 0.0, max(0.0, alpha_max)
    z_high = np.clip(z_mle + high*d, lo, hi)
    if objective(mod, p_ref, z_high, data, freqs11, augmented) <= cutoff:
        return z_high

    for _ in range(60):
        mid = 0.5*(low+high)
        z_mid = np.clip(z_mle + mid*d, lo, hi)
        if objective(mod, p_ref, z_mid, data, freqs11, augmented) <= cutoff:
            low = mid
        else:
            high = mid
    return np.clip(z_mle + 0.95*low*d, lo, hi)

def endpoint_search(mod, p_ref, data, freqs11, augmented, target,
                    mle, lo, hi, rng, side):
    z_mle = np.asarray(mle["z"], float)
    smin = float(mle["objective"])
    cutoff = smin + CHI2
    q0 = q_value(mod, p_ref, z_mle, data, target)

    # finite-difference target gradient at MLE
    g = np.zeros_like(z_mle)
    h = 1e-5
    for i in range(len(g)):
        zp = z_mle.copy(); zm = z_mle.copy()
        if z_mle[i] + h <= hi[i] and z_mle[i] - h >= lo[i]:
            zp[i] += h; zm[i] -= h
            g[i] = (q_value(mod,p_ref,zp,data,target) -
                    q_value(mod,p_ref,zm,data,target))/(2*h)
        elif z_mle[i] + h <= hi[i]:
            zp[i] += h
            g[i] = (q_value(mod,p_ref,zp,data,target)-q0)/h
        else:
            zm[i] -= h
            g[i] = (q0-q_value(mod,p_ref,zm,data,target))/h

    directions = []
    if np.linalg.norm(g) > 0:
        directions.extend([g, -g])
    # axes
    for i in range(len(z_mle)):
        e = np.zeros_like(z_mle); e[i] = 1
        directions.extend([e, -e])
    # random
    for _ in range(N_ENDPOINT_RANDOM_STARTS):
        directions.append(rng.normal(size=len(z_mle)))

    starts = [z_mle.copy()]
    for d in directions:
        starts.append(feasible_boundary_seed(
            mod,p_ref,z_mle,d,data,freqs11,augmented,cutoff,lo,hi))

    sign = 1.0 if side == "min" else -1.0
    cons = {"type":"ineq", "fun":lambda z:
            cutoff - objective(mod,p_ref,z,data,freqs11,augmented)}
    records = []
    best = None

    for k, x0 in enumerate(starts):
        r = minimize(
            lambda z: sign*q_value(mod,p_ref,z,data,target),
            x0=x0, method="SLSQP", bounds=list(zip(lo,hi)),
            constraints=[cons],
            options={"maxiter":5000,"ftol":1e-10,"disp":False}
        )
        s = objective(mod,p_ref,r.x,data,freqs11,augmented)
        q = q_value(mod,p_ref,r.x,data,target)
        feasible = (s <= cutoff + 1e-5 and
                    np.all(r.x >= lo-1e-8) and np.all(r.x <= hi+1e-8))
        rec = {
            "start_index":k,
            "q_pp":q,
            "objective":s,
            "deltaS":s-smin,
            "feasible":bool(feasible),
            "success":bool(r.success),
            "message":str(r.message),
            "active_bounds":int(np.sum(
                np.isclose(r.x,lo,atol=1e-5) |
                np.isclose(r.x,hi,atol=1e-5))),
            "z":r.x.tolist(),
        }
        records.append(rec)
        if feasible:
            if best is None or (q < best["q_pp"] if side=="min" else q > best["q_pp"]):
                best = rec

    if best is None:
        raise RuntimeError(f"No feasible {target} {side} endpoint.")
    return best, records

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    pilot_root = Path(__file__).resolve().parent.parent
    out = pilot_root / "results_stage2"
    out.mkdir(parents=True, exist_ok=True)
    mod = load_stage1(pilot_root)

    p1_root = REPO / "data" / "third_party" / "human_model"
    fit, datafile, _ = mod.find_sources(p1_root)
    p_ref = mod.load_ref(fit)
    freqs, data = mod.load_data(datafile)
    freqs11 = freqs[:11]

    rng = np.random.default_rng(RNG_SEED)
    summary_rows = []
    detailed = {}
    mle_rows = []

    for radius in RADII:
        lo, hi = domain_bounds(mod, p_ref, radius)
        detailed[str(radius)] = {}

        for design, augmented in (("baseline",False),("augmented",True)):
            best_mle, mle_records = fit_mle(
                mod,p_ref,data,freqs11,augmented,lo,hi,rng)
            detailed[str(radius)][design] = {
                "mle":best_mle,
                "mle_multistart":mle_records,
                "targets":{}
            }
            mle_rows.append({
                "trust_radius_log":radius,
                "design":design,
                "best_objective":best_mle["objective"],
                "success":best_mle["success"],
                "nfev":best_mle["nfev"],
                "active_bounds":int(np.sum(
                    np.isclose(np.asarray(best_mle["z"]),lo,atol=1e-5) |
                    np.isclose(np.asarray(best_mle["z"]),hi,atol=1e-5))),
            })

            for target in ("ATP1","Pi10"):
                mn, mnrec = endpoint_search(
                    mod,p_ref,data,freqs11,augmented,target,
                    best_mle,lo,hi,rng,"min")
                mx, mxrec = endpoint_search(
                    mod,p_ref,data,freqs11,augmented,target,
                    best_mle,lo,hi,rng,"max")
                detailed[str(radius)][design]["targets"][target] = {
                    "min":mn,"max":mx,
                    "min_multistart":mnrec,"max_multistart":mxrec,
                    "width_pp":mx["q_pp"]-mn["q_pp"]
                }

        for target in ("ATP1","Pi10"):
            b = detailed[str(radius)]["baseline"]["targets"][target]
            a = detailed[str(radius)]["augmented"]["targets"][target]
            reduction = 100.0*(1.0-a["width_pp"]/b["width_pp"])
            summary_rows.append({
                "trust_radius_log":radius,
                "target":target,
                "baseline_mle_objective":detailed[str(radius)]["baseline"]["mle"]["objective"],
                "augmented_mle_objective":detailed[str(radius)]["augmented"]["mle"]["objective"],
                "baseline_low_pp":b["min"]["q_pp"],
                "baseline_high_pp":b["max"]["q_pp"],
                "baseline_width_pp":b["width_pp"],
                "augmented_low_pp":a["min"]["q_pp"],
                "augmented_high_pp":a["max"]["q_pp"],
                "augmented_width_pp":a["width_pp"],
                "width_reduction_percent":reduction,
                "baseline_min_deltaS":b["min"]["deltaS"],
                "baseline_max_deltaS":b["max"]["deltaS"],
                "augmented_min_deltaS":a["min"]["deltaS"],
                "augmented_max_deltaS":a["max"]["deltaS"],
                "all_endpoints_feasible":all([
                    b["min"]["feasible"],b["max"]["feasible"],
                    a["min"]["feasible"],a["max"]["feasible"]
                ]),
            })

    write_csv(out/"stage2_summary.csv", summary_rows)
    write_csv(out/"stage2_mle_summary.csv", mle_rows)
    (out/"stage2_detailed_results.json").write_text(
        json.dumps(detailed,indent=2), encoding="utf-8")

    by_radius={}
    for r in RADII:
        rows=[x for x in summary_rows if abs(x["trust_radius_log"]-r)<1e-12]
        by_radius[str(r)]={x["target"]:x for x in rows}

    result = {
        "classification":"HUMAN_PILOT_STAGE2_BOUNDED_NONLINEAR",
        "trust_radii_log":RADII,
        "nominal_deltaS_cutoff":CHI2,
        "summary":by_radius,
        "interpretation_boundary":[
            "Working group-mean objective, not patient-level calibrated inference.",
            "Human reference/model historically used the same dataset.",
            "Bounded local domains centered on published ND x_p.",
            "No individual fitted-parameter mechanistic interpretation."
        ],
        "decision_rule":{
            "HUMAN_GENERALIZATION_POSITIVE":
                "ATP1 width reduction materially exceeds Pi10 in both 0.25 and 0.50 domains with feasible endpoints.",
            "HUMAN_GENERALIZATION_WEAK_OR_DOMAIN_DEPENDENT":
                "Ranking is clear in only one domain or becomes modest/ambiguous.",
            "HUMAN_GENERALIZATION_NEGATIVE":
                "Target ranking disappears or reverses robustly."
        }
    }
    (out/"stage2_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (out/"RUN_STATUS.md").write_text(
        "# Human Pilot Stage 2 — Run Status\n\n"
        "**Execution: PASS**\n\n"
        "Upload `Human_Atrial_Pilot_Stage2_RESULTS_2026-09-13.zip` to ChatGPT. "
        "Do not start diabetic or manuscript-integration work automatically.\n",
        encoding="utf-8")
    print(json.dumps({"status":"PASS","summary":by_radius},indent=2))

if __name__ == "__main__":
    main()
