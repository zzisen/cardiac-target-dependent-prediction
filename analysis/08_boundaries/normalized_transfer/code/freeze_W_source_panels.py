#!/usr/bin/env python
"""Freeze kinetic-normalized source panel coordinates before destination scoring."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))


import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve()
UX = HERE.parents[2]
PROJECT = HERE.parents[3]
OLD = PROJECT / "P2_QT_UPLIFT_2026-10-02"
OUT = HERE.parents[1] / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(OLD / "common"))
sys.path.insert(0, str(OLD / "PRIOR_A_TO_P_BASELINE" / "K" / "code"))
from dynamic_prediction import load_human  # noqa: E402
from p2.model16d import _rates_and_states  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_empirical_c(freqs: np.ndarray, y: np.ndarray, *, max_starts: int = 16, max_nfev: int = 3000) -> dict:
    """Fit the source-paper complex-modulus equation to a group-mean spectrum."""
    freqs = np.asarray(freqs, float)
    omega = 2 * np.pi * freqs
    n = len(freqs)
    y = np.asarray(y, float)[:n] + 1j * np.asarray(y, float)[n:]
    scale = max(float(np.median(np.abs(y))), 1e-6)
    s = 1j * omega

    def pred(q):
        A, k, B, b, C, c = np.exp(q[0]), q[1], np.exp(q[2]), np.exp(q[3]), np.exp(q[4]), np.exp(q[5])
        return A * s**k - B * s / (2 * np.pi * b + s) + C * s / (2 * np.pi * c + s)

    def residual(q):
        err = (pred(q) - y) / scale
        return np.r_[err.real, err.imag]

    lo = np.array([np.log(1e-12), .01, np.log(1e-8), np.log(.01), np.log(1e-8), np.log(.01)])
    hi = np.array([np.log(1e3), 1.8, np.log(1e3), np.log(1000), np.log(1e3), np.log(1000)])
    all_starts = []
    for b0 in (.2, 1., 5., 20.):
        for c0 in (1., 5., 20., 60.):
            all_starts.append([np.log(max(scale / (2 * np.pi * 10) ** .5, 1e-8)), .5,
                               np.log(scale), np.log(b0), np.log(scale), np.log(c0)])
    # Keep balanced starts across the full range for ordinary fold fits, while the source freeze
    # can retain all 16 for a stronger stability audit.
    starts = all_starts if max_starts >= len(all_starts) else [all_starts[i] for i in (0, 5, 10, 15)][:max_starts]
    fits = [least_squares(residual, np.clip(np.asarray(s0), lo + 1e-9, hi - 1e-9),
                          bounds=(lo, hi), max_nfev=max_nfev, xtol=1e-9, ftol=1e-9, gtol=1e-9)
            for s0 in starts]
    fits.sort(key=lambda z: float(z.fun @ z.fun))
    best = fits[0]
    c_values = np.asarray([np.exp(z.x[5]) for z in fits if np.isfinite(z.cost)])
    # Only near-optimal solutions describe an empirical clock; retain their full range as a stability audit.
    near = [z for z in fits if z.cost <= best.cost + max(1e-8, best.cost * .01)]
    near_c = np.asarray([np.exp(z.x[5]) for z in near])
    q = np.exp(best.x[[0, 2, 3, 4, 5]])
    return {"A": float(q[0]), "k": float(best.x[1]), "B": float(q[1]), "b_Hz": float(q[2]),
            "C": float(q[3]), "c_Hz": float(q[4]), "normalized_residual_norm": float(np.linalg.norm(best.fun)),
            "best_cost": float(best.cost), "n_starts": len(fits), "n_near_optimal_starts": len(near),
            "near_optimal_c_min_Hz": float(np.min(near_c)), "near_optimal_c_max_Hz": float(np.max(near_c)),
            "all_start_c_min_Hz": float(np.min(c_values)), "all_start_c_max_Hz": float(np.max(c_values)),
            "clock_fit_stable_1pct": bool(np.max(near_c) / np.min(near_c) <= 1.25)}


def rat_dominant_mode(params: np.ndarray, atp: float) -> dict:
    """Largest-output-residue stable pole of public Model16D at target ATP, Pi=1 mM."""
    st = _rates_and_states(params, atp, 1.0)
    k1, km1, k2, km2, k3, km3 = [st[k] for k in ("k1", "km1", "k2", "km2", "k3", "km3")]
    B0, C0, A0 = st["B0"], st["C0"], st["A0"]
    phi_x, phi_v, phi_l = st["phi_x"], st["phi_v"], st["phi_l"]
    ps_m2, ps3 = st["phi_s_m2"], st["phi_s3"]
    d11 = -(k1 + km1 + k2); d21 = km2 - k1
    d12 = k2 - km3; d22 = -(k3 + km2 + km3)
    d31 = 0.0; d32 = 0.0
    d41 = ps_m2 * km2 * C0
    d42 = -ps_m2 * km2 * C0 - ps3 * k3 * C0
    du1 = k1 * phi_l / 2.3; du2 = km3 * phi_l / 2.3
    d33 = -phi_x / B0 * (A0 * k1 + C0 * km2)
    d44 = -phi_x / C0 * (B0 * k2 + A0 * km3)
    poles = list(np.linalg.eigvals(np.array([[d11, d12], [d21, d22]], float))) + [d33, d44]

    def transfer(s):
        hB = phi_v * s / (s - d33); hC = phi_v * s / (s - d44)
        den = (s - d22) * (s - d11) - d12 * d21
        hc_s = (d12 * (d31 * hB + d41 * hC) + (d32 * hB + d42 * hC) * (s - d11)) / den
        hc_l = (du1 * d12 + du2 * (s - d11)) / den
        return 2.2 / 1000 * st["K"] * (B0 * hB + C0 * hC + (hc_s + hc_l) * .01) + 2.2 / 1000 * st["Ks"]

    rows = []
    for pole in poles:
        if np.real(pole) >= 0:
            raise ValueError(f"Unstable/nondecaying Model16D pole: {pole}")
        eps = max(1e-7, abs(pole) * 1e-7)
        residue = eps * transfer(pole + eps)
        rows.append({"pole_s_inv": complex(pole), "pole_frequency_Hz": float(-np.real(pole) / (2 * np.pi)),
                     "output_residue_abs": float(abs(residue))})
    dom = max(rows, key=lambda r: r["output_residue_abs"])
    return {"characteristic_frequency_Hz": dom["pole_frequency_Hz"],
            "dominant_pole_s_inv": str(dom["pole_s_inv"]), "dominant_output_residue_abs": dom["output_residue_abs"],
            "all_modes": rows}


def main():
    inputs = {
        "r_robust": REPO / "analysis/02_robustness/robust/input/R_SHARED_VOCAB_ROBUST_PANELS.csv",
        "n_fit": REPO / "analysis/02_robustness/variants/results" / "N_VARIANT_FITS.csv",
        "human_source_panels": REPO / "analysis/08_boundaries/transfer/frozen_panels",
        "human_observations": REPO / "analysis/08_boundaries/preparation_support/input" / "B_HUMAN_CONDITION_OBSERVATIONS.csv",
    }
    r = pd.read_csv(inputs["r_robust"])
    n = pd.read_csv(inputs["n_fit"])
    nrow = n.loc[n.variant.eq("published_Model16D")].iloc[0]
    params = np.asarray([float(x) for x in nrow.parameter_vector.split(";")])
    human = load_human(inputs["human_observations"])
    rows = []; clocks = []

    for target, atp in (("ATP0.1", .1), ("ATP1", 1.0)):
        mode = rat_dominant_mode(params, atp)
        clocks.append({"system": "Rat_Model16D", "target": target, "clock_type": "largest-output-residue stable pole",
                       "n_source_units": 1, **{k: v for k, v in mode.items() if k != "all_modes"}})
        rr = r.loc[r.target.eq(target)].sort_values("budget")
        for _, rec in rr.iterrows():
            hz = [float(x) for x in str(rec.selected_frequency_Hz).split(";")]
            for i, f in enumerate(hz):
                rows.append({"source_system": "Rat_Model16D", "target": target, "budget": int(rec.budget),
                             "panel_index": i + 1, "source_frequency_hz": f,
                             "source_characteristic_hz": mode["characteristic_frequency_Hz"],
                             "normalized_coordinate_f_over_fchar": f / mode["characteristic_frequency_Hz"],
                             "source_panel_id": rec.panel_id, "source_clock_method": "dominant target-condition Model16D pole by output residue"})

    for target in ("ATP0.1", "ATP1"):
        samples = [s for s in human if s["target"] == target and s["group"] == "non-diabetic"]
        mean_y = np.mean(np.vstack([s["y"] for s in samples]), axis=0)
        fit = fit_empirical_c(samples[0]["freqs"], mean_y)
        clocks.append({"system": "Human_ND", "target": target, "clock_type": "effective c from empirical complex-modulus fit",
                       "n_source_units": len(samples), **fit})
        for budget in (1, 2, 3):
            p = inputs["human_source_panels"] / f"S_FROZEN_PANEL_human_ND_source_{target}_B{budget}.json"
            if not p.exists():
                continue
            panel = json.loads(p.read_text(encoding="utf-8"))
            for i, f in enumerate(panel["nominal_frequency_hz"]):
                rows.append({"source_system": "Human_ND", "target": target, "budget": budget,
                             "panel_index": i + 1, "source_frequency_hz": float(f),
                             "source_characteristic_hz": fit["c_Hz"],
                             "normalized_coordinate_f_over_fchar": float(f) / fit["c_Hz"],
                             "source_panel_id": panel["panel_id"], "source_clock_method": "effective c from empirical complex-modulus fit"})

    # Record an Awinda source recipe for the reverse direction; the rat destination has no preparation-level outcome set.
    u = pd.read_csv(REPO / "analysis/03_awinda/continuum/results" / "U_MGATP_CONTINUUM_SELECTIONS.csv")
    v = pd.read_csv(REPO / "analysis/08_boundaries/kinetics/results" / "V_TARGET_KINETIC_TIMESCALES.csv")
    for target in ("ATP0.1", "ATP1"):
        atp = .1 if target == "ATP0.1" else 1.0
        us = u.loc[np.isclose(u.target_ATP_mM, atp)]
        vs = v.loc[np.isclose(v.target_ATP_mM, atp)]
        merged = us.merge(vs, left_on="heldout_mouse", right_on="AnimalID", how="inner")
        ratio = np.asarray(merged.selected_frequency_Hz, float) / np.asarray(merged.c_Hz, float)
        clocks.append({"system": "Awinda", "target": target, "clock_type": "retrospective source c from held-out U mice",
                       "n_source_units": int(len(merged)), "median_source_ratio": float(np.median(ratio)),
                       "q25_source_ratio": float(np.quantile(ratio, .25)), "q75_source_ratio": float(np.quantile(ratio, .75)),
                       "source_clock_method": "median of U nested frequency / source-fitted target c; retrospective"})
        for i, f in enumerate([float(us.selected_frequency_Hz.median())]):
            c = float(vs.c_Hz.median())
            rows.append({"source_system": "Awinda", "target": target, "budget": 1, "panel_index": 1,
                         "source_frequency_hz": f, "source_characteristic_hz": c,
                         "normalized_coordinate_f_over_fchar": f / c,
                         "source_panel_id": "U_mouse_median_nested_frequency", "source_clock_method": "retrospective source c; not prospective"})

    tab = pd.DataFrame(rows).sort_values(["source_system", "target", "budget", "panel_index"])
    tab.to_csv(OUT / "W_SOURCE_NORMALIZED_PANELS.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(clocks).to_json(OUT / "W_SOURCE_CLOCK_DIAGNOSTICS.json", orient="records", indent=2)
    modes_path = OUT / "W_RAT_MODE_SENSITIVITY.json"
    modes = {target: rat_dominant_mode(params, atp) for target, atp in (("ATP0.1", .1), ("ATP1", 1.0))}
    modes_path.write_text(json.dumps(modes, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")
    manifest = {"freeze_time_utc": datetime.now(timezone.utc).isoformat(),
                "destination_outcomes_scored_at_freeze": False,
                "source_panel_file_sha256": sha(OUT / "W_SOURCE_NORMALIZED_PANELS.csv"),
                "source_clock_file_sha256": sha(OUT / "W_SOURCE_CLOCK_DIAGNOSTICS.json"),
                "source_inputs_sha256": {k: sha(v) for k, v in inputs.items() if v.is_file()},
                "source_rule": "retain frozen source panel frequencies as source-relative coordinates f/f_char; no destination outcomes used",
                "unavailable_route": "Awinda-to-rat empirical held-out prediction is not scoreable: the allowed rat evidence is model-level robust utility, not independent preparation-level outcomes."}
    (OUT / "W_SOURCE_PANEL_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"source_panels": len(tab), "source_clocks": len(clocks), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
