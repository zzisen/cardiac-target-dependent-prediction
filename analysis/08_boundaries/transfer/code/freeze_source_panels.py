"""Select and cryptographically freeze every source panel before T/S scoring."""
from __future__ import annotations

import hashlib
import itertools
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "PRIOR_A_TO_P_BASELINE"
OUT = HERE.parent / "frozen_panels"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "common"))
from dynamic_prediction import COMMON_HZ, load_human, predict_fold, spectrum_metrics


def panel_variance(V, g, R):
    base = max(0.0, float(g @ V @ g))
    rv = R @ V
    cross = rv @ g
    return max(0.0, base - max(0.0, float(cross @ np.linalg.solve(np.eye(R.shape[0]) + rv @ R.T, cross))))


def panel_covariance(V, R):
    rv = R @ V
    return V - V @ R.T @ np.linalg.solve(np.eye(R.shape[0]) + R @ V @ R.T, R @ V)


def k_scenarios(target):
    kb = pd.read_csv(REPO / "analysis/01_core_target_value/sparse/results/K_BASELINE_SENSITIVITIES.csv")
    kc = pd.read_csv(REPO / "analysis/01_core_target_value/sparse/results/K_LOCAL_SENSITIVITIES.csv")
    kg = pd.read_csv(REPO / "analysis/01_core_target_value/sparse/results/K_TARGET_GRADIENTS.csv")
    candidate_condition = "ATP1" if target == "ATP0.1" else "ATP0.1"
    scenarios = []
    for domain in sorted(kb.domain.unique()):
        b = kb[(kb.domain.eq(domain)) & kb.target.eq(target)]
        bmat = b.pivot(index="row_id", columns="parameter_index", values="sensitivity").sort_index().sort_index(axis=1).to_numpy(float)
        V = np.linalg.pinv(bmat.T @ bmat, rcond=1e-11)
        g = kg[(kg.domain.eq(domain)) & kg.target.eq(target)].sort_values("parameter_index").target_gradient.to_numpy(float)
        cand = kc[(kc.domain.eq(domain)) & kc.target.eq(target) & kc.condition.eq(candidate_condition)]
        actions = {}
        for sid, sub in cand.groupby("scalar_id"):
            parts = str(sid).rsplit("_", 1)
            if len(parts) != 2 or parts[1] not in {"Re", "Im"} or not str(sid).startswith("CM_"):
                continue
            f = float(str(sid).split("_")[-2].replace("Hz", ""))
            hidx = int(np.argmin(np.abs(np.asarray(COMMON_HZ) - f)))
            if abs(COMMON_HZ[hidx] - f) / COMMON_HZ[hidx] > 0.05:
                continue
            nominal = float(COMMON_HZ[hidx])
            actions.setdefault(nominal, {})[parts[1]] = sub.sort_values("parameter_index").whitened_sensitivity.to_numpy(float)
        matrices = {f: np.vstack([parts["Re"], parts["Im"]]) for f, parts in actions.items()
                    if set(parts) == {"Re", "Im"}}
        if set(matrices) != set(COMMON_HZ):
            raise ValueError(f"K {domain} {target}: common frequency action set incomplete: {sorted(matrices)}")
        scenarios.append({"domain": domain, "V": V, "g": g, "actions": matrices})
    return scenarios, candidate_condition


def choose_k_panel(target, budget, criterion):
    scenarios, candidate_condition = k_scenarios(target)
    combos = list(itertools.combinations(COMMON_HZ, budget))
    ret = np.zeros((len(combos), len(scenarios)), float)
    for j, sc in enumerate(scenarios):
        if criterion == "target_utility":
            utilities = np.asarray([panel_variance(sc["V"], sc["g"], np.vstack([sc["actions"][f] for f in combo])) for combo in combos])
            oracle = utilities.max()
            ret[:, j] = utilities / oracle if oracle else 0
        elif criterion == "parameter_A_opt":
            btrace = float(np.trace(sc["V"]))
            utilities = np.asarray([btrace - float(np.trace(panel_covariance(sc["V"], np.vstack([sc["actions"][f] for f in combo])))) for combo in combos])
            oracle = utilities.max()
            ret[:, j] = utilities / oracle if oracle else 0
        else:
            raise ValueError(criterion)
    worst = ret.min(axis=1); mean = ret.mean(axis=1); regret = (1-ret).max(axis=1)
    ix = min(range(len(combos)), key=lambda i: (-worst[i], regret[i], -mean[i], combos[i]))
    return {"source": "rat_Model16D_two_K_domains", "design_method": criterion,
            "target": target, "candidate_condition": candidate_condition, "budget": budget,
            "nominal_frequency_hz": list(combos[ix]), "panel_id": ";".join(f"{candidate_condition}_{f:g}Hz" for f in combos[ix]),
            "source_domain_fraction_oracle": {sc["domain"]: float(ret[ix, j]) for j, sc in enumerate(scenarios)},
            "worst_domain_fraction_oracle": float(worst[ix]), "mean_domain_fraction_oracle": float(mean[ix]),
            "rule": "maximin normalized source utility across the two frozen K support domains"}


def choose_human_nd(target, budget, samples):
    source = [s for s in samples if s["target"] == target and s["group"] == "non-diabetic"]
    combos = list(itertools.combinations(COMMON_HZ, budget))
    all_clusters = np.asarray([s["cluster_id"] for s in source], object)
    y = np.vstack([s["y"] for s in source])
    scores = []
    for combo in combos:
        fold_scores = []
        for held in sorted(set(all_clusters.tolist())):
            te = np.flatnonzero(all_clusters == held); tr = np.flatnonzero(all_clusters != held)
            pred = predict_fold(source, y, tr, te, combo)
            fold_scores.extend(spectrum_metrics(y[i], pred[j], source[i]["freqs"])["whole_spectrum_nrmse_pct"] for j, i in enumerate(te))
        scores.append(float(np.mean(fold_scores)))
    ix = min(range(len(combos)), key=lambda i: (scores[i], combos[i]))
    return {"source": "human_non_diabetic_public_preparations", "design_method": "source_only_leave_one_preparation_out_dynamic_prediction",
            "target": target, "candidate_condition": "ATP1" if target == "ATP0.1" else "ATP0.1", "budget": budget,
            "nominal_frequency_hz": list(combos[ix]),
            "panel_id": ";".join(f"{('ATP1' if target == 'ATP0.1' else 'ATP0.1')}_{f:g}Hz" for f in combos[ix]),
            "source_cv_mean_nrmse_pct": scores[ix], "source_cv_n_preparations": len(source),
            "rule": "choose minimum source-only LO-preparation whole-spectrum NRMSE among all common-vocabulary panels of this budget"}


def main():
    human = load_human(REPO / "analysis/08_boundaries/preparation_support/input/B_HUMAN_CONDITION_OBSERVATIONS.csv")
    r_shared = pd.read_csv(REPO / "analysis/02_robustness/robust/input/R_SHARED_VOCAB_ROBUST_PANELS.csv")
    frozen = []
    static_methods = {}
    for target in ("ATP0.1", "ATP1"):
        candidate_condition = "ATP1" if target == "ATP0.1" else "ATP0.1"
        for budget in (1, 2, 3):
            static_methods[("rat_K_target", target, budget)] = choose_k_panel(target, budget, "target_utility")
            static_methods[("rat_parameter_A", target, budget)] = choose_k_panel(target, budget, "parameter_A_opt")
            rr = r_shared[(r_shared.target.eq(target)) & (r_shared.budget.eq(budget))].iloc[0]
            hz = [float(x) for x in str(rr.selected_frequency_Hz).split(";")]
            static_methods[("rat_R_robust", target, budget)] = {
                "source": "rat_N_model_ensemble_plus_K_domains", "design_method": "R_maximin_normalized_oracle_utility",
                "target": target, "candidate_condition": candidate_condition, "budget": budget,
                "nominal_frequency_hz": hz, "panel_id": str(rr.panel_id),
                "worst_case_fraction_oracle": float(rr.worst_case_fraction_oracle_retained),
                "rule": "frozen R shared-vocabulary robust panel selected across five prespecified model variants and two Model16D domains"}
            static_methods[("human_ND_source", target, budget)] = choose_human_nd(target, budget, human)
            rng = np.random.default_rng(20261002 + (0 if target == "ATP0.1" else 100) + budget)
            random_hz = sorted(float(x) for x in rng.choice(np.asarray(COMMON_HZ), size=budget, replace=False))
            static_methods[("random_seeded", target, budget)] = {
                "source": "prespecified_rng_20261002", "design_method": "uniform_without_replacement",
                "target": target, "candidate_condition": candidate_condition, "budget": budget,
                "nominal_frequency_hz": random_hz, "panel_id": ";".join(f"{candidate_condition}_{f:g}Hz" for f in random_hz),
                "seed": 20261002 + (0 if target == "ATP0.1" else 100) + budget,
                "rule": "one seeded uniform draw from four shared frequencies"}
            conventional = [1.0, 10.0, 31.622776601683793][:budget]
            static_methods[("fixed_conventional", target, budget)] = {
                "source": "prespecified_shared_vocabulary", "design_method": "fixed_conventional",
                "target": target, "candidate_condition": candidate_condition, "budget": budget,
                "nominal_frequency_hz": conventional,
                "panel_id": ";".join(f"{candidate_condition}_{f:g}Hz" for f in conventional),
                "rule": "first budget frequencies in frozen conventional order 1, 10, 31.6228 Hz"}

    # Each file is serialized and hashed before any T or S destination scoring begins.
    for (source, target, budget), payload in sorted(static_methods.items()):
        now = datetime.now(timezone.utc).isoformat()
        payload = {**payload, "frozen_at_utc": now, "destination_evaluation_started": False,
                   "shared_vocabulary_rule": "rat/human exact nominal and Awinda nearest within 5%; no interpolation"}
        filename = f"S_FROZEN_PANEL_{source}_{target}_B{budget}.json"
        dest = OUT / filename
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        dest.write_bytes(encoded)
        frozen.append({"filename": filename, "source": source, "target": target, "budget": budget,
                       "frozen_at_utc": now, "sha256": hashlib.sha256(encoded).hexdigest(),
                       "destination_evaluation_started": False, "panel_id": payload["panel_id"]})
    manifest = pd.DataFrame(frozen)
    manifest.to_csv(HERE.parent / "S_PANEL_FREEZE_REGISTER.csv", index=False, encoding="utf-8-sig")
    print(manifest[["filename", "panel_id", "sha256"]].to_string(index=False))


if __name__ == "__main__":
    main()
