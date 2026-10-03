#!/usr/bin/env python
"""Freeze and score robust ordinal-band/frequency rules from rat model scenarios."""
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


import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "PRIOR_A_TO_X_BASELINE/shared_code"))
from dynamic_prediction import load_human, predict_fold, spectrum_metrics  # noqa: E402

HERE = REPO / "analysis/08_boundaries/robust_bands"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
RAT_INPUT = REPO / "analysis/08_boundaries/state_surface/results/AA_CROSS_SYSTEM_TARGET_SURFACE.csv"
AW_INPUT = REPO / "analysis/03_awinda/landscape/results/Y_MOUSE_LEVEL_UTILITY.csv"
HU_INPUT = REPO / "analysis/08_boundaries/state_surface/results/AA_HUMAN_PREPARATION_UTILITY.csv"
RULES = ["AB0_global_fixed_frequency", "AB1_target_conditioned_mean",
         "AB2_target_conditioned_minimax", "AB3_global_minimax_target_agnostic"]
TARGETS = ["ATP0.1", "ATP1"]
TARGET_MMS = {"ATP0.1": 0.1, "ATP1": 1.0}


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bands_for_freq(freqs):
    out = np.empty(len(freqs), dtype=object)
    for label, ix in zip(("low", "mid", "high"), np.array_split(np.arange(len(freqs)), 3)):
        out[ix] = label
    return out


def choose_policies(data, train_contexts):
    # The frequency grid is common across frozen rat Model16D/N scenarios.
    sub = data[data.context.isin(train_contexts)]
    freqs = np.sort(sub.frequency_Hz.unique().astype(float))
    if len(freqs) != 17:
        raise ValueError(f"expected common 17-point rat grid; found {len(freqs)}")
    score_by = {}
    for t in TARGETS:
        a = sub[sub.future_target_label.eq(t)].pivot_table(
            index="context", columns="frequency_Hz", values="utility_rank_score_0_to_1", aggfunc="mean"
        ).reindex(index=train_contexts, columns=freqs)
        if a.isna().any().any():
            raise ValueError(f"incomplete model/domain × target × frequency grid in {t}")
        score_by[t] = a.to_numpy(float)
    pooled = np.concatenate([score_by[t] for t in TARGETS], axis=0)
    objectives = {
        "AB0_global_fixed_frequency": pooled.mean(axis=0),
        "AB3_global_minimax_target_agnostic": pooled.min(axis=0),
    }
    result = {}
    for rule, obj in objectives.items():
        i = int(np.argmax(obj))
        result[rule] = {"ATP0.1": float(freqs[i]), "ATP1": float(freqs[i]),
                        "global_frequency_Hz": float(freqs[i]), "objective_score": float(obj[i]),
                        "policy_objective": "mean rank score" if rule.startswith("AB0") else "minimum rank score"}
    for rule, mode in (("AB1_target_conditioned_mean", "mean"), ("AB2_target_conditioned_minimax", "min")):
        targets = {}
        for t in TARGETS:
            mat = score_by[t]
            obj = mat.mean(axis=0) if mode == "mean" else mat.min(axis=0)
            i = int(np.argmax(obj))
            targets[t] = {"frequency_Hz": float(freqs[i]), "objective_score": float(obj[i]),
                          "policy_objective": "mean scenario rank score" if mode == "mean" else "worst-case scenario rank score"}
        result[rule] = targets
    band = bands_for_freq(freqs)
    for rule, bytarget in result.items():
        if rule in ("AB0_global_fixed_frequency", "AB3_global_minimax_target_agnostic"):
            bytarget["band_by_target"] = {t: str(band[int(np.argmin(np.abs(freqs - bytarget[t])))]) for t in TARGETS}
        else:
            bytarget["band_by_target"] = {t: str(band[int(np.argmin(np.abs(freqs - bytarget[t]["frequency_Hz"])))]) for t in TARGETS}
    return result, freqs


def build_freeze():
    rat = pd.read_csv(RAT_INPUT)
    rat = rat[(rat.system == "Rat_Model16D") & rat.future_target_label.isin(TARGETS)].copy()
    contexts = sorted(rat.context.astype(str).unique())
    full, freqs = choose_policies(rat, contexts)
    folds = []
    for held in contexts:
        train = [c for c in contexts if c != held]
        policy, _ = choose_policies(rat, train)
        folds.append({"heldout_scenario_context": held, "training_scenario_contexts": train,
                      "policy_recommendations": policy})
    rule = {
        "status": "FROZEN_BEFORE_AB_OUTCOME_SCORING",
        "version": "YAB-AB-1.0",
        "utility_selection_metric": "Within-context frequency rank score (1 best, 0 worst), frozen to avoid unstable raw local Fisher magnitudes.",
        "candidate_frequency_grid_Hz": [float(x) for x in freqs],
        "candidate_scenario_contexts": contexts,
        "target_states": TARGETS,
        "rule_definitions": {
            "AB0_global_fixed_frequency": "maximize mean rank score across training model/domain contexts and both target states; one frequency for every target",
            "AB1_target_conditioned_mean": "for each target state, maximize mean rank score across training contexts",
            "AB2_target_conditioned_minimax": "for each target state, maximize worst-case rank score across training contexts",
            "AB3_global_minimax_target_agnostic": "maximize worst-case rank score across training contexts and both targets; one frequency for every target",
        },
        "full_scenario_policy": full,
        "leave_one_scenario_out_frozen_policies": folds,
        "cross_system_band_policy": "Project the selected rat-grid frequency to its pre-frozen ordinal rank band only; within Awinda/human that band, choose the frequency using outer-training target/preparation utilities, then score the held-out unit/target.",
        "cross_system_band_definition_file": "03_AA_CROSS_MODEL_STATE_SURFACE/AA_BAND_DEFINITION_FREEZE.md",
        "frequency_grid_sha256": hashlib.sha256(np.asarray(freqs, dtype="<f8").tobytes()).hexdigest(),
        "code_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in (RAT_INPUT,)},
    }
    p = OUT / "AB_DESIGN_RULE_FREEZE.json"
    p.write_text(json.dumps(rule, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = sha256(p)
    (OUT / "AB_DESIGN_RULE_SHA256.txt").write_text(f"{digest}  AB_DESIGN_RULE_FREEZE.json\n", encoding="ascii")
    return rule, digest


def score_rat_loto(rule, digest):
    rat = pd.read_csv(RAT_INPUT)
    rat = rat[(rat.system == "Rat_Model16D") & rat.future_target_label.isin(TARGETS)].copy()
    rows = []
    for fold in rule["leave_one_scenario_out_frozen_policies"]:
        held = fold["heldout_scenario_context"]
        policy = fold["policy_recommendations"]
        for t in TARGETS:
            h = rat[(rat.context.astype(str) == held) & rat.future_target_label.eq(t)].sort_values("frequency_Hz")
            fgrid = h.frequency_Hz.to_numpy(float)
            utility = h.utility_raw.to_numpy(float)
            rank_score = h.utility_rank_score_0_to_1.to_numpy(float)
            oracle = float(np.nanmax(utility))
            for name in RULES:
                if name in ("AB0_global_fixed_frequency", "AB3_global_minimax_target_agnostic"):
                    freq = float(policy[name][t])
                else:
                    freq = float(policy[name][t]["frequency_Hz"])
                i = int(np.argmin(np.abs(fgrid - freq)))
                rows.append({"evaluation": "rat_leave_one_scenario_out", "heldout_scenario_context": held,
                             "training_scenario_contexts": ";".join(fold["training_scenario_contexts"]),
                             "future_target_label": t, "design_rule": name,
                             "selected_frequency_Hz": float(fgrid[i]), "selected_band": str(h.frequency_rank_band.iloc[i]),
                             "heldout_utility_raw": float(utility[i]), "heldout_context_oracle_utility": oracle,
                             "heldout_oracle_retention_fraction": float(utility[i] / oracle) if oracle > 0 else np.nan,
                             "heldout_frequency_rank_score": float(rank_score[i]),
                             "heldout_regret_fraction_of_oracle": float(1.0 - utility[i] / oracle) if oracle > 0 else np.nan,
                             "heldout_outcome_used_for_selection": False, "freeze_sha256": digest})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "AB_RAT_HELDOUT_SCENARIO_RESULTS.csv", index=False)
    summary = (result.groupby("design_rule", as_index=False)
               .agg(n_scenario_target_folds=("heldout_scenario_context", "size"),
                    mean_oracle_retention=("heldout_oracle_retention_fraction", "mean"),
                    worst_case_oracle_retention=("heldout_oracle_retention_fraction", "min"),
                    mean_rank_score=("heldout_frequency_rank_score", "mean"),
                    worst_rank_score=("heldout_frequency_rank_score", "min"),
                    mean_regret_fraction=("heldout_regret_fraction_of_oracle", "mean"),
                    selected_band_stability=("selected_band", lambda x: x.value_counts(normalize=True).max())))
    summary.to_csv(OUT / "AB_RAT_HELDOUT_SCENARIO_SUMMARY.csv", index=False)
    return result, summary


def _band_freqs(freqs, band):
    labels = bands_for_freq(np.asarray(freqs, float))
    return np.flatnonzero(labels == band)


def score_awinda(rule, digest):
    aw = pd.read_csv(AW_INPUT)
    freqs = np.sort(aw.frequency_Hz.unique().astype(float))
    band_labels = bands_for_freq(freqs)
    targets = np.sort(aw.target_ATP_mM.unique().astype(float))
    rows = []
    policy = rule["full_scenario_policy"]
    mouse_mean = (aw.groupby(["target_ATP_mM", "heldout_mouse", "frequency_Hz"], as_index=False)
                 .gain_pp.mean())
    curves = {float(t): mouse_mean[mouse_mean.target_ATP_mM.eq(t)].pivot(
        index="heldout_mouse", columns="frequency_Hz", values="gain_pp").reindex(columns=freqs).mean(axis=0).to_numpy(float)
              for t in targets}
    for test_t in targets:
        train_t = [float(t) for t in targets if float(t) != float(test_t)]
        training = np.mean(np.vstack([curves[t] for t in train_t]), axis=0)
        observed = curves[float(test_t)]
        oracle_idx = int(np.argmax(observed))
        for name in RULES:
            # Map only the target-conditioned anchor label to the nearest frozen rat target state.
            mapped = min(TARGETS, key=lambda t: abs(np.log10(TARGET_MMS[t]) - np.log10(float(test_t))))
            if name in ("AB0_global_fixed_frequency", "AB3_global_minimax_target_agnostic"):
                chosen_band = policy[name]["band_by_target"][mapped]
            else:
                chosen_band = policy[name]["band_by_target"][mapped]
            cand = _band_freqs(freqs, chosen_band)
            i = int(cand[np.argmax(training[cand])])
            gain = float(observed[i])
            rows.append({"evaluation": "Awinda_outer_target_holdout_ordinal_band_transfer",
                         "heldout_target_ATP_mM": float(test_t), "outer_training_targets_mM": ";".join(f"{x:g}" for x in train_t),
                         "rat_anchor_target_state": mapped, "design_rule": name,
                         "selected_frequency_band": chosen_band, "selected_frequency_Hz": float(freqs[i]),
                         "heldout_gain_pp": gain, "heldout_oracle_gain_pp": float(observed[oracle_idx]),
                         "heldout_regret_pp": float(observed[oracle_idx] - gain),
                         "heldout_oracle_retention_fraction": float(gain / observed[oracle_idx]) if observed[oracle_idx] > 0 else np.nan,
                         "heldout_band_matches_oracle": bool(band_labels[i] == band_labels[oracle_idx]),
                         "heldout_target_outcome_used_for_selection": False, "freeze_sha256": digest})
    return pd.DataFrame(rows)


def score_human(rule, digest):
    all_samples = load_human(REPO / "analysis/08_boundaries/preparation_support/input/B_HUMAN_CONDITION_OBSERVATIONS.csv")
    freqs = np.asarray(all_samples[0]["freqs"], float)
    human_bands = bands_for_freq(freqs)
    rows = []
    policy = rule["full_scenario_policy"]
    for disease in ("non-diabetic", "diabetic"):
        for t in TARGETS:
            samples = [s for s in all_samples if s["group"] == disease and s["target"] == t]
            if len(samples) != 10:
                raise ValueError(f"Expected 10 human preparations in {disease}/{t}, got {len(samples)}")
            y = np.vstack([s["y"] for s in samples])
            for held in range(len(samples)):
                outer_train = np.asarray([i for i in range(len(samples)) if i != held], int)
                outer_test = np.asarray([held], int)
                inner_gain = np.zeros(len(freqs), float)
                for inner in outer_train:
                    inner_test = np.asarray([inner], int)
                    inner_train = outer_train[outer_train != inner]
                    base_pred = predict_fold(samples, y, inner_train, inner_test, None)
                    base_err = spectrum_metrics(y[inner], base_pred[0], freqs)["whole_spectrum_nrmse_pct"]
                    for j, f in enumerate(freqs):
                        pred = predict_fold(samples, y, inner_train, inner_test, [float(f)])
                        err = spectrum_metrics(y[inner], pred[0], freqs)["whole_spectrum_nrmse_pct"]
                        inner_gain[j] += base_err - err
                inner_gain /= len(outer_train)
                base_outer_pred = predict_fold(samples, y, outer_train, outer_test, None)
                baseline_outer = spectrum_metrics(y[held], base_outer_pred[0], freqs)["whole_spectrum_nrmse_pct"]
                oracle_idx = None
                true_curve = []
                for f in freqs:
                    p = predict_fold(samples, y, outer_train, outer_test, [float(f)])
                    true_curve.append(baseline_outer - spectrum_metrics(y[held], p[0], freqs)["whole_spectrum_nrmse_pct"])
                true_curve = np.asarray(true_curve, float)
                oracle_idx = int(np.argmax(true_curve))
                for name in RULES:
                    rat_band = policy[name]["band_by_target"][t]
                    cand = np.flatnonzero(human_bands == rat_band)
                    i = int(cand[np.argmax(inner_gain[cand])])
                    gain = float(true_curve[i])
                    rows.append({"evaluation": "Human_outer_preparation_holdout_ordinal_band_transfer",
                                 "disease_group": disease, "future_target_label": t,
                                 "heldout_preparation_id": samples[held]["unit_id"], "design_rule": name,
                                 "rat_anchor_band": rat_band, "selected_human_frequency_band": str(human_bands[i]),
                                 "selected_frequency_Hz": float(freqs[i]), "nested_inner_selection_mean_gain_pp": float(inner_gain[i]),
                                 "heldout_baseline_nrmse_pct": float(baseline_outer), "heldout_gain_pp": gain,
                                 "heldout_oracle_gain_pp": float(true_curve[oracle_idx]),
                                 "heldout_regret_pp": float(true_curve[oracle_idx] - gain),
                                 "heldout_oracle_retention_fraction": float(gain / true_curve[oracle_idx]) if true_curve[oracle_idx] > 0 else np.nan,
                                 "heldout_band_matches_oracle": bool(human_bands[i] == human_bands[oracle_idx]),
                                 "heldout_preparation_outcome_used_for_selection": False,
                                 "donor_linkage_available": False, "freeze_sha256": digest})
    return pd.DataFrame(rows)


def score():
    p = OUT / "AB_DESIGN_RULE_FREEZE.json"
    frozen = json.loads(p.read_text(encoding="utf-8"))
    expected = (OUT / "AB_DESIGN_RULE_SHA256.txt").read_text(encoding="ascii").split()[0]
    digest = sha256(p)
    if frozen.get("status") != "FROZEN_BEFORE_AB_OUTCOME_SCORING" or expected != digest:
        raise RuntimeError("AB frozen-rule integrity check failed")
    rat, rat_summary = score_rat_loto(frozen, digest)
    aw = score_awinda(frozen, digest)
    hu = score_human(frozen, digest)
    aw.to_csv(OUT / "AB_AWINDA_HELDOUT_TARGET_BAND_RESULTS.csv", index=False)
    hu.to_csv(OUT / "AB_HUMAN_HELDOUT_PREPARATION_BAND_RESULTS.csv", index=False)
    ext = pd.concat([aw.assign(system="Awinda"), hu.assign(system="Human")], ignore_index=True, sort=False)
    ext.to_csv(OUT / "AB_EXTERNAL_HELDOUT_RESULTS.csv", index=False)
    ext_summary = (ext.groupby(["system", "design_rule"], as_index=False)
                   .agg(n_heldout_units=("heldout_regret_pp", "size"), mean_regret_pp=("heldout_regret_pp", "mean"),
                        mean_oracle_retention=("heldout_oracle_retention_fraction", "mean"),
                        median_oracle_retention=("heldout_oracle_retention_fraction", "median"),
                        band_hit_fraction=("heldout_band_matches_oracle", "mean"),
                        mean_gain_pp=("heldout_gain_pp", "mean")))
    ext_summary.to_csv(OUT / "AB_EXTERNAL_HELDOUT_SUMMARY.csv", index=False)
    print("AB frozen hash:", digest)
    print("Rat scenario LOTO:\n", rat_summary.to_string(index=False))
    print("External heldout:\n", ext_summary.to_string(index=False))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"freeze", "score"}:
        raise SystemExit("usage: run_ab_robust.py freeze|score")
    if sys.argv[1] == "freeze":
        rule, digest = build_freeze()
        print(json.dumps(rule["full_scenario_policy"], indent=2))
        print("AB rule freeze sha256:", digest)
    else:
        score()


if __name__ == "__main__":
    main()
