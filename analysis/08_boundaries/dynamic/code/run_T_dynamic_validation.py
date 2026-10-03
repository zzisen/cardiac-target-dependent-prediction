"""LO-cluster prediction of held-out future complex-modulus spectra."""
from __future__ import annotations

import itertools
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "PRIOR_A_TO_P_BASELINE"
PANEL_DIR = REPO / "analysis/08_boundaries/transfer/frozen_panels"
OUT = HERE.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "common"))
from dynamic_prediction import (COMMON_HZ, load_awinda, load_human,
                                nearest_frequency_index, predict_fold, spectrum_metrics)


def read_panel(method, target, budget):
    path = PANEL_DIR / f"S_FROZEN_PANEL_{method}_{target}_B{budget}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fold_iter(samples):
    clusters = sorted(set(s["cluster_id"] for s in samples))
    for held in clusters:
        test = np.asarray([i for i, s in enumerate(samples) if s["cluster_id"] == held], int)
        train = np.asarray([i for i, s in enumerate(samples) if s["cluster_id"] != held], int)
        yield held, train, test, clusters


def evaluate_fixed_method(system, target, samples, method, budget, panel_hz, method_detail):
    y = np.vstack([s["y"] for s in samples])
    rows = []
    split_rows = []
    for held, train, test, clusters in fold_iter(samples):
        pred = predict_fold(samples, y, train, test, panel_hz)
        split_rows.append({"system": system, "target": target, "method": method,
                           "budget": budget, "heldout_cluster": held,
                           "train_clusters": ";".join(x for x in clusters if x != held),
                           "test_clusters": held, "n_train_units": len(train), "n_test_units": len(test)})
        for j, i in enumerate(test):
            metrics = spectrum_metrics(y[i], pred[j], samples[i]["freqs"])
            rows.append({"system": system, "target": target, "method": method, "budget": budget,
                         "panel_id": method_detail.get("panel_id", method),
                         "panel_frequency_hz": ";".join(f"{x:g}" for x in (panel_hz or [])),
                         "cluster_id": samples[i]["cluster_id"], "unit_id": samples[i]["unit_id"],
                         "group": samples[i]["group"], "n_fibres": samples[i].get("n_fibres", np.nan),
                         **metrics, "y_true_json": json.dumps(y[i].tolist(), separators=(",", ":")),
                         "y_pred_json": json.dumps(pred[j].tolist(), separators=(",", ":")),
                         "frequencies_json": json.dumps(samples[i]["freqs"].tolist(), separators=(",", ":")),
                         "train_clusters": ";".join(x for x in clusters if x != held),
                         "test_cluster": held})
    return rows, split_rows


def evaluate_nested_destination_oracle(system, target, samples, budget):
    y = np.vstack([s["y"] for s in samples])
    clusters = sorted(set(s["cluster_id"] for s in samples))
    panels = list(itertools.combinations(COMMON_HZ, budget))
    rows = []
    selections = []
    split_rows = []
    for held in clusters:
        test = np.asarray([i for i, s in enumerate(samples) if s["cluster_id"] == held], int)
        outer_train = np.asarray([i for i, s in enumerate(samples) if s["cluster_id"] != held], int)
        inner_clusters = sorted(set(samples[i]["cluster_id"] for i in outer_train))
        panel_errors = []
        for panel in panels:
            errs = []
            for inner in inner_clusters:
                inner_test = np.asarray([i for i in outer_train if samples[i]["cluster_id"] == inner], int)
                inner_train = np.asarray([i for i in outer_train if samples[i]["cluster_id"] != inner], int)
                inner_pred = predict_fold(samples, y, inner_train, inner_test, panel)
                errs.extend(spectrum_metrics(y[i], inner_pred[j], samples[i]["freqs"])["whole_spectrum_nrmse_pct"]
                            for j, i in enumerate(inner_test))
            panel_errors.append(float(np.mean(errs)))
        selected = min(range(len(panels)), key=lambda ix: (panel_errors[ix], panels[ix]))
        chosen = panels[selected]
        pred = predict_fold(samples, y, outer_train, test, chosen)
        panel_id = ";".join(f"{('ATP1' if target == 'ATP0.1' else 'ATP0.1')}_{f:g}Hz" for f in chosen)
        selections.append({"system": system, "target": target, "budget": budget,
                           "heldout_cluster": held, "selected_panel_id": panel_id,
                           "inner_cv_mean_nrmse_pct": panel_errors[selected],
                           "inner_candidate_panels": len(panels), "inner_clusters": len(inner_clusters),
                           "selection_uses_outer_test_outcome": False})
        split_rows.append({"system": system, "target": target, "method": "destination_nested_oracle",
                           "budget": budget, "heldout_cluster": held,
                           "train_clusters": ";".join(x for x in clusters if x != held),
                           "test_clusters": held, "n_train_units": len(outer_train), "n_test_units": len(test)})
        for j, i in enumerate(test):
            metrics = spectrum_metrics(y[i], pred[j], samples[i]["freqs"])
            rows.append({"system": system, "target": target, "method": "destination_nested_oracle", "budget": budget,
                         "panel_id": panel_id, "panel_frequency_hz": ";".join(f"{x:g}" for x in chosen),
                         "cluster_id": samples[i]["cluster_id"], "unit_id": samples[i]["unit_id"],
                         "group": samples[i]["group"], "n_fibres": samples[i].get("n_fibres", np.nan),
                         **metrics, "y_true_json": json.dumps(y[i].tolist(), separators=(",", ":")),
                         "y_pred_json": json.dumps(pred[j].tolist(), separators=(",", ":")),
                         "frequencies_json": json.dumps(samples[i]["freqs"].tolist(), separators=(",", ":")),
                         "train_clusters": ";".join(x for x in clusters if x != held), "test_cluster": held})
    return rows, split_rows, selections


def summarize(predictions: pd.DataFrame):
    metric_cols = ["whole_spectrum_rmse", "whole_spectrum_nrmse_pct", "re_rmse", "im_rmse",
                   "spectral_shape_correlation", "log_frequency_weighted_rmse", "omit_1hz_nrmse_pct"]
    by_cluster = predictions.groupby(["system", "target", "method", "budget", "cluster_id"], as_index=False)[metric_cols].mean()
    summary = by_cluster.groupby(["system", "target", "method", "budget"], as_index=False).agg(
        n_heldout_clusters=("cluster_id", "nunique"),
        **{f"equal_cluster_mean_{m}": (m, "mean") for m in metric_cols},
        **{f"median_cluster_{m}": (m, "median") for m in metric_cols})
    baseline = by_cluster[by_cluster.method.eq("baseline_only")][["system", "target", "cluster_id", "whole_spectrum_nrmse_pct"]].rename(
        columns={"whole_spectrum_nrmse_pct": "baseline_nrmse_pct"})
    summary = summary.merge(baseline.groupby(["system", "target"], as_index=False).baseline_nrmse_pct.mean(),
                            on=["system", "target"], how="left")
    summary["improvement_vs_baseline_pp"] = summary["baseline_nrmse_pct"] - summary["equal_cluster_mean_whole_spectrum_nrmse_pct"]
    return summary, by_cluster


def main():
    human = load_human(REPO / "analysis/08_boundaries/preparation_support/input/B_HUMAN_CONDITION_OBSERVATIONS.csv")
    awinda = load_awinda(REPO / "analysis/03_awinda/frequency/results/Q_AWINDA_ANIMAL_CONDITION_AGGREGATES.csv")
    all_rows = []; all_splits = []; oracle_selections = []
    for system, raw in (("Human", human), ("Awinda", awinda)):
        for target in ("ATP0.1", "ATP1"):
            samples = [s for s in raw if s["target"] == target]
            # Baseline-only keeps group, two train-only PCs of the baseline spectrum, and baseline stress.
            rows, splits = evaluate_fixed_method(system, target, samples, "baseline_only", 0, [], {"panel_id": "baseline_only"})
            all_rows.extend(rows); all_splits.extend(splits)
            for budget in (1, 2, 3):
                for source in ("rat_K_target", "rat_R_robust", "rat_parameter_A", "human_ND_source",
                               "random_seeded", "fixed_conventional"):
                    detail = read_panel(source, target, budget)
                    hz = [float(x) for x in detail["nominal_frequency_hz"]]
                    rows, splits = evaluate_fixed_method(system, target, samples, source, budget, hz, detail)
                    all_rows.extend(rows); all_splits.extend(splits)
                rows, splits, selected = evaluate_nested_destination_oracle(system, target, samples, budget)
                all_rows.extend(rows); all_splits.extend(splits); oracle_selections.extend(selected)
            # Unbudgeted candidate-condition whole spectrum is a reference only.
            full_hz = samples[0]["freqs"].tolist()
            rows, splits = evaluate_fixed_method(system, target, samples, "full_spectrum_reference", len(full_hz), full_hz,
                                                 {"panel_id": "all_candidate_condition_frequencies_reference"})
            all_rows.extend(rows); all_splits.extend(splits)
    pred = pd.DataFrame(all_rows)
    pred.to_csv(OUT / "T_DYNAMIC_HELDOUT_PREDICTIONS.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_splits).drop_duplicates().to_csv(OUT / "T_CV_SPLITS.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(oracle_selections).to_csv(OUT / "T_DESTINATION_ORACLE_PANEL_SELECTIONS.csv", index=False, encoding="utf-8-sig")
    summary, cluster = summarize(pred)
    summary.to_csv(OUT / "T_DYNAMIC_HELDOUT_RESULTS.csv", index=False, encoding="utf-8-sig")
    cluster.to_csv(OUT / "T_CLUSTER_WEIGHTED_RESULTS.csv", index=False, encoding="utf-8-sig")
    sensitivity = summary[["system", "target", "method", "budget", "n_heldout_clusters", "baseline_nrmse_pct",
                          "equal_cluster_mean_whole_spectrum_nrmse_pct", "improvement_vs_baseline_pp",
                          "equal_cluster_mean_re_rmse", "equal_cluster_mean_im_rmse",
                          "equal_cluster_mean_spectral_shape_correlation", "equal_cluster_mean_log_frequency_weighted_rmse",
                          "equal_cluster_mean_omit_1hz_nrmse_pct"]].copy()
    sensitivity.to_csv(OUT / "T_METRIC_SENSITIVITY.csv", index=False, encoding="utf-8-sig")
    print(summary[summary.method.isin(["baseline_only", "rat_K_target", "rat_R_robust", "rat_parameter_A",
                                       "human_ND_source", "random_seeded", "fixed_conventional",
                                       "destination_nested_oracle"])][["system", "target", "method", "budget",
                                       "equal_cluster_mean_whole_spectrum_nrmse_pct", "improvement_vs_baseline_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
