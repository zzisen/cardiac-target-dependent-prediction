#!/usr/bin/env python
"""Score frozen absolute-Hz and kinetic-normalized panels on held-out destinations."""
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

HERE = Path(__file__).resolve()
UX = HERE.parents[2]
PROJECT = HERE.parents[3]
OLD = PROJECT / "P2_QT_UPLIFT_2026-10-02"
OUT = HERE.parents[1] / "results"
sys.path.insert(0, str(OLD / "common"))
sys.path.insert(0, str(HERE.parent))
from dynamic_prediction import load_awinda, load_human, nearest_frequency_index, predict_fold, spectrum_metrics  # noqa: E402
from prepare_W_destination_panels import nearest_panel, source_records  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics_rows(samples, y, train_idx, test_idx, panel, route, target, budget, method):
    pred = predict_fold(samples, y, train_idx, test_idx, panel)
    rows = []
    for j, i in enumerate(test_idx):
        m = spectrum_metrics(y[i], pred[j], samples[i]["freqs"])
        rows.append({"route": route, "target": target, "budget": budget, "method": method,
                     "cluster_id": str(samples[i]["cluster_id"]), "unit_id": str(samples[i]["unit_id"]),
                     "group": str(samples[i]["group"]), "panel_hz": ";".join(f"{x:g}" for x in (panel or [])),
                     "n_unique_panel_frequencies": len(set(panel or [])), **m})
    return rows


def main():
    panel_maps = pd.read_csv(OUT / "W_DESTINATION_PANEL_MAPS_FROZEN.csv")
    frozen_sources = pd.read_csv(OUT / "W_SOURCE_NORMALIZED_PANELS.csv")
    freeze_manifest = json.loads((OUT / "W_DESTINATION_PANEL_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    assert freeze_manifest["destination_outcomes_scored_at_freeze"] is False
    assert sha(OUT / "W_DESTINATION_PANEL_MAPS_FROZEN.csv") == freeze_manifest["destination_panel_maps_sha256"]
    assert sha(OUT / "W_DESTINATION_CLOCKS_TRAIN_ONLY.csv") == freeze_manifest["training_only_clocks_sha256"]

    awinda = load_awinda(REPO / "analysis/03_awinda/frequency/results" / "Q_AWINDA_ANIMAL_CONDITION_AGGREGATES.csv")
    human = load_human(REPO / "analysis/08_boundaries/preparation_support/input" / "B_HUMAN_CONDITION_OBSERVATIONS.csv")
    route_specs = [
        ("rat_Model16D_to_Awinda", "Awinda", "Rat_Model16D", awinda, None),
        ("rat_Model16D_to_Human_ND", "Human", "Rat_Model16D", human, "non-diabetic"),
        ("rat_Model16D_to_Human_T2D", "Human", "Rat_Model16D", human, "diabetic"),
        ("human_ND_to_Human_T2D", "Human", "Human_ND", human, "diabetic"),
    ]
    all_rows = []
    for route, system, source_system, raw, group_keep in route_specs:
        for target in ("ATP0.1", "ATP1"):
            samples = [s for s in raw if s["target"] == target and (group_keep is None or s["group"] == group_keep)]
            if not samples:
                continue
            y = np.vstack([s["y"] for s in samples])
            clusters = sorted({str(s["cluster_id"]) for s in samples})
            for held in clusters:
                test_idx = np.asarray([i for i, s in enumerate(samples) if str(s["cluster_id"]) == held], int)
                train_idx = np.asarray([i for i, s in enumerate(samples) if str(s["cluster_id"]) != held], int)
                all_rows.extend(metrics_rows(samples, y, train_idx, test_idx, [], route, target, 0, "baseline_only"))
                for budget in (1, 2, 3):
                    src = source_records(frozen_sources, source_system, target, budget)
                    raw_hz = src.source_frequency_hz.to_numpy(float).tolist()
                    abs_map = nearest_panel(samples[test_idx[0]]["freqs"], raw_hz)
                    abs_panel = [z["mapped_hz"] for z in abs_map]
                    all_rows.extend(metrics_rows(samples, y, train_idx, test_idx, abs_panel, route, target, budget, "absolute_Hz"))

                    # Awinda source has multiple genotype/drug strata within one held-out mouse.
                    test_groups = sorted({str(samples[i]["group"]) for i in test_idx})
                    norm_rows = panel_maps[(panel_maps.route == route) & (panel_maps.target == target) &
                                           (panel_maps.budget == budget) & (panel_maps.heldout_cluster.astype(str) == held)]
                    if not len(norm_rows):
                        raise ValueError(f"No frozen normalized map: {route}/{target}/B{budget}/{held}")
                    for group in test_groups:
                        tr = norm_rows[norm_rows.destination_state.eq(group)]
                        if len(tr) != budget:
                            raise ValueError(f"Expected {budget} map rows for {route}/{target}/B{budget}/{held}/{group}; got {len(tr)}")
                        test_state_idx = np.asarray([i for i in test_idx if str(samples[i]["group"]) == group], int)
                        mapped = tr.sort_values("panel_index").mapped_destination_hz.astype(float).tolist()
                        all_rows.extend(metrics_rows(samples, y, train_idx, test_state_idx, mapped,
                                                     route, target, budget, "kinetic_normalized"))

    unit = pd.DataFrame(all_rows)
    unit.to_csv(OUT / "W_HELDOUT_TRANSFER_METRICS.csv", index=False, encoding="utf-8-sig")
    cluster = unit.groupby(["route", "target", "budget", "method", "cluster_id"], as_index=False).agg(
        whole_spectrum_nrmse_pct=("whole_spectrum_nrmse_pct", "mean"),
        log_frequency_weighted_rmse=("log_frequency_weighted_rmse", "mean"),
        spectral_shape_correlation=("spectral_shape_correlation", "mean"))
    cluster.to_csv(OUT / "W_CLUSTER_TRANSFER_METRICS.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    rng = np.random.default_rng(20261002)
    for (route, target, budget), sub in cluster[cluster.budget.gt(0)].groupby(["route", "target", "budget"]):
        wide = sub.pivot(index="cluster_id", columns="method", values="whole_spectrum_nrmse_pct").dropna()
        base = cluster[(cluster.route == route) & (cluster.target == target) & (cluster.budget == 0)].set_index("cluster_id").whole_spectrum_nrmse_pct
        oracle_src = pd.read_csv(REPO / "analysis/08_boundaries/dynamic/results" / "T_DYNAMIC_HELDOUT_PREDICTIONS.csv")
        oracle_src = oracle_src[(oracle_src.system == ("Awinda" if route.endswith("Awinda") else "Human")) &
                                (oracle_src.target == target) & (oracle_src.method == "destination_nested_oracle") &
                                (oracle_src.budget == budget)]
        if route == "rat_Model16D_to_Human_ND":
            oracle_src = oracle_src[oracle_src.group.eq("non-diabetic")]
        elif route.endswith("Human_T2D"):
            oracle_src = oracle_src[oracle_src.group.eq("diabetic")]
        oracle = oracle_src.groupby(oracle_src.cluster_id.astype(str)).whole_spectrum_nrmse_pct.mean()
        wide = wide.join(base.rename("baseline"), how="inner").join(oracle.rename("oracle"), how="inner").dropna()
        if not len(wide):
            continue
        d = wide.kinetic_normalized - wide.absolute_Hz
        boots = np.empty(20000, float)
        arr = d.to_numpy(float)
        for b in range(len(boots)):
            boots[b] = np.mean(rng.choice(arr, size=len(arr), replace=True))
        for method in ("absolute_Hz", "kinetic_normalized"):
            gain = wide.baseline - wide[method]
            regret = wide[method] - wide.oracle
            retention = gain / (wide.baseline - wide.oracle).replace(0, np.nan)
            summary_rows.append({"route": route, "target": target, "budget": budget, "method": method,
                                 "n_heldout_clusters": int(len(wide)), "mean_gain_vs_baseline_pp": float(gain.mean()),
                                 "median_gain_vs_baseline_pp": float(gain.median()),
                                 "n_clusters_gain_positive": int((gain > 0).sum()),
                                 "n_clusters_gain_nonpositive": int((gain <= 0).sum()),
                                 "mean_absolute_regret_vs_destination_oracle_pp": float(regret.mean()),
                                 "mean_destination_oracle_retention": float(retention.mean()) if retention.notna().any() else np.nan})
        summary_rows.append({"route": route, "target": target, "budget": budget, "method": "paired_normalized_minus_absolute",
                             "n_heldout_clusters": int(len(wide)), "mean_gain_vs_baseline_pp": float((-d).mean()),
                             "median_gain_vs_baseline_pp": float((-d).median()),
                             "paired_delta_nrmse_normalized_minus_absolute_pp": float(d.mean()),
                             "paired_delta_bootstrap_95_low": float(np.quantile(boots, .025)),
                             "paired_delta_bootstrap_95_high": float(np.quantile(boots, .975)),
                             "n_clusters_normalized_better": int((d < 0).sum()),
                             "n_clusters_normalized_worse": int((d > 0).sum())})
    # One primary paired summary for the pre-existing R->Awinda transfer route: first average
    # the six target-by-budget contrasts within each mouse, then bootstrap mice as clusters.
    core = cluster[(cluster.route == "rat_Model16D_to_Awinda") & cluster.budget.isin([1, 2, 3])]
    core_wide = core.pivot_table(index="cluster_id", columns=["target", "budget", "method"],
                                 values="whole_spectrum_nrmse_pct")
    if len(core_wide):
        delta = (core_wide.xs("kinetic_normalized", axis=1, level="method") -
                 core_wide.xs("absolute_Hz", axis=1, level="method"))
        per_mouse = delta.mean(axis=1).to_numpy(float)
        pooled_boot = np.asarray([np.mean(rng.choice(per_mouse, size=len(per_mouse), replace=True)) for _ in range(50000)])
        summary_rows.append({"route": "rat_Model16D_to_Awinda", "target": "ATP0.1_and_ATP1", "budget": "1_to_3_equal_weight",
                             "method": "pooled_normalized_minus_absolute_equal_mouse",
                             "n_heldout_clusters": int(len(per_mouse)),
                             "paired_delta_nrmse_normalized_minus_absolute_pp": float(np.mean(per_mouse)),
                             "paired_delta_bootstrap_95_low": float(np.quantile(pooled_boot, .025)),
                             "paired_delta_bootstrap_95_high": float(np.quantile(pooled_boot, .975)),
                             "n_target_budget_cells_normalized_better": int((delta.mean(axis=0) < 0).sum()),
                             "n_target_budget_cells_normalized_worse": int((delta.mean(axis=0) > 0).sum()),
                             "n_mice_with_normalized_mean_better": int((per_mouse < 0).sum()),
                             "n_mice_with_normalized_mean_worse": int((per_mouse > 0).sum())})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "W_TRANSFER_SUMMARY.csv", index=False, encoding="utf-8-sig")
    freeze_manifest["scoring_status"] = "complete"
    freeze_manifest["scoring_time_utc"] = datetime.now(timezone.utc).isoformat()
    freeze_manifest["heldout_metrics_sha256"] = sha(OUT / "W_HELDOUT_TRANSFER_METRICS.csv")
    freeze_manifest["summary_sha256"] = sha(OUT / "W_TRANSFER_SUMMARY.csv")
    (OUT / "W_DESTINATION_PANEL_FREEZE_MANIFEST.json").write_text(json.dumps(freeze_manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
