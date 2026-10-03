#!/usr/bin/env python
"""Create and hash training-only destination panel mappings before outcome scoring."""
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
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(OLD / "common"))
sys.path.insert(0, str(HERE.parent))
from dynamic_prediction import load_awinda, load_human  # noqa: E402
from freeze_W_source_panels import fit_empirical_c  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cluster_folds(samples):
    clusters = sorted({str(s["cluster_id"]) for s in samples})
    for held in clusters:
        train = [s for s in samples if str(s["cluster_id"]) != held]
        test = [s for s in samples if str(s["cluster_id"]) == held]
        yield held, train, test, clusters


def source_records(source_panels: pd.DataFrame, source_system: str, target: str, budget: int):
    r = source_panels[(source_panels.source_system == source_system) &
                      (source_panels.target == target) & (source_panels.budget == budget)].sort_values("panel_index")
    if len(r) != budget:
        raise ValueError(f"Expected {budget} source coordinates for {source_system}/{target}/B{budget}; found {len(r)}")
    return r


def nearest_panel(freqs, requested):
    grid = np.asarray(freqs, float)
    mapped = []
    for x in requested:
        idx = int(np.argmin(np.abs(np.log(grid) - np.log(float(x)))))
        actual = float(grid[idx])
        mapped.append({"requested_hz": float(x), "mapped_hz": actual,
                       "relative_mapping_error": abs(actual - float(x)) / float(x)})
    return mapped


def main():
    frozen = pd.read_csv(OUT / "W_SOURCE_NORMALIZED_PANELS.csv")
    awinda = load_awinda(REPO / "analysis/03_awinda/frequency/results" / "Q_AWINDA_ANIMAL_CONDITION_AGGREGATES.csv")
    human = load_human(REPO / "analysis/08_boundaries/preparation_support/input" / "B_HUMAN_CONDITION_OBSERVATIONS.csv")
    v = pd.read_csv(REPO / "analysis/08_boundaries/kinetics/results" / "V_TARGET_KINETIC_TIMESCALES.csv")
    aw_sets = [s for s in awinda if s["target"] in ("ATP0.1", "ATP1")]
    hu_sets = [s for s in human if s["target"] in ("ATP0.1", "ATP1")]
    rows = []; clock_rows = []

    # Rat Model16D -> Awinda: c is estimated only from training animals in the target and state.
    for target in ("ATP0.1", "ATP1"):
        dest = [s for s in aw_sets if s["target"] == target]
        for held, train, test, clusters in cluster_folds(dest):
            tr_ids = {str(s["cluster_id"]) for s in train}
            for genotype, drug in sorted({(s["group"].split("|")[0], s["group"].split("|")[1]) for s in test}):
                vt = v[(v.target_ATP_mM.round(2) == (.1 if target == "ATP0.1" else 1.0)) &
                       v.genotype.eq(genotype) & v.drug_state.eq(drug) & v.AnimalID.astype(str).isin(tr_ids)]
                by_animal = vt.groupby("AnimalID").c_Hz.median()
                fallback = False
                if by_animal.nunique() < 2:
                    vt = v[(v.target_ATP_mM.round(2) == (.1 if target == "ATP0.1" else 1.0)) &
                           v.AnimalID.astype(str).isin(tr_ids)]
                    by_animal = vt.groupby("AnimalID").c_Hz.median()
                    fallback = True
                c_dest = float(by_animal.median())
                state_id = f"{genotype}|{drug}"
                clock_rows.append({"route": "rat_Model16D_to_Awinda", "target": target, "heldout_cluster": held,
                                   "destination_state": state_id, "clock_estimate_hz": c_dest,
                                   "n_training_animals_clock": int(by_animal.nunique()), "state_fallback": fallback,
                                   "training_clusters": ";".join(sorted(tr_ids)), "leakage_check": "heldout animal excluded"})
                for budget in (1, 2, 3):
                    src = source_records(frozen, "Rat_Model16D", target, budget)
                    requested = (src.normalized_coordinate_f_over_fchar.to_numpy(float) * c_dest).tolist()
                    mapped = nearest_panel(test[0]["freqs"], requested)
                    for i, m in enumerate(mapped):
                        rows.append({"route": "rat_Model16D_to_Awinda", "target": target, "budget": budget,
                                     "source_system": "Rat_Model16D", "destination_system": "Awinda",
                                     "heldout_cluster": held, "destination_state": state_id,
                                     "panel_index": i + 1, "source_frequency_hz": float(src.iloc[i].source_frequency_hz),
                                     "source_f_char_hz": float(src.iloc[i].source_characteristic_hz),
                                     "source_normalized_coordinate": float(src.iloc[i].normalized_coordinate_f_over_fchar),
                                     "destination_f_char_hz": c_dest, "requested_destination_hz": m["requested_hz"],
                                     "mapped_destination_hz": m["mapped_hz"], "relative_mapping_error": m["relative_mapping_error"],
                                     "mapping_method": "nearest measured frequency in log distance; no interpolation",
                                     "train_only_clock": True, "n_training_animals_clock": int(by_animal.nunique()),
                                     "state_fallback": fallback})

    # Rat Model16D -> Human disease strata and Human ND -> T2D.
    routes = [("Rat_Model16D", "Human", "non-diabetic", "rat_Model16D_to_Human_ND"),
              ("Rat_Model16D", "Human", "diabetic", "rat_Model16D_to_Human_T2D"),
              ("Human_ND", "Human", "diabetic", "human_ND_to_Human_T2D")]
    for source_system, _, dest_group, route in routes:
        for target in ("ATP0.1", "ATP1"):
            dest = [s for s in hu_sets if s["target"] == target and s["group"] == dest_group]
            if not dest:
                continue
            for held, train, test, clusters in cluster_folds(dest):
                train_y = np.mean(np.vstack([s["y"] for s in train]), axis=0)
                fit = fit_empirical_c(test[0]["freqs"], train_y, max_starts=4, max_nfev=800)
                c_dest = fit["c_Hz"]
                train_ids = sorted({str(s["cluster_id"]) for s in train})
                clock_rows.append({"route": route, "target": target, "heldout_cluster": held,
                                   "destination_state": dest_group, "clock_estimate_hz": c_dest,
                                   "n_training_clusters_clock": len(train_ids),
                                   "near_optimal_c_min_hz": fit["near_optimal_c_min_Hz"],
                                   "near_optimal_c_max_hz": fit["near_optimal_c_max_Hz"],
                                   "clock_fit_stable_1pct": fit["clock_fit_stable_1pct"],
                                   "training_clusters": ";".join(train_ids), "leakage_check": "heldout preparation excluded"})
                for budget in (1, 2, 3):
                    src = source_records(frozen, source_system, target, budget)
                    requested = (src.normalized_coordinate_f_over_fchar.to_numpy(float) * c_dest).tolist()
                    mapped = nearest_panel(test[0]["freqs"], requested)
                    for i, m in enumerate(mapped):
                        rows.append({"route": route, "target": target, "budget": budget,
                                     "source_system": source_system, "destination_system": f"Human_{dest_group}",
                                     "heldout_cluster": held, "destination_state": dest_group,
                                     "panel_index": i + 1, "source_frequency_hz": float(src.iloc[i].source_frequency_hz),
                                     "source_f_char_hz": float(src.iloc[i].source_characteristic_hz),
                                     "source_normalized_coordinate": float(src.iloc[i].normalized_coordinate_f_over_fchar),
                                     "destination_f_char_hz": c_dest, "requested_destination_hz": m["requested_hz"],
                                     "mapped_destination_hz": m["mapped_hz"], "relative_mapping_error": m["relative_mapping_error"],
                                     "mapping_method": "nearest measured frequency in log distance; no interpolation",
                                     "train_only_clock": True, "n_training_clusters_clock": len(train_ids),
                                     "clock_fit_stable_1pct": fit["clock_fit_stable_1pct"]})

    panel_path = OUT / "W_DESTINATION_PANEL_MAPS_FROZEN.csv"
    clock_path = OUT / "W_DESTINATION_CLOCKS_TRAIN_ONLY.csv"
    pd.DataFrame(rows).to_csv(panel_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(clock_rows).to_csv(clock_path, index=False, encoding="utf-8-sig")
    manifest = {"freeze_time_utc": datetime.now(timezone.utc).isoformat(), "destination_outcomes_scored_at_freeze": False,
                "source_panel_sha256": sha(OUT / "W_SOURCE_NORMALIZED_PANELS.csv"),
                "destination_panel_maps_sha256": sha(panel_path), "training_only_clocks_sha256": sha(clock_path),
                "mapping_rule": "source f/fchar multiplied by training-only destination fchar, snapped to nearest observed candidate frequency by log-distance; no interpolation",
                "scoring_status": "not started", "n_frozen_frequency_assignments": len(rows),
                "n_train_only_clock_rows": len(clock_rows)}
    (OUT / "W_DESTINATION_PANEL_FREEZE_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
