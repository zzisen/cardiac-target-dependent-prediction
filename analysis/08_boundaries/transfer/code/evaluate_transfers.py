"""Hash-check frozen panels and replay their held-out destination metrics."""
from __future__ import annotations

import hashlib
import json
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
S_DIR = HERE.parent
PANEL_DIR = S_DIR / "frozen_panels"
TRES = REPO / "analysis/08_boundaries/dynamic/results"
OUT = S_DIR / "results"
OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def nrmse(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return 100.0 * float(np.sqrt(np.mean((p - y) ** 2))) / float(np.sqrt(np.mean(y ** 2)))


def mean_by_cluster(frame):
    return frame.groupby("cluster_id", as_index=False).whole_spectrum_nrmse_pct.mean()


def main():
    register = pd.read_csv(S_DIR / "S_PANEL_FREEZE_REGISTER.csv")
    preds = pd.read_csv(TRES / "T_DYNAMIC_HELDOUT_PREDICTIONS.csv")
    scoring_time = datetime.now(timezone.utc).isoformat()
    # This marker is written once at the start of destination scoring, after all source files exist.
    marker = S_DIR / "S_DESTINATION_SCORING_STARTED_UTC.txt"
    if not marker.exists():
        marker.write_text(scoring_time + "\n", encoding="utf-8")
    scoring_time = marker.read_text(encoding="utf-8").strip()

    hash_checks = []
    for _, row in register.iterrows():
        path = PANEL_DIR / str(row.filename)
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual = sha256(path)
        frozen = pd.Timestamp(payload["frozen_at_utc"])
        scored = pd.Timestamp(scoring_time)
        hash_checks.append({"filename": row.filename, "registered_sha256": row.sha256,
                            "current_sha256": actual, "hash_match": actual == row.sha256,
                            "frozen_before_destination_scoring": frozen < scored,
                            "frozen_at_utc": payload["frozen_at_utc"], "destination_scoring_started_utc": scoring_time})

    # Independent replay of every stored held-out prediction's headline NRMSE.
    max_metric_diff = 0.0
    replay_rows = []
    for _, row in preds.iterrows():
        y = json.loads(row.y_true_json); p = json.loads(row.y_pred_json)
        calculated = nrmse(y, p)
        diff = abs(calculated - float(row.whole_spectrum_nrmse_pct))
        max_metric_diff = max(max_metric_diff, diff)
        replay_rows.append({"system": row.system, "target": row.target, "method": row.method,
                            "budget": row.budget, "unit_id": row.unit_id,
                            "stored_nrmse_pct": float(row.whole_spectrum_nrmse_pct),
                            "replayed_nrmse_pct": calculated, "abs_diff_pp": diff,
                            "no_cluster_leakage": row.test_cluster not in str(row.train_clusters).split(";")})
    pd.DataFrame(replay_rows).to_csv(OUT / "S_DESTINATION_METRIC_REPLAY.csv", index=False, encoding="utf-8-sig")

    routes = [
        ("rat_R_robust", "Human", "non-diabetic", "rat→human non-DM"),
        ("rat_R_robust", "Human", "diabetic", "rat→human T2D"),
        ("rat_R_robust", "Awinda", None, "rat→Awinda"),
        ("human_ND_source", "Human", "diabetic", "human non-DM→T2D"),
    ]
    transfer_rows = []
    for source, system, group, route in routes:
        for target in ("ATP0.1", "ATP1"):
            for budget in (1, 2, 3):
                freeze_row = register[(register.source.eq(source)) & register.target.eq(target) & register.budget.eq(budget)].iloc[0]
                frozen = preds[(preds.system.eq(system)) & preds.target.eq(target) & preds.method.eq(source) & preds.budget.eq(budget)]
                oracle = preds[(preds.system.eq(system)) & preds.target.eq(target) & preds.method.eq("destination_nested_oracle") & preds.budget.eq(budget)]
                baseline = preds[(preds.system.eq(system)) & preds.target.eq(target) & preds.method.eq("baseline_only")]
                if group is not None:
                    frozen = frozen[frozen.group.eq(group)]; oracle = oracle[oracle.group.eq(group)]; baseline = baseline[baseline.group.eq(group)]
                fm = mean_by_cluster(frozen); om = mean_by_cluster(oracle); bm = mean_by_cluster(baseline)
                joined = fm.merge(om, on="cluster_id", suffixes=("_frozen", "_oracle")).merge(bm, on="cluster_id")
                joined = joined.rename(columns={"whole_spectrum_nrmse_pct": "baseline_nrmse_pct"})
                fmean = float(joined.whole_spectrum_nrmse_pct_frozen.mean())
                omean = float(joined.whole_spectrum_nrmse_pct_oracle.mean())
                bmean = float(joined.baseline_nrmse_pct.mean())
                denom = bmean - omean
                utility = bmean - fmean
                payload = json.loads((PANEL_DIR / str(freeze_row.filename)).read_text(encoding="utf-8"))
                transfer_rows.append({"transfer_route": route, "source_panel_family": source,
                                      "destination_system": system, "destination_group": group or "all",
                                      "target": target, "budget_frequency_pairs": budget,
                                      "frozen_panel_id": payload["panel_id"], "frozen_frequencies_hz": ";".join(map(str, payload["nominal_frequency_hz"])),
                                      "frozen_panel_sha256": freeze_row.sha256,
                                      "frozen_before_destination_scoring": bool(freeze_row.frozen_at_utc < scoring_time),
                                      "n_heldout_clusters": int(joined.cluster_id.nunique()),
                                      "frozen_panel_mean_nrmse_pct": fmean,
                                      "destination_nested_oracle_mean_nrmse_pct": omean,
                                      "baseline_only_mean_nrmse_pct": bmean,
                                      "transfer_regret_nrmse_pp": fmean - omean,
                                      "empirical_utility_vs_baseline_nrmse_pp": utility,
                                      "destination_oracle_utility_vs_baseline_nrmse_pp": denom,
                                      "fraction_destination_oracle_utility_retained": utility / denom if denom > 1e-12 else np.nan,
                                      "oracle_is_nested_source_only_selection": True})
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(OUT / "S_TRANSFER_RESULTS.csv", index=False, encoding="utf-8-sig")
    transfer.to_csv(OUT / "S_DESTINATION_ORACLE_COMPARISON.csv", index=False, encoding="utf-8-sig")
    audit = {"status": "PASS" if all(r["hash_match"] and r["frozen_before_destination_scoring"] for r in hash_checks)
             and max_metric_diff < 1e-10 and all(x["no_cluster_leakage"] for x in replay_rows) else "FAIL",
             "freeze_checks": hash_checks, "n_prediction_rows_replayed": len(replay_rows),
             "max_abs_nrmse_replay_difference_pp": max_metric_diff,
             "no_cluster_leakage_in_prediction_splits": all(x["no_cluster_leakage"] for x in replay_rows),
             "destination_scoring_started_utc": scoring_time,
             "oracle_note": "Destination oracle is selected within each outer training fold by nested leave-cluster-out CV; transfer regret is relative to that nested, target-specific oracle."}
    (OUT / "S_TRANSFER_REPLAY_VERIFICATION.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(transfer[["transfer_route", "target", "budget_frequency_pairs", "frozen_panel_mean_nrmse_pct",
                    "destination_nested_oracle_mean_nrmse_pct", "fraction_destination_oracle_utility_retained"]].to_string(index=False))
    print("verification", audit["status"], "max metric replay delta", max_metric_diff)


if __name__ == "__main__":
    main()
