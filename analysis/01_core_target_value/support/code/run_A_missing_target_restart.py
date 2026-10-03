"""Closure-only full-target restart for the two previously omitted targets."""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

HERE = Path(__file__).resolve().parent
BRANCH = HERE.parents[1]
CLOSURE_ROOT = REPO / "analysis/01_core_target_value"
sys.path.insert(0, str(HERE))
from aj_core import (OUT, PublicLikelihood, load_public_rat, load_seeds, domain_for,
                     candidate_definitions, save_json, sha256)
from aj_core_v2 import support_endpoint_v2

SOURCE = OUT / "A_measurement_target_map.json"
EXISTING = OUT / "A_full_target_rank_stability_v2_existing_ATP1_Pi5.json"
RESULT = OUT / "A_missing_target_restart.json"
PROGRESS = OUT / "A_missing_target_restart_progress.json"
TARGETS = ["ATP0.1", "Pi0"]
MARGINS = [0.5, 1.0]


def eligible_ids(target: str) -> list[str]:
    return [x["id"] for x in candidate_definitions() if x["condition"] != target]


def main() -> None:
    original = json.loads(SOURCE.read_text(encoding="utf-8"))
    data = load_public_rat()
    seeds = load_seeds()
    if data.raw_sha256 != original["input_sha256"]:
        raise RuntimeError("A input hash mismatch")
    candidates = {x["id"]: x for x in candidate_definitions()}
    out = {"source_result_sha256": sha256(SOURCE), "input_sha256": data.raw_sha256,
           "scope": "Missing targets ATP0.1 and Pi0; all six actions eligible after same-condition exclusion; both frozen pair-domain margins; restart-safe endpoints with archived endpoint incumbents and six-start cap.",
           "max_starts": 6, "maxiter": 1300, "domains": {}, "endpoint_searches": [],
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.time()
    for margin in MARGINS:
        domain_name = f"P1_pair_plus_{margin:.1f}"
        old_domain = original["domains"][domain_name]
        domain = domain_for(margin, seeds)
        baseline_like = PublicLikelihood(data)
        record = {"margin": margin, "baseline": {}, "actions": {}}
        for target in TARGETS:
            old = old_domain["baseline_support"][target]
            ends = {}
            for side in ("low", "high"):
                opposite = "high" if side == "low" else "low"
                end = support_endpoint_v2(baseline_like, domain, old_domain["baseline_fit"], target,
                    side, seeds, warm=[np.asarray(old[side]["params"]), np.asarray(old[opposite]["params"])],
                    max_starts=6)
                ends[side] = end
                out["endpoint_searches"].append({"domain": domain_name, "measurement": "baseline_only",
                    "target": target, "side": side, "old_q": old[side]["q"], "new_q": end["q"],
                    "abs_q_shift": abs(end["q"] - old[side]["q"]), "old_objective": old[side]["objective"],
                    "new_objective": end["objective"], "selected_start": end["selected_start"],
                    "optimizer_success": end["optimizer_success"], "n_starts": len(end["all_attempts"]),
                    "n_feasible_starts": sum(bool(x.get("feasible")) for x in end["all_attempts"]),
                    "attempts": end["all_attempts"]})
                out["_partial"] = {"n_searches": len(out["endpoint_searches"]),
                                   "elapsed_seconds": time.time() - started}
                save_json(PROGRESS, out)
            record["baseline"][target] = {"old_width": old["high"]["q"] - old["low"]["q"],
                "new_width": ends["high"]["q"] - ends["low"]["q"], "low": ends["low"], "high": ends["high"]}
        for target in TARGETS:
            for candidate_id in eligible_ids(target):
                candidate = candidates[candidate_id]
                like = PublicLikelihood(data, candidate)
                old_measurement = old_domain["measurements"][candidate_id]
                old_support = old_measurement["support"][target]
                ends = {}
                for side in ("low", "high"):
                    opposite = "high" if side == "low" else "low"
                    baseline_endpoint = old_domain["baseline_support"][target][side]
                    warm = [np.asarray(old_support[side]["params"]),
                            np.asarray(baseline_endpoint["params"]),
                            np.asarray(old_support[opposite]["params"])]
                    end = support_endpoint_v2(like, domain, old_measurement["fit"], target, side,
                                              seeds, warm=warm, max_starts=6)
                    ends[side] = end
                    out["endpoint_searches"].append({"domain": domain_name, "measurement": candidate_id,
                        "target": target, "side": side, "old_q": old_support[side]["q"], "new_q": end["q"],
                        "abs_q_shift": abs(end["q"] - old_support[side]["q"]),
                        "old_objective": old_support[side]["objective"], "new_objective": end["objective"],
                        "selected_start": end["selected_start"], "optimizer_success": end["optimizer_success"],
                        "n_starts": len(end["all_attempts"]),
                        "n_feasible_starts": sum(bool(x.get("feasible")) for x in end["all_attempts"]),
                        "attempts": end["all_attempts"]})
                    out["_partial"] = {"n_searches": len(out["endpoint_searches"]),
                                       "elapsed_seconds": time.time() - started}
                    save_json(PROGRESS, out)
                baseline_width = record["baseline"][target]["new_width"]
                width = ends["high"]["q"] - ends["low"]["q"]
                record["actions"].setdefault(candidate_id, {})
                record["actions"][candidate_id][target] = {
                    "old_relative_reduction_pct": old_support["relative_width_reduction_pct"],
                    "new_relative_reduction_pct": 100.0 * (baseline_width - width) / baseline_width,
                    "old_width": old_support["high"]["q"] - old_support["low"]["q"],
                    "new_width": width, "baseline_old_width": old_support["baseline_width_q_pct_points"],
                    "baseline_new_width": baseline_width, "low": ends["low"], "high": ends["high"]}
        out["domains"][domain_name] = record
        out["_partial"] = {"n_searches": len(out["endpoint_searches"]),
                           "elapsed_seconds": time.time() - started}
        save_json(PROGRESS, out)
    rankings = []
    for domain_name, domain_record in out["domains"].items():
        for target in TARGETS:
            ordered = sorted(({"measurement": cid,
                "relative_width_reduction_pct": item[target]["new_relative_reduction_pct"]}
                for cid, item in domain_record["actions"].items() if target in item),
                key=lambda x: x["relative_width_reduction_pct"], reverse=True)
            for rank, item in enumerate(ordered, 1):
                item.update({"rank": rank, "domain": domain_name, "target": target})
                rankings.append(item)
    out["equal_budget_rankings"] = rankings
    out["optimizer_summary"] = {"endpoint_searches": len(out["endpoint_searches"]),
        "total_optimizer_attempts": sum(len(x.get("attempts", [])) for x in out["endpoint_searches"]),
        "non_successful_attempts": sum(1 for x in out["endpoint_searches"]
            for a in x.get("attempts", []) if not a.get("success", False)),
        "max_abs_endpoint_q_shift": max(x["abs_q_shift"] for x in out["endpoint_searches"])}
    max_excess = -float("inf")
    for item in out["endpoint_searches"]:
        source_domain = original["domains"][item["domain"]]
        mle_obj = (source_domain["baseline_fit"]["objective"] if item["measurement"] == "baseline_only"
                   else source_domain["measurements"][item["measurement"]]["fit"]["objective"])
        max_excess = max(max_excess, item["new_objective"] - (mle_obj + 3.841458820694124))
    out["optimizer_summary"]["max_lr_excess_vs_source_mle"] = max_excess
    out["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["elapsed_seconds"] = time.time() - started
    out.pop("_partial", None)
    save_json(RESULT, out)
    merge_full_matrix(original, out, json.loads(EXISTING.read_text(encoding="utf-8")))
    print(json.dumps({k: v for k, v in out.items() if k not in ("endpoint_searches", "domains", "equal_budget_rankings")}, indent=2))
    print("FULL_MATRIX_PATH=" + str(CLOSURE_ROOT / "A_FULL_RESTART_AUDIT.csv"))


def merge_full_matrix(original: dict, missing: dict, previous: dict) -> None:
    source_rankings = list(csv.DictReader((OUT / "A_rankings.csv").open(encoding="utf-8-sig", newline="")))
    rankings = {(x["domain"], x["context"], x["measurement_id"]): int(float(x["rank"]))
                for x in source_rankings if x["ranking_axis"] == "measurement_within_target"}
    rows = []
    domains = ["P1_pair_plus_0.5", "P1_pair_plus_1.0"]
    for domain in domains:
        prior = previous["domains"][domain]
        fresh = missing["domains"][domain]
        for target in ("ATP0.1", "ATP1", "Pi0", "Pi5"):
            baseline = fresh["baseline"][target] if target in fresh["baseline"] else prior["baseline"][target]
            actions = fresh["actions"] if target in ("ATP0.1", "Pi0") else prior["actions"]
            eligible = eligible_ids(target)
            ranked = sorted(((cid, actions[cid][target]["new_relative_reduction_pct"]) for cid in eligible),
                            key=lambda x: x[1], reverse=True)
            new_rank = {cid: rank for rank, (cid, _) in enumerate(ranked, 1)}
            rows.append({"domain": domain, "target": target, "measurement_id": "baseline_only",
                "status": "baseline_reference", "old_rank": "", "new_rank": "", "rank_changed": "",
                "old_width_q_pct_points": baseline["old_width"],
                "new_width_q_pct_points": baseline["new_width"], "old_relative_reduction_pct": 0,
                "new_relative_reduction_pct": 0, "n_scalar_observations": 0,
                "low_q": baseline["low"]["q"], "high_q": baseline["high"]["q"],
                "max_abs_endpoint_q_shift": max(abs(baseline[side]["q"] - original["domains"][domain]["baseline_support"][target][side]["q"]) for side in ("low", "high"))})
            for cid in eligible:
                item = actions[cid][target]
                old_rank = rankings.get((domain, target, cid), "")
                ss = 1 if cid.startswith("stress_") else 34
                rows.append({"domain": domain, "target": target, "measurement_id": cid, "status": "computed",
                    "old_rank": old_rank, "new_rank": new_rank[cid],
                    "rank_changed": bool(old_rank and int(old_rank) != new_rank[cid]),
                    "old_width_q_pct_points": item["old_width"], "new_width_q_pct_points": item["new_width"],
                    "old_relative_reduction_pct": item["old_relative_reduction_pct"],
                    "new_relative_reduction_pct": item["new_relative_reduction_pct"],
                    "n_scalar_observations": ss, "low_q": item["low"]["q"], "high_q": item["high"]["q"],
                    "max_abs_endpoint_q_shift": max(abs(item[side]["q"] - original["domains"][domain]["measurements"][cid]["support"][target][side]["q"]) for side in ("low", "high"))})
    out_path = CLOSURE_ROOT / "A_FULL_RESTART_AUDIT.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Pairwise ranking comparison over exactly the six eligible actions for each target.
    reversals = 0; comparable_pairs = 0
    for domain in domains:
        for target in ("ATP0.1", "ATP1", "Pi0", "Pi5"):
            actions = [x for x in rows if x["domain"] == domain and x["target"] == target and x["status"] == "computed"]
            for i, left in enumerate(actions):
                for right in actions[i + 1:]:
                    olddiff = int(left["old_rank"]) - int(right["old_rank"])
                    newdiff = int(left["new_rank"]) - int(right["new_rank"])
                    comparable_pairs += 1
                    reversals += olddiff * newdiff < 0
    all_shifts = [float(x["max_abs_endpoint_q_shift"]) for x in rows]
    report = ["# A full-target restart closure audit", "",
        "The restart completes the two missing targets (ATP0.1, Pi0) using the frozen 14-parameter model, likelihood, domains, deterministic seed bank, same-condition exclusions, and restart-safe endpoint routine. ATP1/Pi5 rows reuse the already completed closure restart output. Values are descriptive optimizer/ranking checks; the LR-based support is conditional on the model and diagonal SEM likelihood.", "",
        f"- Coverage: {len(rows)} rows = 2 pair-domain margins × 4 targets × (baseline reference + 6 eligible actions).",
        f"- Pairwise rank reversals versus source ranks: {reversals}/{comparable_pairs} eligible action pairs.",
        f"- Maximum absolute endpoint shift versus source: {max(all_shifts):.6g} q percentage points.",
        f"- Missing-target restart endpoint searches: {missing['optimizer_summary']['endpoint_searches']}; optimizer attempts: {missing['optimizer_summary']['total_optimizer_attempts']}; nonsuccess attempts: {missing['optimizer_summary']['non_successful_attempts']}.",
        f"- Missing-target maximum LR cutoff excess: {missing['optimizer_summary']['max_lr_excess_vs_source_mle']:.3g} objective units.",
        "- Read `A_FULL_RESTART_AUDIT.csv` for old/new ranks, widths, endpoint values, scalar burden, and per-cell shifts.",
        "- Caveat: endpoint maxima/minima are bounded multistart optimizer solutions. Solver non-success flags are retained in the detailed JSON; feasibility is checked independently in the closure verifier.", ""]
    (CLOSURE_ROOT / "A_FULL_RESTART_REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
