"""Reproduce the frozen RLC-1 nested leave-one-rat-out analysis."""
from __future__ import annotations

from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))

from collections import Counter
import csv
import hashlib
import itertools
import math
import zipfile

import numpy as np
import openpyxl

BASE = Path(__file__).resolve().parent.parent / "results"
SOURCE = BASE / "RLC1_SOURCE_DATA"
ARCHIVE = SOURCE / "24354289.zip"
WORKBOOK = SOURCE / "RLC-1_Fig6.xlsx"
ARCHIVE_SHA = "bfdc0f36cd0369ad9392d20fa5b4c9c55fc7b8037cc0d8b206f41972cf40a816"
WORKBOOK_SHA = "77e1b8b3a59e90138e5bc9335706b61a33aab1103804ad9a882058bad56e857e"
DOSES = ["ND", "1uM", "3uM", "10uM"]
FAMILIES = ["PeakTension", "TTP", "RT50"]
SHEETS = {"PeakTension": "Fig6B_TPeak", "TTP": "Fig6D_TTP", "RT50": "Fig6E_RT50"}
LABELS = {"PeakTension": "Peak twitch tension", "TTP": "Time to peak tension", "RT50": "Time to 50% relaxation"}
UNITS = {"PeakTension": "relative source units (ND=100)", "TTP": "ms (interpreted from Fig6A time axis)", "RT50": "ms (interpreted from Fig6A time axis)"}
CANDIDATES = ["1uM_PeakTension", "1uM_TTP", "1uM_RT50"]
TARGETS = [f"{dose}_{family}" for dose in ("3uM", "10uM") for family in FAMILIES]
MODELS = ["C0", "C1", "C2", "C3"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(name: str, rows: list[dict]) -> None:
    path = BASE / name
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(v) for k, v in row.items()})


def fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, (float, np.floating)):
        return format(float(v), ".12g") if math.isfinite(float(v)) else ""
    if isinstance(v, (int, np.integer)):
        return int(v)
    return v


def parts(target: str):
    return target.split("_", 1)


def sd(x: np.ndarray) -> float:
    out = float(np.std(x, ddof=1))
    if not math.isfinite(out) or out == 0:
        raise ValueError("A training-fold SD is zero or non-finite")
    return out


def fit_predict(train: list[int], test: int, candidate: str, target: str,
                xdata: dict[str, np.ndarray], ydata: dict[str, np.ndarray]) -> dict:
    x, y = xdata[candidate], ydata[target]
    xm, xs = float(np.mean(x[train])), sd(x[train])
    ym, ys = float(np.mean(y[train])), sd(y[train])
    xz, yz = (x[train] - xm) / xs, (y[train] - ym) / ys
    design = np.column_stack((np.ones(len(train)), xz))
    b0, b1 = np.linalg.lstsq(design, yz, rcond=None)[0]
    pred_z = float(b0 + b1 * ((float(x[test]) - xm) / xs))
    actual_z = float((float(y[test]) - ym) / ys)
    pred_native = float(ym + ys * pred_z)
    return {
        "pred_z": pred_z, "actual_z": actual_z,
        "pred_native": pred_native, "actual_native": float(y[test]),
        "y_mean": ym, "y_sd": ys, "x_mean": xm, "x_sd": xs,
        "intercept_z": float(b0), "slope_z": float(b1),
        "ae_std": abs(actual_z - pred_z),
        "ae_native": abs(float(y[test]) - pred_native),
    }


def inner_scores(train: list[int], target: str,
                 xdata: dict[str, np.ndarray], ydata: dict[str, np.ndarray]) -> dict[str, float]:
    errors = {c: [] for c in CANDIDATES}
    for hold in train:
        fit_rows = [i for i in train if i != hold]
        for candidate in CANDIDATES:
            pred = fit_predict(fit_rows, hold, candidate, target, xdata, ydata)
            errors[candidate].append(pred["ae_std"])
    return {c: float(np.mean(errors[c])) for c in CANDIDATES}


def winner(scores: dict[str, float]) -> str:
    return min(CANDIDATES, key=lambda c: (scores[c], CANDIDATES.index(c)))


def main() -> None:
    canonical_input = REPO / "analysis/05_rlc1/input/RLC1_CANONICAL_DATA.csv"
    with canonical_input.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    xdata, ydata, mats = {}, {}, {}
    for family in FAMILIES:
        matrix = []
        for row in range(2, 9):
            matrix.append([float(next(r["value"] for r in records if r["family_key"] == family and int(r["source_excel_row"]) == row and r["dose"] == dose)) for dose in DOSES])
        mats[family] = matrix
        xdata[f"1uM_{family}"] = np.asarray([r[1] for r in matrix], float)
        for dose, col in (("3uM", 2), ("10uM", 3)):
            ydata[f"{dose}_{family}"] = np.asarray([r[col] for r in matrix], float)
    # The frozen parser verified that RT90 exactly duplicated RT50; only the retained families enter fitting.

    # Canonical long table keeps source coordinates and values; rat IDs are inferred only from row order.
    canonical = []
    for family, sheet in SHEETS.items():
        for i, row in enumerate(mats[family], start=1):
            for j, dose in enumerate(DOSES):
                canonical.append({
                    "source_archive": "24354289.zip", "source_workbook": "RLC-1_Fig6.xlsx",
                    "source_sheet": sheet, "source_excel_row": i + 1,
                    "rat_id_row_order_inferred": f"Rat_{i:02d}",
                    "outcome_family": LABELS[family], "family_key": family,
                    "dose": dose, "value": row[j], "unit_note": UNITS[family],
                    "pairing_basis": "matching 7-row order across the three 7x4 tables; no explicit ID column",
                })
    write_csv("RLC1_CANONICAL_DATA.csv", canonical)

    all_idx = list(range(7))
    # C3 selection deliberately sees all seven subjects; it is retrospective only.
    oracle_scores = {t: inner_scores(all_idx, t, xdata, ydata) for t in TARGETS}
    oracle = {t: winner(oracle_scores[t]) for t in TARGETS}
    predictions, selections = [], []
    scores_by_fold, c2_by_fold, outer_by_candidate = {}, {}, {}

    for held in all_idx:
        train = [i for i in all_idx if i != held]
        fold_scores = {t: inner_scores(train, t, xdata, ydata) for t in TARGETS}
        scores_by_fold[held] = fold_scores
        c1_scores = {c: float(np.mean([fold_scores[t][c] for t in TARGETS])) for c in CANDIDATES}
        c1 = winner(c1_scores)
        c2 = {t: winner(fold_scores[t]) for t in TARGETS}
        c2_by_fold[held] = c2
        outer_by_candidate[held] = {}

        for target in TARGETS:
            outer_by_candidate[held][target] = {
                c: fit_predict(train, held, c, target, xdata, ydata) for c in CANDIDATES
            }
            ytrain = ydata[target][train]
            ymean, ysd = float(np.mean(ytrain)), sd(ytrain)
            actual = float(ydata[target][held])
            c0 = {"candidate": "NO_ADDED_MEASUREMENT", "pred_z": 0.0,
                  "actual_z": (actual-ymean)/ysd, "pred_native": ymean,
                  "actual_native": actual, "y_mean": ymean, "y_sd": ysd,
                  "x_mean": "", "x_sd": "", "intercept_z": 0.0, "slope_z": 0.0,
                  "ae_std": abs((actual-ymean)/ysd), "ae_native": abs(actual-ymean)}
            chosen = {"C0": c0, "C1": {**outer_by_candidate[held][target][c1], "candidate": c1},
                      "C2": {**outer_by_candidate[held][target][c2[target]], "candidate": c2[target]},
                      "C3": {**outer_by_candidate[held][target][oracle[target]], "candidate": oracle[target]}}
            family = parts(target)[1]
            for model in MODELS:
                fit = chosen[model]
                predictions.append({
                    "heldout_rat": f"Rat_{held+1:02d}", "heldout_row_index": held+1,
                    "target": target, "target_dose": parts(target)[0], "target_family": family,
                    "comparator": model, "candidate": fit["candidate"],
                    "candidate_1uM_value": (float(xdata[fit["candidate"]][held]) if model != "C0" else ""),
                    "target_actual_native": fit["actual_native"], "target_predicted_native": fit["pred_native"],
                    "outer_training_target_mean": fit["y_mean"], "outer_training_target_sd": fit["y_sd"],
                    "target_actual_standardized": fit["actual_z"], "target_prediction_standardized": fit["pred_z"],
                    "absolute_error_standardized": fit["ae_std"], "absolute_error_native": fit["ae_native"],
                    "outer_training_n": len(train), "C1_global_candidate_this_fold": c1,
                    "C2_candidate_for_target_this_fold": c2[target],
                    "C3_all_data_oracle_candidate": oracle[target],
                    "C3_retrospective_only": "yes" if model == "C3" else "no",
                })
            for candidate in CANDIDATES:
                selections.append({
                    "heldout_rat": f"Rat_{held+1:02d}", "target": target, "candidate": candidate,
                    "inner_cv_standardized_mae_for_target": fold_scores[target][candidate],
                    "C1_mean_inner_mae_across_6_targets": c1_scores[candidate],
                    "C1_selected_global_this_fold": int(candidate == c1),
                    "C2_selected_for_this_target": int(candidate == c2[target]),
                    "C3_full_data_oracle_selected_for_this_target": int(candidate == oracle[target]),
                })

    pred_lookup = {(r["heldout_rat"], r["target"], r["comparator"]): r for r in predictions}
    for r in predictions:
        key = (r["heldout_rat"], r["target"])
        e0 = pred_lookup[(key[0], key[1], "C0")]["absolute_error_standardized"]
        e1 = pred_lookup[(key[0], key[1], "C1")]["absolute_error_standardized"]
        e2 = pred_lookup[(key[0], key[1], "C2")]["absolute_error_standardized"]
        r["improved_vs_C0"] = int(r["comparator"] != "C0" and r["absolute_error_standardized"] < e0)
        r["C2_beats_C1_on_this_cell"] = int(e2 < e1) if r["comparator"] == "C2" else ""
    write_csv("RLC1_OUTER_LOO_PREDICTIONS.csv", predictions)
    write_csv("RLC1_CANDIDATE_SELECTION.csv", selections)

    # Comparator summaries: overall, per target (including native units), and per held-out rat.
    summary = []
    for model in MODELS:
        model_rows = [r for r in predictions if r["comparator"] == model]
        groups = [("overall", "ALL_42_RAT_TARGET_CELLS", model_rows)]
        groups += [("target", t, [r for r in model_rows if r["target"] == t]) for t in TARGETS]
        groups += [("heldout_rat", f"Rat_{i:02d}", [r for r in model_rows if r["heldout_row_index"] == i]) for i in range(1,8)]
        for level, group, rows in groups:
            keys = [(r["heldout_rat"], r["target"]) for r in rows]
            ref0 = [float(pred_lookup[(rat,t,"C0")]["absolute_error_standardized"]) for rat,t in keys]
            c1e = [float(pred_lookup[(rat,t,"C1")]["absolute_error_standardized"]) for rat,t in keys]
            c2e = [float(pred_lookup[(rat,t,"C2")]["absolute_error_standardized"]) for rat,t in keys]
            ae = np.asarray([float(r["absolute_error_standardized"]) for r in rows])
            mae = float(np.mean(ae))
            summary.append({
                "summary_level": level, "group": group, "comparator": model, "n_predictions": len(rows),
                "mean_standardized_mae": mae, "median_standardized_absolute_error": float(np.median(ae)),
                "native_unit_mae": (float(np.mean([r["absolute_error_native"] for r in rows])) if level == "target" else ""),
                "fraction_improved_vs_C0": (float(np.mean([float(r["absolute_error_standardized"]) < b for r,b in zip(rows,ref0)])) if model != "C0" else 0.0),
                "fraction_C2_beats_C1": float(np.mean([b < a for a,b in zip(c1e,c2e)])),
                "percent_improvement_vs_C0": (100*(float(np.mean(ref0))-mae)/float(np.mean(ref0)) if np.mean(ref0) else ""),
                "C2_percent_improvement_vs_C1": (100*(float(np.mean(c1e))-float(np.mean(c2e)))/float(np.mean(c1e)) if model == "C2" and level == "overall" and np.mean(c1e) else ""),
                "retrospective_only": "yes" if model == "C3" else "no",
            })
    write_csv("RLC1_COMPARATOR_SUMMARY.csv", summary)

    c1_counts = Counter()
    c2_counts = Counter()
    ranking = []
    modal = {}
    for held in all_idx:
        c1_scores = {c: float(np.mean([scores_by_fold[held][t][c] for t in TARGETS])) for c in CANDIDATES}
        c1_counts[winner(c1_scores)] += 1
        for t in TARGETS:
            c2_counts[(t, c2_by_fold[held][t])] += 1
    for target in TARGETS:
        avg = {c: float(np.mean([scores_by_fold[i][target][c] for i in all_idx])) for c in CANDIDATES}
        order = sorted(CANDIDATES, key=lambda c: (avg[c], CANDIDATES.index(c)))
        counts = {c: c2_counts[(target,c)] for c in CANDIDATES}
        chosen = min(CANDIDATES, key=lambda c: (-counts[c], CANDIDATES.index(c)))
        modal[target] = chosen
        for rank, candidate in enumerate(order, 1):
            ranking.append({
                "target": target, "target_dose": parts(target)[0], "target_family": parts(target)[1],
                "rank": rank, "candidate": candidate,
                "mean_inner_cv_standardized_mae_across_outer_folds": avg[candidate],
                "C2_selected_count_of_7_outer_folds": counts[candidate],
                "C2_selection_frequency": counts[candidate]/7,
                "modal_C2_winner_for_target": int(candidate == chosen),
            })
    write_csv("RLC1_TARGET_RANKING.csv", ranking)
    distinct_modal = len(set(modal.values()))

    # Exact 6! mapping-label permutations; fold-specific nested C2 assignments are moved across target slots.
    perm_rows = []
    for perm in itertools.permutations(range(6)):
        errs = []
        for held in all_idx:
            for pos, target in enumerate(TARGETS):
                assigned_target = TARGETS[perm[pos]]
                assigned_candidate = c2_by_fold[held][assigned_target]
                errs.append(outer_by_candidate[held][target][assigned_candidate]["ae_std"])
        perm_rows.append((perm, float(np.mean(errs))))
    observed = float(np.mean([r["absolute_error_standardized"] for r in predictions if r["comparator"] == "C2"]))
    n_le = sum(error <= observed + 1e-12 for _,error in perm_rows)
    p_exact = n_le / len(perm_rows)
    write_csv("RLC1_TARGET_LABEL_PERMUTATION.csv", [
        {"permutation_index": i+1, "assigned_target_indices_zero_based": "-".join(map(str,perm)), "mean_standardized_mae": err}
        for i,(perm,err) in enumerate(perm_rows)
    ])

    c1 = {(r["heldout_rat"],r["target"]): float(r["absolute_error_standardized"]) for r in predictions if r["comparator"] == "C1"}
    c2 = {(r["heldout_rat"],r["target"]): float(r["absolute_error_standardized"]) for r in predictions if r["comparator"] == "C2"}
    all_c1, all_c2 = float(np.mean(list(c1.values()))), float(np.mean(list(c2.values())))
    influence = []
    for i in range(1,8):
        rat = f"Rat_{i:02d}"
        c1rat = float(np.mean([v for (r,_),v in c1.items() if r == rat]))
        c2rat = float(np.mean([v for (r,_),v in c2.items() if r == rat]))
        remain = [k for k in c1 if k[0] != rat]
        delta_ex = float(np.mean([c1[k]-c2[k] for k in remain]))
        influence.append((rat,c1rat,c2rat,c1rat-c2rat,delta_ex))
    lines = ["# RLC-1 minimal robustness", "", "## R1 — Held-out-rat influence", "",
             "Each row gives the rat's own six held-out errors and the aggregate C1−C2 difference after removing that rat's six held-out prediction cells. Subjects remain in training data for other outer folds, so this is a compact held-out-block influence check, not a six-rat refit.", "",
             "| Held-out rat | C1 mean std AE | C2 mean std AE | C1−C2 | C1−C2 after deleting this held-out block |", "|---|---:|---:|---:|---:|"]
    for rat,a,b,d,e in influence:
        lines.append(f"| {rat} | {a:.6f} | {b:.6f} | {d:.6f} | {e:.6f} |")
    lines += ["", f"Overall C1={all_c1:.6f}; C2={all_c2:.6f}; C2 improvement={(all_c1-all_c2)/all_c1*100:.2f}%. Delete-block deltas range {min(x[4] for x in influence):.6f} to {max(x[4] for x in influence):.6f}.",
              "", "## R2 — Exact target-label permutation", "",
              f"All 720 target-label permutations were enumerated as specified in `RLC1_PRIMARY_FREEZE.md`. Identity mapping C2 standardized MAE={observed:.6f}; {n_le}/720 mappings had MAE no greater than identity; exact one-sided mapping p={p_exact:.6f}. This permutes learned candidate-to-target assignments; it is not a randomization of rats or drug treatment.",
              "", "## R3 — Candidate winner stability", "", "| Candidate | C1 selections across 7 folds | C2 selections across 42 target-folds |", "|---|---:|---:|"]
    for cand in CANDIDATES:
        lines.append(f"| {cand} | {c1_counts[cand]} | {sum(c2_counts[(t,cand)] for t in TARGETS)} |")
    lines += ["", f"Distinct modal C2 winners across the six targets: {distinct_modal}.", "", "## R4 — Direct independent recomputation", "",
              "`RLC1_independent_recompute.py` independently recomputes headline C0/C1/C2 standardized MAEs from the final prediction CSV; see `RLC1_INDEPENDENT_RECOMPUTE.csv`."]
    (BASE / "RLC1_MINIMAL_ROBUSTNESS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")

    # Write frozen target-wise rankings and headline values to stdout for report preparation.
    overall = {m: float(np.mean([r["absolute_error_standardized"] for r in predictions if r["comparator"] == m])) for m in MODELS}
    frac_c2_c1 = float(np.mean([c2[k] < c1[k] for k in c1]))
    frac_c1_c0 = float(np.mean([pred_lookup[(r["heldout_rat"],r["target"],"C1")]["absolute_error_standardized"] < pred_lookup[(r["heldout_rat"],r["target"],"C0")]["absolute_error_standardized"] for r in predictions if r["comparator"] == "C1"]))
    frac_c2_c0 = float(np.mean([pred_lookup[(r["heldout_rat"],r["target"],"C2")]["absolute_error_standardized"] < pred_lookup[(r["heldout_rat"],r["target"],"C0")]["absolute_error_standardized"] for r in predictions if r["comparator"] == "C2"]))
    print(f"archive_sha256={digest(ARCHIVE)}")
    print(f"workbook_sha256={digest(WORKBOOK)}")
    print(f"N_rat=7; targets=6; prediction cells=42")
    print("mean_standardized_MAE="+"; ".join(f"{m}:{overall[m]:.9f}" for m in MODELS))
    print(f"C2_vs_C1_percent={(overall['C1']-overall['C2'])/overall['C1']*100:.6f}; C2_better_fraction={frac_c2_c1:.6f}")
    print(f"C1_beats_C0_fraction={frac_c1_c0:.6f}; C2_beats_C0_fraction={frac_c2_c0:.6f}")
    print(f"distinct_modal_C2_winners={distinct_modal}; modal_winners="+"; ".join(f"{t}:{modal[t]}({c2_counts[(t,modal[t])]}/7)" for t in TARGETS))
    print(f"permutation={n_le}/720; exact_p={p_exact:.6f}")
    print("C1 selection counts="+str({c:c1_counts[c] for c in CANDIDATES}))
    print("C2 selection counts="+str({c:sum(c2_counts[(t,c)] for t in TARGETS) for c in CANDIDATES}))
    print("delete-block C1-C2 differences="+str({x[0]:round(x[4],6) for x in influence}))


if __name__ == "__main__":
    main()
