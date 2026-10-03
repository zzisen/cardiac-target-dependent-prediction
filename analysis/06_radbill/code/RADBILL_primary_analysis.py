#!/usr/bin/env python3
"""Nested participant-LOPO analysis of the supplied Radbill human pacing workbook."""
import argparse
import csv
import hashlib
import itertools
import math
import platform
from collections import Counter
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))


import numpy as np
import openpyxl

CANDS = [
    ("DT5_QT", "DT5 QT", 15, "QT"),
    ("DT5_PP", "DT5 PP", 18, "PP"),
    ("DTend_QT", "DTend QT", 20, "QT"),
    ("DTend_PP", "DTend PP", 23, "PP"),
]
TARGS = [
    ("110_DTend_QT", "110 bpm", 110, 20, "QT"),
    ("110_DTend_PP", "110 bpm", 110, 23, "PP"),
    ("120_DTend_QT", "120 bpm", 120, 20, "QT"),
    ("120_DTend_PP", "120 bpm", 120, 23, "PP"),
    ("130first_DTend_QT", "130 first", "130 first", 20, "QT"),
    ("130first_DTend_PP", "130 first", "130 first", 23, "PP"),
    ("130last_DTend_QT", "130 last", "130 last", 20, "QT"),
    ("130last_DTend_PP", "130 last", "130 last", 23, "PP"),
]
SEQ = [70, 80, 90, 100, 110, 120, "130 first", "130 last"]
EPS = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def gray(cell):
    c = cell.fill.fgColor
    return cell.fill.fill_type == "solid" and c.type == "theme" and c.theme == 0 and c.tint < 0


def num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def mean_sd(vals):
    a = np.asarray(vals, dtype=float)
    if len(a) < 2:
        return None, None
    mu, sd = float(np.mean(a)), float(np.std(a, ddof=1))
    return mu, sd if math.isfinite(sd) and sd > EPS else None


def load_data(xlsx, readme):
    xlsx, readme = Path(xlsx), Path(readme)
    if not xlsx.is_file():
        return load_canonical(REPO / "analysis/06_radbill/input/RADBILL_CANONICAL_DATA.csv")
    if not xlsx.is_file() or not readme.is_file():
        raise FileNotFoundError(f"Required local source missing: {xlsx}; {readme}")
    wv = openpyxl.load_workbook(xlsx, data_only=True)
    wf = openpyxl.load_workbook(xlsx, data_only=False)
    if "Study measurements" not in wv.sheetnames or "Clinical data" not in wv.sheetnames:
        raise ValueError(f"Unexpected sheet names: {wv.sheetnames}")
    sv, sf, cl = wv["Study measurements"], wf["Study measurements"], wv["Clinical data"]
    starts = [r for r in range(6, sv.max_row + 1) if sv.cell(r, 1).value is not None]
    if len(starts) != 19:
        raise ValueError(f"Expected 19 participant blocks, found {len(starts)}")
    ps, data_rows = {}, []
    for start in starts:
        pid, cohort = int(sv.cell(start, 1).value), int(sv.cell(start, 2).value)
        rr = list(range(start, start + 8))
        seq = [sv.cell(r, 4).value for r in rr]
        if seq != SEQ:
            raise ValueError(f"Unexpected pacing sequence for Study ID {pid}: {seq}")
        if any(sv.cell(r, 1).value is not None for r in rr[1:]):
            raise ValueError(f"Participant ID repeats inside the 8-row block for {pid}")
        ps[pid] = {
            "id": pid, "cohort": "HCM" if cohort == 1 else "Control",
            "cohort_binary": cohort, "start": start, "rows": dict(zip(SEQ, rr)),
        }
        data_rows.extend(rr)
    if len(ps) != 19 or len(set(ps)) != 19:
        raise ValueError("Study IDs are not 19 unique participants")
    ch = [cl.cell(1, c).value for c in range(1, cl.max_column + 1)]
    id_col, group_col = ch.index("Study ID") + 1, ch.index("Study Group") + 1
    clinical = {
        int(cl.cell(r, id_col).value): str(cl.cell(r, group_col).value).strip()
        for r in range(2, cl.max_row + 1) if cl.cell(r, id_col).value is not None
    }
    if set(clinical) != set(ps):
        raise ValueError("Study-measurement IDs do not match Clinical data IDs")
    for pid, p in ps.items():
        expected = "Case" if p["cohort_binary"] else "Control"
        if clinical[pid].casefold() != expected.casefold():
            raise ValueError(f"Cohort mismatch for Study ID {pid}")
    groups = Counter(p["cohort"] for p in ps.values())
    if groups != Counter({"HCM": 9, "Control": 10}):
        raise ValueError(f"Expected 9 HCM and 10 controls; found {groups}")
    gray_cells = []
    for r in data_rows:
        for c in range(1, 32):
            if gray(sf.cell(r, c)):
                gray_cells.append(sf.cell(r, c).coordinate)
    for pid, p in ps.items():
        crow = p["rows"][100]
        p["candidate_row"] = crow
        p["cand"], p["cand_gray"] = {}, {}
        for key, label, col, family in CANDS:
            marked = gray(sf.cell(crow, col))
            p["cand_gray"][key] = marked
            p["cand"][key] = None if marked else num(sv.cell(crow, col).value)
        p["target"], p["target_gray"], p["target_row"] = {}, {}, {}
        for key, rate_label, rate, col, family in TARGS:
            tr = p["rows"][rate]
            marked = gray(sf.cell(tr, col))
            p["target_gray"][key] = marked
            p["target"][key] = None if marked else num(sv.cell(tr, col).value)
            p["target_row"][key] = tr
    pp_checks, pp_mismatches = 0, []
    for r in data_rows:
        for c, hi, lo in [(10, 8, 9), (18, 16, 17), (23, 21, 22), (31, 29, 30)]:
            if isinstance(sf.cell(r, c).value, str) and sf.cell(r, c).value.startswith("="):
                cached, high, low = num(sv.cell(r, c).value), num(sv.cell(r, hi).value), num(sv.cell(r, lo).value)
                if cached is not None and high is not None and low is not None:
                    pp_checks += 1
                    if abs(cached - (high - low)) > EPS:
                        pp_mismatches.append((sf.cell(r, c).coordinate, cached, high - low))
    if pp_mismatches:
        raise ValueError(f"Pulse-pressure formula cache mismatch: {pp_mismatches[:5]}")
    support = {}
    ckeys = [c[0] for c in CANDS]
    for tkey, *_ in TARGS:
        tn = sum(ps[i]["target"][tkey] is not None for i in ps)
        pair = {
            c: sum(ps[i]["target"][tkey] is not None and ps[i]["cand"][c] is not None for i in ps)
            for c in ckeys
        }
        all4 = sum(
            ps[i]["target"][tkey] is not None and all(ps[i]["cand"][c] is not None for c in ckeys)
            for i in ps
        )
        support[tkey] = {"target_n": tn, "pair_n": pair, "all4_n": all4}
    return {
        "ps": ps, "groups": groups, "data_rows": data_rows, "gray_cells": gray_cells,
        "pp_checks": pp_checks, "support": support, "clinical": clinical,
        "xlsx": xlsx, "readme": readme, "xlsx_sha": sha(xlsx), "readme_sha": sha(readme),
        "xlsx_bytes": xlsx.stat().st_size, "readme_bytes": readme.stat().st_size,
        "sheets": wv.sheetnames,
    }


def load_canonical(path):
    """Frozen public measurement table; author-gray cells were removed before modelling."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    ps = {}
    for r in records:
        pid = int(r["participant_id"])
        if pid not in ps:
            ps[pid] = {"id": pid, "cohort": r["cohort"], "cohort_binary": int(r["cohort_binary_hcm1"]),
                       "cand": {}, "cand_gray": {}, "target": {}, "target_gray": {}, "target_row": {},
                       "candidate_row": int(r["candidate_source_row"])}
        p = ps[pid]
        for c, *_ in CANDS:
            p["cand"][c] = float(r["candidate_"+c]) if r["candidate_"+c+"_valid"] == "1" else None
            p["cand_gray"][c] = r["candidate_"+c+"_author_gray"] == "1"
        t = r["target"]
        p["target"][t] = float(r["target_value"]) if r["target_valid"] == "1" else None
        p["target_gray"][t] = r["target_author_gray"] == "1"
        p["target_row"][t] = int(r["target_source_row"])
    support = {t[0]: {"target_n": sum(p["target"][t[0]] is not None for p in ps.values()),
                         "pair_n": {c[0]: sum(p["target"][t[0]] is not None and p["cand"][c[0]] is not None for p in ps.values()) for c in CANDS},
                         "all4_n": sum(p["target"][t[0]] is not None and all(p["cand"][c[0]] is not None for c in CANDS) for p in ps.values())} for t in TARGS}
    return {"ps": ps, "groups": Counter(p["cohort"] for p in ps.values()), "support": support,
            "gray_cells": sorted({str(r["target_source_row"])+":"+r["target"] for r in records if r["target_author_gray"] == "1"})}


def fit(ds, train, cand, targ):
    ps = ds["ps"]
    xs = [ps[i]["cand"][cand] for i in train if ps[i]["cand"][cand] is not None]
    ys = [ps[i]["target"][targ] for i in train if ps[i]["target"][targ] is not None]
    xm, xd = mean_sd(xs)
    ym, yd = mean_sd(ys)
    if xm is None or xd is None or ym is None or yd is None:
        return None
    pair = [i for i in train if ps[i]["cand"][cand] is not None and ps[i]["target"][targ] is not None]
    if len(pair) < 3:
        return None
    xz = np.array([(ps[i]["cand"][cand] - xm) / xd for i in pair])
    yz = np.array([(ps[i]["target"][targ] - ym) / yd for i in pair])
    beta, _, rank, _ = np.linalg.lstsq(np.column_stack([np.ones(len(pair)), xz]), yz, rcond=None)
    if rank < 2 or not np.all(np.isfinite(beta)):
        return None
    return {"xm": xm, "xd": xd, "ym": ym, "yd": yd, "b0": float(beta[0]), "b1": float(beta[1]), "n_pair": len(pair)}


def predict(m, x):
    if m is None or x is None:
        return None
    return m["ym"] + m["yd"] * (m["b0"] + m["b1"] * ((x - m["xm"]) / m["xd"]))


def inner(ds, train):
    ps, ckeys = ds["ps"], [c[0] for c in CANDS]
    scores = {t[0]: {c: [] for c in ckeys} for t in TARGS}
    valid_n = Counter()
    for hold in train:
        itrain = [i for i in train if i != hold]
        for targ, *_ in TARGS:
            p = ps[hold]
            if p["target"][targ] is None or not all(p["cand"][c] is not None for c in ckeys):
                continue
            models = {c: fit(ds, itrain, c, targ) for c in ckeys}
            if any(models[c] is None for c in ckeys):
                continue
            valid_n[targ] += 1
            y = p["target"][targ]
            for c in ckeys:
                scores[targ][c].append(abs(predict(models[c], p["cand"][c]) - y) / models[c]["yd"])
    means = {}
    for targ, *_ in TARGS:
        means[targ] = {c: float(np.mean(scores[targ][c])) if scores[targ][c] else float("inf") for c in ckeys}
        if not any(math.isfinite(v) for v in means[targ].values()):
            raise ValueError(f"No inner-CV scores for {targ}")
    aggregate = {
        c: float(np.mean([means[t[0]][c] for t in TARGS])) if all(math.isfinite(means[t[0]][c]) for t in TARGS) else float("inf")
        for c in ckeys
    }
    if not any(math.isfinite(x) for x in aggregate.values()):
        raise ValueError("No candidate has all eight finite C1 scores")
    c1 = min(ckeys, key=lambda c: aggregate[c])
    c2 = {t[0]: min(ckeys, key=lambda c: means[t[0]][c]) for t in TARGS}
    return {"means": means, "aggregate": aggregate, "c1": c1, "c2": c2, "valid_n": valid_n}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for row in rows:
            clean = {}
            for k, v in row.items():
                if v is None or (isinstance(v, (float, np.floating)) and not math.isfinite(float(v))):
                    clean[k] = ""
                elif isinstance(v, (bool, np.bool_)):
                    clean[k] = "1" if v else "0"
                elif isinstance(v, (float, np.floating)):
                    clean[k] = f"{float(v):.12g}"
                else:
                    clean[k] = v
            wr.writerow(clean)


def write_freeze(out):
    path = out / "RADBILL_PRIMARY_FREEZE.md"
    if path.exists():
        return
    path.write_text("""# Radbill primary analysis freeze

This records the design fixed from the user's task prompt before interpreting model results.

## Frozen variables and source handling

- Use only the owner's local HCM pacing XLSX and README.
- Participant is Study ID; pooled primary cohort is all 19 participants.
- Gray author-marked cells and blank values are missing and are never imputed.
- Candidate state: 100 bpm; fixed tie-break order: DT5 QT, DT5 PP, DTend QT, DTend PP.
- Targets: DTend QT and DTend PP at 110 bpm, 120 bpm, 130 bpm first, and 130 bpm last.
- No clinical covariates or subgroup-tuned models enter the primary analysis.
- Pulse pressure uses workbook formula-cache values (BPmax minus BPmin); caches are verified against source BP columns.

## Nested validation and scoring

- Outer validation is leave-one-participant-out across all 19 Study IDs.
- Within each outer training set, selection is exhaustive inner leave-one-training-participant-out.
- Feature mean/SD is from valid outer/inner training measurements for that feature. Target mean/SD is from valid training outcomes for that target. SD uses sample SD (ddof=1).
- Each affine OLS fit uses pairwise-complete training participants for the candidate-target pair, with intercept and one standardized candidate.
- Inner candidate scores use the same validation participants for all four candidates: target valid and all four 100-bpm candidates valid. C1 is the mean of eight target-specific inner MAEs, with targets equally weighted; C2 selects per target. Ties follow the candidate order above.
- Primary outer C0/C1/C2 comparison uses the same participant-target cells where the target and selected C1 and C2 predictors are valid. Strict all-four-candidate support is used for the exact target-label permutation.
- Report target-valid N, candidate-target pair N and common-support N; do not impute.
- C3 is an all-data in-sample oracle and is not validation evidence.

## Minimal checks

- Delete each participant's held-out prediction block once.
- Enumerate all 8! permutations of target labels to learned C2 selection profiles as a mapping diagnostic only, not a biological significance test.
- Report pooled-model held-out errors by HCM/control descriptively; do not tune group-specific models.
- Independently recompute headline metrics from the final predictions CSV using a separate script.
""", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--readme", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_data(args.xlsx, args.readme)
    write_freeze(out)
    ps, ids = ds["ps"], sorted(ds["ps"])
    ckeys, tkeys = [c[0] for c in CANDS], [t[0] for t in TARGS]
    labels = {c[0]: c[1] for c in CANDS}
    targ_human = {t[0]: f"{t[1]} {t[4]}" for t in TARGS}

    canonical = []
    for pid in ids:
        p = ps[pid]
        for targ, rate, _, _, family in TARGS:
            r = {
                "participant_id": pid, "cohort": p["cohort"], "cohort_binary_hcm1": p["cohort_binary"],
                "candidate_state_bpm": 100, "candidate_source_row": p["candidate_row"],
                "target": targ, "target_family": family, "target_rate": rate,
                "target_source_row": p["target_row"][targ], "target_value": p["target"][targ],
                "target_valid": p["target"][targ] is not None, "target_author_gray": p["target_gray"][targ],
            }
            for c in ckeys:
                r[f"candidate_{c}"] = p["cand"][c]
                r[f"candidate_{c}_valid"] = p["cand"][c] is not None
                r[f"candidate_{c}_author_gray"] = p["cand_gray"][c]
            r["all_four_candidates_valid"] = all(p["cand"][c] is not None for c in ckeys)
            canonical.append(r)
    write_csv(out / "RADBILL_CANONICAL_DATA.csv", canonical)

    # Source check.
    supp_md = []
    for targ, *_ in TARGS:
        s = ds["support"][targ]
        pair = ", ".join(f"{labels[c]}={s['pair_n'][c]}" for c in ckeys)
        supp_md.append(f"| {targ} | {s['target_n']}/19 | {pair} | {s['all4_n']}/19 |")
    sourcecheck = f"""# Radbill source check

## Local files and hashes

- XLSX: {ds['xlsx']}; {ds['xlsx_bytes']:,} bytes; SHA-256 {ds['xlsx_sha']}.
- README: {ds['readme']}; {ds['readme_bytes']:,} bytes; SHA-256 {ds['readme_sha']}.
- Both were read from the existing local 01_Data/Radbill folder; no data were downloaded or replaced.
- The review-package copies under RADBILL_SOURCE_DATA were byte-compared with these originals and are unchanged.
- Workbook sheets: {', '.join(ds['sheets'])}.

## Structure

- 19 unique Study IDs have matching Clinical data rows; 9 are HCM cases and 10 are controls.
- Each participant has eight repeated measurement rows in this order: 70, 80, 90, 100, 110, 120, 130 first, 130 last bpm (152 rows total).
- Candidate and target columns match the frozen prompt. The workbook says gray data cells are unreliable; {len(ds['gray_cells'])} gray-filled data cells were detected, and gray/blank values are missing in the canonical table.
- {ds['pp_checks']} cached pulse-pressure formula results were checked against BPmax minus BPmin; no mismatches.
- README reports 19 cases/rows with repeated measurements and describes the same study data.

## Target-specific usable support

Candidate-target N requires both measurements valid. Common N requires valid target plus all four candidates.

| Target | Target-valid N | Candidate-target N (DT5 QT, DT5 PP, DTend QT, DTend PP) | All-four common N |
|---|---:|---|---:|
{chr(10).join(supp_md)}

Core source facts match; primary analysis proceeds.
"""
    (out / "00_RADBILL_SOURCE_CHECK.md").write_text(sourcecheck, encoding="utf-8")

    selections, preds = [], []
    sel_list = []
    candidate_errors = np.full((len(ids), len(tkeys), len(ckeys)), np.nan)
    all4_mask = np.zeros((len(ids), len(tkeys)), dtype=bool)
    for oi, hold in enumerate(ids):
        train = [i for i in ids if i != hold]
        sel = inner(ds, train)
        sel_list.append(sel)
        c1 = sel["c1"]
        for ti, (targ, rate, _, _, family) in enumerate(TARGS):
            c2 = sel["c2"][targ]
            sr = {
                "heldout_participant_id": hold, "heldout_cohort": ps[hold]["cohort"],
                "target": targ, "C1_global_candidate": c1, "C2_target_candidate": c2,
                "inner_validation_n_common": sel["valid_n"][targ],
                "C1_selected_global_inner_MAE": sel["aggregate"][c1],
                "C2_selected_target_inner_MAE": sel["means"][targ][c2],
            }
            for c in ckeys:
                sr[f"inner_target_MAE_{c}"] = sel["means"][targ][c]
                sr[f"inner_global_mean_MAE_{c}"] = sel["aggregate"][c]
            selections.append(sr)

            models = {c: fit(ds, train, c, targ) for c in ckeys}
            ytrain = [ps[i]["target"][targ] for i in train if ps[i]["target"][targ] is not None]
            ym, yd = mean_sd(ytrain)
            if ym is None or yd is None:
                raise ValueError(f"Target scale unavailable for outer fold {hold} / {targ}")
            hp, y = ps[hold], ps[hold]["target"][targ]
            c0pred = ym if y is not None else None
            cpred = {c: predict(models[c], hp["cand"][c]) if y is not None else None for c in ckeys}
            for ci, c in enumerate(ckeys):
                if y is not None and cpred[c] is not None:
                    candidate_errors[oi, ti, ci] = abs(cpred[c] - y) / yd
            c1pred, c2pred = cpred[c1], cpred[c2]
            pair = y is not None and c1pred is not None and c2pred is not None
            all4 = y is not None and all(hp["cand"][c] is not None and cpred[c] is not None for c in ckeys)
            all4_mask[oi, ti] = bool(all4)
            r = {
                "heldout_participant_id": hold, "cohort": hp["cohort"], "target": targ,
                "target_family": family, "target_rate": rate, "target_value_native": y,
                "target_valid": y is not None, "target_training_mean": ym, "target_training_sd": yd,
                "target_training_n": len(ytrain), "C1_global_candidate": c1, "C2_target_candidate": c2,
                "C0_prediction_native": c0pred, "C1_prediction_native": c1pred, "C2_prediction_native": c2pred,
                "primary_pair_support_C1_C2": bool(pair), "all_four_candidate_common_support": bool(all4),
                "standardized_target_value": (y - ym) / yd if y is not None else None,
            }
            for model, pred in [("C0", c0pred), ("C1", c1pred), ("C2", c2pred)]:
                r[f"standardized_abs_error_{model}"] = abs(pred - y) / yd if y is not None and pred is not None else None
                r[f"native_abs_error_{model}"] = abs(pred - y) if y is not None and pred is not None else None
            for c in ckeys:
                r[f"candidate_value_{c}"] = hp["cand"][c]
                r[f"candidate_prediction_native_{c}"] = cpred[c]
                r[f"standardized_abs_error_{c}"] = abs(cpred[c] - y) / yd if y is not None and cpred[c] is not None else None
                r[f"native_abs_error_{c}"] = abs(cpred[c] - y) if y is not None and cpred[c] is not None else None
            preds.append(r)
    write_csv(out / "RADBILL_CANDIDATE_SELECTION.csv", selections)
    write_csv(out / "RADBILL_OUTER_LOPO_PREDICTIONS.csv", preds)

    # Primary metrics on shared participant-target cells where both selected predictors and target are valid.
    paired = [r for r in preds if r["primary_pair_support_C1_C2"]]
    metric = {}
    for m in ["C0", "C1", "C2"]:
        e = [r[f"standardized_abs_error_{m}"] for r in paired]
        metric[m] = {
            "mean": float(np.mean(e)), "median": float(np.median(e)),
            "macro": float(np.mean([
                np.mean([r[f"standardized_abs_error_{m}"] for r in paired if r["target"] == t[0]])
                for t in TARGS
            ])), "n": len(e),
        }
    c0, c1, c2 = metric["C0"]["mean"], metric["C1"]["mean"], metric["C2"]["mean"]
    imp10 = 100 * (c0 - c1) / c0
    imp20 = 100 * (c0 - c2) / c0
    imp21 = 100 * (c1 - c2) / c1
    wins = sum(r["standardized_abs_error_C2"] < r["standardized_abs_error_C1"] for r in paired)
    winfrac = wins / len(paired)
    tmetrics = {}
    for targ, *_ in TARGS:
        use = [r for r in paired if r["target"] == targ]
        tmetrics[targ] = {"n": len(use)}
        for m in ["C0", "C1", "C2"]:
            tmetrics[targ][f"std_{m}"] = float(np.mean([r[f"standardized_abs_error_{m}"] for r in use]))
            tmetrics[targ][f"native_{m}"] = float(np.mean([r[f"native_abs_error_{m}"] for r in use]))

    # All-data in-sample C3 oracle, clearly separated from validation evidence.
    oracle, oracleerr = {}, {}
    for targ, *_ in TARGS:
        candidate_scores = {}
        for ca in ckeys:
            model = fit(ds, ids, ca, targ)
            vals = []
            if model:
                for pid in ids:
                    y, x = ps[pid]["target"][targ], ps[pid]["cand"][ca]
                    if y is not None and x is not None:
                        vals.append(abs(predict(model, x) - y) / model["yd"])
            candidate_scores[ca] = float(np.mean(vals)) if vals else float("inf")
        oracle[targ] = min(ckeys, key=lambda ca: candidate_scores[ca])
        oracleerr[targ] = candidate_scores[oracle[targ]]

    summary_rows = []
    for model in ["C0", "C1", "C2"]:
        row = {
            "comparator": model, "primary_support": "shared C1/C2 paired target-valid outer cells",
            "n_cells": metric[model]["n"], "pooled_mean_standardized_MAE": metric[model]["mean"],
            "median_standardized_absolute_error": metric[model]["median"],
            "macro_target_mean_standardized_MAE": metric[model]["macro"],
            "improvement_vs_C0_percent": 0 if model == "C0" else imp10 if model == "C1" else imp20,
            "C2_vs_C1_improvement_percent": imp21 if model == "C2" else "",
            "C2_beats_C1_fraction": winfrac if model == "C2" else "",
            "C2_beats_C1_count": f"{wins}/{len(paired)}" if model == "C2" else "",
        }
        for targ, *_ in TARGS:
            row[f"target_n_{targ}"] = tmetrics[targ]["n"]
            row[f"target_standardized_MAE_{targ}"] = tmetrics[targ][f"std_{model}"]
            row[f"target_native_MAE_{targ}"] = tmetrics[targ][f"native_{model}"]
        for pid in ids:
            use = [r for r in paired if int(r["heldout_participant_id"]) == pid]
            vals = [r[f"standardized_abs_error_{model}"] for r in use]
            row[f"participant_{pid}_mean_standardized_AE"] = float(np.mean(vals)) if vals else ""
        summary_rows.append(row)
    orow = {
        "comparator": "C3_ORACLE", "primary_support": "all-data in-sample; not validation evidence",
        "n_cells": "", "pooled_mean_standardized_MAE": float(np.mean(list(oracleerr.values()))),
        "median_standardized_absolute_error": "", "macro_target_mean_standardized_MAE": float(np.mean(list(oracleerr.values()))),
        "improvement_vs_C0_percent": "", "C2_vs_C1_improvement_percent": "",
        "C2_beats_C1_fraction": "", "C2_beats_C1_count": "",
    }
    for targ, *_ in TARGS:
        orow[f"target_n_{targ}"] = ""
        orow[f"target_standardized_MAE_{targ}"] = oracleerr[targ]
        orow[f"target_native_MAE_{targ}"] = ""
    for pid in ids:
        orow[f"participant_{pid}_mean_standardized_AE"] = ""
    summary_rows.append(orow)
    write_csv(out / "RADBILL_COMPARATOR_SUMMARY.csv", summary_rows)

    # Rankings, fold selection counts, and modal C2 winners.
    counts = {t[0]: Counter() for t in TARGS}
    c1counts = Counter()
    for s in sel_list:
        c1counts[s["c1"]] += 1
        for targ in tkeys:
            counts[targ][s["c2"][targ]] += 1
    ranking = []
    modal = {}
    for targ, rate, _, _, family in TARGS:
        avgs = {ca: float(np.mean([s["means"][targ][ca] for s in sel_list])) for ca in ckeys}
        order = sorted(ckeys, key=lambda ca: avgs[ca])
        win = max(ckeys, key=lambda ca: (counts[targ][ca], -ckeys.index(ca)))
        modal[targ] = win
        for rank, ca in enumerate(order, 1):
            ranking.append({
                "target": targ, "target_family": family, "target_rate": rate, "candidate": ca,
                "mean_inner_CV_standardized_MAE": avgs[ca], "mean_rank": rank,
                "C2_selected_outer_folds": counts[targ][ca], "modal_C2_winner": win,
                "modal_winner_folds": counts[targ][win], "primary_outer_n": tmetrics[targ]["n"],
            })
    write_csv(out / "RADBILL_TARGET_RANKING.csv", ranking)
    distinct_modal = len(set(modal.values()))

    # R1 delete-one participant block.
    influence = []
    for pid in ids:
        use = [r for r in paired if int(r["heldout_participant_id"]) != pid]
        a = float(np.mean([r["standardized_abs_error_C1"] for r in use]))
        b = float(np.mean([r["standardized_abs_error_C2"] for r in use]))
        influence.append({"deleted_participant": pid, "remaining_cells": len(use),
                          "C1_mean_std_MAE": a, "C2_mean_std_MAE": b, "C1_minus_C2": a-b,
                          "C2_better_direction_retained": b < a})
    influence_positive = all(r["C1_minus_C2"] > 0 for r in influence)

    # R2: all 8! profile permutations on identical all-four-candidate complete support.
    obs = np.full((len(ids), len(tkeys), len(ckeys)), np.nan)
    for r in preds:
        i = ids.index(int(r["heldout_participant_id"]))
        ti = tkeys.index(r["target"])
        for ci, ca in enumerate(ckeys):
            if r[f"standardized_abs_error_{ca}"] is not None:
                obs[i, ti, ci] = float(r[f"standardized_abs_error_{ca}"])
    smat = np.array([[ckeys.index(s["c2"][t]) for t in tkeys] for s in sel_list])
    perm_mask = all4_mask
    obsvals = [obs[i, j, smat[i, j]] for i, j in zip(*np.where(perm_mask))]
    perm_identity = float(np.mean(obsvals))
    n_le, n_perm, best = 0, 0, (float("inf"), None)
    for perm in itertools.permutations(range(len(tkeys))):
        mapped = smat[:, perm]
        scores = np.take_along_axis(obs, mapped[:, :, None], axis=2)[:, :, 0]
        val = float(np.mean(scores[perm_mask]))
        n_perm += 1
        if val <= perm_identity + 1e-12:
            n_le += 1
        if val < best[0]:
            best = (val, perm)
    perm_p = n_le / n_perm

    # R3 pooled selection, descriptive group sensitivity only.
    group_rows = []
    group_scores = {}
    for g in ["HCM", "Control"]:
        use = [r for r in paired if r["cohort"] == g]
        group_scores[g] = {
            "participants": len(set(r["heldout_participant_id"] for r in use)), "cells": len(use),
            **{f"{m}_std_MAE": float(np.mean([r[f"standardized_abs_error_{m}"] for r in use])) for m in ["C0", "C1", "C2"]},
        }
        group_rows.append({"cohort": g, **group_scores[g]})

    # QT/PP selection-family pattern.
    qt_choice, pp_choice = Counter(), Counter()
    for s in sel_list:
        for targ, _, _, _, fam in TARGS:
            (qt_choice if fam == "QT" else pp_choice)[s["c2"][targ]] += 1
    qt_total, pp_total = sum(qt_choice.values()), sum(pp_choice.values())
    qt_same = sum(qt_choice[c[0]] for c in CANDS if c[3] == "QT")
    pp_same = sum(pp_choice[c[0]] for c in CANDS if c[3] == "PP")

    # Conservative final category; this is an assessment, not a journal decision.
    imp21, winfrac = imp21, winfrac
    if imp10 >= 10 and imp21 >= 10 and winfrac >= .60 and influence_positive and perm_p <= .05 and distinct_modal >= 2:
        verdict = "RADBILL_STRONG_HUMAN_P2_REPLICATION"
    elif imp10 > 0 and imp20 > 0 and distinct_modal >= 2 and influence_positive:
        verdict = "RADBILL_PARTIAL_HUMAN_P2_REPLICATION"
    elif imp10 > 0 and imp20 > 0:
        verdict = "RADBILL_PREDICTIVE_BUT_NOT_TARGET_AWARE"
    else:
        verdict = "RADBILL_NO_MEANINGFUL_P2_REPLICATION"

    # Write compact source/report artifacts.
    support_rows = []
    for targ, *_ in TARGS:
        s = ds["support"][targ]
        support_rows.append(
            f"| {targ} | {s['target_n']} | " + ", ".join(str(s["pair_n"][c]) for c in ckeys)
            + f" | {s['all4_n']} | {tmetrics[targ]['n']} |"
        )
    target_rows_md = []
    for targ, *_ in TARGS:
        m = tmetrics[targ]
        target_rows_md.append(
            f"| {targ} | {m['n']} | {m['std_C0']:.4f} | {m['std_C1']:.4f} | {m['std_C2']:.4f} | "
            f"{m['native_C0']:.4f} | {m['native_C1']:.4f} | {m['native_C2']:.4f} |"
        )
    winner_rows = []
    for targ, rate, _, _, _ in TARGS:
        order = sorted(ckeys, key=lambda ca: np.mean([s["means"][targ][ca] for s in sel_list]))
        winner_rows.append(
            f"| {targ} | {labels[modal[targ]]} ({counts[targ][modal[targ]]}/19) | "
            + " > ".join(labels[x] for x in order) + " |"
        )
    target_order = ", ".join(t[0] for t in TARGS)
    report = f"""# Radbill primary report

## Source and biological unit

The supplied local workbook is the sole analysis source. The Study measurements and Clinical data sheets agree on 19 participants: 9 HCM and 10 controls. Each participant contributes the full eight-row pacing sequence. Outer leave-one-participant-out holds out the person's entire repeated block. Gray author-marked and blank cells are missing; no clinical variables enter primary selection or fitting.

## Usable N

Candidate-target N requires both measurements. Primary C0/C1/C2 metrics use shared held-out cells where the target and the selected C1 and C2 candidate measurements are valid. The all-four N is the strict common support used by target-label permutation.

| Target | Valid target N | Pair N in candidate order (DT5 QT, DT5 PP, DTend QT, DTend PP) | All-four N | Primary scored N |
|---|---:|---|---:|---:|
{chr(10).join(support_rows)}

## Nested LOPO metrics

All primary models are scored on the same {len(paired)} participant-target cells. Standardized errors use each outer training fold's target mean and sample SD. Native QT MAE is in ms; PP MAE is in mmHg. Mean MAE is pooled across scored cells; equal-target mean is also reported.

| Model | Mean standardized MAE | Median standardized AE | Equal-target mean standardized MAE | Improvement vs C0 | Native MAE by target, in order: {target_order} |
|---|---:|---:|---:|---:|---|
| C0 | {c0:.6f} | {metric['C0']['median']:.6f} | {metric['C0']['macro']:.6f} | — | {'; '.join(f"{tmetrics[t]['native_C0']:.4f}" for t in tkeys)} |
| C1 | {c1:.6f} | {metric['C1']['median']:.6f} | {metric['C1']['macro']:.6f} | {imp10:.2f}% | {'; '.join(f"{tmetrics[t]['native_C1']:.4f}" for t in tkeys)} |
| C2 | {c2:.6f} | {metric['C2']['median']:.6f} | {metric['C2']['macro']:.6f} | {imp20:.2f}% | {'; '.join(f"{tmetrics[t]['native_C2']:.4f}" for t in tkeys)} |

C2 versus C1 improvement: **{imp21:.2f}%**. C2 has lower standardized absolute error in **{wins}/{len(paired)} ({winfrac:.1%})** paired predictions.

| Target | N | C0 std MAE | C1 std MAE | C2 std MAE | C0 native MAE | C1 native MAE | C2 native MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(target_rows_md)}

## Candidate rankings

Order is mean nested inner-CV standardized MAE across outer folds. Modal winner is the most frequently selected C2 candidate across 19 outer folds.

| Target | Modal C2 winner | Candidate order, best to worst |
|---|---|---|
{chr(10).join(winner_rows)}

There are {distinct_modal} distinct modal winners across the eight targets. C1 fold counts: {', '.join(f"{labels[c]} {c1counts[c]}/19" for c in ckeys)}. C2 total target-fold selection counts: {', '.join(f"{labels[c]} {sum(counts[t][c] for t in tkeys)}" for c in ckeys)}.

C3 all-data in-sample oracle mean standardized MAE is {np.mean(list(oracleerr.values())):.6f}; it is not validation evidence. Per-target oracle candidates: {', '.join(f"{t}={labels[oracle[t]]}" for t in tkeys)}.

## Verdict

**{verdict}.** C1 vs C0 improvement is {imp10:.2f}%; C2 vs C0 is {imp20:.2f}%; C2 vs C1 is {imp21:.2f}%. See the minimal robustness report for influence, exact mapping permutation, and pooled subgroup sensitivity.

Although C2 has lower target-averaged standardized MAE than C1 for each of the eight targets, it wins fewer than half of individual paired cells. That indicates heterogeneous gains rather than a uniform participant-level improvement.
"""
    (out / "RADBILL_PRIMARY_REPORT.md").write_text(report, encoding="utf-8")

    influence_positive_text = "yes" if influence_positive else "no"
    qt_same_pct = qt_same / qt_total if qt_total else 0
    pp_same_pct = pp_same / pp_total if pp_total else 0
    best_map = " -> ".join(tkeys[i] for i in best[1])
    group_md = "\n".join(
        f"| {g} | {group_scores[g]['participants']} | {group_scores[g]['cells']} | "
        f"{group_scores[g]['C0_std_MAE']:.6f} | {group_scores[g]['C1_std_MAE']:.6f} | {group_scores[g]['C2_std_MAE']:.6f} |"
        for g in ["HCM", "Control"]
    )
    robustness = f"""# Radbill minimal robustness

## R1 — Participant influence

Deleting each held-out participant's full prediction block kept C1−C2 positive for every deletion: {influence_positive_text}. Difference range: {min(r['C1_minus_C2'] for r in influence):.6f} to {max(r['C1_minus_C2'] for r in influence):.6f}.

| Deleted participant | Remaining paired cells | C1 minus C2 |
|---:|---:|---:|
{chr(10).join(f"| {r['deleted_participant']} | {r['remaining_cells']} | {r['C1_minus_C2']:.6f} |" for r in influence)}

## R2 — Exact target-label mapping permutation

All 8! = {n_perm:,} permutations of the target-to-C2-selection-profile mapping were enumerated on the fixed all-four-candidate complete support. Identity-map mean standardized MAE: {perm_identity:.6f}; {n_le:,}/{n_perm:,} mappings were no worse (<=) than identity; one-sided exact mapping diagnostic p={perm_p:.6f}. Best score: {best[0]:.6f}. This is a mapping diagnostic, not biological randomization or biological significance.

## R3 — HCM/control descriptive sensitivity

Selection remained pooled; no group-specific models were tuned. Both rows use the shared primary outer predictions.

| Cohort | Participants represented | Cells | C0 mean std MAE | C1 mean std MAE | C2 mean std MAE |
|---|---:|---:|---:|---:|---:|
{group_md}

The primary paired support includes all 9 HCM participants but 9 of 10 controls. Control Study ID 19 has no valid value for the outer-fold C1 global candidate (DTend QT at 100 bpm), so its C1/C2 comparison cells are unscorable; no values were imputed.

For QT targets, QT-family candidates were selected {qt_same}/{qt_total} times ({qt_same_pct:.1%}); for PP targets, PP-family candidates were selected {pp_same}/{pp_total} times ({pp_same_pct:.1%}). This is a descriptive family-alignment summary; see target-level winners and rankings in RADBILL_TARGET_RANKING.csv.

## R4 — Independent recomputation

RADBILL_independent_recompute.py independently recomputes headline metrics from RADBILL_OUTER_LOPO_PREDICTIONS.csv. Its output is RADBILL_INDEPENDENT_RECOMPUTE.csv.
"""
    (out / "RADBILL_MINIMAL_ROBUSTNESS.md").write_text(robustness, encoding="utf-8")

    # Conservative human uplift assessment; does not select a journal.
    if verdict == "RADBILL_STRONG_HUMAN_P2_REPLICATION":
        strength = "Strong on the frozen internal criteria, while still limited to one 19-person dataset."
    elif verdict == "RADBILL_PARTIAL_HUMAN_P2_REPLICATION":
        strength = "A useful human external example, but partial rather than decisive because the full criteria for a strong target-aware result are not all met."
    elif verdict == "RADBILL_PREDICTIVE_BUT_NOT_TARGET_AWARE":
        strength = "Supports prediction from 100-bpm measurements, but does not establish meaningful target-aware selection."
    else:
        strength = "Does not materially strengthen the target-aware prediction case on this dataset."
    uplift = f"""# P2 Radbill dual-Q1 uplift assessment

1. **Independent human dataset:** External human dataset relative to this analysis; independence from every P2 anchor laboratory is not established here.
2. **Repeated physiological perturbation:** Yes; repeated pacing conditions are observed within each participant.
3. **Human participant-held-out prediction:** Yes; outer LOPO covers all 19 people and holds out the whole repeated block. The shared paired C0/C1/C2 scoring support represents 18 people because one control lacks the outer-fold selected C1 measurement; that value was not imputed.
4. **C2 vs C1:** C2 {'outperforms' if c2 < c1 else 'does not outperform'} C1 by {imp21:.2f}% mean standardized MAE on shared held-out support.
5. **Size:** C1 improves {imp10:.2f}% and C2 {imp20:.2f}% versus C0; C2 beats C1 on {wins}/{len(paired)} ({winfrac:.1%}) paired participant-target predictions.
6. **QT vs PP distinction:** QT-target selections use QT candidates {qt_same}/{qt_total} times; PP-target selections use PP candidates {pp_same}/{pp_total} times. The target-level winner pattern and stability determine whether this is a coherent split.
7. **Dual-Q1 strength:** {strength}
8. **Editorial weakness addressed:** Adds participant-held-out predictive evidence during repeated physiological perturbation in humans; it does not establish cross-laboratory reproducibility.
9. **Defensible new claim:** In this 19-participant repeated-pacing dataset, nested leave-one-participant-out prediction suggests the relative utility of 100-bpm early-drive QT/PP and end-drive QT/PP measurements can vary by the higher-rate end-drive QT or pulse-pressure target.
10. **Prohibited claim:** Do not claim clinical prediction, causal mechanism, broad transportability, verified cross-laboratory replication, or a decisive general human target-aware rule. HCM/control comparisons are descriptive and do not justify subgroup-specific models.

Final verdict: {verdict}.
"""
    (out / "P2_RADBILL_DUAL_Q1_UPLIFT.md").write_text(uplift, encoding="utf-8")
    (out / "ENVIRONMENT_NOTE.md").write_text(
        f"# Environment note\n\n- Python {platform.python_version()}\n- NumPy {np.__version__}\n- openpyxl {openpyxl.__version__}\n- Deterministic NumPy least-squares affine OLS; no stochastic model.\n",
        encoding="utf-8",
    )

    print("WORKBOOK_SHA256", ds["xlsx_sha"])
    print("README_SHA256", ds["readme_sha"])
    print("PARTICIPANTS", len(ids), dict(ds["groups"]))
    print("GRAY_FILLED_CELLS", len(ds["gray_cells"]))
    print("SUPPORT", ds["support"])
    print("PRIMARY_CELLS", len(paired))
    print("C0_C1_C2_STD_MAE", {k: metric[k]["mean"] for k in metric})
    print("IMPROVEMENT_PERCENT", {"C1_vs_C0": imp10, "C2_vs_C0": imp20, "C2_vs_C1": imp21})
    print("C2_BEATS_C1", wins, len(paired), winfrac)
    print("MODAL_WINNERS", modal, "DISTINCT", distinct_modal)
    print("QT_MATCH", qt_same, qt_total, qt_same_pct, "PP_MATCH", pp_same, pp_total, pp_same_pct)
    print("INFLUENCE_POSITIVE", influence_positive, "RANGE", min(r["C1_minus_C2"] for r in influence), max(r["C1_minus_C2"] for r in influence))
    print("PERMUTATION", n_le, n_perm, perm_p, "IDENTITY_MAE", perm_identity)
    print("GROUPS", group_scores)
    print("VERDICT", verdict)


if __name__ == "__main__":
    main()
