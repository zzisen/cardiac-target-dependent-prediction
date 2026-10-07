# Project-authored code: MIT; scientific functions preserved verbatim.
from __future__ import annotations

import argparse

import csv

import hashlib

import itertools

import json

import math

import platform

import sys

import zipfile

from datetime import datetime, timezone

from pathlib import Path

from typing import Any

import numpy as np

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[3]

RLC = Path(__file__).resolve().parents[1] / 'inputs' / 'project' / 'P2_RLC1_DUAL_Q1_UPLIFT_2026-10-03'

RAD = ROOT / 'analysis' / '06_radbill'

EPS = 1e-12

RLC_TARGETS = [
    "3uM_PeakTension", "3uM_TTP", "3uM_RT50",
    "10uM_PeakTension", "10uM_TTP", "10uM_RT50",
]

RLC_CANDIDATES = ["1uM_PeakTension", "1uM_TTP", "1uM_RT50"]

RLC_SHEETS = {"PeakTension": "Fig6B_TPeak", "TTP": "Fig6D_TTP", "RT50": "Fig6E_RT50"}

RAD_TARGETS = [
    "110_DTend_QT", "110_DTend_PP", "120_DTend_QT", "120_DTend_PP",
    "130first_DTend_QT", "130first_DTend_PP", "130last_DTend_QT", "130last_DTend_PP",
]

RAD_CANDIDATES = ["DT5_QT", "DT5_PP", "DTend_QT", "DTend_PP"]

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def canonical_blocks(blocks: list[tuple[int, ...]]) -> tuple[tuple[int, ...], ...]:
    return tuple(sorted((tuple(sorted(b)) for b in blocks)))

def all_pairings(indices: tuple[int, ...]) -> list[tuple[tuple[int, ...], ...]]:
    if not indices:
        return [()]
    first, rest = indices[0], indices[1:]
    found = []
    for j, partner in enumerate(rest):
        remaining = rest[:j] + rest[j + 1:]
        for suffix in all_pairings(remaining):
            found.append(canonical_blocks([(first, partner), *suffix]))
    return sorted(set(found))

def partition_registries() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rlc_parts = all_pairings(tuple(range(len(RLC_TARGETS))))
    rlc_natural = canonical_blocks([(0, 3), (1, 4), (2, 5)])
    rlc_rows = []
    for i, blocks in enumerate(rlc_parts, start=1):
        rlc_rows.append({
            "partition_id": f"RLC1_2X2_{i:03d}",
            "system": "RLC-1",
            "universe": "all_unlabeled_2+2+2_partitions_of_6_targets",
            "block_count": 3,
            "block_size": 2,
            "block_1_targets": "|".join(RLC_TARGETS[j] for j in blocks[0]),
            "block_2_targets": "|".join(RLC_TARGETS[j] for j in blocks[1]),
            "block_3_targets": "|".join(RLC_TARGETS[j] for j in blocks[2]),
            "is_natural_partition": blocks == rlc_natural,
            "canonical_signature": ";".join(",".join(str(j) for j in b) for b in blocks),
        })

    rad_parts = []
    for rest in itertools.combinations(range(1, len(RAD_TARGETS)), 3):
        b1 = (0, *rest)
        b2 = tuple(i for i in range(len(RAD_TARGETS)) if i not in b1)
        rad_parts.append((tuple(b1), b2))
    rad_parts.sort(key=lambda p: (p[0], p[1]))
    qt_idx = tuple(i for i, t in enumerate(RAD_TARGETS) if t.endswith("_QT"))
    pp_idx = tuple(i for i, t in enumerate(RAD_TARGETS) if t.endswith("_PP"))
    rad_natural = (qt_idx, pp_idx)
    rad_rows = []
    for i, blocks in enumerate(rad_parts, start=1):
        rad_rows.append({
            "partition_id": f"RADBILL_4X4_{i:03d}",
            "system": "Radbill",
            "universe": "all_unlabeled_4+4_partitions_of_8_targets",
            "block_count": 2,
            "block_size": 4,
            "block_1_targets": "|".join(RAD_TARGETS[j] for j in blocks[0]),
            "block_2_targets": "|".join(RAD_TARGETS[j] for j in blocks[1]),
            "block_3_targets": "",
            "is_natural_partition": blocks == rad_natural,
            "canonical_signature": ";".join(",".join(str(j) for j in b) for b in blocks),
        })
    assert len(rlc_parts) == 15 and len({r["canonical_signature"] for r in rlc_rows}) == 15
    assert len(rad_parts) == 35 and len({r["canonical_signature"] for r in rad_rows}) == 35
    assert sum(bool(r["is_natural_partition"]) for r in rlc_rows) == 1
    assert sum(bool(r["is_natural_partition"]) for r in rad_rows) == 1
    return rlc_rows, rad_rows, {
        "RLC1": {r["partition_id"]: tuple(tuple(int(x) for x in b.split(",")) for b in r["canonical_signature"].split(";")) for r in rlc_rows},
        "RADBILL": {r["partition_id"]: tuple(tuple(int(x) for x in b.split(",")) for b in r["canonical_signature"].split(";")) for r in rad_rows},
    }

def truth(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}

def rlc_fit(x: np.ndarray, y: np.ndarray, train: list[int], test: int) -> tuple[float, float, float]:
    xmu = float(np.mean(x[train]))
    xsd = float(np.std(x[train], ddof=1))
    ymu = float(np.mean(y[train]))
    ysd = float(np.std(y[train], ddof=1))
    if not all(math.isfinite(z) and z > 0.0 for z in (xsd, ysd)):
        raise RuntimeError("RLC nonpositive/nonfinite fold-local sample SD")
    xz = (x[train] - xmu) / xsd
    yz = (y[train] - ymu) / ysd
    beta, _, rank, _ = np.linalg.lstsq(np.column_stack((np.ones(len(train)), xz)), yz, rcond=None)
    if rank != 2 or not np.isfinite(beta).all():
        raise RuntimeError("RLC affine OLS rank/finite failure")
    pred = ymu + ysd * (float(beta[0]) + float(beta[1]) * ((float(x[test]) - xmu) / xsd))
    return float(pred), ymu, ysd

def load_rlc_data() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    wb = load_workbook(RLC / "RLC1_SOURCE_DATA" / "RLC-1_Fig6.xlsx", read_only=True, data_only=True)
    xs: dict[str, np.ndarray] = {}
    ys: dict[str, np.ndarray] = {}
    for family, sheet in RLC_SHEETS.items():
        rows = list(wb[sheet].iter_rows(values_only=True))
        values = np.asarray(rows[1:], dtype=float)
        xs[f"1uM_{family}"] = values[:, 1]
        ys[f"3uM_{family}"] = values[:, 2]
        ys[f"10uM_{family}"] = values[:, 3]
    wb.close()
    return xs, ys

def compute_rlc(xs: dict[str, np.ndarray], ys: dict[str, np.ndarray], parts: dict[str, tuple[tuple[int, ...], ...]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_family = {t: t.split("_", 1)[1] for t in RLC_TARGETS}
    sel_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for pid, blocks in parts.items():
        for outer in range(7):
            outer_train = [i for i in range(7) if i != outer]
            chosen_by_target: dict[str, str] = {}
            for bidx, block_idx in enumerate(blocks, start=1):
                block_targets = [RLC_TARGETS[i] for i in block_idx]
                scores = {}
                for candidate in RLC_CANDIDATES:
                    per_inner = []
                    for inner in outer_train:
                        fit_rows = [i for i in outer_train if i != inner]
                        e = []
                        for target in block_targets:
                            pred, _, ysd = rlc_fit(xs[candidate], ys[target], fit_rows, inner)
                            e.append(abs(float(ys[target][inner]) - pred) / ysd)
                        per_inner.append(float(np.mean(e)))
                    scores[candidate] = float(np.mean(per_inner))
                selected = min(RLC_CANDIDATES, key=lambda c: (scores[c], RLC_CANDIDATES.index(c)))
                for target in block_targets:
                    chosen_by_target[target] = selected
                sel_rows.append({
                    "partition_id": pid,
                    "outer_heldout_rat": f"Rat_{outer + 1:02d}",
                    "block_id": f"B{bidx}",
                    "block_targets": "|".join(block_targets),
                    "selected_candidate_action": selected,
                    "score_1uM_PeakTension": scores["1uM_PeakTension"],
                    "score_1uM_TTP": scores["1uM_TTP"],
                    "score_1uM_RT50": scores["1uM_RT50"],
                    "tie_rule": "unrounded_binary64_exact_tie_fixed_candidate_order",
                })
            for target in RLC_TARGETS:
                candidate = chosen_by_target[target]
                pred, ymu, ysd = rlc_fit(xs[candidate], ys[target], outer_train, outer)
                actual = float(ys[target][outer])
                err = abs(actual - pred) / ysd
                pred_rows.append({
                    "partition_id": pid,
                    "heldout_rat": f"Rat_{outer + 1:02d}",
                    "target": target,
                    "target_family": target_family[target],
                    "block_id": next(r["block_id"] for r in reversed(sel_rows) if r["partition_id"] == pid and r["outer_heldout_rat"] == f"Rat_{outer + 1:02d}" and target in r["block_targets"].split("|")),
                    "candidate_action": candidate,
                    "candidate_value": float(xs[candidate][outer]),
                    "observed_native": actual,
                    "predicted_native": pred,
                    "outer_training_target_mean": ymu,
                    "outer_training_target_sd": ysd,
                    "standardized_absolute_error": err,
                    "native_absolute_error": abs(actual - pred),
                    "outer_training_n": 6,
                })
    # Attach paired cell-loss differences to the natural partition after all predictions exist.
    natural_id = next(pid for pid, sig in parts.items() if canonical_blocks([tuple(b) for b in sig]) == canonical_blocks([(0, 3), (1, 4), (2, 5)]))
    natural_errors = {(r["heldout_rat"], r["target"]): float(r["standardized_absolute_error"]) for r in pred_rows if r["partition_id"] == natural_id}
    for row in pred_rows:
        row["paired_cell_loss_difference_vs_natural"] = float(row["standardized_absolute_error"]) - natural_errors[(row["heldout_rat"], row["target"])]
    return pred_rows, sel_rows, [{"natural_partition_id": natural_id}]

def rad_mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 2:
        return None, None
    mu = float(np.mean(np.asarray(values, dtype=float)))
    sd = float(np.std(np.asarray(values, dtype=float), ddof=1))
    if not math.isfinite(sd) or sd <= EPS:
        return mu, None
    return mu, sd

def load_radbill() -> tuple[dict[int, dict[str, Any]], set[tuple[int, str]], dict[tuple[int, str], dict[str, str]]]:
    rows = read_csv(RAD / "input" / "RADBILL_CANONICAL_DATA.csv")
    ds: dict[int, dict[str, Any]] = {}
    rowmap = {}
    for r in rows:
        pid, target = int(r["participant_id"]), r["target"]
        if pid not in ds:
            ds[pid] = {"cand": {}, "target": {}}
        ds[pid]["target"][target] = float(r["target_value"]) if r["target_valid"] == "1" else None
        for cand in RAD_CANDIDATES:
            ds[pid]["cand"][cand] = float(r[f"candidate_{cand}"]) if r[f"candidate_{cand}_valid"] == "1" else None
        rowmap[(pid, target)] = r
    # A24 support is a source-derived structural mask. The policy-dependent
    # 132-cell fair-comparator mask is retained only as frozen A23 context.
    support = {
        (pid, target) for (pid, target), row in rowmap.items()
        if row["target_valid"] == "1"
        and all(row[f"candidate_{cand}_valid"] == "1" for cand in RAD_CANDIDATES)
    }
    return ds, support, rowmap

def rad_fit(ds: dict[int, dict[str, Any]], train: list[int], cand: str, target: str) -> dict[str, float] | None:
    xs = [ds[i]["cand"][cand] for i in train if ds[i]["cand"][cand] is not None]
    ys = [ds[i]["target"][target] for i in train if ds[i]["target"][target] is not None]
    xm, xd = rad_mean_sd(xs)
    ym, yd = rad_mean_sd(ys)
    if xm is None or xd is None or ym is None or yd is None:
        return None
    paired = [i for i in train if ds[i]["cand"][cand] is not None and ds[i]["target"][target] is not None]
    if len(paired) < 3:
        return None
    xz = np.asarray([(ds[i]["cand"][cand] - xm) / xd for i in paired], dtype=float)
    yz = np.asarray([(ds[i]["target"][target] - ym) / yd for i in paired], dtype=float)
    beta, _, rank, _ = np.linalg.lstsq(np.column_stack((np.ones(len(paired)), xz)), yz, rcond=None)
    if rank < 2 or not np.isfinite(beta).all():
        return None
    return {"xm": xm, "xd": xd, "ym": ym, "yd": yd, "b0": float(beta[0]), "b1": float(beta[1])}

def rad_predict(model: dict[str, float] | None, x: float | None) -> float | None:
    if model is None or x is None:
        return None
    return model["ym"] + model["yd"] * (model["b0"] + model["b1"] * ((x - model["xm"]) / model["xd"]))

def rad_inner(ds: dict[int, dict[str, Any]], train: list[int]) -> dict[str, Any]:
    scores = {t: {c: [] for c in RAD_CANDIDATES} for t in RAD_TARGETS}
    for hold in train:
        inner_train = [i for i in train if i != hold]
        for target in RAD_TARGETS:
            if ds[hold]["target"][target] is None or not all(ds[hold]["cand"][c] is not None for c in RAD_CANDIDATES):
                continue
            models = {c: rad_fit(ds, inner_train, c, target) for c in RAD_CANDIDATES}
            if any(models[c] is None for c in RAD_CANDIDATES):
                continue
            y = ds[hold]["target"][target]
            for c in RAD_CANDIDATES:
                pred = rad_predict(models[c], ds[hold]["cand"][c])
                scores[target][c].append(abs(float(pred) - float(y)) / float(models[c]["yd"]))
    means = {}
    for target in RAD_TARGETS:
        means[target] = {c: (float(np.mean(scores[target][c])) if scores[target][c] else math.inf) for c in RAD_CANDIDATES}
        if not any(math.isfinite(v) for v in means[target].values()):
            raise RuntimeError(f"No frozen-compatible inner scores for Radbill target {target}")
    return {"means": means}

def rank_summaries(pred_rows: list[dict[str, Any]], registry_rows: list[dict[str, Any]], unit_key: str, target_count: int, partition_count: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    natural_id = next(r["partition_id"] for r in registry_rows if truth(r["is_natural_partition"]))
    by_partition = {r["partition_id"]: [p for p in pred_rows if p["partition_id"] == r["partition_id"]] for r in registry_rows}
    unit_risks: dict[str, dict[str, float]] = {}
    risks = {}
    for pid, rows in by_partition.items():
        risks[pid] = float(np.mean([float(r["standardized_absolute_error"]) for r in rows]))
        units = {}
        for unit in sorted({str(r[unit_key]) for r in rows}):
            losses = [float(r["standardized_absolute_error"]) for r in rows if str(r[unit_key]) == unit]
            units[unit] = float(np.mean(losses))
        unit_risks[pid] = units
    ordered = sorted(registry_rows, key=lambda r: (risks[r["partition_id"]], r["partition_id"]))
    ranks = {r["partition_id"]: i + 1 for i, r in enumerate(ordered)}
    nat = risks[natural_id]
    sorted_risks = [risks[r["partition_id"]] for r in ordered]
    median_risk = sorted_risks[len(sorted_risks) // 2]
    best, worst = sorted_risks[0], sorted_risks[-1]
    lower = sum(v < nat - EPS for v in risks.values())
    equal = sum(abs(v - nat) <= EPS for v in risks.values())
    higher = sum(v > nat + EPS for v in risks.values())
    out = []
    for row in registry_rows:
        pid = row["partition_id"]
        result = {
            "partition_id": pid,
            "is_natural_partition": row["is_natural_partition"],
            "primary_mean_standardized_absolute_error": risks[pid],
            "median_unit_standardized_absolute_error": float(np.median(list(unit_risks[pid].values()))),
            "rank_lower_is_better": ranks[pid],
            "paired_risk_difference_vs_natural": risks[pid] - nat,
            "n_scored_cells": len(by_partition[pid]),
            "n_scored_units": len(unit_risks[pid]),
        }
        if pid == natural_id:
            result.update({
                "natural_rank_denominator": partition_count,
                "matched_partitions_lower_count": lower,
                "matched_partitions_lower_fraction": lower / partition_count,
                "matched_partitions_equal_within_1e-12_count": equal,
                "matched_partitions_equal_within_1e-12_fraction": equal / partition_count,
                "matched_partitions_higher_count": higher,
                "matched_partitions_higher_fraction": higher / partition_count,
                "natural_minus_best_risk": nat - best,
                "natural_minus_median_partition_risk": nat - median_risk,
                "natural_minus_worst_risk": nat - worst,
            })
        else:
            result.update({
                "natural_rank_denominator": "",
                "matched_partitions_lower_count": "",
                "matched_partitions_lower_fraction": "",
                "matched_partitions_equal_within_1e-12_count": "",
                "matched_partitions_equal_within_1e-12_fraction": "",
                "matched_partitions_higher_count": "",
                "matched_partitions_higher_fraction": "",
                "natural_minus_best_risk": "",
                "natural_minus_median_partition_risk": "",
                "natural_minus_worst_risk": "",
            })
        out.append(result)
    return out, unit_risks

def leave_one_unit(pred_rows: list[dict[str, Any]], registry_rows: list[dict[str, Any]], unit_key: str, all_units: list[str], support_weighted: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    natural_id = next(r["partition_id"] for r in registry_rows if truth(r["is_natural_partition"]))
    out = []
    natural_ranks = []
    n = len(registry_rows)
    thresholds = {"top10_threshold": math.ceil(0.10 * n), "top_quartile_threshold": math.ceil(0.25 * n), "top_half_threshold": math.ceil(0.50 * n)}
    for deleted in all_units:
        risks = {}
        for reg in registry_rows:
            rows = [r for r in pred_rows if r["partition_id"] == reg["partition_id"] and str(r[unit_key]) != deleted]
            if not rows:
                risks[reg["partition_id"]] = math.inf
            else:
                # RLC has equal cells per rat; Radbill preserves pooled equal-cell weighting.
                risks[reg["partition_id"]] = float(np.mean([float(r["standardized_absolute_error"]) for r in rows]))
        ranked = sorted(registry_rows, key=lambda r: (risks[r["partition_id"]], r["partition_id"]))
        rank_map = {r["partition_id"]: i + 1 for i, r in enumerate(ranked)}
        rank = rank_map[natural_id]
        natural_ranks.append(rank)
        for reg in registry_rows:
            out.append({
                "deleted_biological_unit": deleted,
                "partition_id": reg["partition_id"],
                "partition_risk_after_omission": risks[reg["partition_id"]],
                "rank_lower_is_better": rank_map[reg["partition_id"]],
                "is_natural_partition": reg["partition_id"] == natural_id,
                "frozen_support_cell_weighting": "pooled_cells" if support_weighted else "equal_unit_then_equal_target",
            })
    meta = {
        "natural_partition_id": natural_id,
        "natural_leave_one_unit_ranks": natural_ranks,
        "best_count": sum(r == 1 for r in natural_ranks),
        "top10_count": sum(r <= thresholds["top10_threshold"] for r in natural_ranks),
        "top_quartile_count": sum(r <= thresholds["top_quartile_threshold"] for r in natural_ranks),
        "top_half_count": sum(r <= thresholds["top_half_threshold"] for r in natural_ranks),
        "n_omissions": len(all_units),
        **thresholds,
    }
    return out, meta

def compute_radbill_selections(ds: dict[int, dict[str, Any]], parts: dict[str, tuple[tuple[int, ...], ...]]) -> tuple[list[dict[str, Any]], dict[str, dict[int, dict[str, str]]]]:
    selection_rows = []
    maps: dict[str, dict[int, dict[str, str]]] = {pid: {} for pid in parts}
    participants = sorted(ds)
    for pid, blocks in parts.items():
        for outer in participants:
            train = [i for i in participants if i != outer]
            inn = rad_inner(ds, train)
            by_block = {}
            for bidx, block in enumerate(blocks, start=1):
                targets = [RAD_TARGETS[i] for i in block]
                scores = {c: float(np.mean([inn["means"][t][c] for t in targets])) for c in RAD_CANDIDATES}
                selected = min(RAD_CANDIDATES, key=lambda c: (scores[c], RAD_CANDIDATES.index(c)))
                by_block[f"B{bidx}"] = selected
                selection_rows.append({
                    "partition_id": pid,
                    "outer_heldout_participant": outer,
                    "block_id": f"B{bidx}",
                    "block_targets": "|".join(targets),
                    "selected_candidate_action": selected,
                    "score_DT5_QT": scores["DT5_QT"],
                    "score_DT5_PP": scores["DT5_PP"],
                    "score_DTend_QT": scores["DTend_QT"],
                    "score_DTend_PP": scores["DTend_PP"],
                    "tie_rule": "unrounded_binary64_exact_tie_frozen_candidate_order",
                })
            maps[pid][outer] = by_block
    return selection_rows, maps

def compute_radbill_predictions(ds: dict[int, dict[str, Any]], support: set[tuple[int, str]], registry: list[dict[str, Any]], parts: dict[str, tuple[tuple[int, ...], ...]], maps: dict[str, dict[int, dict[str, str]]]) -> list[dict[str, Any]]:
    out = []
    participants = sorted(ds)
    for reg in registry:
        pid = reg["partition_id"]
        blocks = parts[pid]
        target_to_block = {RAD_TARGETS[tidx]: f"B{bidx}" for bidx, block in enumerate(blocks, start=1) for tidx in block}
        for heldout, target in sorted(support):
            train = [i for i in participants if i != heldout]
            block_id = target_to_block[target]
            candidate = maps[pid][heldout][block_id]
            model = rad_fit(ds, train, candidate, target)
            if model is None:
                raise RuntimeError(f"Frozen Radbill outer fit unavailable on common support: {pid}/{target}/{candidate}")
            y = ds[heldout]["target"][target]
            x = ds[heldout]["cand"][candidate]
            if y is None or x is None:
                raise RuntimeError(f"Radbill action/support gate changed after check: {pid}/{target}/{candidate}")
            pred = float(rad_predict(model, x))
            out.append({
                "partition_id": pid,
                "heldout_participant": heldout,
                "target": target,
                "target_family": "QT" if target.endswith("_QT") else "PP",
                "block_id": block_id,
                "candidate_action": candidate,
                "observed_native": float(y),
                "predicted_native": pred,
                "outer_training_target_mean": model["ym"],
                "outer_training_target_sd": model["yd"],
                "standardized_absolute_error": abs(pred - float(y)) / model["yd"],
                "native_absolute_error": abs(pred - float(y)),
                "outer_training_participants": len(train),
            })
    nat_id = next(r["partition_id"] for r in registry if truth(r["is_natural_partition"]))
    nat_errors = {(r["heldout_participant"], r["target"]): float(r["standardized_absolute_error"]) for r in out if r["partition_id"] == nat_id}
    for row in out:
        row["paired_cell_loss_difference_vs_natural"] = float(row["standardized_absolute_error"]) - nat_errors[(row["heldout_participant"], row["target"])]
    return out
