# Project-authored code: MIT; scientific functions preserved verbatim.
from __future__ import annotations

import argparse

import csv

import hashlib

import itertools

import json

import math

import platform

import re

import shutil

import sys

import time

import zipfile

from collections import Counter, defaultdict

from pathlib import Path

import numpy as np

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1] / 'inputs' / 'project'

V2 = Path(__file__).resolve().parents[3]

A23_PRE = Path(__file__).resolve().parents[1] / 'inputs' / 'project' / 'P2_V3_ABT' / 'A23_PREDECLARATION_LUNA5'

TARGETS_RLC = [f"{dose}_{family}" for dose in ("3uM", "10uM") for family in ("PeakTension", "TTP", "RT50")]

RLC_FAMILIES = ["PeakTension", "TTP", "RT50"]

RLC_CANDIDATES = ["1uM_PeakTension", "1uM_TTP", "1uM_RT50"]

RLC_SHEETS = {"PeakTension": "Fig6B_TPeak", "TTP": "Fig6D_TTP", "RT50": "Fig6E_RT50"}

RLC_ACTION_FOR_TARGET = {t: t.split("_", 1)[1] for t in TARGETS_RLC}

RAD_CANDIDATES = ["DT5_QT", "DT5_PP", "DTend_QT", "DTend_PP"]

RAD_TARGETS = [
    "110_DTend_QT", "110_DTend_PP", "120_DTend_QT", "120_DTend_PP",
    "130first_DTend_QT", "130first_DTend_PP", "130last_DTend_QT", "130last_DTend_PP",
]

RAD_FAMILY = {t: t.rsplit("_", 1)[1] for t in RAD_TARGETS}

RAD_FAMILY_CANDS = {"QT": ["DT5_QT", "DTend_QT"], "PP": ["DT5_PP", "DTend_PP"]}

TANNER_TARGETS = [25, 100, 250, 1000]

TANNER_CANDIDATES = [100, 250]

AW_TARGETS = [0.05, 0.10, 0.25, 0.50, 1.00, 2.50]

RESOLUTION_ORDER = {
    "RLC1_RAT": ["SHARED", "FAMILY", "EXACT"],
    "RADBILL_HUMAN_PACING": ["SHARED", "FAMILY", "EXACT"],
    "TANNER_RAT_TRABECULA": ["SHARED", "EXACT"],
    "AWINDA_MOUSE": ["SHARED", "EXACT"],
}

COUNTERS = Counter()

PREDICTION_CACHE: dict[tuple, object] = {}

MODEL_CACHE: dict[tuple, object] = {}

SELECTION_CACHE: dict[tuple, object] = {}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def verify_radbill_strict_support() -> dict:
    """Verify the amended policy-independent support using source flags only."""
    path = V2 / "analysis" / "06_radbill" / "input" / "RADBILL_CANONICAL_DATA.csv"
    frame = pd.read_csv(path)
    if frame["participant_id"].nunique() != 19:
        raise RuntimeError("Radbill canonical data must contain 19 source participants")
    if any(set(sub["target"].astype(str)) != set(RAD_TARGETS) or len(sub) != len(RAD_TARGETS)
           for _, sub in frame.groupby("participant_id", sort=True)):
        raise RuntimeError("Radbill canonical support must contain exactly one row for each of the eight frozen targets per participant")
    if ((frame["target_valid"].astype(int) == 1) & (frame["target_author_gray"].astype(int) != 0)).any():
        raise RuntimeError("Radbill target-valid flags conflict with author-gray target markings")
    eligible, excluded, target_counts = [], [], {}
    candidate_status = {}
    for pid, sub in frame.groupby("participant_id", sort=True):
        p = int(pid)
        candidate_flags = {}
        for candidate in RAD_CANDIDATES:
            valid = sub[f"candidate_{candidate}_valid"].astype(int).eq(1).all()
            non_author_gray = sub[f"candidate_{candidate}_author_gray"].astype(int).eq(0).all()
            candidate_flags[candidate] = bool(valid and non_author_gray)
        complete = all(candidate_flags.values())
        (eligible if complete else excluded).append(p)
        candidate_status[str(p)] = candidate_flags
        target_counts[str(p)] = int(sub["target_valid"].astype(int).eq(1).sum()) if complete else 0
    eligible_target_rows = frame.loc[
        frame["participant_id"].isin(eligible) & frame["target_valid"].astype(int).eq(1)
    ]
    target_counts_by_target = {
        str(target): int(eligible_target_rows["target"].astype(str).eq(target).sum())
        for target in RAD_TARGETS
    }
    result = {
        "support_rule": "STRICT ALL-FOUR-CANDIDATE POLICY-EVALUABLE PARTICIPANT SUPPORT",
        "candidate_eligibility_source": "canonical candidate_*_valid == 1 and candidate_*_author_gray == 0 for every frozen candidate measurement",
        "target_cell_rule": "within eligible participants, retain each row with target_valid == 1; no imputation",
        "source_participants": int(frame["participant_id"].nunique()),
        "excluded_participant_ids": excluded,
        "eligible_participant_ids": eligible,
        "eligible_participants": len(eligible),
        "target_valid_cells": int(sum(target_counts.values())),
        "target_valid_cells_by_participant": target_counts,
        "target_valid_cells_by_target": target_counts_by_target,
        "candidate_completeness_by_participant": candidate_status,
        "expected_values_verified": False,
    }
    if excluded != [9, 19] or len(eligible) != 17 or result["target_valid_cells"] != 126:
        raise RuntimeError(f"Amended Radbill support did not match expected 19/2/17/126 values: {result}")
    if not all(all(flags.values()) for pid, flags in candidate_status.items() if int(pid) in eligible):
        raise RuntimeError("At least one eligible Radbill participant lacks a valid non-author-gray candidate")
    if sum(result["target_valid_cells_by_target"].values()) != result["target_valid_cells"]:
        raise RuntimeError("Radbill per-target and pooled target-support counts disagree")
    result["expected_values_verified"] = True
    return result

def sample_sd(values: np.ndarray) -> float:
    return float(np.std(np.asarray(values, dtype=float), ddof=1))

def cached(cache: dict, key: tuple, function):
    if key in cache:
        COUNTERS["cache_hits"] += 1
        return cache[key]
    COUNTERS["cache_misses"] += 1
    value = function()
    cache[key] = value
    return value

def load_rlc() -> dict:
    workbook = PROJECT / "P2_RLC1_DUAL_Q1_UPLIFT_2026-10-03" / "RLC1_SOURCE_DATA" / "RLC-1_Fig6.xlsx"
    xdata, ydata = {}, {}
    for family, sheet in RLC_SHEETS.items():
        frame = pd.read_excel(workbook, sheet_name=sheet)
        if frame.shape != (7, 4):
            raise RuntimeError(f"RLC sheet shape mismatch: {sheet} {frame.shape}")
        xdata[f"1uM_{family}"] = pd.to_numeric(frame["1uM"], errors="coerce").to_numpy(float)
        for dose in ("3uM", "10uM"):
            target = f"{dose}_{family}"
            ydata[target] = pd.to_numeric(frame[dose], errors="coerce").to_numpy(float)
    if not all(np.isfinite(x).all() for x in list(xdata.values()) + list(ydata.values())):
        raise RuntimeError("RLC support contains nonfinite values")
    units = [f"Rat_{i:02d}" for i in range(1, 8)]
    return {"units": units, "x": xdata, "y": ydata}

def load_radbill() -> dict:
    path = V2 / "analysis" / "06_radbill" / "input" / "RADBILL_CANONICAL_DATA.csv"
    frame = pd.read_csv(path)
    support_verification = verify_radbill_strict_support()
    eligible_ids = set(support_verification["eligible_participant_ids"])
    ps: dict[int, dict] = {}
    for pid, sub in frame.groupby("participant_id", sort=True):
        p = int(pid)
        if p not in eligible_ids:
            continue
        cand = {}
        for c in RAD_CANDIDATES:
            val_col, valid_col, gray_col = f"candidate_{c}", f"candidate_{c}_valid", f"candidate_{c}_author_gray"
            valid = sub[valid_col].astype(int).eq(1) & sub[gray_col].astype(int).eq(0)
            vals = pd.to_numeric(sub.loc[valid, val_col], errors="coerce").dropna().unique()
            if len(vals) > 1:
                raise RuntimeError(f"Radbill candidate value varies by target: {p}/{c}")
            if len(vals) != 1:
                raise RuntimeError(f"Eligible Radbill participant lacks a structurally valid candidate: {p}/{c}")
            cand[c] = float(vals[0])
        targets = {}
        for r in sub.itertuples(index=False):
            targets[str(r.target)] = float(r.target_value) if int(r.target_valid) == 1 and pd.notna(r.target_value) else None
        ps[p] = {"cand": cand, "target": targets}
    support = set(
        (p, target) for p in sorted(ps) for target in RAD_TARGETS
        if ps[p]["target"][target] is not None
    )
    if len(ps) != 17 or len(support) != 126 or sorted(set(range(1, 20)) - set(ps)) != [9, 19]:
        raise RuntimeError(f"Radbill amended strict support mismatch: participants={len(ps)}, target_cells={len(support)}")
    return {
        "units": sorted(ps), "ps": ps, "support": support,
        "source_participant_count": 19, "excluded_participant_ids": [9, 19],
        "support_label": "A25_STRICT_SUPPORT_FIXED_POLICY_RECOMPUTE",
    }

def load_tanner() -> dict:
    root = V2 / "analysis" / "07_tanner"
    freeze_path = root / "01_TANNER_SCIENTIFIC_FREEZE.json"
    sha_path = root / "01_TANNER_SCIENTIFIC_FREEZE.sha256"
    if sha256(freeze_path) != sha_path.read_text(encoding="ascii").split()[0]:
        raise RuntimeError("Tanner scientific freeze hash mismatch")
    frame = pd.read_csv(root / "input" / "TANNER_CANONICAL_SUBJECT_LEVEL_DATA.csv")
    outcome = "peak_stress_response_mN_per_mm2"
    units = sorted(frame["trabecula_id_as_listed"].astype(int).unique().tolist())
    x, y = {c: {} for c in TANNER_CANDIDATES}, {t: {} for t in TANNER_TARGETS}
    for r in frame.itertuples(index=False):
        sid = int(r.trabecula_id_as_listed)
        tm, rate = float(r.time_to_stretch_ms), int(r.strain_rate_s_inv)
        value = getattr(r, outcome)
        if pd.isna(value):
            continue
        if tm == 10.0 and rate in TANNER_CANDIDATES:
            x[rate][sid] = float(value)
        if tm == 100.0 and rate in TANNER_TARGETS:
            y[rate][sid] = float(value)
    if any(set(x[c]) != set(units) for c in TANNER_CANDIDATES) or any(set(y[t]) != set(units) for t in TANNER_TARGETS):
        raise RuntimeError("Tanner primary source grid is incomplete")
    return {"units": units, "x": x, "y": y}

def read_awinda_frequency_grid() -> list[float]:
    spec = (A23_PRE / "A23_AWINDA_TARGET_SPECIFIC_RECOMPUTE_SPEC.md").read_text(encoding="utf-8")
    start = spec.index("candidate actions are exactly one matched Re/Im complex-modulus pair")
    end = spec.index("Each action costs exactly two scalar components.", start)
    values = re.findall(r"\d+\.\d+", spec[start:end])
    if len(values) != 72:
        raise RuntimeError(f"Frozen Awinda candidate grid contains {len(values)} frequencies")
    return [float(v) for v in values]

def load_awinda() -> dict:
    src = V2 / "data" / "derived" / "awinda_source"
    xld = pd.read_csv(src / "WT_RLC_N47K_Control_Mavacamten_XLD_GWN_DQ1_2019_09_09.csv", low_memory=False)
    force = pd.read_csv(src / "Transgenic_mice_Fiber_Tensions_with_Fits_DQ1_2019_09_06.csv", low_memory=False)
    link = force[["AnimalID", "Fiber", "Mutation", "Treatment"]].drop_duplicates()
    if link["Fiber"].duplicated().any():
        raise RuntimeError("Awinda fibre map is not one-to-one")
    files = xld[["Filename", "Mutation", "Cond"]].dropna(subset=["Filename"]).drop_duplicates()
    fmap = files.merge(link, left_on="Filename", right_on="Fiber", how="left", validate="one_to_one")
    if fmap["AnimalID"].isna().any() or not fmap["Mutation_x"].eq(fmap["Mutation_y"]).all():
        raise RuntimeError("Awinda fibre linkage or genotype check failed")
    fmap["AnimalID"] = fmap["AnimalID"].astype(str)
    fmap["drug_state"] = np.where(fmap["Cond"].eq("Control"), "Control", "0.3_uM_mavacamten")
    fmap["group"] = fmap["Mutation_x"].astype(str) + "|" + fmap["drug_state"]
    d = xld.loc[
        xld["pCa"].eq(4.8) & pd.to_numeric(xld["Quality"], errors="coerce").eq(1)
    ].copy()
    d = d.merge(fmap[["Filename", "AnimalID", "Mutation_x", "drug_state", "group"]], on="Filename", how="left", validate="many_to_one")
    for c in ("ATP (mM)", "Freq (Hz)", "Em (kPa)", "Vm (kPa)"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if d[["ATP (mM)", "Freq (Hz)", "Em (kPa)", "Vm (kPa)"]].isna().any().any() or d["AnimalID"].isna().any():
        raise RuntimeError("Awinda eligible rows have missing/nonfinite model fields")
    freq = np.sort(d.loc[d["ATP (mM)"].eq(5.0), "Freq (Hz)"].unique())
    if len(freq) != 72 or [f"{v:.4f}" for v in freq] != [f"{v:.4f}" for v in read_awinda_frequency_grid()]:
        raise RuntimeError("Awinda source grid differs from the frozen 72-frequency grid")
    cells_cols = ["AnimalID", "Mutation_x", "drug_state", "group"]
    means = d.groupby(cells_cols + ["ATP (mM)", "Freq (Hz)"], as_index=False).agg(
        Re=("Em (kPa)", "mean"), Im=("Vm (kPa)", "mean"), n_fibres=("Filename", "nunique")
    )
    cell_meta, xrows = [], []
    yrows = {target: [] for target in AW_TARGETS}
    for key, sub in means.groupby(cells_cols, sort=True):
        meta = dict(zip(cells_cols, key))
        cell_meta.append(meta)
        base = sub[sub["ATP (mM)"].eq(5.0)].set_index("Freq (Hz)").reindex(freq)
        if base[["Re", "Im"]].isna().any().any():
            raise RuntimeError(f"Awinda incomplete baseline spectrum: {key}")
        xrows.append(np.stack([base["Re"].to_numpy(float), base["Im"].to_numpy(float)], axis=1))
        for target in AW_TARGETS:
            y = sub[sub["ATP (mM)"].eq(target)].set_index("Freq (Hz)").reindex(freq)
            if y[["Re", "Im"]].isna().any().any():
                raise RuntimeError(f"Awinda incomplete target spectrum: {key}/{target}")
            yrows[target].append(np.r_[y["Re"].to_numpy(float), y["Im"].to_numpy(float)])
    cells = pd.DataFrame(cell_meta)
    cells["AnimalID"] = cells["AnimalID"].astype(str)
    mice = sorted(cells["AnimalID"].unique().tolist())
    groups = sorted(cells["group"].astype(str).unique().tolist())
    X = np.asarray(xrows, dtype=float)
    Y = {target: np.asarray(rows, dtype=float) for target, rows in yrows.items()}
    if len(mice) != 10 or len(cells) != 19 or X.shape != (19, 72, 2):
        raise RuntimeError(f"Awinda frozen support mismatch: mice={len(mice)}, cells={len(cells)}, X={X.shape}")
    for target in AW_TARGETS:
        if Y[target].shape != (19, 144) or not np.isfinite(Y[target]).all():
            raise RuntimeError(f"Awinda response support mismatch at target {target}")
    G = np.column_stack([(cells["group"].astype(str).to_numpy() == group).astype(float) for group in groups])
    return {
        "units": mice, "cells": cells, "X": X, "Y": Y, "freq": freq,
        "groups": groups, "G": G,
        "mouse_cell_ix": {m: np.flatnonzero(cells["AnimalID"].to_numpy() == m) for m in mice},
        "freq_ids": [f"CM_ATP5_{float(f):.4f}Hz_pair" for f in freq],
        "group_support": cells.groupby("group")["AnimalID"].nunique().to_dict(),
    }

def load_all_data() -> dict:
    return {
        "RLC1_RAT": load_rlc(),
        "RADBILL_HUMAN_PACING": load_radbill(),
        "TANNER_RAT_TRABECULA": load_tanner(),
        "AWINDA_MOUSE": load_awinda(),
    }

def make_fit_cache(cache: dict, key: tuple, function):
    return cached(cache, key, function)

def rlc_fit_predict(data: dict, train: list[str], test: str, candidate: str, target: str) -> tuple[float, float]:
    idx = {u: i for i, u in enumerate(data["units"])}
    ti = tuple(sorted(idx[u] for u in train))
    test_i = idx[test]
    key = ("rlc", ti, test_i, candidate, target)

    def compute():
        x, y = data["x"][candidate], data["y"][target]
        xm, xs = float(np.mean(x[list(ti)])), sample_sd(x[list(ti)])
        ym, ys = float(np.mean(y[list(ti)])), sample_sd(y[list(ti)])
        if not all(math.isfinite(v) and v > 0 for v in (xs, ys)):
            raise RuntimeError(f"RLC invalid training SD: {candidate}/{target}/{train}")
        xz = (x[list(ti)] - xm) / xs
        yz = (y[list(ti)] - ym) / ys
        beta, _, rank, _ = np.linalg.lstsq(np.column_stack([np.ones(len(ti)), xz]), yz, rcond=None)
        if rank < 2 or not np.isfinite(beta).all():
            raise RuntimeError(f"RLC fit rank failure: {candidate}/{target}/{train}")
        pred_z = float(beta[0] + beta[1] * ((x[test_i] - xm) / xs))
        return ym + ys * pred_z, ys
    return make_fit_cache(PREDICTION_CACHE, key, compute)

def rlc_inner_error_cube(data: dict, train: list[str]) -> dict:
    key = ("rlc_selector", tuple(sorted(train)))
    if key in SELECTION_CACHE:
        COUNTERS["selector_cache_hits"] += 1
        return SELECTION_CACHE[key]
    COUNTERS["selector_cache_misses"] += 1
    errors = {c: {t: {} for t in TARGETS_RLC} for c in RLC_CANDIDATES}
    for hold in sorted(train):
        inner_train = [u for u in train if u != hold]
        for candidate in RLC_CANDIDATES:
            for target in TARGETS_RLC:
                pred, ysd = rlc_fit_predict(data, inner_train, hold, candidate, target)
                actual = data["y"][target][data["units"].index(hold)]
                errors[candidate][target][hold] = abs(float(actual - pred) / ysd)
                COUNTERS["inner_action_score_evaluations"] += 1
    SELECTION_CACHE[key] = errors
    return errors

def rlc_select(data: dict, train: list[str], resolution: str, context: dict) -> tuple[dict, list[dict]]:
    cube = rlc_inner_error_cube(data, train)
    rows, actions = [], {}
    if resolution == "SHARED":
        block_targets = {"ALL_SIX": TARGETS_RLC}
    elif resolution == "FAMILY":
        block_targets = {f: [t for t in TARGETS_RLC if RLC_ACTION_FOR_TARGET[t] == f] for f in RLC_FAMILIES}
    elif resolution == "EXACT":
        block_targets = {t: [t] for t in TARGETS_RLC}
    else:
        raise ValueError(resolution)
    for block, targets in block_targets.items():
        scores = {}
        for candidate in RLC_CANDIDATES:
            if resolution == "FAMILY":
                per_hold = [
                    float(np.mean([cube[candidate][t][hold] for t in targets]))
                    for hold in sorted(train)
                ]
                score = float(np.mean(per_hold))
            else:
                target_means = [
                    float(np.mean([cube[candidate][t][hold] for hold in sorted(train)]))
                    for t in targets
                ]
                score = float(np.mean(target_means))
            scores[candidate] = score
            COUNTERS["candidate_action_aggregate_scores"] += 1
        chosen = min(RLC_CANDIDATES, key=lambda c: (scores[c], RLC_CANDIDATES.index(c)))
        actions[block] = chosen
        for candidate in RLC_CANDIDATES:
            rows.append({
                **context, "system_id": "RLC1_RAT", "resolution_id": resolution,
                "block_or_target": block, "candidate_action": candidate,
                "inner_cv_score": scores[candidate], "selected": candidate == chosen,
                "n_inner_units": len(train), "tie_rule": "exact score tie; A23 fixed candidate order",
            })
    return actions, rows

def tanner_predict(data: dict, train: list[int], test: int, candidate: int, target: int) -> tuple[float, float, float]:
    key = ("tanner", tuple(sorted(train)), test, int(candidate), int(target))

    def compute():
        x = np.asarray([data["x"][candidate][u] for u in train], dtype=float)
        y = np.asarray([data["y"][target][u] for u in train], dtype=float)
        xm, ym = float(np.mean(x)), float(np.mean(y))
        denom = float(np.sum((x - xm) ** 2))
        slope = 0.0 if denom <= 1e-14 else float(np.sum((x - xm) * (y - ym)) / denom)
        intercept = ym - slope * xm
        pred = slope * float(data["x"][candidate][test]) + intercept
        return pred, slope, intercept
    return make_fit_cache(PREDICTION_CACHE, key, compute)

def tanner_inner_score_cube(data: dict, train: list[int]) -> dict:
    key = ("tanner_selector", tuple(sorted(train)))
    if key in SELECTION_CACHE:
        COUNTERS["selector_cache_hits"] += 1
        return SELECTION_CACHE[key]
    COUNTERS["selector_cache_misses"] += 1
    errors = {c: {t: {} for t in TANNER_TARGETS} for c in TANNER_CANDIDATES}
    for hold in sorted(train):
        inner_train = [u for u in train if u != hold]
        for candidate in TANNER_CANDIDATES:
            for target in TANNER_TARGETS:
                pred, _, _ = tanner_predict(data, inner_train, hold, candidate, target)
                errors[candidate][target][hold] = abs(pred - data["y"][target][hold])
                COUNTERS["inner_action_score_evaluations"] += 1
    SELECTION_CACHE[key] = errors
    return errors

def tanner_select(data: dict, train: list[int], resolution: str, context: dict) -> tuple[dict, list[dict]]:
    cube = tanner_inner_score_cube(data, train)
    if resolution == "SHARED":
        blocks = {"ALL_FOUR": TANNER_TARGETS}
    elif resolution == "EXACT":
        blocks = {str(t): [t] for t in TANNER_TARGETS}
    else:
        raise ValueError(resolution)
    actions, rows = {}, []
    for block, targets in blocks.items():
        scores = {}
        for candidate in TANNER_CANDIDATES:
            target_means = [
                float(np.mean([cube[candidate][t][hold] for hold in sorted(train)]))
                for t in targets
            ]
            scores[candidate] = float(np.mean(target_means))
            COUNTERS["candidate_action_aggregate_scores"] += 1
        chosen = min(TANNER_CANDIDATES, key=lambda c: (scores[c], c))
        actions[block] = chosen
        for candidate in TANNER_CANDIDATES:
            rows.append({
                **context, "system_id": "TANNER_RAT_TRABECULA", "resolution_id": resolution,
                "block_or_target": block, "candidate_action": candidate,
                "inner_cv_score": scores[candidate], "selected": candidate == chosen,
                "n_inner_units": len(train), "tie_rule": "exact score tie; lower strain rate",
            })
    return actions, rows

def radbill_fit_model(data: dict, train: list[int], candidate: str, target: str):
    key = ("radbill_model", tuple(sorted(train)), candidate, target)
    if key in MODEL_CACHE:
        COUNTERS["model_cache_hits"] += 1
        return MODEL_CACHE[key]
    COUNTERS["model_cache_misses"] += 1
    ps = data["ps"]
    xs = [ps[i]["cand"][candidate] for i in train if ps[i]["cand"][candidate] is not None]
    ys = [ps[i]["target"][target] for i in train if ps[i]["target"][target] is not None]
    if len(xs) < 2 or len(ys) < 2:
        MODEL_CACHE[key] = None
        return None
    xm, xd = float(np.mean(xs)), sample_sd(np.asarray(xs))
    ym, yd = float(np.mean(ys)), sample_sd(np.asarray(ys))
    if not all(math.isfinite(v) and v > 1e-12 for v in (xd, yd)):
        MODEL_CACHE[key] = None
        return None
    pair = [i for i in train if ps[i]["cand"][candidate] is not None and ps[i]["target"][target] is not None]
    if len(pair) < 3:
        MODEL_CACHE[key] = None
        return None
    xz = np.asarray([(ps[i]["cand"][candidate] - xm) / xd for i in pair], dtype=float)
    yz = np.asarray([(ps[i]["target"][target] - ym) / yd for i in pair], dtype=float)
    beta, _, rank, _ = np.linalg.lstsq(np.column_stack([np.ones(len(pair)), xz]), yz, rcond=None)
    if rank < 2 or not np.isfinite(beta).all():
        MODEL_CACHE[key] = None
        return None
    model = {"xm": xm, "xd": xd, "ym": ym, "yd": yd, "b0": float(beta[0]), "b1": float(beta[1]), "n_pair": len(pair)}
    MODEL_CACHE[key] = model
    return model

def radbill_predict(data: dict, train: list[int], test: int, candidate: str, target: str) -> tuple[float, float] | None:
    key = ("radbill", tuple(sorted(train)), int(test), candidate, target)

    def compute():
        m = radbill_fit_model(data, train, candidate, target)
        x = data["ps"][test]["cand"][candidate]
        if m is None or x is None:
            return None
        pred = m["ym"] + m["yd"] * (m["b0"] + m["b1"] * ((x - m["xm"]) / m["xd"]))
        return float(pred), float(m["yd"])
    return make_fit_cache(PREDICTION_CACHE, key, compute)

def radbill_inner_score_cube(data: dict, train: list[int]) -> dict:
    key = ("radbill_selector", tuple(sorted(train)))
    if key in SELECTION_CACHE:
        COUNTERS["selector_cache_hits"] += 1
        return SELECTION_CACHE[key]
    COUNTERS["selector_cache_misses"] += 1
    scores = {t: {c: [] for c in RAD_CANDIDATES} for t in RAD_TARGETS}
    for hold in sorted(train):
        inner_train = [i for i in train if i != hold]
        for target in RAD_TARGETS:
            p = data["ps"][hold]
            if p["target"][target] is None or not all(p["cand"][c] is not None for c in RAD_CANDIDATES):
                continue
            predictions = {c: radbill_predict(data, inner_train, hold, c, target) for c in RAD_CANDIDATES}
            if any(predictions[c] is None for c in RAD_CANDIDATES):
                continue
            for candidate in RAD_CANDIDATES:
                pred, ysd = predictions[candidate]
                scores[target][candidate].append(abs(pred - p["target"][target]) / ysd)
                COUNTERS["inner_action_score_evaluations"] += 1
    means = {}
    for target in RAD_TARGETS:
        means[target] = {
            c: float(np.mean(scores[target][c])) if scores[target][c] else math.inf
            for c in RAD_CANDIDATES
        }
    SELECTION_CACHE[key] = means
    return means

def radbill_select(data: dict, train: list[int], resolution: str, context: dict) -> tuple[dict, list[dict]]:
    means = radbill_inner_score_cube(data, train)
    if resolution == "SHARED":
        blocks = {"ALL_EIGHT": (RAD_TARGETS, RAD_CANDIDATES)}
    elif resolution == "FAMILY":
        blocks = {
            family: ([t for t in RAD_TARGETS if RAD_FAMILY[t] == family], candidates)
            for family, candidates in RAD_FAMILY_CANDS.items()
        }
    elif resolution == "EXACT":
        blocks = {t: ([t], RAD_CANDIDATES) for t in RAD_TARGETS}
    else:
        raise ValueError(resolution)
    actions, rows = {}, []
    for block, (targets, candidates) in blocks.items():
        scores = {
            c: float(np.mean([means[t][c] for t in targets]))
            if all(math.isfinite(means[t][c]) for t in targets) else math.inf
            for c in candidates
        }
        chosen = min(candidates, key=lambda c: (scores[c], RAD_CANDIDATES.index(c)))
        actions[block] = chosen
        for candidate in candidates:
            rows.append({
                **context, "system_id": "RADBILL_HUMAN_PACING", "resolution_id": resolution,
                "block_or_target": block, "candidate_action": candidate,
                "inner_cv_score": scores[candidate], "selected": candidate == chosen,
                "n_inner_units": len(train), "tie_rule": "exact score tie; frozen A23 candidate order",
            })
    return actions, rows

def awinda_predict(data: dict, train: list[str], test: str, target: float, candidate_id: str | None) -> np.ndarray:
    key = ("awinda", tuple(sorted(train)), str(test), float(target), candidate_id)

    def compute():
        train_ix = np.concatenate([data["mouse_cell_ix"][m] for m in sorted(train)])
        test_ix = data["mouse_cell_ix"][str(test)]
        G = data["G"]
        if candidate_id is None:
            Xt, Xv = G[train_ix], G[test_ix]
            penalty = np.zeros(G.shape[1])
        else:
            ci = data["freq_ids"].index(candidate_id)
            features = data["X"][:, ci, :]
            mu = features[train_ix].mean(axis=0)
            sd = features[train_ix].std(axis=0, ddof=0)
            sd = np.where(sd == 0.0, 1.0, sd)
            Xt = np.column_stack([G[train_ix], (features[train_ix] - mu) / sd])
            Xv = np.column_stack([G[test_ix], (features[test_ix] - mu) / sd])
            penalty = np.r_[np.zeros(G.shape[1]), np.ones(2)]
        lhs = Xt.T @ Xt + np.diag(penalty)
        beta = np.linalg.solve(lhs, Xt.T @ data["Y"][target][train_ix])
        pred = Xv @ beta
        if not np.isfinite(pred).all():
            raise RuntimeError("Nonfinite Awinda prediction")
        return pred
    return make_fit_cache(PREDICTION_CACHE, key, compute)

def awinda_score_cube(data: dict, train: list[str]) -> np.ndarray:
    key = ("awinda_selector", tuple(sorted(train)))
    if key in SELECTION_CACHE:
        COUNTERS["selector_cache_hits"] += 1
        return SELECTION_CACHE[key]
    COUNTERS["selector_cache_misses"] += 1
    mice, ids = sorted(train), data["freq_ids"]
    cube = np.full((len(mice), len(ids), len(AW_TARGETS)), np.nan, dtype=float)
    for mi, hold in enumerate(mice):
        inner_train = [m for m in mice if m != hold]
        test_ix = data["mouse_cell_ix"][hold]
        for ci, candidate in enumerate(ids):
            for ti, target in enumerate(AW_TARGETS):
                pred = awinda_predict(data, inner_train, hold, target, candidate)
                obs = data["Y"][target][test_ix]
                denominator = np.mean(obs ** 2, axis=1)
                if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
                    raise RuntimeError("Awinda inner outcome spectrum has invalid RMS")
                per_cell = np.mean((pred - obs) ** 2, axis=1) / denominator
                cube[mi, ci, ti] = float(np.mean(per_cell))
                COUNTERS["inner_action_score_evaluations"] += 1
    if not np.isfinite(cube).all():
        raise RuntimeError("Awinda inner score support is incomplete")
    SELECTION_CACHE[key] = cube
    return cube

def awinda_select(data: dict, train: list[str], resolution: str, context: dict) -> tuple[dict, list[dict]]:
    cube = awinda_score_cube(data, train)
    ids = data["freq_ids"]
    lexical = sorted(range(len(ids)), key=lambda i: ids[i])
    shared_scores = np.mean(np.mean(cube, axis=2), axis=0)
    exact_scores = np.mean(cube, axis=0)
    rows, actions = [], {}
    if resolution == "SHARED":
        scores = {ids[i]: float(shared_scores[i]) for i in range(len(ids))}
        chosen = min(lexical, key=lambda i: (float(shared_scores[i]), ids[i]))
        actions["ALL_SIX"] = ids[chosen]
        for candidate in ids:
            rows.append({
                **context, "system_id": "AWINDA_MOUSE", "resolution_id": resolution,
                "block_or_target": "ALL_SIX", "candidate_action": candidate,
                "inner_cv_score": scores[candidate], "selected": candidate == ids[chosen],
                "n_inner_units": len(train), "tie_rule": "exact score tie; ascending candidate ID",
            })
    elif resolution == "EXACT":
        for ti, target in enumerate(AW_TARGETS):
            chosen = min(lexical, key=lambda i: (float(exact_scores[i, ti]), ids[i]))
            actions[str(target)] = ids[chosen]
            for candidate_i, candidate in enumerate(ids):
                rows.append({
                    **context, "system_id": "AWINDA_MOUSE", "resolution_id": resolution,
                    "block_or_target": str(target), "candidate_action": candidate,
                    "inner_cv_score": float(exact_scores[candidate_i, ti]),
                    "selected": candidate_i == chosen,
                    "n_inner_units": len(train), "tie_rule": "exact score tie; ascending candidate ID",
                })
    else:
        raise ValueError(resolution)
    COUNTERS["candidate_action_aggregate_scores"] += len(rows)
    return actions, rows

SELECTORS = {
    "RLC1_RAT": rlc_select,
    "RADBILL_HUMAN_PACING": radbill_select,
    "TANNER_RAT_TRABECULA": tanner_select,
    "AWINDA_MOUSE": awinda_select,
}

def action_for(actions: dict, resolution: str, target: object, system: str) -> object:
    if resolution == "SHARED":
        return actions["ALL_SIX" if system == "AWINDA_MOUSE" else ("ALL_EIGHT" if system == "RADBILL_HUMAN_PACING" else ("ALL_FOUR" if system == "TANNER_RAT_TRABECULA" else "ALL_SIX"))]
    if system == "RLC1_RAT" and resolution == "FAMILY":
        return actions[RLC_ACTION_FOR_TARGET[str(target)]]
    if system == "RADBILL_HUMAN_PACING" and resolution == "FAMILY":
        return actions[RAD_FAMILY[str(target)]]
    return actions[str(target)]

def predict_non_awinda(system: str, data: dict, train: list, test, resolution: str, actions: dict) -> list[dict]:
    rows = []
    if system == "RLC1_RAT":
        for target in TARGETS_RLC:
            candidate = action_for(actions, resolution, target, system)
            pred, ysd = rlc_fit_predict(data, train, test, candidate, target)
            actual = float(data["y"][target][data["units"].index(test)])
            rows.append({
                "outer_unit": test, "target": target, "candidate_action": candidate,
                "actual": actual, "predicted": pred, "training_target_sd": ysd,
                "loss": abs(actual - pred) / ysd, "metric": "standardized absolute error",
            })
    elif system == "TANNER_RAT_TRABECULA":
        for target in TANNER_TARGETS:
            candidate = int(action_for(actions, resolution, target, system))
            pred, slope, intercept = tanner_predict(data, train, test, candidate, target)
            actual = float(data["y"][target][test])
            rows.append({
                "outer_unit": test, "target": target, "candidate_action": candidate,
                "actual": actual, "predicted": pred, "fit_slope": slope, "fit_intercept": intercept,
                "loss": abs(actual - pred), "metric": "native absolute error",
            })
    elif system == "RADBILL_HUMAN_PACING":
        for target in RAD_TARGETS:
            if (int(test), target) not in data["support"]:
                continue
            candidate = str(action_for(actions, resolution, target, system))
            result = radbill_predict(data, train, int(test), candidate, target)
            if result is None:
                raise RuntimeError(f"Radbill selected action not estimable for fixed support: {test}/{target}/{candidate}")
            pred, ysd = result
            actual = float(data["ps"][int(test)]["target"][target])
            rows.append({
                "outer_unit": int(test), "target": target, "candidate_action": candidate,
                "actual": actual, "predicted": pred, "training_target_sd": ysd,
                "loss": abs(actual - pred) / ysd, "metric": "standardized absolute error",
            })
    else:
        raise ValueError(system)
    return rows

def predict_awinda(data: dict, train: list[str], test: str, resolution: str, actions: dict) -> tuple[list[dict], list[dict]]:
    rows, cell_rows = [], []
    test_ix = data["mouse_cell_ix"][str(test)]
    for target in AW_TARGETS:
        candidate = str(action_for(actions, resolution, target, "AWINDA_MOUSE"))
        pred = awinda_predict(data, train, str(test), target, candidate)
        obs = data["Y"][target][test_ix]
        denom = np.mean(obs ** 2, axis=1)
        nrmse = 100.0 * np.sqrt(np.mean((pred - obs) ** 2, axis=1)) / np.sqrt(denom)
        if not np.isfinite(nrmse).all():
            raise RuntimeError("Awinda middle/outer NRMSE nonfinite")
        for local, cell_ix in enumerate(test_ix):
            meta = data["cells"].iloc[cell_ix]
            cell_rows.append({
                "outer_unit": str(test), "target": target,
                "cell_index": int(cell_ix), "AnimalID": str(meta["AnimalID"]),
                "genotype": str(meta["Mutation_x"]), "drug_state": str(meta["drug_state"]),
                "group": str(meta["group"]), "candidate_action": candidate,
                "loss": float(nrmse[local]), "metric": "per-spectrum NRMSE percent",
            })
            for component_i in range(144):
                typ = "Re" if component_i < 72 else "Im"
                f_i = component_i if component_i < 72 else component_i - 72
                rows.append({
                    "heldout_mouse": str(test), "target_ATP_mM": target,
                    "selected_resolution": resolution, "candidate_id": candidate,
                    "AnimalID": str(meta["AnimalID"]), "genotype": str(meta["Mutation_x"]),
                    "drug_state": str(meta["drug_state"]), "group": str(meta["group"]),
                    "component_index": component_i, "component_type": typ,
                    "frequency_Hz": float(data["freq"][f_i]),
                    "observed": float(obs[local, component_i]), "predicted": float(pred[local, component_i]),
                })
    return rows, cell_rows

def middle_system_risk(system: str, rows: list[dict]) -> float:
    if system != "AWINDA_MOUSE":
        vals = [float(r["loss"]) for r in rows]
        if not vals:
            raise RuntimeError(f"Middle risk has no eligible rows: {system}")
        return float(np.mean(vals))
    by_target = []
    for target in AW_TARGETS:
        per_mouse = []
        for mouse in sorted({str(r["outer_unit"]) for r in rows if float(r["target"]) == target}):
            per_mouse.append(float(np.mean([r["loss"] for r in rows if str(r["outer_unit"]) == mouse and float(r["target"]) == target])))
        if not per_mouse:
            raise RuntimeError(f"Awinda middle support absent at target {target}")
        by_target.append(float(np.mean(per_mouse)))
    return float(np.mean(by_target))

def support_count(system: str, data: dict, unit) -> int:
    if system == "RADBILL_HUMAN_PACING":
        return sum((int(unit), t) in data["support"] for t in RAD_TARGETS)
    if system == "AWINDA_MOUSE":
        return len(data["mouse_cell_ix"][str(unit)]) * len(AW_TARGETS)
    if system == "RLC1_RAT":
        return len(TARGETS_RLC)
    return len(TANNER_TARGETS)

def exact_tied_coarser(system: str, risks: dict[str, float]) -> str:
    order = RESOLUTION_ORDER[system]
    min_risk = min(float(v) for v in risks.values())
    tied = [resolution for resolution in order if float(risks[resolution]) <= min_risk + 1e-12]
    return tied[0]

def recompute_radbill_strict_fixed_policies(data: dict) -> tuple[list[dict], list[dict]]:
    """Re-run the frozen fixed Radbill selectors on the amended 17-participant support."""
    policy_rows: list[dict] = []
    conventional = {"QT": "DTend_QT", "PP": "DTend_PP"}
    for outer in data["units"]:
        train = [unit for unit in data["units"] if unit != outer]
        selected = {}
        for resolution in ("SHARED", "FAMILY", "EXACT"):
            selected[resolution], _ = radbill_select(data, train, resolution, {"selection_layer": "STRICT_SUPPORT_FIXED_RECOMPUTE"})
        for target in RAD_TARGETS:
            if (outer, target) not in data["support"]:
                continue
            observed = float(data["ps"][outer]["target"][target])
            training_targets = [data["ps"][unit]["target"][target] for unit in train if data["ps"][unit]["target"][target] is not None]
            if len(training_targets) < 2:
                raise RuntimeError(f"Radbill strict fixed C0 has fewer than two training targets: {outer}/{target}")
            target_mean = float(np.mean(training_targets))
            target_sd = sample_sd(np.asarray(training_targets, dtype=float))
            if not math.isfinite(target_sd) or target_sd <= 1e-12:
                raise RuntimeError(f"Radbill strict fixed C0 has invalid training target SD: {outer}/{target}")
            predictions = {
                "C0": (None, target_mean, target_sd),
            }
            for resolution in ("SHARED", "FAMILY", "EXACT"):
                candidate = str(action_for(selected[resolution], resolution, target, "RADBILL_HUMAN_PACING"))
                result = radbill_predict(data, train, outer, candidate, target)
                if result is None:
                    raise RuntimeError(f"Radbill strict fixed policy is not estimable: {resolution}/{outer}/{target}/{candidate}")
                predictions[resolution] = (candidate, result[0], result[1])
            conv_candidate = conventional[RAD_FAMILY[target]]
            conv_result = radbill_predict(data, train, outer, conv_candidate, target)
            if conv_result is None:
                raise RuntimeError(f"Radbill strict conventional comparator is not estimable: {outer}/{target}/{conv_candidate}")
            predictions["CONVENTIONAL"] = (conv_candidate, conv_result[0], conv_result[1])
            for policy, (candidate, predicted, target_scale) in predictions.items():
                policy_rows.append({
                    "support_label": "A25_STRICT_SUPPORT_FIXED_POLICY_RECOMPUTE",
                    "policy_id": policy, "heldout_participant": outer, "target": target,
                    "candidate_action": candidate or "NO_ADDED_MEASUREMENT",
                    "observed_native": observed, "predicted_native": float(predicted),
                    "training_target_sd": float(target_scale),
                    "standardized_AE": abs(float(predicted) - observed) / float(target_scale),
                    "n_training_units": len(train), "n_training_target_values": len(training_targets),
                })
    frame = pd.DataFrame(policy_rows)
    expected_cells = {(unit, target) for unit, target in data["support"]}
    summaries = []
    for policy in ("C0", "SHARED", "FAMILY", "EXACT", "CONVENTIONAL"):
        subset = frame.loc[frame["policy_id"] == policy]
        observed_cells = set((int(r.heldout_participant), str(r.target)) for r in subset.itertuples(index=False))
        if len(subset) != len(expected_cells) or observed_cells != expected_cells or not np.isfinite(subset["standardized_AE"].to_numpy(float)).all():
            raise RuntimeError(f"Radbill strict fixed policy does not cover exactly the common 126-cell support: {policy}")
        summaries.append({
            "support_label": "A25_STRICT_SUPPORT_FIXED_POLICY_RECOMPUTE",
            "policy_id": policy, "n_biological_units": len(data["units"]),
            "n_target_cells": len(subset), "standardized_AE_risk": float(subset["standardized_AE"].mean()),
            "aggregation": "pooled mean standardized absolute error over the 126 target-valid participant-target cells",
        })
    return policy_rows, summaries

def unit_loss_from_predictions(system: str, records: list[dict]) -> list[dict]:
    if system != "AWINDA_MOUSE":
        return [{
            "unit": r["outer_unit"], "target": r["target"], "loss": float(r["loss"])
        } for r in records]
    grouped: dict[tuple, list] = defaultdict(list)
    for r in records:
        key = (
            str(r["heldout_mouse"]), float(r["target_ATP_mM"]), str(r["AnimalID"]),
            str(r["genotype"]), str(r["drug_state"]), str(r["group"]),
        )
        grouped[key].append((float(r["observed"]), float(r["predicted"])))
    rows = []
    for (mouse, target, animal, genotype, drug, group), vals in grouped.items():
        obs = np.asarray([v[0] for v in vals], dtype=float)
        pred = np.asarray([v[1] for v in vals], dtype=float)
        denom = float(np.mean(obs ** 2))
        if not math.isfinite(denom) or denom <= 0:
            raise RuntimeError("Invalid Awinda outer spectrum denominator during independent recomputation")
        loss = float(100.0 * np.sqrt(np.mean((pred - obs) ** 2)) / np.sqrt(denom))
        rows.append({"unit": mouse, "target": target, "cell": animal + "|" + group, "loss": loss})
    return rows

def native_risk(system: str, unit_target_rows: list[dict]) -> float:
    if not unit_target_rows:
        return math.nan
    if system == "AWINDA_MOUSE":
        target_risks = []
        for target in AW_TARGETS:
            mouse_risks = []
            for mouse in sorted({str(r["unit"]) for r in unit_target_rows if abs(float(r["target"]) - target) < 1e-12}):
                mouse_cells = [r["loss"] for r in unit_target_rows if str(r["unit"]) == mouse and abs(float(r["target"]) - target) < 1e-12]
                if mouse_cells:
                    mouse_risks.append(float(np.mean(mouse_cells)))
            if not mouse_risks:
                raise RuntimeError(f"Awinda held-out support absent for target {target}")
            target_risks.append(float(np.mean(mouse_risks)))
        return float(np.mean(target_risks))
    return float(np.mean([float(r["loss"]) for r in unit_target_rows]))

def middle_unit_loss(system: str, unit, rows: list[dict]) -> float | None:
    if system == "AWINDA_MOUSE":
        values = []
        for target in AW_TARGETS:
            cell_losses = [r["loss"] for r in rows if str(r["outer_unit"]) == str(unit) and abs(float(r["target"]) - target) < 1e-12]
            if cell_losses:
                values.append(float(np.mean(cell_losses)))
        return float(np.mean(values)) if values else None
    values = [float(r["loss"]) for r in rows if str(r["outer_unit"]) == str(unit)]
    return float(np.mean(values)) if values else None

def selected_actions_json(actions: dict) -> str:
    return json.dumps(actions, sort_keys=True, ensure_ascii=False, default=str)

def run_adaptive_system(system: str, data: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    units = data["units"]
    predictions, middle_trace, outer_selections, action_score_rows = [], [], [], []
    for outer in units:
        douter = [u for u in units if u != outer]
        assert outer not in douter
        resolution_risks: dict[str, float] = {}
        for resolution in RESOLUTION_ORDER[system]:
            middle_rows = []
            for middle in douter:
                dmid = [u for u in douter if u != middle]
                assert outer not in dmid and middle not in dmid
                context = {
                    "outer_unit": str(outer), "middle_unit": str(middle),
                    "selection_layer": "MIDDLE_ACTION_SELECTION",
                    "training_units_json": json.dumps(dmid, ensure_ascii=False),
                }
                actions, score_rows = SELECTORS[system](data, dmid, resolution, context)
                action_score_rows.extend(score_rows)
                action_map = selected_actions_json(actions)
                if system == "AWINDA_MOUSE":
                    component_rows, score_units = predict_awinda(data, dmid, str(middle), resolution, actions)
                    for r in component_rows:
                        r["outer_unit"] = str(outer)
                        r["outer_training_units_json"] = json.dumps(douter, ensure_ascii=False)
                        r["middle_training_units_json"] = json.dumps(dmid, ensure_ascii=False)
                        r["middle_action_map_json"] = action_map
                        r["middle_resolution_risk_context"] = resolution
                    # score_units records are unit-cell summaries for this held-out middle mouse.
                    middle_rows.extend(score_units)
                else:
                    score_units = predict_non_awinda(system, data, dmid, middle, resolution, actions)
                    middle_rows.extend(score_units)
                unit_risk = middle_unit_loss(system, middle, middle_rows)
                middle_trace.append({
                    "system_id": system, "outer_unit": str(outer), "candidate_resolution": resolution,
                    "middle_unit": str(middle), "outer_train_units_json": json.dumps(douter, ensure_ascii=False),
                    "middle_train_units_json": json.dumps(dmid, ensure_ascii=False),
                    "selected_action_map_json": action_map, "middle_unit_risk": unit_risk,
                    "n_middle_scored_cells": support_count(system, data, middle),
                })
            resolution_risks[resolution] = middle_system_risk(system, middle_rows)
        chosen_resolution = exact_tied_coarser(system, resolution_risks)
        context = {
            "outer_unit": str(outer), "middle_unit": "",
            "selection_layer": "OUTER_ACTION_SELECTION",
            "training_units_json": json.dumps(douter, ensure_ascii=False),
        }
        actions, score_rows = SELECTORS[system](data, douter, chosen_resolution, context)
        action_score_rows.extend(score_rows)
        action_map = selected_actions_json(actions)
        outer_selections.append({
            "system_id": system, "outer_unit": str(outer),
            "n_source_outer_units": len(units), "n_outer_scored_cells": support_count(system, data, outer),
            "selected_resolution": chosen_resolution, "middle_risks_json": json.dumps(resolution_risks, sort_keys=True),
            "selected_action_map_json": action_map,
            "outer_training_units_json": json.dumps(douter, ensure_ascii=False),
            "resolution_tie_rule": "minimum middle risk; within 1e-12 coarser registry order",
        })
        if system == "AWINDA_MOUSE":
            component_rows, _ = predict_awinda(data, douter, str(outer), chosen_resolution, actions)
            for r in component_rows:
                r["outer_unit"] = str(outer)
                r["selected_resolution"] = chosen_resolution
                r["selected_action_map_json"] = action_map
                r["outer_training_units_json"] = json.dumps(douter, ensure_ascii=False)
                r["middle_risks_json"] = json.dumps(resolution_risks, sort_keys=True)
            predictions.extend(component_rows)
        else:
            rows = predict_non_awinda(system, data, douter, outer, chosen_resolution, actions)
            for r in rows:
                r["system_id"] = system
                r["selected_resolution"] = chosen_resolution
                r["selected_action_map_json"] = action_map
                r["middle_risks_json"] = json.dumps(resolution_risks, sort_keys=True)
                r["outer_training_units_json"] = json.dumps(douter, ensure_ascii=False)
                r["outer_scored_support"] = True
            predictions.extend(rows)
    return predictions, middle_trace, outer_selections, action_score_rows

def resolutions_freq_rows(system: str, selections: list[dict], source_units: list) -> list[dict]:
    scored = [r for r in selections if int(r["n_outer_scored_cells"]) > 0]
    selected_all = Counter(r["selected_resolution"] for r in selections)
    selected_scored = Counter(r["selected_resolution"] for r in scored)
    out = []
    for resolution in RESOLUTION_ORDER[system]:
        out.append({
            "system_id": system, "resolution_id": resolution,
            "n_outer_folds_selected": selected_all[resolution], "n_outer_folds_total": len(source_units),
            "frequency_all_outer_folds": selected_all[resolution] / len(source_units),
            "n_scored_outer_units_selected": selected_scored[resolution], "n_scored_outer_units": len(scored),
            "frequency_scored_outer_units": selected_scored[resolution] / len(scored) if scored else "",
            "status": "PASS",
        })
    return out
