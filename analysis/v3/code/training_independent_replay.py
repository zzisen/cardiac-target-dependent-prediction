#!/usr/bin/env python3
"""Independent B1 replay from the canonical summary CSV.

This implementation uses numpy.linalg.lstsq rather than the analytic scalar
OLS routine in main_execution.py. It reads no FCS data and imports no code from
the main execution.

Usage:
  python independent_replay.py [--work-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


PATIENTS = [
    "UPN1", "UPN2", "UPN3", "UPN4", "UPN5", "UPN6", "UPN7", "UPN8", "UPN9",
    "UPN11", "UPN12", "UPN13", "UPN14", "UPN15", "UPN16", "UPN17", "UPN18", "UPN19",
    "UPN20", "UPN21", "UPN23", "UPN24", "UPN25", "UPN26", "UPN27", "UPN28", "UPN29",
    "UPN30", "UPN31", "UPN47", "UPN48", "UPN49", "UPN50", "UPN51", "UPN52", "UPN53",
    "UPN54", "UPN55", "UPN56", "UPN57", "UPN58", "UPN67", "UPN68", "UPN69",
]
CHANNELS = [
    "pS6", "p4EBP1", "pSTAT5", "pSYK",
    "pPLC\u03b32", "pAKT", "pERK1/2", "pIKAROS",
]
TARGETS = [
    "BCR-Crosslink", "BEZ-235", "Dasatinib", "IL-7",
    "Pervanadate", "TSLP", "Tofacitinib",
]
POLICIES = ["C0", "SHARED", "EXACT"]
ATOL = 1e-12
RTOL = 1e-10


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def scaled_fit(x_train: np.ndarray, y_train: np.ndarray, x_test: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve the standardized intercept OLS model with one predictor."""
    x_mean = np.mean(x_train, dtype=np.float64)
    x_sd = np.std(x_train, ddof=1, dtype=np.float64)
    if x_sd == 0.0:
        x_sd = 1.0
    y_mean = np.mean(y_train, axis=0, dtype=np.float64)
    y_sd = np.std(y_train, axis=0, ddof=1, dtype=np.float64)
    y_sd = np.where(y_sd == 0.0, 1.0, y_sd)
    xz = (x_train - x_mean) / x_sd
    yz = (y_train - y_mean) / y_sd
    design = np.column_stack((np.ones(len(x_train), dtype=np.float64), xz))
    beta, _, _, _ = np.linalg.lstsq(design, yz, rcond=None)
    xtest_z = np.asarray([1.0, (float(x_test) - x_mean) / x_sd], dtype=np.float64)
    pred_z = xtest_z @ beta
    pred_raw = y_mean + y_sd * pred_z
    return pred_z, pred_raw


def load_matrices(work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = read_csv(work_dir / "B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv")
    expected_keys = {
        (p, condition, channel)
        for p in PATIENTS
        for condition in ["Basal"] + TARGETS
        for channel in CHANNELS
    }
    seen: set[tuple[str, str, str]] = set()
    vals: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["patient_id"], row["condition"], row["channel"])
        if key in seen:
            raise RuntimeError(f"duplicate summary key: {key}")
        seen.add(key)
        if key not in expected_keys:
            raise RuntimeError(f"unexpected summary key: {key}")
        value = float(row["positive_fraction"])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"invalid fraction at {key}: {value}")
        vals[key] = value
    if seen != expected_keys or len(rows) != 44 * 8 * 8:
        raise RuntimeError("summary coverage does not equal 44 x 8 x 8")
    x = np.empty((44, 8), dtype=np.float64)
    y = np.empty((44, 7, 8), dtype=np.float64)
    pidx = {p: i for i, p in enumerate(PATIENTS)}
    cidx = {c: i for i, c in enumerate(CHANNELS)}
    tidx = {t: i for i, t in enumerate(TARGETS)}
    for p in PATIENTS:
        for c in CHANNELS:
            x[pidx[p], cidx[c]] = vals[(p, "Basal", c)]
        for t in TARGETS:
            for c in CHANNELS:
                y[pidx[p], tidx[t], cidx[c]] = vals[(p, t, c)]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("independent replay matrix contains non-finite values")
    return x, y


def replay(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    n = len(PATIENTS)
    losses = np.empty((n, len(TARGETS), 3), dtype=np.float64)
    shared_rows: list[dict[str, object]] = []
    exact_rows: list[dict[str, object]] = []
    for outer_test in range(n):
        outer_train = np.asarray([i for i in range(n) if i != outer_test], dtype=int)
        score_samples = np.empty((len(outer_train), 8, 7), dtype=np.float64)
        for inner_position, inner_test in enumerate(outer_train):
            inner_train = outer_train[outer_train != inner_test]
            if len(inner_train) != 42 or inner_test in inner_train or outer_test in inner_train:
                raise RuntimeError("independent replay patient-fold isolation failed")
            for ti in range(7):
                y_train = y[inner_train, ti, :]
                y_test = y[inner_test, ti, :]
                y_mean = np.mean(y_train, axis=0, dtype=np.float64)
                y_sd = np.std(y_train, axis=0, ddof=1, dtype=np.float64)
                y_sd = np.where(y_sd == 0.0, 1.0, y_sd)
                y_test_z = (y_test - y_mean) / y_sd
                for ci in range(8):
                    pred_z, _ = scaled_fit(x[inner_train, ci], y_train, x[inner_test, ci])
                    score_samples[inner_position, ci, ti] = np.mean(
                        (y_test_z - pred_z) ** 2, dtype=np.float64
                    )
        if not np.isfinite(score_samples).all():
            raise RuntimeError("independent inner score is non-finite")
        scores_by_target = np.mean(score_samples, axis=0, dtype=np.float64)
        shared_scores = np.mean(scores_by_target, axis=1, dtype=np.float64)
        shared_choice = int(np.argmin(shared_scores))
        exact_choices = np.argmin(scores_by_target, axis=0)
        shared_rows.append({
            "outer_fold": outer_test + 1,
            "held_out_patient": PATIENTS[outer_test],
            "selected_shared_channel": CHANNELS[shared_choice],
        })
        for ti, target in enumerate(TARGETS):
            exact_rows.append({
                "outer_fold": outer_test + 1,
                "held_out_patient": PATIENTS[outer_test],
                "target": target,
                "selected_exact_channel": CHANNELS[int(exact_choices[ti])],
            })
            outer_train_y = y[outer_train, ti, :]
            y_mean = np.mean(outer_train_y, axis=0, dtype=np.float64)
            y_sd = np.std(outer_train_y, axis=0, ddof=1, dtype=np.float64)
            y_sd = np.where(y_sd == 0.0, 1.0, y_sd)
            y_test_z = (y[outer_test, ti, :] - y_mean) / y_sd
            pred_c0_z = np.zeros(8, dtype=np.float64)
            pred_shared_z, _ = scaled_fit(
                x[outer_train, shared_choice], outer_train_y, x[outer_test, shared_choice]
            )
            exact_choice = int(exact_choices[ti])
            pred_exact_z, _ = scaled_fit(
                x[outer_train, exact_choice], outer_train_y, x[outer_test, exact_choice]
            )
            for pi, pred_z in enumerate([pred_c0_z, pred_shared_z, pred_exact_z]):
                losses[outer_test, ti, pi] = np.mean(
                    (y_test_z - pred_z) ** 2, dtype=np.float64
                )
    return losses, shared_rows, exact_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    x, y = load_matrices(work_dir)
    replay_losses, replay_shared, replay_exact = replay(x, y)
    summary_rows = read_csv(work_dir / "B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv")
    main_loss_rows = read_csv(work_dir / "B1_OUTER_PATIENT_TARGET_LOSSES.csv")
    main_shared_rows = read_csv(work_dir / "B1_SHARED_SELECTION_MAP.csv")
    main_exact_rows = read_csv(work_dir / "B1_EXACT_SELECTION_MAP.csv")
    expected_loss_keys = {(p, t) for p in PATIENTS for t in TARGETS}
    main_loss_map = {(r["patient_id"], r["target"]): r for r in main_loss_rows}
    if set(main_loss_map) != expected_loss_keys or len(main_loss_rows) != 308:
        raise RuntimeError("main loss table does not contain the exact 308 patient-target cells")

    action_mismatches: list[str] = []
    shared_main = {int(r["outer_fold"]): r["selected_shared_channel"] for r in main_shared_rows}
    for row in replay_shared:
        if shared_main.get(int(row["outer_fold"])) != row["selected_shared_channel"]:
            action_mismatches.append(f"shared fold {row['outer_fold']}")
    exact_main = {
        (int(r["outer_fold"]), r["target"]): r["selected_exact_channel"]
        for r in main_exact_rows
    }
    for row in replay_exact:
        key = (int(row["outer_fold"]), str(row["target"]))
        if exact_main.get(key) != row["selected_exact_channel"]:
            action_mismatches.append(f"exact fold-target {key}")

    loss_differences: list[float] = []
    matched_policy_cells = 0
    for pi, policy in enumerate(POLICIES):
        column = "loss_" + policy
        for patient_i, patient in enumerate(PATIENTS):
            for target_i, target in enumerate(TARGETS):
                main_value = float(main_loss_map[(patient, target)][column])
                replay_value = float(replay_losses[patient_i, target_i, pi])
                loss_differences.append(abs(main_value - replay_value))
                if np.isclose(main_value, replay_value, rtol=RTOL, atol=ATOL):
                    matched_policy_cells += 1
    replay_risks = np.mean(replay_losses, axis=(0, 1), dtype=np.float64)
    main_risk_rows = {
        r["policy"]: float(r["overall_equal_patient_equal_target_risk"])
        for r in read_csv(work_dir / "B1_PRIMARY_RISK_SUMMARY.csv")
    }
    risk_mismatches = {
        policy: {"main": main_risk_rows[policy], "replay": float(replay_risks[i])}
        for i, policy in enumerate(POLICIES)
        if not np.isclose(main_risk_rows[policy], replay_risks[i], rtol=RTOL, atol=ATOL)
    }
    main_contrasts = {
        r["contrast"]: float(r["risk_difference"])
        for r in read_csv(work_dir / "B1_PRIMARY_CONTRASTS.csv")
    }
    replay_contrasts = {
        "C0_minus_SHARED": float(replay_risks[0] - replay_risks[1]),
        "C0_minus_EXACT": float(replay_risks[0] - replay_risks[2]),
        "SHARED_minus_EXACT": float(replay_risks[1] - replay_risks[2]),
    }
    contrast_mismatches = {
        key: {"main": main_contrasts[key], "replay": value}
        for key, value in replay_contrasts.items()
        if not np.isclose(main_contrasts[key], value, rtol=RTOL, atol=ATOL)
    }
    max_abs_loss_difference = max(loss_differences, default=float("nan"))
    pass_replay = (
        not action_mismatches and not risk_mismatches and not contrast_mismatches
        and len(main_loss_rows) == 308 and matched_policy_cells == 924
    )
    status = "PASS" if pass_replay else "FAIL"
    lines = [
        "# B1 independent replay report",
        "",
        f"## Status: {status}",
        "",
        "The replay starts from B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv and uses a separate implementation based on numpy.linalg.lstsq. It does not import the main execution code or read FCS files.",
        "",
        f"- Summary input rows: {len(summary_rows)} (expected 2,816).",
        f"- Main outer loss rows: {len(main_loss_rows)}/308.",
        f"- Shared selected actions identical: {44-len([x for x in action_mismatches if x.startswith('shared')])}/44.",
        f"- Exact selected actions identical: {308-len([x for x in action_mismatches if x.startswith('exact')])}/308.",
        f"- Matched policy loss values: {matched_policy_cells}/924 within rtol={RTOL:g}, atol={ATOL:g}.",
        f"- Maximum absolute per-policy loss difference: {max_abs_loss_difference:.17g}.",
        f"- Primary risks matched: {not risk_mismatches}.",
        f"- Primary contrasts matched: {not contrast_mismatches}.",
        "",
        "## Replayed risks",
        "",
        "| Policy | Main execution | Independent replay |",
        "|---|---:|---:|",
    ]
    for i, policy in enumerate(POLICIES):
        lines.append(f"| {policy} | {main_risk_rows[policy]:.17g} | {replay_risks[i]:.17g} |")
    lines += [
        "",
        "## Replayed contrasts",
        "",
        "| Contrast | Main execution | Independent replay |",
        "|---|---:|---:|",
    ]
    for key in ["C0_minus_SHARED", "C0_minus_EXACT", "SHARED_minus_EXACT"]:
        lines.append(f"| {key} | {main_contrasts[key]:.17g} | {replay_contrasts[key]:.17g} |")
    lines += [
        "",
        f"Action mismatches: {json.dumps(action_mismatches, ensure_ascii=False)}",
        f"Risk mismatches: {json.dumps(risk_mismatches, ensure_ascii=False)}",
        f"Contrast mismatches: {json.dumps(contrast_mismatches, ensure_ascii=False)}",
        "",
        "PASS means the frozen selectors and held-out predictions/losses reproduce; it does not depend on whether the outcome is favorable.",
        "",
    ]
    (work_dir / "B1_INDEPENDENT_REPLAY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if pass_replay else 3


if __name__ == "__main__":
    raise SystemExit(main())
