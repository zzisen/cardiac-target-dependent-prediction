#!/usr/bin/env python
"""Reproducible numerical checks for the finite-action T* formulation."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "TSTAR_M_NUMERICAL_CHECKS.csv"
N_MC = 2_000_000
BATCH = 200_000
CDF_MAXPTS = 5_000_000
CDF_ABS_TOL = 1e-8
CDF_REL_TOL = 1e-8
INTEGRATION_ALLOWANCE = 2e-4


def penalty(d: float, v: float, n: float) -> float:
    if d == 0.0 or v == 0.0:
        return 0.0
    return abs(d) * norm.cdf(-np.sqrt(n) * abs(d) / np.sqrt(v))


def two_action_benchmark():
    q = np.array([0.8, 0.2], dtype=float)
    delta = np.array([0.5, -0.25], dtype=float)
    Sigma = np.array([[0.25, -0.4], [-0.4, 1.0]], dtype=float)
    eig = np.linalg.eigvalsh(Sigma)
    assert eig.min() > 0.0

    d_shared = float(q @ delta)
    v_shared = float(q @ Sigma @ q)
    d_exact = q * delta
    v_exact = (q ** 2) * np.diag(Sigma)
    oracle_gain = 0.5 * (np.abs(d_exact).sum() - abs(d_shared))
    assert np.isclose(d_shared, 0.35, atol=1e-14)
    assert np.isclose(v_shared, 0.072, atol=1e-14)
    assert np.allclose(d_exact, [0.4, -0.05], atol=1e-14)
    assert np.allclose(v_exact, [0.16, 0.04], atol=1e-14)
    assert np.isclose(oracle_gain, 0.05, atol=1e-14)

    def shared_minus_exact(n):
        return oracle_gain + penalty(d_shared, v_shared, n) - sum(
            penalty(float(d), float(v), n)
            for d, v in zip(d_exact, v_exact)
        )

    root = brentq(shared_minus_exact, 0.5, 1.5, xtol=1e-14, rtol=1e-14)
    expected_root = 0.9922849575458863
    assert abs(root - expected_root) < 5e-8

    rows = []
    root_diff = abs(root - expected_root)
    rows.append({
        "check_id": "M2_ROOT",
        "case": "Sol two-target shared-versus-exact benchmark",
        "quantity": "continuous crossover n",
        "n": "",
        "action_index": "",
        "analytic_value": f"{root:.12f}",
        "reference_value": f"{expected_root:.12f}",
        "monte_carlo_value": "",
        "mc_standard_error": "",
        "absolute_difference": f"{root_diff:.12g}",
        "tolerance": "5e-8",
        "pass": str(root_diff < 5e-8).upper(),
        "notes": "Closed-form normal-CDF risks; continuous n root."
    })

    refs = [
        (0.001, -0.000555),
        (0.01, -0.001732),
        (0.1, -0.004793),
        (0.5, -0.005030),
        (1.0, 0.000092),
        (2.0, 0.011838),
        (10.0, 0.038964),
    ]
    for n, ref in refs:
        value = shared_minus_exact(n)
        diff = abs(value - ref)
        tol = 1e-6
        assert diff <= tol, (n, value, ref, diff)
        rows.append({
            "check_id": f"M2_D_N_{n:g}",
            "case": "Sol two-target shared-versus-exact benchmark",
            "quantity": "Rbar_shared minus Rbar_exact",
            "n": f"{n:g}",
            "action_index": "",
            "analytic_value": f"{value:.12g}",
            "reference_value": f"{ref:.6f}",
            "monte_carlo_value": "",
            "mc_standard_error": "",
            "absolute_difference": f"{diff:.12g}",
            "tolerance": f"{tol:g}",
            "pass": "TRUE",
            "notes": "Published benchmark value is rounded to 6 decimals."
        })
    return rows, root


def orthant_choice_probabilities(r, V, n):
    """MVN integral for examples whose contrast law is nonsingular."""
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    m = len(r)
    probs = []
    for a in range(m):
        other = [b for b in range(m) if b != a]
        H = np.zeros((m - 1, m), dtype=float)
        for row, b in enumerate(other):
            H[row, b] = 1.0
            H[row, a] = -1.0
        mu = H @ r
        cov = H @ V @ H.T / n
        if np.linalg.eigvalsh(cov).min() <= 1e-12:
            raise ValueError("This CDF helper is reserved for nonsingular contrast examples.")
        # D = H Y; action a wins when D >= 0. Equivalently -D <= 0.
        p = multivariate_normal.cdf(
            np.zeros(m - 1),
            mean=-mu,
            cov=cov,
            allow_singular=False,
            maxpts=CDF_MAXPTS,
            abseps=CDF_ABS_TOL,
            releps=CDF_REL_TOL,
        )
        probs.append(float(p))
    probs = np.asarray(probs)
    if not np.isclose(probs.sum(), 1.0, atol=2e-5, rtol=0.0):
        raise AssertionError(f"MVN action probabilities sum to {probs.sum()}")
    return probs


def simulate_choices(r, V, n, draws, seed):
    rng = np.random.default_rng(seed)
    counts = np.zeros(len(r), dtype=np.int64)
    remaining = draws
    while remaining:
        size = min(BATCH, remaining)
        Y = rng.multivariate_normal(
            mean=np.asarray(r, dtype=float),
            cov=np.asarray(V, dtype=float) / n,
            size=size,
            check_valid="raise",
        )
        chosen = np.argmin(Y, axis=1)  # fixed priority: lower action index wins ties
        counts += np.bincount(chosen, minlength=len(r))
        remaining -= size
    return counts / draws


def probability_rows(case, r, V, n, seed, draws=N_MC, tied_oracle=False):
    r = np.asarray(r, dtype=float)
    V = np.asarray(V, dtype=float)
    assert np.linalg.eigvalsh(V).min() > 1e-10
    exact = orthant_choice_probabilities(r, V, n)
    simulated = simulate_choices(r, V, n, draws, seed)
    rows = []
    for a, (p, f) in enumerate(zip(exact, simulated), start=1):
        se = float(np.sqrt(p * (1.0 - p) / draws))
        tol = 6.0 * se + INTEGRATION_ALLOWANCE
        diff = abs(float(p - f))
        assert diff <= tol, (case, a, p, f, se, tol, diff)
        rows.append({
            "check_id": f"{case}_ACTION_{a}",
            "case": case,
            "quantity": "selection probability",
            "n": f"{n:g}",
            "action_index": a,
            "analytic_value": f"{p:.12g}",
            "reference_value": "",
            "monte_carlo_value": f"{f:.12g}",
            "mc_standard_error": f"{se:.12g}",
            "absolute_difference": f"{diff:.12g}",
            "tolerance": f"{tol:.12g}",
            "pass": "TRUE",
            "notes": f"N={draws}; six binomial SE plus {INTEGRATION_ALLOWANCE:g} integration allowance."
        })
    if tied_oracle:
        if not np.isclose(r[0], r[1]) or r[2] <= r[0]:
            raise AssertionError("Tie case must have two tied oracle actions and one worse action.")
        if abs(exact[0] - exact[1]) > 2e-5:
            raise AssertionError("Symmetric tied-oracle actions should have matching selection probabilities.")
        oracle_excess = float((r - r.min()) @ exact)
        mc_excess = float((r - r.min()) @ simulated)
        excess_se = float((r[2] - r.min()) * np.sqrt(exact[2] * (1.0 - exact[2]) / draws))
        tol = 6.0 * excess_se + INTEGRATION_ALLOWANCE
        diff = abs(oracle_excess - mc_excess)
        assert diff <= tol
        rows.append({
            "check_id": f"{case}_EXCESS",
            "case": case,
            "quantity": "expected excess risk",
            "n": f"{n:g}",
            "action_index": "",
            "analytic_value": f"{oracle_excess:.12g}",
            "reference_value": "",
            "monte_carlo_value": f"{mc_excess:.12g}",
            "mc_standard_error": f"{excess_se:.12g}",
            "absolute_difference": f"{diff:.12g}",
            "tolerance": f"{tol:.12g}",
            "pass": "TRUE",
            "notes": "The two tied oracle actions each have zero gap and contribute zero excess."
        })
    return rows, exact, simulated


def singular_case_rows():
    # V has rank one: estimates for actions 1 and 2 are identical N(0,1);
    # action 3 is deterministic at 0.5. Priority selects action 1 on the
    # positive-mass tie between actions 1 and 2.
    r = np.array([0.0, 0.0, 0.5])
    V = np.array([[1.0, 1.0, 0.0],
                  [1.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0]])
    assert np.linalg.eigvalsh(V).min() >= -1e-12
    n = 1.0
    p1 = float(norm.cdf(0.5 * np.sqrt(n)))
    p2 = 0.0
    p3 = float(norm.cdf(-0.5 * np.sqrt(n)))
    assert np.isclose(p1 + p2 + p3, 1.0, atol=1e-14)
    probs = [p1, p2, p3]
    rows = []
    for a, p in enumerate(probs, start=1):
        rows.append({
            "check_id": f"SINGULAR_ACTION_{a}",
            "case": "singular PSD with permanent estimator tie",
            "quantity": "selection probability",
            "n": f"{n:g}",
            "action_index": a,
            "analytic_value": f"{p:.12g}",
            "reference_value": "",
            "monte_carlo_value": "",
            "mc_standard_error": "",
            "absolute_difference": "0",
            "tolerance": "0",
            "pass": "TRUE",
            "notes": "Exact 1D reduction; actions 1 and 2 tie on every draw; priority gives the tie to action 1."
        })
    excess = 0.5 * p3
    rows.append({
        "check_id": "SINGULAR_EXCESS",
        "case": "singular PSD with permanent estimator tie",
        "quantity": "expected excess risk",
        "n": f"{n:g}",
        "action_index": "",
        "analytic_value": f"{excess:.12g}",
        "reference_value": "",
        "monte_carlo_value": "",
        "mc_standard_error": "",
        "absolute_difference": "0",
        "tolerance": "0",
        "pass": "TRUE",
        "notes": "Only action 3 is non-oracle and its true gap is 0.5."
    })
    return rows


def main():
    rows, root = two_action_benchmark()
    print(f"M=2 Sol crossover: {root:.12f}")

    A3 = np.array([[1.0, 0.0, 0.0],
                   [0.30, 0.80, 0.0],
                   [-0.20, 0.25, 0.70]])
    V3 = A3 @ A3.T
    r3 = np.array([0.20, 0.00, 0.16])
    r3_rows, p3, f3 = probability_rows(
        "M3_POSITIVE_DEFINITE", r3, V3, n=1.0, seed=301606
    )
    rows.extend(r3_rows)
    print("M=3 probabilities:", np.round(p3, 8), "MC:", np.round(f3, 8))

    A4 = np.array([[1.00, 0.00, 0.00, 0.00],
                   [0.35, 0.85, 0.00, 0.00],
                   [-0.15, 0.25, 0.75, 0.00],
                   [0.20, -0.10, 0.25, 0.65]])
    V4 = A4 @ A4.T
    r4 = np.array([0.14, 0.04, 0.12, 0.22])
    r4_rows, p4, f4 = probability_rows(
        "M4_POSITIVE_DEFINITE", r4, V4, n=1.0, seed=401606
    )
    rows.extend(r4_rows)
    print("M=4 probabilities:", np.round(p4, 8), "MC:", np.round(f4, 8))

    rt = np.array([0.0, 0.0, 0.4])
    Vt = np.eye(3)
    tie_rows, pt, ft = probability_rows(
        "TIED_ORACLE_M3", rt, Vt, n=1.0, seed=501606, tied_oracle=True
    )
    rows.extend(tie_rows)
    print("Tied-oracle probabilities:", np.round(pt, 8), "MC:", np.round(ft, 8))

    rows.extend(singular_case_rows())

    fields = [
        "check_id", "case", "quantity", "n", "action_index",
        "analytic_value", "reference_value", "monte_carlo_value",
        "mc_standard_error", "absolute_difference", "tolerance", "pass", "notes"
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if row["pass"] != "TRUE"]
    if failed:
        raise SystemExit(f"{len(failed)} numerical checks failed.")
    print(f"Wrote {len(rows)} passing checks to {CSV_PATH}")


if __name__ == "__main__":
    main()
