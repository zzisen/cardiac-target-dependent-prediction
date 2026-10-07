"""Independent reproduction of the two Sol T* benchmarks.

Uses only the Python standard library. It writes
TSTAR_FINAL_BENCHMARK_REPRODUCTION.csv beside this file.
Convention: D(n) = learned shared risk minus learned exact/fine risk.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


OUT = Path(__file__).with_name("TSTAR_FINAL_BENCHMARK_REPRODUCTION.csv")


def phi_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def quadratic(q, sigma):
    return sum(q[i] * sigma[i][j] * q[j] for i in range(len(q)) for j in range(len(q)))


def exact_components(q, delta, sigma, n):
    weighted = [q[i] * delta[i] for i in range(len(q))]
    d_shared = sum(weighted)
    v_shared = quadratic(q, sigma)
    oracle_gain = 0.5 * (sum(abs(x) for x in weighted) - abs(d_shared))
    if v_shared > 0.0 and d_shared != 0.0:
        e_shared = abs(d_shared) * phi_cdf(
            -math.sqrt(n) * abs(d_shared) / math.sqrt(v_shared)
        )
    else:
        e_shared = 0.0
    e_fine = sum(
        abs(weighted[i])
        * phi_cdf(-math.sqrt(n) * abs(delta[i]) / math.sqrt(sigma[i][i]))
        for i in range(len(q))
    )
    d_risk = oracle_gain + e_shared - e_fine
    return {
        "D_shared_minus_fine": d_risk,
        "oracle_gain": oracle_gain,
        "E_shared": e_shared,
        "E_fine": e_fine,
        "d_shared": d_shared,
        "v_shared": v_shared,
    }


def bisect_root(q, delta, sigma, left, right, iterations=100):
    f_left = exact_components(q, delta, sigma, left)["D_shared_minus_fine"]
    f_right = exact_components(q, delta, sigma, right)["D_shared_minus_fine"]
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if f_left * f_right >= 0.0:
        raise ValueError(f"Root is not bracketed: [{left}, {right}]")
    for _ in range(iterations):
        mid = (left + right) / 2.0
        f_mid = exact_components(q, delta, sigma, mid)["D_shared_minus_fine"]
        if f_left * f_mid <= 0.0:
            right, f_right = mid, f_mid
        else:
            left, f_left = mid, f_mid
    return (left + right) / 2.0


def scan_roots(q, delta, sigma, lower=1e-7, upper=1e3, steps=60000):
    log_lo = math.log(lower)
    log_hi = math.log(upper)
    roots = []
    prev_n = lower
    prev_value = exact_components(q, delta, sigma, prev_n)["D_shared_minus_fine"]
    for index in range(1, steps + 1):
        current_n = math.exp(log_lo + (log_hi - log_lo) * index / steps)
        current_value = exact_components(q, delta, sigma, current_n)["D_shared_minus_fine"]
        if prev_value * current_value < 0.0:
            roots.append(bisect_root(q, delta, sigma, prev_n, current_n))
        prev_n, prev_value = current_n, current_value
    return roots


def determinant_2(s):
    return s[0][0] * s[1][1] - s[0][1] * s[1][0]


def determinant_3(s):
    return (
        s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1])
        - s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0])
        + s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0])
    )


rows = []


def add(benchmark, statistic, value, reference="", tolerance="", note=""):
    if reference == "":
        status = "PASS"
        difference = ""
    else:
        difference = abs(float(value) - float(reference))
        status = "PASS" if difference <= float(tolerance) else "CHECK"
    rows.append({
        "benchmark": benchmark,
        "statistic": statistic,
        "independent_value": f"{value:.15g}" if isinstance(value, float) else str(value),
        "reference_value": reference,
        "absolute_difference": "" if difference == "" else f"{difference:.8g}",
        "tolerance": tolerance,
        "status": status,
        "notes": note,
    })
    if status != "PASS":
        raise AssertionError(f"{benchmark} {statistic}: mismatch {value} vs {reference}")


# B1: two-target crossover.
q1 = [0.8, 0.2]
delta1 = [0.5, -0.25]
sigma1 = [[0.25, -0.4], [-0.4, 1.0]]
det1 = determinant_2(sigma1)
if sigma1[0][0] <= 0.0 or det1 <= 0.0:
    raise AssertionError("B1 covariance is not positive definite")
add("B1", "weight_sum", sum(q1), "1.0", "1e-15", "Supplied weights.")
add("B1", "Sigma_determinant", det1, "0.09", "1e-14", "Sylvester criterion.")
b1_at_1 = exact_components(q1, delta1, sigma1, 1.0)
add("B1", "oracle_refinement_gain", b1_at_1["oracle_gain"], "0.05", "1e-14")
add("B1", "shared_variance_numerator", b1_at_1["v_shared"], "0.072", "1e-14")
roots1 = scan_roots(q1, delta1, sigma1)
if len(roots1) != 1:
    raise AssertionError(f"B1 expected one root in scan; found {roots1}")
add(
    "B1",
    "continuous_crossover_n",
    roots1[0],
    "0.9922849575458896",
    "1e-9",
    "D=shared risk minus exact risk; negative favors shared.",
)
for n, reference in ((0.1, "-0.00479263971745536"), (2.0, "0.0118384954803831")):
    value = exact_components(q1, delta1, sigma1, n)["D_shared_minus_fine"]
    add(
        "B1",
        f"D_at_n_{n:g}",
        value,
        reference,
        "1e-10",
        "Small n favors shared; later n favors exact.",
    )

# B2: three-target multiple-crossing benchmark.
q2 = [0.34032923, 0.03045246, 0.62921832]
delta2 = [-3.80674, 4.44024572, -4.54254381]
sigma2 = [
    [40.84734174, -3.40947653, 0.15188810],
    [-3.40947653, 0.39622421, -0.07107592],
    [0.15188810, -0.07107592, 0.06041956],
]
minor1 = sigma2[0][0]
minor2 = sigma2[0][0] * sigma2[1][1] - sigma2[0][1] ** 2
det2 = determinant_3(sigma2)
if not (minor1 > 0.0 and minor2 > 0.0 and det2 > 0.0):
    raise AssertionError("B2 covariance fails Sylvester's criterion")
add("B2_raw", "weight_sum_as_printed", sum(q2), "1.00000001", "1e-14",
    "Input decimals have a 1e-8 rounding excess.")
add("B2_raw", "Sigma_leading_minor_1", minor1, "", "",
    "Positive leading principal minor.")
add("B2_raw", "Sigma_leading_minor_2", minor2, "", "",
    "Positive leading principal minor.")
add("B2_raw", "Sigma_determinant", det2, "", "",
    "Positive; symmetric Sylvester criterion establishes positive definiteness.")
roots2 = scan_roots(q2, delta2, sigma2)
if len(roots2) != 2:
    raise AssertionError(f"B2 expected exactly two scanned roots; found {roots2}")
b2_base = exact_components(q2, delta2, sigma2, 1.0)
add("B2_raw", "oracle_refinement_gain", b2_base["oracle_gain"],
    "0.13521640517847144", "1e-10", "Weights used exactly as printed.")
add("B2_raw", "crossing_1", roots2[0], "0.6539384968639277", "1e-6")
add("B2_raw", "crossing_2", roots2[1], "4.447944456201809", "1e-6")
for n, reference in (
    (0.1, "0.7071220240900151"),
    (1.0, "-0.09113538514000064"),
    (2.0, "-0.1053554500180569"),
    (5.0, "0.01680806800613195"),
    (10.0, "0.0965901999866946"),
):
    value = exact_components(q2, delta2, sigma2, n)["D_shared_minus_fine"]
    add(
        "B2_raw",
        f"D_at_n_{n:g}",
        value,
        reference,
        "1e-9",
        "Sign sequence verifies two crossings and non-monotonicity.",
    )

# Renormalize the rounded q vector to honor sum(q)=1 exactly.
q2_norm = [x / sum(q2) for x in q2]
norm_gain = exact_components(q2_norm, delta2, sigma2, 1.0)["oracle_gain"]
norm_roots = scan_roots(q2_norm, delta2, sigma2)
add("B2_normalized", "weight_sum", sum(q2_norm), "1.0", "1e-15")
add("B2_normalized", "oracle_refinement_gain", norm_gain, "", "",
    "Uniform q normalization scales risks and gain; roots are invariant.")
add("B2_normalized", "crossing_1", norm_roots[0], "", "")
add("B2_normalized", "crossing_2", norm_roots[1], "", "")
add("B2_normalized", "root_change_1", norm_roots[0] - roots2[0], "0.0", "1e-10")
add("B2_normalized", "root_change_2", norm_roots[1] - roots2[1], "0.0", "1e-10")

with OUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "benchmark",
            "statistic",
            "independent_value",
            "reference_value",
            "absolute_difference",
            "tolerance",
            "status",
            "notes",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"{len(rows)} benchmark records: PASS")
print(f"B1 root={roots1[0]:.15g}")
print(f"B2 raw gain={b2_base['oracle_gain']:.15g}; roots={roots2}")
print(f"B2 normalized gain={norm_gain:.15g}; roots={norm_roots}")
print(f"CSV: {OUT}")
