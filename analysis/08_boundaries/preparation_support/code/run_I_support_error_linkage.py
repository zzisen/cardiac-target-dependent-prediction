"""Preparation-paired, descriptive link between local support widths and errors."""
from __future__ import annotations
import csv, math, statistics
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))


ROOT = Path(__file__).resolve().parents[2]
BOUT = REPO / "analysis/08_boundaries/preparation_support/results"
IOUT = REPO / "analysis/08_boundaries/preparation_support/paired"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        midrank = (i + 1 + j) / 2.0
        for k in order[i:j]: result[k] = midrank
        i = j
    return result


def corr(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    if len(x) < 3: return None, None
    def pearson(a, b):
        ma, mb = statistics.mean(a), statistics.mean(b)
        da = [v - ma for v in a]; db = [v - mb for v in b]
        den = math.sqrt(sum(v*v for v in da) * sum(v*v for v in db))
        return sum(a*b for a, b in zip(da, db)) / den if den else None
    return pearson(x, y), pearson(ranks(x), ranks(y))


def mean(values): return statistics.mean(values) if values else float("nan")
def median(values): return statistics.median(values) if values else float("nan")


def main() -> None:
    IOUT.mkdir(parents=True, exist_ok=True)
    raw = read_csv(BOUT / "B_HUMAN_HELDOUT_RESULTS.csv")
    keyed = {(r["preparation_id"], r["target"], r["design"]): r for r in raw}
    results = []
    corr_rows = []
    for target in ("ATP1", "Pi10"):
        pairs = []
        for prep_id in sorted({r["preparation_id"] for r in raw}):
            baseline = keyed[(prep_id, target, "baseline_only")]
            augmented = keyed[(prep_id, target, "baseline_plus_ATP0.1_stress")]
            width_drop = float(baseline["support_width_percentage_points"]) - float(augmented["support_width_percentage_points"])
            abs_error_gain = float(baseline["absolute_error_pct"]) - float(augmented["absolute_error_pct"])
            cm_nrmse_gain = float(baseline["CM_target_NRMSE_pct_of_observed_RMS"]) - float(augmented["CM_target_NRMSE_pct_of_observed_RMS"])
            pairs.append({"preparation_id": prep_id, "group": baseline["group"], "target": target,
                "baseline_support_width_pp": float(baseline["support_width_percentage_points"]),
                "augmented_support_width_pp": float(augmented["support_width_percentage_points"]),
                "support_width_reduction_pp": width_drop,
                "baseline_absolute_stress_error_pp": float(baseline["absolute_error_pct"]),
                "augmented_absolute_stress_error_pp": float(augmented["absolute_error_pct"]),
                "absolute_stress_error_improvement_pp": abs_error_gain,
                "baseline_CM_target_NRMSE_pct": float(baseline["CM_target_NRMSE_pct_of_observed_RMS"]),
                "augmented_CM_target_NRMSE_pct": float(augmented["CM_target_NRMSE_pct_of_observed_RMS"]),
                "CM_target_NRMSE_improvement_pp": cm_nrmse_gain,
                "nominal_support_inclusion_baseline": baseline["observed_inside_nominal_support"].lower() == "true",
                "nominal_support_inclusion_augmented": augmented["observed_inside_nominal_support"].lower() == "true"})
        results.extend(pairs)
        groups = [("all_preparations", pairs), ("non-diabetic", [p for p in pairs if p["group"] == "non-diabetic"]),
                  ("diabetic", [p for p in pairs if p["group"] == "diabetic"])]
        for label, subset in groups:
            for outcome, key in (("absolute_stress_error_improvement", "absolute_stress_error_improvement_pp"),
                                 ("CM_target_NRMSE_improvement", "CM_target_NRMSE_improvement_pp")):
                pearson, spearman = corr([p["support_width_reduction_pp"] for p in subset], [p[key] for p in subset])
                corr_rows.append({"target": target, "subset": label, "n_preparations": len(subset),
                    "support_width_metric": "baseline width - augmented width (percentage points)",
                    "error_metric": outcome, "pearson_r": pearson, "spearman_rho": spearman,
                    "width_reduction_mean_pp": mean([p["support_width_reduction_pp"] for p in subset]),
                    "error_improvement_mean_pp": mean([p[key] for p in subset]),
                    "error_improvement_median_pp": median([p[key] for p in subset]),
                    "positive_width_reduction_count": sum(p["support_width_reduction_pp"] > 0 for p in subset),
                    "positive_error_improvement_count": sum(p[key] > 0 for p in subset),
                    "co_improvement_count": sum(p["support_width_reduction_pp"] > 0 and p[key] > 0 for p in subset),
                    "interpretation": "descriptive only; preparation-to-patient linkage unavailable"})
    with (IOUT / "I_PREPARATION_PAIRED_DATA.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    with (IOUT / "I_SUPPORT_ERROR_LINKAGE_RESULTS.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0])); w.writeheader(); w.writerows(corr_rows)
    make_report(raw, results, corr_rows)
    make_figure(results)
    print(f"paired_rows={len(results)} correlation_rows={len(corr_rows)}")


def make_report(raw, pairs, corrs):
    lines = ["# I. Preparation-paired support/error linkage", "",
        "This branch is now estimable as a descriptive preparation-level analysis because B produced held-out ATP1 and Pi10 outcomes for all 20 public trabeculae. It remains non-inferential: donor linkage is absent, multiple trabeculae may come from a donor, and the nominal profile support is not a calibrated interval.", "",
        "The paired exposure is baseline-only support width minus baseline-plus-ATP0.1-stress support width. Positive values mean the augmented local support is narrower. Error improvement is baseline absolute stress-response error minus augmented error; positive values mean smaller error after augmentation. A second outcome uses target complex-modulus NRMSE.", "",
        "## Target-level paired outcomes", "",
        "| Target | n | Prep count with narrower support | Stress absolute error improved | CM NRMSE improved | Nominal support inclusion baseline → augmented | MAE baseline → augmented (pp) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for target in ("ATP1", "Pi10"):
        subset = [p for p in pairs if p["target"] == target]
        raw0 = [r for r in raw if r["target"] == target and r["design"] == "baseline_only"]
        raw1 = [r for r in raw if r["target"] == target and r["design"] == "baseline_plus_ATP0.1_stress"]
        mae0 = mean([float(r["absolute_error_pct"]) for r in raw0]); mae1 = mean([float(r["absolute_error_pct"]) for r in raw1])
        inc0 = sum(p["nominal_support_inclusion_baseline"] for p in subset)
        inc1 = sum(p["nominal_support_inclusion_augmented"] for p in subset)
        cm_improved = sum(p["CM_target_NRMSE_improvement_pp"] > 0 for p in subset)
        lines.append(f"| {target} | 20 | {sum(p['support_width_reduction_pp'] > 0 for p in subset)}/20 | {sum(p['absolute_stress_error_improvement_pp'] > 0 for p in subset)}/20 | {cm_improved}/20 | {inc0}/20 → {inc1}/20 | {mae0:.2f} → {mae1:.2f} |")
    lines.extend(["", "## Width/error association", "",
        "| Target | Subset | n | Spearman ρ, width reduction vs stress-error improvement | Pearson r |", "|---|---|---:|---:|---:|"])
    for row in corrs:
        if row["error_metric"] == "absolute_stress_error_improvement":
            rho = "NA" if row["spearman_rho"] is None else f"{float(row['spearman_rho']):.3f}"
            pearson = "NA" if row["pearson_r"] is None else f"{float(row['pearson_r']):.3f}"
            lines.append(f"| {row['target']} | {row['subset']} | {row['n_preparations']} | {rho} | {pearson} |")
    lines.extend(["", "The correlation is an exploratory descriptive relationship, not calibration, coverage, a treatment effect, or proof that narrowing support improves prediction. We do not report p-values or confidence intervals because donor clusters are unknown. Group-specific summaries use ten preparations and are especially unstable.", "",
        "Reproducible outputs: `I_PREPARATION_PAIRED_DATA.csv`, `I_SUPPORT_ERROR_LINKAGE_RESULTS.csv`, and `I_SUPPORT_ERROR_LINKAGE.png`. Input rows are in B's public-data-derived held-out table; no patient identifiers are present.", ""])
    (IOUT / "I_SUPPORT_ERROR_LINKAGE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def make_figure(pairs):
    try: import matplotlib.pyplot as plt
    except ImportError: return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    colors = {"non-diabetic": "#237c9b", "diabetic": "#c45935"}
    for ax, target in zip(axes, ("ATP1", "Pi10")):
        subset = [p for p in pairs if p["target"] == target]
        for group, color in colors.items():
            values = [p for p in subset if p["group"] == group]
            ax.scatter([p["support_width_reduction_pp"] for p in values],
                       [p["absolute_stress_error_improvement_pp"] for p in values],
                       color=color, label=group, alpha=.82, edgecolor="white", linewidth=.35)
        ax.axhline(0, color="#777777", linewidth=.7); ax.axvline(0, color="#777777", linewidth=.7)
        ax.set_title(f"{target}: n=20 preparations")
        ax.set_xlabel("Support-width reduction (pp)")
        ax.set_ylabel("Absolute stress-error improvement (pp)")
        ax.grid(alpha=.18)
    axes[1].legend(frameon=False)
    fig.suptitle("Narrower local support did not map uniformly to lower held-out error")
    fig.tight_layout()
    fig.savefig(IOUT / "I_SUPPORT_ERROR_LINKAGE.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__": main()
