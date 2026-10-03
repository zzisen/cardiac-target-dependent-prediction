"""Aggregate the frozen N model ensemble and K parameter domains into R panels.

R is deliberately a local design-sensitivity analysis. It does not judge model
adequacy. Model scenarios come from N's frozen five-variant set; two separate
Model16D support-domain scenarios come from the frozen A/K +0.5 and +1.0 boxes.
Those support boxes were only defined for Model16D, so a full model x domain
factorial is not asserted.
"""
from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RDIR = HERE.parent
ROOT = RDIR.parent
BASE = ROOT / "PRIOR_A_TO_P_BASELINE"
KRES = REPO / "analysis/01_core_target_value/sparse/results"
NRES = REPO / "analysis/02_robustness/variants/results"
OUT = RDIR / "results"
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = ["ATP0.1", "ATP1", "Pi0", "Pi5"]
CONDITIONS = ["ATP0.1", "ATP1", "Pi0", "Pi5"]
COMMON_RAT_HUMAN_AWINDA_HZ = [1.0, 3.1622776601683795, 10.0, 31.622776601683793]
FREQ_TOL = 1e-4
RCOND = 1e-11


def panel_variance(V: np.ndarray, g: np.ndarray, R: np.ndarray) -> float:
    base = max(0.0, float(g @ V @ g))
    if not len(R):
        return base
    A = np.eye(R.shape[0]) + R @ V @ R.T
    c = R @ V @ g
    red = float(c @ np.linalg.solve(A, c))
    return max(0.0, base - max(0.0, red))


def read_scenario_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(KRES / "K_BASELINE_SENSITIVITIES.csv"),
        pd.read_csv(KRES / "K_LOCAL_SENSITIVITIES.csv"),
        pd.read_csv(KRES / "K_TARGET_GRADIENTS.csv"),
        pd.read_csv(NRES / "N_BASELINE_SENSITIVITIES.csv"),
        pd.read_csv(NRES / "N_LOCAL_SENSITIVITIES.csv"),
        pd.read_csv(NRES / "N_TARGET_GRADIENTS.csv"),
    )


def pivot_matrix(df: pd.DataFrame, row_col: str, coord_col: str, value_col: str) -> np.ndarray:
    p = df.pivot(index=row_col, columns=coord_col, values=value_col).sort_index(axis=0).sort_index(axis=1)
    return p.to_numpy(float)


def scalar_sensitivity_map(df: pd.DataFrame, coord_col: str, value_col: str) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    vectors: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for scalar, g in df.groupby("scalar_id", sort=False):
        g = g.sort_values(coord_col)
        vectors[str(scalar)] = g[value_col].to_numpy(float)
        meta[str(scalar)] = {c: g.iloc[0][c] for c in g.columns if c not in {"parameter_index", "local_coordinate", value_col}}
    return vectors, meta


def make_actions(scalars: dict[str, np.ndarray], meta: dict[str, dict], target: str,
                 condition_col: str, type_col: str, frequency_col: str | None = None) -> list[dict]:
    actions: list[dict] = []
    grouped: dict[tuple[str, str, float | None], dict[str, np.ndarray]] = {}
    details: dict[tuple[str, str, float | None], dict] = {}
    for sid, row in scalars.items():
        md = meta[sid]
        cond = str(md[condition_col])
        if cond == target:
            continue
        typ = str(md[type_col])
        if typ == "stress":
            key = ("stress", cond, None)
            component = "stress"
        else:
            match = re.match(r"^CM_(.+)_([0-9.]+)Hz_(Re|Im)$", sid)
            if not match:
                continue
            cond_from_id, freq_text, component = match.groups()
            f = float(freq_text)
            key = ("cm", cond_from_id, f)
        grouped.setdefault(key, {})[component] = scalars[sid]
        details[key] = {"type": typ, "condition": cond if typ == "stress" else cond_from_id,
                        "frequency_Hz": None if typ == "stress" else f}
    for key, vals in grouped.items():
        kind, cond, f = key
        if kind == "cm" and not {"Re", "Im"}.issubset(vals):
            continue
        if kind == "stress":
            action_id = f"stress_{cond}"
            rows = np.asarray(vals["stress"], float)[None, :]
            typ = "stress"
        else:
            action_id = f"CM_{cond}_{f:g}Hz"
            rows = np.vstack([vals["Re"], vals["Im"]])
            typ = "CM_frequency_pair"
        actions.append({"action_id": action_id, "type": typ, "condition": cond,
                        "frequency_Hz": f, "rows": rows})
    return sorted(actions, key=lambda a: a["action_id"])


def load_scenarios() -> tuple[list[dict], pd.DataFrame]:
    kb, kc, kg, nb, nc, ng = read_scenario_inputs()
    scenarios: list[dict] = []
    table_rows = []
    for domain in sorted(kb["domain"].unique()):
        scenario_id = f"Model16D|{domain}"
        scenarios.append({"scenario_id": scenario_id, "scenario_family": "A_K_parameter_domain",
                          "model": "published_Model16D", "domain": domain,
                          "base": kb[kb["domain"].eq(domain)], "cand": kc[kc["domain"].eq(domain)],
                          "grad": kg[kg["domain"].eq(domain)], "coord_col": "parameter_index",
                          "value_col": "sensitivity", "cand_value_col": "whitened_sensitivity",
                          "condition_col": "condition", "type_col": "measurement_type"})
        table_rows.append({"scenario_id": scenario_id, "scenario_family": "A_K_parameter_domain",
                           "model": "published_Model16D", "parameter_domain": domain,
                           "domain_source": "frozen A/K P1-pair-derived support box",
                           "baseline_fit_source": "A_measurement_target_map.json; frozen before K",
                           "crossed_with_all_model_variants": False,
                           "scope_note": "valid only for the published Model16D support construction"})
    fitfile = pd.read_csv(REPO / "analysis/02_robustness/variants/results" / "N_VARIANT_FITS.csv")
    for variant in fitfile["variant"].tolist():
        scenario_id = f"{variant}|published_optimizer_box"
        scenarios.append({"scenario_id": scenario_id, "scenario_family": "N_model_variant",
                          "model": variant, "domain": "published_optimizer_box",
                          "base": nb[nb["variant"].eq(variant)], "cand": nc[nc["variant"].eq(variant)],
                          "grad": ng[ng["variant"].eq(variant)], "coord_col": "local_coordinate",
                          "value_col": "sensitivity", "cand_value_col": "whitened_sensitivity",
                          "condition_col": "condition", "type_col": "type"})
        fit = fitfile[fitfile["variant"].eq(variant)].iloc[0]
        table_rows.append({"scenario_id": scenario_id, "scenario_family": "N_model_variant",
                           "model": variant, "parameter_domain": "published_optimizer_box",
                           "domain_source": "N preselected model-specific published fitting bounds",
                           "baseline_fit_source": "N frozen baseline-only multistart fit; 8 starts per variant",
                           "crossed_with_all_model_variants": False,
                           "n_successful_starts": int(fit["successful_starts"]),
                           "scope_note": "distinct published permutations; local design sensitivity only; no adequacy test"})
    return scenarios, pd.DataFrame(table_rows)


def scenario_target_matrices(sc: dict, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    base = sc["base"]
    cand = sc["cand"]
    grad = sc["grad"]
    if sc["scenario_family"] == "A_K_parameter_domain":
        b = base[base["target"].eq(target)]
        c = cand[cand["target"].eq(target)]
        q = grad[grad["target"].eq(target)].sort_values("parameter_index")
        bmat = pivot_matrix(b, "row_id", sc["coord_col"], sc["value_col"])
        vectors, meta = scalar_sensitivity_map(c, sc["coord_col"], sc["cand_value_col"])
        g = q["target_gradient"].to_numpy(float)
    else:
        bmat = pivot_matrix(base, "row_id", sc["coord_col"], sc["value_col"])
        c = cand[cand["target"].eq(target)]
        q = grad[grad["target"].eq(target)].sort_values("local_coordinate")
        vectors, meta = scalar_sensitivity_map(c, sc["coord_col"], sc["cand_value_col"])
        g = q["target_gradient"].to_numpy(float)
    V = np.linalg.pinv(bmat.T @ bmat, rcond=RCOND)
    actions = make_actions(vectors, meta, target, sc["condition_col"], sc["type_col"])
    return V, g, bmat, actions


def panel_pattern(chosen: tuple[dict, ...]) -> str:
    n_stress = sum(a["type"] == "stress" for a in chosen)
    n_cm = len(chosen) - n_stress
    if n_stress == 0:
        return f"{n_cm}_CM_frequency_pairs"
    if n_cm == 0:
        return f"{n_stress}_stress_scalars"
    return f"{n_stress}_stress_plus_{n_cm}_CM_frequency_pairs"


def make_panel_specs(actions: list[dict], budget: int) -> list[dict]:
    specs = []
    for chosen in itertools.combinations(actions, budget):
        ids = tuple(sorted(a["action_id"] for a in chosen))
        rows = np.vstack([a["rows"] for a in chosen])
        specs.append({"panel_id": ";".join(ids), "pattern": panel_pattern(chosen),
                      "rows": rows, "actions": chosen,
                      "frequency_Hz": ";".join(f"{a['frequency_Hz']:g}" for a in chosen if a["frequency_Hz"] is not None),
                      "conditions": ";".join(sorted(set(a["condition"] for a in chosen))),
                      "n_stress": sum(a["type"] == "stress" for a in chosen),
                      "n_CM_pairs": sum(a["type"] == "CM_frequency_pair" for a in chosen)})
    return specs


def aggregate_panel_space(scenarios: list[dict], target: str, budget: int,
                          shared_only: bool = False) -> tuple:
    prepared = []
    for sc in scenarios:
        V, g, bmat, actions = scenario_target_matrices(sc, target)
        if shared_only:
            if target == "ATP1":
                eligible_condition = "ATP0.1"
            elif target == "ATP0.1":
                eligible_condition = "ATP1"
            else:
                eligible_condition = None
            actions = [a for a in actions if a["type"] == "CM_frequency_pair"
                       and a["condition"] == eligible_condition
                       and any(abs(float(a["frequency_Hz"]) - f) <= FREQ_TOL for f in COMMON_RAT_HUMAN_AWINDA_HZ)]
        prepared.append((sc, V, g, actions))
    candidate_sets = [set(a["action_id"] for a in x[3]) for x in prepared]
    common_ids = set.intersection(*candidate_sets) if candidate_sets else set()
    if not common_ids:
        raise ValueError(f"No common candidate actions for {target}")
    prepared = [(sc, V, g, [a for a in acts if a["action_id"] in common_ids]) for sc, V, g, acts in prepared]
    canonical_actions = prepared[0][3]
    specs = make_panel_specs(canonical_actions, budget)
    score_columns: list[np.ndarray] = []
    scenario_oracles: list[float] = []
    scenario_rows = []
    scenario_opt_rows = []
    # Since sensitivity coordinates differ across scenario types, panel IDs are the shared unit.
    for sc, V, g, actions in prepared:
        base_var = max(0.0, float(g @ V @ g))
        scenario_specs = make_panel_specs(actions, budget)
        if [p["panel_id"] for p in scenario_specs] != [p["panel_id"] for p in specs]:
            raise ValueError(f"Candidate panel vocabulary differs across scenario {sc['scenario_id']}")
        util = np.asarray([base_var - panel_variance(V, g, p["rows"]) for p in scenario_specs], float)
        util = np.maximum(util, 0.0)
        oracle = float(np.max(util)) if len(util) else 0.0
        retention = util / oracle if oracle > 0 else np.zeros_like(util)
        score_columns.append(retention)
        scenario_oracles.append(oracle)
        best_idx = int(np.argmax(util))
        bestp = scenario_specs[best_idx]
        scenario_opt_rows.append({"target": target, "budget": budget, "scenario_id": sc["scenario_id"],
                                  "scenario_family": sc["scenario_family"], "model": sc["model"], "domain": sc["domain"],
                                  "oracle_panel_id": bestp["panel_id"], "oracle_panel_pattern": bestp["pattern"],
                                  "oracle_frequency_Hz": bestp["frequency_Hz"], "oracle_utility": oracle,
                                  "baseline_prediction_variance": base_var, "oracle_after_panel_variance": base_var-oracle,
                                  "oracle_relative_variance_reduction_pct": 100*oracle/base_var if base_var else np.nan,
                                  "panel_count_exactly_enumerated": len(specs)})
    X = np.column_stack(score_columns)
    panel_ids = [p["panel_id"] for p in specs]
    worst = X.min(axis=1)
    mean = X.mean(axis=1)
    std = X.std(axis=1, ddof=0)
    max_regret = (1.0 - X).max(axis=1)
    # Equal weight is assigned to each explicitly listed model/domain scenario.
    winner_idx = min(range(len(specs)), key=lambda i: (-worst[i], max_regret[i], -mean[i], panel_ids[i]))
    winner = specs[winner_idx]
    robust_summary = pd.DataFrame([{
        "target": target, "budget": budget, "panel_id": winner["panel_id"], "panel_pattern": winner["pattern"],
        "selected_frequency_Hz": winner["frequency_Hz"], "selected_conditions": winner["conditions"],
        "n_stress_scalars": winner["n_stress"], "n_CM_frequency_pairs": winner["n_CM_pairs"],
        "worst_case_fraction_oracle_retained": worst[winner_idx],
        "minimax_regret_fraction_of_scenario_oracle": max_regret[winner_idx],
        "equal_weight_mean_fraction_oracle_retained": mean[winner_idx],
        "cross_scenario_sd_fraction_oracle_retained": std[winner_idx],
        "scenario_oracle_exact_match_fraction": float(np.mean(np.isclose(X[winner_idx], 1.0))),
        "n_scenarios": X.shape[1], "scenario_weighting": "equal per listed scenario",
        "n_candidate_panels": len(specs), "independent_recomputation": "run_R_robust_panels.py",
    }])
    # Full scenario x robust-panel table supports the required panel heatmap/replay.
    for j, (sc, _, _, _) in enumerate(prepared):
        scenario_rows.append({"target": target, "budget": budget, "scenario_id": sc["scenario_id"],
                              "scenario_family": sc["scenario_family"], "model": sc["model"], "domain": sc["domain"],
                              "panel_id": winner["panel_id"], "panel_fraction_of_scenario_oracle": X[winner_idx, j],
                              "scenario_specific_oracle_panel_id": scenario_opt_rows[j]["oracle_panel_id"],
                              "scenario_specific_oracle_utility": scenario_oracles[j],
                              "robust_panel_utility": scenario_oracles[j] * X[winner_idx, j],
                              "relative_regret": 1.0-X[winner_idx, j]})
    # Prespecified finalist Pareto set: union of the top 25 panels in every scenario
    # and top 25 by worst-case/mean retention. This avoids a large all-pairs frontier.
    finalist = set()
    for j in range(X.shape[1]):
        finalist.update(np.argsort(-X[:, j])[:25].tolist())
    finalist.update(np.argsort(-worst)[:25].tolist())
    finalist.update(np.argsort(-mean)[:25].tolist())
    fi = sorted(finalist, key=lambda i: panel_ids[i])
    XF = X[fi]
    pareto_ids = []
    for i, idx in enumerate(fi):
        others = np.delete(XF, i, axis=0)
        dominates = np.any(np.all(others >= XF[i] - 1e-12, axis=1) & np.any(others > XF[i] + 1e-12, axis=1))
        if not dominates:
            pareto_ids.append(idx)
    pareto_rows = []
    for i in pareto_ids:
        p = specs[i]
        pareto_rows.append({"target": target, "budget": budget, "panel_id": p["panel_id"],
                            "panel_pattern": p["pattern"], "selected_frequency_Hz": p["frequency_Hz"],
                            "worst_case_fraction_oracle_retained": worst[i],
                            "equal_weight_mean_fraction_oracle_retained": mean[i],
                            "cross_scenario_sd_fraction_oracle_retained": std[i],
                            "finalist_pool_size": len(fi), "frontier_scope": "exact non-dominated set within prespecified finalist pool"})
    # Include no forecast on raw cross-scenario variances, only normalized, scenario-relative utility.
    return robust_summary, pd.DataFrame(scenario_opt_rows), pd.DataFrame(scenario_rows), pd.DataFrame(pareto_rows), specs, X


def main() -> None:
    scenarios, scenario_table = load_scenarios()
    optima = []
    robust = []
    scenario_heat = []
    pareto = []
    pattern_rows = []
    shared_rows = []
    freq_map = []
    for target in TARGETS:
        for budget in (1, 2, 3):
            summary, opts, heat, front, specs, X = aggregate_panel_space(scenarios, target, budget)
            robust.append(summary)
            optima.extend(opts.to_dict("records"))
            scenario_heat.extend(heat.to_dict("records"))
            pareto.extend(front.to_dict("records"))
            for pattern, group in pd.DataFrame({"i": range(len(specs)), "pattern": [p["pattern"] for p in specs]}).groupby("pattern"):
                ix = group["i"].to_numpy(int)
                XP = X[ix]
                w = XP.min(axis=1); m = XP.mean(axis=1); mr = (1-XP).max(axis=1)
                best_local = min(range(len(ix)), key=lambda k: (-w[k], mr[k], -m[k], specs[ix[k]]["panel_id"]))
                p = specs[ix[best_local]]
                pattern_rows.append({"target": target, "budget": budget, "panel_pattern": pattern,
                                     "panel_id": p["panel_id"], "selected_frequency_Hz": p["frequency_Hz"],
                                     "worst_case_fraction_oracle_retained": w[best_local],
                                     "max_regret_fraction_oracle": mr[best_local],
                                     "equal_weight_mean_fraction_oracle_retained": m[best_local],
                                     "panel_count_in_pattern": len(ix), "scenario_count": X.shape[1]})
            if target in {"ATP0.1", "ATP1"}:
                ss, so, sh, sp, _, _ = aggregate_panel_space(scenarios, target, budget, shared_only=True)
                shared_rows.append(ss.iloc[0].to_dict())
    summary_df = pd.concat(robust, ignore_index=True)
    opts_df = pd.DataFrame(optima)
    scenario_table.to_csv(OUT / "R_SCENARIO_DEFINITIONS.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUT / "R_ROBUST_PANEL_RESULTS.csv", index=False, encoding="utf-8-sig")
    opts_df.to_csv(OUT / "R_SCENARIO_OPTIMA.csv", index=False, encoding="utf-8-sig")
    opts_df.merge(scenario_table, on=["scenario_id", "scenario_family", "model"], how="left").to_csv(
        OUT / "R_MODEL_DOMAIN_SCENARIO_TABLE.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(scenario_heat).to_csv(OUT / "R_SCENARIO_PANEL_UTILITY.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pareto).to_csv(OUT / "R_PANEL_PARETO_SHORTLIST.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pattern_rows).to_csv(OUT / "R_PATTERN_ROBUST_RESULTS.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(shared_rows).to_csv(RDIR / "R_SHARED_VOCAB_ROBUST_PANELS.csv", index=False, encoding="utf-8-sig")
    # Select model/domain oracle frequencies into a target-specific stability map.
    for (target, budget), sub in opts_df.groupby(["target", "budget"]):
        for _, row in sub.iterrows():
            for f in str(row["oracle_frequency_Hz"]).split(";"):
                if f and f != "nan":
                    freq_map.append({"target": target, "budget": budget, "scenario_id": row["scenario_id"],
                                     "oracle_frequency_Hz": float(f)})
    pd.DataFrame(freq_map).to_csv(OUT / "R_TARGET_SPECIFIC_ROBUST_FREQUENCY_MAP.csv", index=False, encoding="utf-8-sig")

    # Figures use normalized utility, avoiding cross-model raw-variance pooling.
    heat = pd.DataFrame(scenario_heat)
    scenario_order = list(scenario_table["scenario_id"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), sharex=True, sharey=True)
    for ax, target in zip(axes.ravel(), TARGETS):
        s = heat[heat["target"].eq(target)]
        mat = s.pivot(index="scenario_id", columns="budget", values="panel_fraction_of_scenario_oracle").reindex(scenario_order)
        im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(target)
        ax.set_xticks(range(len(mat.columns)), [str(c) for c in mat.columns])
        ax.set_xlabel("Budget (assay units)")
        ax.set_yticks(range(len(mat.index)), mat.index, fontsize=7)
    axes[0, 0].set_ylabel("Frozen model/domain scenario")
    axes[1, 0].set_ylabel("Frozen model/domain scenario")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Robust panel / scenario oracle utility", shrink=0.82)
    fig.suptitle("R: target-specific robust panels across frozen mechanism/domain scenarios")
    fig.savefig(RDIR / "figures" / "R_model_domain_panel_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for target, sub in summary_df.groupby("target"):
        sub = sub.sort_values("budget")
        ax.plot(sub["budget"], sub["worst_case_fraction_oracle_retained"], marker="o", label=target)
    ax.set(xlabel="Budget (assay units)", ylabel="Worst-case fraction of scenario oracle retained",
           ylim=(0, 1.02), title="R: robust panel performance by budget")
    ax.grid(True, alpha=0.25); ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(RDIR / "figures" / "R_robust_regret_curves.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    meta = {"scenario_count": len(scenarios), "model_variant_scenarios": int(sum(s["scenario_family"] == "N_model_variant" for s in scenarios)),
            "domain_scenarios": int(sum(s["scenario_family"] == "A_K_parameter_domain" for s in scenarios)),
            "targets": TARGETS, "budgets": [1, 2, 3], "scenario_weighting": "equal per scenario",
            "utility": "fraction of scenario-specific local Fisher/Gauss-Newton oracle utility; no raw variance pooling",
            "pareto_method": "exact non-dominated frontier within union of top 25 per scenario, top 25 worst-case, and top 25 mean panels",
            "parameter_domains_crossed_with_all_models": False,
            "limitations": ["A/K pair-derived +0.5 and +1.0 boxes only exist for published Model16D; they are not transplanted to alternate structures.",
                            "N alternate structures use their model-specific published optimizer bounds and fitted baseline-only local sensitivities.",
                            "Local Fisher utilities are rank/selection diagnostics and inherit pseudoinverse conditioning sensitivity."]}
    (OUT / "R_RUN_METADATA.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(shared_rows)[["target", "budget", "panel_id", "worst_case_fraction_oracle_retained",
                                    "equal_weight_mean_fraction_oracle_retained"]].to_string(index=False))
    print("scenario table", len(scenarios), "robust rows", len(summary_df), "candidate target-budget rows")


if __name__ == "__main__":
    main()
