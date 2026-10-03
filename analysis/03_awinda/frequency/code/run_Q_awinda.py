"""Independent Awinda frequency-domain validation with animal-held-out prediction.

The analysis aggregates multiple fibres within each animal x genotype x drug cell,
holds out whole animals, and predicts active-state complex modulus spectra at ATP
0.1 or 1 mM from one baseline (5 mM ATP) scalar or one baseline frequency feature.
The target spectrum is never used as an input. Sparse frequency selection is nested
inside each outer animal-held-out fold.
"""
from __future__ import annotations

import itertools
import json
import hashlib
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
QDIR = HERE.parent
ROOT = QDIR.parent
SOURCE = REPO / "data/derived/awinda_source"
OUT = QDIR / "results"
OUT.mkdir(parents=True, exist_ok=True)
FREQ_FILE = "WT_RLC_N47K_Control_Mavacamten_XLD_GWN_DQ1_2019_09_09.xlsx"
TENSION_FILE = "Transgenic_mice_Fiber_Tensions_with_Fits_DQ1_2019_09_06.xlsx"
KINETIC_FILE = "Noise_analysis_ABC_GWN_WithFits_DQ1_ReFits_2020_08_04_Ver_2024.xlsx"
TARGET_ATP = [0.1, 1.0]
BASELINE_ATP = 5.0
ACTIVE_PCA = 4.8
RIDGE_ALPHA = 1.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def weighted_metrics(y: np.ndarray, pred: np.ndarray, freqs: np.ndarray) -> dict[str, float]:
    err = pred - y
    nfreq = len(freqs)
    re = err[:, :nfreq]
    im = err[:, nfreq:]
    yr = y[:, :nfreq]
    yi = y[:, nfreq:]
    rmse = float(np.sqrt(np.mean(err**2)))
    rms = float(np.sqrt(np.mean(y**2)))
    order = np.argsort(freqs)
    logf = np.log(freqs[order])
    w = np.empty(nfreq, float)
    if nfreq == 1:
        w[:] = 1.0
    else:
        edges = np.empty(nfreq + 1)
        edges[1:-1] = (logf[:-1] + logf[1:]) / 2
        edges[0] = logf[0] - (logf[1] - logf[0]) / 2
        edges[-1] = logf[-1] + (logf[-1] - logf[-2]) / 2
        w_sorted = np.diff(edges)
        w[order] = w_sorted / w_sorted.sum()
    w2 = np.r_[w / 2, w / 2]
    log_rmse = float(np.sqrt(np.mean(np.sum(err**2 * w2[None, :], axis=1))))
    log_rms = float(np.sqrt(np.mean(np.sum(y**2 * w2[None, :], axis=1))))
    corrs = []
    for a, b in zip(y, pred):
        if np.std(a) > 0 and np.std(b) > 0:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "rmse_kPa": rmse,
        "nrmse_pct_global_rms": 100 * rmse / rms if rms else np.nan,
        "re_rmse_kPa": float(np.sqrt(np.mean(re**2))),
        "im_rmse_kPa": float(np.sqrt(np.mean(im**2))),
        "shape_correlation_mean": float(np.mean(corrs)) if corrs else np.nan,
        "log_frequency_weighted_rmse_kPa": log_rmse,
        "log_frequency_weighted_nrmse_pct": 100 * log_rmse / log_rms if log_rms else np.nan,
    }


def design(group: np.ndarray, feats: np.ndarray, train: np.ndarray, groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Unpenalized group means plus prespecified ridge-penalized measurement slopes."""
    G = np.column_stack([(group == g).astype(float) for g in groups])
    if feats.shape[1] == 0:
        return G, G
    mu = feats[train].mean(axis=0)
    sd = feats[train].std(axis=0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (feats - mu) / sd
    X = np.column_stack([G, Z])
    return X, np.r_[np.zeros(G.shape[1]), np.full(feats.shape[1], RIDGE_ALPHA)]


def predict_fold(group: np.ndarray, feats: np.ndarray, y: np.ndarray, train: np.ndarray,
                 test: np.ndarray, groups: list[str]) -> np.ndarray:
    X, penalty = design(group, feats, train, groups)
    if penalty.ndim == 1:
        coef = np.linalg.solve(X[train].T @ X[train] + np.diag(penalty), X[train].T @ y[train])
    else:
        coef = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
    return X[test] @ coef


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xld = pd.read_csv(SOURCE / FREQ_FILE.replace(".xlsx", ".csv"))
    tension = pd.read_csv(SOURCE / TENSION_FILE.replace(".xlsx", ".csv"))
    kinetics = pd.read_csv(SOURCE / KINETIC_FILE.replace(".xlsx", ".csv"))
    return xld, tension, kinetics


def main() -> None:
    xld, tension, kinetics = read_sources()
    # Publicly released animal linkage is in the accompanying force-pCa workbook.
    link = tension[["AnimalID", "Fiber", "Mutation", "Treatment"]].drop_duplicates()
    if link["Fiber"].duplicated().any():
        raise ValueError("Fiber-to-animal map is not one-to-one; stop before modelling")
    xld_samples = xld[["Filename", "Mutation", "Cond", "Concentration"]].dropna(subset=["Filename"]).drop_duplicates()
    sample_map = xld_samples.merge(link, left_on="Filename", right_on="Fiber", how="left", validate="one_to_one")
    sample_map["drug_state"] = np.where(sample_map["Cond"].eq("Control"), "Control", "0.3_uM_mavacamten")
    sample_map["group"] = sample_map["Mutation_x"].astype(str) + "|" + sample_map["drug_state"]
    if sample_map["AnimalID"].isna().any():
        raise ValueError("Some dynamic files do not map to public animal IDs")
    if not sample_map["Mutation_x"].eq(sample_map["Mutation_y"]).all():
        raise ValueError("Genotype mismatch between the frequency and tension workbook")

    active = xld.loc[
        xld["pCa"].eq(ACTIVE_PCA)
        & pd.to_numeric(xld["Quality"], errors="coerce").eq(1)
        & xld["Freq (Hz)"].notna()
    ].copy()
    active = active.merge(sample_map[["Filename", "AnimalID", "Mutation_x", "drug_state", "group"]],
                          on="Filename", how="left", validate="many_to_one")
    active["ATP (mM)"] = pd.to_numeric(active["ATP (mM)"], errors="coerce")
    active["Freq (Hz)"] = pd.to_numeric(active["Freq (Hz)"], errors="coerce")
    active["Em (kPa)"] = pd.to_numeric(active["Em (kPa)"], errors="coerce")
    active["Vm (kPa)"] = pd.to_numeric(active["Vm (kPa)"], errors="coerce")
    freq = np.sort(active.loc[active["ATP (mM)"].eq(BASELINE_ATP), "Freq (Hz)"].unique())
    if len(freq) != 72:
        raise ValueError(f"Expected the shared 72-frequency grid; found {len(freq)}")
    coverage = active.groupby("Filename", as_index=False).agg(
        n_active_MgATP_conditions=("ATP (mM)", "nunique"),
        n_runs=("Run", "nunique"),
        active_frequency_count=("Freq (Hz)", "nunique"),
    )
    sample_map = sample_map.merge(coverage, left_on="Filename", right_on="Filename", how="left", validate="one_to_one")

    # Create one independent observation per animal x genotype x drug state.
    cell_cols = ["AnimalID", "Mutation_x", "drug_state", "group"]
    mean_spectra = active.groupby(cell_cols + ["ATP (mM)", "Freq (Hz)"], as_index=False).agg(
        Re_kPa=("Em (kPa)", "mean"), Im_kPa=("Vm (kPa)", "mean"), n_fibres=("Filename", "nunique"))
    wide_rows = []
    for key, sub in mean_spectra.groupby(cell_cols, sort=True):
        base = sub[sub["ATP (mM)"].eq(BASELINE_ATP)].set_index("Freq (Hz)").reindex(freq)
        if base[["Re_kPa", "Im_kPa"]].isna().any().any():
            raise ValueError(f"Incomplete baseline spectrum for cell {key}")
        out = dict(zip(cell_cols, key))
        for i, f in enumerate(freq):
            out[f"base_re_{f:.4f}"] = float(base.loc[f, "Re_kPa"])
            out[f"base_im_{f:.4f}"] = float(base.loc[f, "Im_kPa"])
        for target in TARGET_ATP:
            fut = sub[sub["ATP (mM)"].eq(target)].set_index("Freq (Hz)").reindex(freq)
            if fut[["Re_kPa", "Im_kPa"]].isna().any().any():
                raise ValueError(f"Incomplete ATP {target:g} target spectrum for cell {key}")
            for i, f in enumerate(freq):
                out[f"target_{target:g}_re_{f:.4f}"] = float(fut.loc[f, "Re_kPa"])
                out[f"target_{target:g}_im_{f:.4f}"] = float(fut.loc[f, "Im_kPa"])
        out["n_animals"] = 1
        out["n_fibres_total"] = int(active.loc[
            active["AnimalID"].eq(key[0]) & active["Mutation_x"].eq(key[1]) & active["drug_state"].eq(key[2]), "Filename"
        ].nunique())
        wide_rows.append(out)
    cells = pd.DataFrame(wide_rows).sort_values(cell_cols).reset_index(drop=True)

    # Baseline steady active tension is an optional single-scalar candidate.
    tension_active = tension.loc[
        tension["ATPmM"].eq(BASELINE_ATP) & tension["PimM"].eq(0.3) & tension["pCa"].eq(ACTIVE_PCA)
    ].merge(link, on=["AnimalID", "Fiber", "Mutation", "Treatment"], how="left", validate="many_to_one")
    tension_active["drug_state"] = np.where(tension_active["Treatment"].eq("Control"), "Control", "0.3_uM_mavacamten")
    stress_map = tension_active.groupby(["AnimalID", "Mutation", "drug_state"], as_index=False).agg(
        stress_kPa=("ActiveTensionkPa", "mean"), n_stress_fibres=("Fiber", "nunique"))
    cells = cells.merge(stress_map, left_on=["AnimalID", "Mutation_x", "drug_state"],
                        right_on=["AnimalID", "Mutation", "drug_state"], how="left", validate="one_to_one")
    cells["group"] = cells["Mutation_x"].astype(str) + "|" + cells["drug_state"]
    groups = sorted(cells["group"].unique())
    animals = sorted(cells["AnimalID"].astype(str).unique())
    g = cells["group"].to_numpy()

    # Keep the complete target-frequency source data in tidy long form.
    long = mean_spectra.rename(columns={"ATP (mM)": "ATP_mM", "Freq (Hz)": "frequency_Hz"})
    long.to_csv(OUT / "Q_AWINDA_FULL_CM_RESULTS.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(OUT / "Q_AWINDA_ANIMAL_CONDITION_AGGREGATES.csv", index=False, encoding="utf-8-sig")
    sample_map.to_csv(OUT / "Q_AWINDA_SAMPLE_MAP.csv", index=False, encoding="utf-8-sig")

    # Candidate panel vocabulary: baseline active stress, every individual Re/Im scalar,
    # and every matched complex-modulus frequency pair from 5 mM ATP.
    candidates: list[tuple[str, str, np.ndarray, float | None]] = []
    if cells["stress_kPa"].notna().all():
        candidates.append(("stress_baseline_pCa4.8_ATP5", "stress_scalar", cells[["stress_kPa"]].to_numpy(float), None))
    for f in freq:
        candidates.append((f"CM_ATP5_{f:.4f}Hz_Re", "Re_scalar", cells[[f"base_re_{f:.4f}"]].to_numpy(float), float(f)))
        candidates.append((f"CM_ATP5_{f:.4f}Hz_Im", "Im_scalar", cells[[f"base_im_{f:.4f}"]].to_numpy(float), float(f)))
        candidates.append((f"CM_ATP5_{f:.4f}Hz_pair", "complex_frequency_pair",
                           cells[[f"base_re_{f:.4f}", f"base_im_{f:.4f}"]].to_numpy(float), float(f)))

    split_rows = []
    for held in animals:
        tr = np.flatnonzero(cells["AnimalID"].astype(str).to_numpy() != held)
        te = np.flatnonzero(cells["AnimalID"].astype(str).to_numpy() == held)
        split_rows.append({"heldout_animal": held, "heldout_cells": ";".join(cells.iloc[te]["group"].astype(str)),
                           "training_animals": ";".join(sorted(cells.iloc[tr]["AnimalID"].astype(str).unique())),
                           "n_training_animals": len(set(cells.iloc[tr]["AnimalID"].astype(str))),
                           "n_training_cells": len(tr), "n_heldout_cells": len(te), "grouped_by_animal": True})
    pd.DataFrame(split_rows).to_csv(OUT / "Q_AWINDA_CV_SPLITS.csv", index=False, encoding="utf-8-sig")

    heldout_rows = []
    pred_cache: dict[tuple[float, str], tuple[np.ndarray, np.ndarray]] = {}
    for target in TARGET_ATP:
        ycols = [f"target_{target:g}_re_{f:.4f}" for f in freq] + [f"target_{target:g}_im_{f:.4f}" for f in freq]
        y = cells[ycols].to_numpy(float)
        for held in animals:
            tr = np.flatnonzero(cells["AnimalID"].astype(str).to_numpy() != held)
            te = np.flatnonzero(cells["AnimalID"].astype(str).to_numpy() == held)
            pred0 = predict_fold(g, np.empty((len(cells), 0)), y, tr, te, groups)
            m0 = weighted_metrics(y[te], pred0, freq)
            pred_cache[(target, held, "GROUP_ONLY")] = (y[te], pred0)
            for row_i, ix in enumerate(te):
                heldout_rows.append({"target_ATP_mM": target, "heldout_animal": held,
                                     "AnimalID": cells.iloc[ix]["AnimalID"], "group": cells.iloc[ix]["group"],
                                     "candidate_id": "GROUP_ONLY", "candidate_type": "baseline_reference",
                                     "frequency_Hz": np.nan, "candidate_budget_scalars": 0,
                                     **weighted_metrics(y[ix:ix+1], pred0[row_i:row_i+1], freq),
                                     "baseline_only_nrmse_pct": weighted_metrics(y[ix:ix+1], pred0[row_i:row_i+1], freq)["nrmse_pct_global_rms"]})
            for cid, ctype, feats, fhz in candidates:
                pred = predict_fold(g, feats, y, tr, te, groups)
                pred_cache[(target, held, cid)] = (y[te], pred)
                for row_i, ix in enumerate(te):
                    met = weighted_metrics(y[ix:ix+1], pred[row_i:row_i+1], freq)
                    heldout_rows.append({"target_ATP_mM": target, "heldout_animal": held,
                                         "AnimalID": cells.iloc[ix]["AnimalID"], "group": cells.iloc[ix]["group"],
                                         "candidate_id": cid, "candidate_type": ctype, "frequency_Hz": fhz,
                                         "candidate_budget_scalars": feats.shape[1], **met,
                                         "baseline_only_nrmse_pct": weighted_metrics(y[ix:ix+1], pred0[row_i:row_i+1], freq)["nrmse_pct_global_rms"]})
    heldout = pd.DataFrame(heldout_rows)
    heldout["nrmse_improvement_vs_group_only_pp"] = heldout["baseline_only_nrmse_pct"] - heldout["nrmse_pct_global_rms"]
    heldout.to_csv(OUT / "Q_AWINDA_HELDOUT_RESULTS.csv", index=False, encoding="utf-8-sig")

    # Candidate-by-target score matrix is based only on outer-fold held-out spectra.
    cand_scores = heldout.groupby(["target_ATP_mM", "candidate_id", "candidate_type", "frequency_Hz", "candidate_budget_scalars"], dropna=False).agg(
        n_animal_condition_cells=("AnimalID", "size"), n_animals=("AnimalID", "nunique"), mean_rmse_kPa=("rmse_kPa", "mean"),
        mean_nrmse_pct=("nrmse_pct_global_rms", "mean"), median_nrmse_pct=("nrmse_pct_global_rms", "median"),
        mean_re_rmse_kPa=("re_rmse_kPa", "mean"), mean_im_rmse_kPa=("im_rmse_kPa", "mean"),
        mean_shape_correlation=("shape_correlation_mean", "mean"),
        mean_logfreq_nrmse_pct=("log_frequency_weighted_nrmse_pct", "mean"),
        mean_nrmse_improvement_pp=("nrmse_improvement_vs_group_only_pp", "mean"),
        improved_cells=("nrmse_improvement_vs_group_only_pp", lambda x: int((x > 0).sum())),
        worsened_cells=("nrmse_improvement_vs_group_only_pp", lambda x: int((x < 0).sum()))).reset_index()
    cand_scores["rank_by_nrmse"] = cand_scores.groupby("target_ATP_mM")["mean_nrmse_pct"].rank(method="min")
    cand_scores.to_csv(OUT / "Q_AWINDA_CANDIDATE_TARGET_UTILITY.csv", index=False, encoding="utf-8-sig")

    # Pairwise reversals compare the same 72 baseline frequency pairs under both future targets.
    pair = cand_scores[cand_scores["candidate_type"].eq("complex_frequency_pair")]
    piv = pair.pivot(index="candidate_id", columns="target_ATP_mM", values="mean_nrmse_improvement_pp")
    rev = []
    if len(TARGET_ATP) == 2:
        t0, t1 = TARGET_ATP
        ids = list(piv.index)
        nrev = 0
        ncomp = 0
        for a, b in itertools.combinations(ids, 2):
            da = piv.loc[a, t0] - piv.loc[b, t0]
            db = piv.loc[a, t1] - piv.loc[b, t1]
            if np.isfinite(da) and np.isfinite(db):
                ncomp += 1
                nrev += int(da * db < 0)
        rho = float(spearmanr(piv[t0], piv[t1], nan_policy="omit").statistic)
        tau = float(kendalltau(piv[t0], piv[t1], nan_policy="omit").statistic)
        rev.append({"comparison": f"ATP{t0:g}_vs_ATP{t1:g}", "candidate_class": "baseline_CM_frequency_pairs",
                    "n_pairwise_frequency_comparisons": ncomp, "rank_reversals": nrev,
                    "reversal_fraction": nrev / ncomp if ncomp else np.nan,
                    "spearman_rank_correlation_of_heldout_utility": rho, "kendall_tau_b": tau,
                    "interpretation": "descriptive outer-fold rank reversal fraction; not an inferential test"})
    pd.DataFrame(rev).to_csv(OUT / "Q_AWINDA_TARGET_RANK_REVERSALS.csv", index=False, encoding="utf-8-sig")

    strat = heldout[heldout["candidate_type"].isin(["baseline_reference", "complex_frequency_pair", "stress_scalar"])].groupby(
        ["target_ATP_mM", "group", "candidate_id", "candidate_type"], dropna=False).agg(
        n_animals=("AnimalID", "nunique"), mean_nrmse_pct=("nrmse_pct_global_rms", "mean"),
        median_nrmse_pct=("nrmse_pct_global_rms", "median"), mean_improvement_pp=("nrmse_improvement_vs_group_only_pp", "mean"),
        improved_animals=("nrmse_improvement_vs_group_only_pp", lambda x: int((x > 0).sum())),
        worsened_animals=("nrmse_improvement_vs_group_only_pp", lambda x: int((x < 0).sum()))).reset_index()
    strat["rank_within_stratum"] = strat.groupby(["target_ATP_mM", "group"])["mean_nrmse_pct"].rank(method="min")
    strat.to_csv(OUT / "Q_AWINDA_GENOTYPE_DRUG_STRATA.csv", index=False, encoding="utf-8-sig")

    # Nested animal-held-out selection of a one-frequency pair. Each outer test animal
    # remains unseen while the frequency is selected by inner animal-held-out CV.
    nested_rows = []
    for target in TARGET_ATP:
        ycols = [f"target_{target:g}_re_{f:.4f}" for f in freq] + [f"target_{target:g}_im_{f:.4f}" for f in freq]
        y = cells[ycols].to_numpy(float)
        pair_candidates = [(cid, feats, fhz) for cid, ctype, feats, fhz in candidates if ctype == "complex_frequency_pair"]
        for outer in animals:
            tr_outer = np.flatnonzero(cells["AnimalID"].astype(str).to_numpy() != outer)
            te_outer = np.flatnonzero(cells["AnimalID"].astype(str).to_numpy() == outer)
            base_outer = predict_fold(g, np.empty((len(cells), 0)), y, tr_outer, te_outer, groups)
            inner_animals = sorted(set(cells.iloc[tr_outer]["AnimalID"].astype(str)))
            scores = []
            for cid, feats, fhz in pair_candidates:
                errors = []
                for inner in inner_animals:
                    tr_inner = tr_outer[cells.iloc[tr_outer]["AnimalID"].astype(str).to_numpy() != inner]
                    te_inner = tr_outer[cells.iloc[tr_outer]["AnimalID"].astype(str).to_numpy() == inner]
                    pr = predict_fold(g, feats, y, tr_inner, te_inner, groups)
                    errors.append(float(np.mean(((pr - y[te_inner]) / np.sqrt(np.mean(y[te_inner]**2)))**2)))
                scores.append((float(np.mean(errors)), cid, fhz))
            inner_nrmse, selected, selected_f = min(scores, key=lambda z: (z[0], z[1]))
            feats = next(x[1] for x in pair_candidates if x[0] == selected)
            pr_outer = predict_fold(g, feats, y, tr_outer, te_outer, groups)
            for ix, pr, pr0 in zip(te_outer, pr_outer, base_outer):
                met = weighted_metrics(y[ix:ix+1], pr.reshape(1, -1), freq)
                base_met = weighted_metrics(y[ix:ix+1], pr0.reshape(1, -1), freq)
                nested_rows.append({"target_ATP_mM": target, "heldout_animal": outer,
                                    "AnimalID": cells.iloc[ix]["AnimalID"], "group": cells.iloc[ix]["group"],
                                    "selected_panel_id": selected, "selected_frequency_Hz": selected_f,
                                    "inner_cv_mean_scaled_mse": inner_nrmse,
                                    "outer_nrmse_pct": met["nrmse_pct_global_rms"], **met,
                                    "baseline_group_only_nrmse_pct": base_met["nrmse_pct_global_rms"],
                                    "nrmse_improvement_vs_group_only_pp": base_met["nrmse_pct_global_rms"] - met["nrmse_pct_global_rms"],
                                    "inner_selection_grouped_by_animal": True,
                                    "destination_outcome_used_in_selection": False})
    nested = pd.DataFrame(nested_rows)
    nested.to_csv(OUT / "Q_AWINDA_SPARSE_PANEL_RESULTS.csv", index=False, encoding="utf-8-sig")

    # Workbook inventory plus independently inspectable preparation hierarchy counts.
    sample_map.to_csv(OUT / "Q_AWINDA_SAMPLE_MAP.csv", index=False, encoding="utf-8-sig")
    meta = {
        "branch": "Q", "route": "Q3 model-agnostic target-dependence benchmark with repeated MgATP target spectra",
        "baseline_ATP_mM": BASELINE_ATP, "future_targets_ATP_mM": TARGET_ATP, "active_pCa": ACTIVE_PCA,
        "frequency_count": int(len(freq)), "frequency_min_Hz": float(freq.min()), "frequency_max_Hz": float(freq.max()),
        "n_animals": int(cells["AnimalID"].nunique()), "n_animal_condition_cells": int(len(cells)),
        "n_fibres": int(sample_map["Filename"].nunique()), "n_genotype_drug_strata": int(cells["group"].nunique()),
        "animal_holdout": True, "baseline_candidates": int(len(candidates)),
        "ridge_alpha_measurement_features": RIDGE_ALPHA,
        "target_feature_leakage": False,
        "excluded_frequency_rows": int(len(xld) - len(active)),
        "quality_rule": "active-state pCa 4.8; Quality==1; one public mouse ID held out with all of its fibres and drug conditions",
        "source_sha256": {p.name: sha256(p) for p in sorted(SOURCE.glob("*.xlsx"))},
    }
    (OUT / "Q_AWINDA_RUN_METADATA.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Model-source kinetic fits are inventoried as a possible independent descriptive target,
    # but are not used as a prediction target because fitted summaries reuse all MgATP spectra.
    kinetic_link = link[["AnimalID", "Fiber"]].drop_duplicates()
    kinetics = kinetics.merge(kinetic_link, left_on="Filename", right_on="Fiber", how="left", validate="many_to_one")
    kinetic_summary = kinetics.groupby(["Mutation", "Treatment", "Nucleotide_Fit"], dropna=False).agg(
        rows=("Filename", "size"), fibres=("Filename", "nunique"), animals=("AnimalID", "nunique")).reset_index()
    kinetic_summary.to_csv(OUT / "Q_AWINDA_KINETIC_FIT_INVENTORY.csv", index=False, encoding="utf-8-sig")

    pairplot = cand_scores[cand_scores["candidate_type"].eq("complex_frequency_pair")]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    colors = {0.1: "#007C91", 1.0: "#A64037"}
    for target in TARGET_ATP:
        sub = pairplot[pairplot["target_ATP_mM"].eq(target)].sort_values("frequency_Hz")
        axes[0].plot(sub["frequency_Hz"], sub["mean_nrmse_pct"], marker=".", linewidth=1.3,
                     markersize=4, color=colors[target], label=f"future ATP {target:g} mM")
        axes[1].plot(sub["frequency_Hz"], sub["mean_nrmse_improvement_pp"], marker=".", linewidth=1.3,
                     markersize=4, color=colors[target], label=f"future ATP {target:g} mM")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Baseline complex-modulus frequency (Hz)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Animal-held-out whole-spectrum NRMSE (%)")
    axes[1].set_ylabel("NRMSE reduction vs group-only model (percentage points)")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Awinda: baseline frequency value depends on the future MgATP target")
    fig.tight_layout()
    fig.savefig(QDIR / "figures" / "Q_target_specific_frequency_utility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
