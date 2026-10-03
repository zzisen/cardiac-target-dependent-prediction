#!/usr/bin/env python
"""Rebuild comparable ordinal target × frequency surfaces for Awinda, rat and human."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

# Public-release relocation only; numerical routines retain the frozen implementation.
import sys as _release_sys
REPO = next(p for p in Path(__file__).resolve().parents if (p / "CITATION.cff").is_file() and (p / "analysis").is_dir())
_release_sys.path.insert(0, str(REPO / "src" / "common"))
_release_sys.path.insert(0, str(Path(__file__).resolve().parent))


import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = REPO / "analysis/08_boundaries/state_surface"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
PRIOR = ROOT / "PRIOR_A_TO_X_BASELINE"
sys.path.insert(0, str(PRIOR / "shared_code"))
from dynamic_prediction import load_human, predict_fold, spectrum_metrics  # noqa: E402

BOOT_SEED = 20261004
N_BOOT = 10_000
TARGETS = {"ATP0.1": 0.1, "ATP1": 1.0}
FREQ_RE = re.compile(r"CM_(ATP(?:0\.1|1))_([0-9.]+)Hz_(Re|Im)$")


def sha256(p: Path):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ordinal_bands(freqs):
    names = np.empty(len(freqs), dtype=object)
    for label, ix in zip(("low", "mid", "high"), np.array_split(np.arange(len(freqs)), 3)):
        names[ix] = label
    return names


def attach_ranks(df, group_cols):
    x = df.copy()
    x["frequency_rank_band"] = ""
    x["utility_rank"] = np.nan
    x["utility_rank_score_0_to_1"] = np.nan
    x["fraction_of_context_oracle"] = np.nan
    for _, idx in x.groupby(group_cols, sort=False).groups.items():
        loc = list(idx)
        sub = x.loc[loc].sort_values("frequency_Hz")
        freq = sub.frequency_Hz.to_numpy(float)
        bands = ordinal_bands(freq)
        util = sub.utility_raw.to_numpy(float)
        ranks = rankdata(-util, method="average")
        score = 1.0 - (ranks - 1.0) / max(1, len(ranks) - 1)
        peak = float(np.nanmax(util))
        frac = util / peak if peak > 0 else np.full(len(util), np.nan)
        x.loc[sub.index, "frequency_rank_band"] = bands
        x.loc[sub.index, "utility_rank"] = ranks
        x.loc[sub.index, "utility_rank_score_0_to_1"] = score
        x.loc[sub.index, "fraction_of_context_oracle"] = frac
    return x


def build_awinda():
    p = REPO / "analysis/03_awinda/landscape/results/Y_FULL_UTILITY_LANDSCAPE.csv"
    d = pd.read_csv(p)
    out = pd.DataFrame({
        "system": "Awinda_mouse_public",
        "context": "all_mouse_states; ATP5_baseline_measurement",
        "future_target_label": d.target_ATP_mM.map(lambda x: f"ATP{x:g}"),
        "future_target_mM": d.target_ATP_mM.astype(float),
        "frequency_Hz": d.frequency_Hz.astype(float),
        "utility_definition": "outer_mouse_heldout_NRMSE_gain_pp; one_complex_pair",
        "utility_raw": d.mean_gain_pp.astype(float),
        "n_biological_units": d.n_mice.astype(int),
        "ci_low": d.bootstrap_95pct_low_pp.astype(float),
        "ci_high": d.bootstrap_95pct_high_pp.astype(float),
        "utility_rank_source": "mean_mouse_level_gain",
    })
    return out


def build_human():
    samples_all = load_human(REPO / "analysis/08_boundaries/preparation_support/input/B_HUMAN_CONDITION_OBSERVATIONS.csv")
    rows, prep_rows = [], []
    freqs = np.asarray(samples_all[0]["freqs"], float)
    rng = np.random.default_rng(BOOT_SEED)
    for disease in ("non-diabetic", "diabetic"):
        for target in ("ATP0.1", "ATP1"):
            samples = [s for s in samples_all if s["target"] == target and s["group"] == disease]
            if len(samples) != 10:
                raise ValueError(f"Expected 10 public preparations in {disease}/{target}, found {len(samples)}")
            y = np.vstack([s["y"] for s in samples])
            baseline_errors, candidate_errors = {float(f): {} for f in freqs}, {float(f): {} for f in freqs}
            for held in range(len(samples)):
                train = np.asarray([i for i in range(len(samples)) if i != held], int)
                test = np.asarray([held], int)
                base_pred = predict_fold(samples, y, train, test, None)
                base_nrmse = spectrum_metrics(y[held], base_pred[0], freqs)["whole_spectrum_nrmse_pct"]
                for f in freqs:
                    pred = predict_fold(samples, y, train, test, [float(f)])
                    augmented = spectrum_metrics(y[held], pred[0], freqs)["whole_spectrum_nrmse_pct"]
                    gain = float(base_nrmse - augmented)
                    baseline_errors[float(f)][held] = float(base_nrmse)
                    candidate_errors[float(f)][held] = float(augmented)
                    prep_rows.append({"system": "Human_public", "disease_group": disease,
                                      "future_target_label": target, "future_target_mM": TARGETS[target],
                                      "preparation_id": samples[held]["unit_id"],
                                      "frequency_Hz": float(f), "baseline_nrmse_pct": float(base_nrmse),
                                      "augmented_nrmse_pct": float(augmented), "utility_gain_pp": gain,
                                      "measurement_state": samples[held]["candidate_condition"],
                                      "target_state": target})
            fold = pd.DataFrame([r for r in prep_rows if r["disease_group"] == disease and r["future_target_label"] == target])
            for f in freqs:
                vals = np.asarray([candidate_errors[float(f)][i] for i in range(len(samples))], float)
                gain = np.asarray([baseline_errors[float(f)][i] - candidate_errors[float(f)][i]
                                   for i in range(len(samples))], float)
                boot = gain[rng.integers(0, len(gain), size=(N_BOOT, len(gain)))].mean(axis=1)
                rows.append({"system": "Human_public", "context": disease,
                             "future_target_label": target, "future_target_mM": TARGETS[target],
                             "frequency_Hz": float(f),
                             "utility_definition": "leave_one_preparation_out_whole_spectrum_NRMSE_gain_pp; reciprocal_ATP_measurement",
                             "utility_raw": float(gain.mean()), "n_biological_units": len(gain),
                             "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)),
                             "utility_rank_source": "preparation_equal_mean",
                             "measurement_state": samples[0]["candidate_condition"],
                             "donor_linkage": "unavailable; preparation is cluster"})
    pd.DataFrame(prep_rows).to_csv(OUT / "AA_HUMAN_PREPARATION_UTILITY.csv", index=False)
    return pd.DataFrame(rows)


def _pivot_sens(df, id_col, p_col, val_col):
    p = df.pivot_table(index=id_col, columns=p_col, values=val_col, aggfunc="first")
    return p.sort_index(axis=1).to_numpy(float)


def panel_variance_reduction(base, g, candidate_rows):
    H = base.T @ base
    V = np.linalg.pinv((H + H.T) / 2.0, rcond=1e-11)
    v0 = max(0.0, float(g @ V @ g))
    R = np.asarray(candidate_rows, float)
    RV = R @ V
    A = np.eye(R.shape[0]) + RV @ R.T
    c = RV @ g
    try:
        red = float(c @ np.linalg.solve(A, c))
    except np.linalg.LinAlgError:
        red = float(c @ np.linalg.pinv(A, rcond=1e-11) @ c)
    return v0, max(0.0, red)


def compute_model_surfaces(prefix, base_df, sens_df, grad_df, output_name):
    rows = []
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    key_col = "domain" if prefix == "K" else "variant"
    raw_actions = None
    if prefix == "N":
        raw_actions = pd.read_csv(REPO / "analysis/02_robustness/variants/results/N_A_STYLE_ACTION_UTILITIES.csv")
    contexts = sorted(base_df[key_col].astype(str).unique())
    for ctx in contexts:
        for target in ("ATP0.1", "ATP1"):
            measure = "ATP1" if target == "ATP0.1" else "ATP0.1"
            if prefix == "K":
                b = base_df[(base_df.domain.astype(str) == ctx) & (base_df.target == target)]
                s = sens_df[(sens_df.domain.astype(str) == ctx) & (sens_df.target == target) &
                            (sens_df.condition == measure) & (sens_df.measurement_type == "cm")]
                gdf = grad_df[(grad_df.domain.astype(str) == ctx) & (grad_df.target == target)]
                bmat = _pivot_sens(b, "row_id", "parameter_index", "sensitivity")
                g = gdf.sort_values("parameter_index").target_gradient.to_numpy(float)
                info = "Model16D local predictive-variance reduction; K frozen +0.5/+1.0 domains"
            else:
                b = base_df[base_df.variant.astype(str) == ctx]
                s = sens_df[(sens_df.variant.astype(str) == ctx) & (sens_df.target == target) &
                            (sens_df.condition == measure) & (sens_df.type == "cm")]
                gdf = grad_df[(grad_df.variant.astype(str) == ctx) & (grad_df.target == target)]
                bmat = _pivot_sens(b, "row_id", "parameter_index", "sensitivity")
                g = gdf.sort_values("parameter_index").target_gradient.to_numpy(float)
                info = "Model-variant local predictive-variance reduction; N frozen model-specific support"
            if bmat.shape[1] != len(g):
                raise ValueError(f"baseline/gradient dimensions differ for {prefix}/{ctx}/{target}: {bmat.shape}/{len(g)}")
            v0 = max(0.0, float(g @ np.linalg.pinv(bmat.T @ bmat, rcond=1e-11) @ g))
            for scalar_id, sub in s.groupby("scalar_id", sort=False):
                m = FREQ_RE.match(str(scalar_id))
                if not m:
                    continue
                cond, ftext, component = m.groups()
                if cond != measure:
                    continue
                if prefix == "K":
                    R1 = sub.sort_values("parameter_index").whitened_sensitivity.to_numpy(float)
                else:
                    R1 = sub.sort_values("parameter_index").whitened_sensitivity.to_numpy(float)
                # Assemble the real/imaginary rows for this one-frequency pair below.
            freqs = sorted({float(FREQ_RE.match(str(k)).group(2)) for k in s.scalar_id.unique()
                            if FREQ_RE.match(str(k)) and FREQ_RE.match(str(k)).group(1) == measure})
            for f in freqs:
                pair_rows = []
                for comp in ("Re", "Im"):
                    sid = f"CM_{measure}_{f:g}Hz_{comp}"
                    one = s[s.scalar_id.astype(str) == sid]
                    if one.empty:
                        pair_rows = []
                        break
                    pair_rows.append(one.sort_values("parameter_index").whitened_sensitivity.to_numpy(float))
                if len(pair_rows) != 2:
                    continue
                v0_check, utility = panel_variance_reduction(bmat, g, pair_rows)
                rows.append({"system": "Rat_Model16D", "context": ctx, "future_target_label": target,
                             "future_target_mM": TARGETS[target], "measurement_state": measure,
                             "frequency_Hz": f, "utility_definition": info,
                             "utility_raw": utility, "n_biological_units": 0,
                             "ci_low": np.nan, "ci_high": np.nan,
                             "utility_rank_source": "local_linear_design_not_observed_accuracy",
                             "baseline_prediction_variance": v0_check,
                             "variance_reduction_pct": 100.0 * utility / v0_check if v0_check > 0 else np.nan})
                if raw_actions is not None:
                    stored = raw_actions[(raw_actions.variant.astype(str) == ctx) &
                                          (raw_actions.target == target) &
                                          (raw_actions.condition == measure) &
                                          raw_actions.type.eq("CM_frequency_pair") &
                                          np.isclose(raw_actions.frequency_hz.astype(float), f)]
                    if len(stored) == 1:
                        diff = abs(utility - float(stored.variance_reduction.iloc[0]))
                        max_abs_diff = max(max_abs_diff, diff)
                        max_rel_diff = max(max_rel_diff, diff / max(abs(utility), 1e-30))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / output_name, index=False)
    return result, {"independent_reconstruction": len(result) > 0, "reconstructed_rows": int(len(result)),
                    "max_abs_difference_vs_N_stored_utility": max_abs_diff,
                    "max_relative_difference_vs_N_stored_utility": max_rel_diff,
                    "status": ("PASS" if prefix != "N" else
                              "PASS_WITH_FLOATING_POINT_QUALIFICATION" if len(result) > 0 and max_rel_diff < 1e-6 else "CHECK")}


def build_rat():
    kbase = pd.read_csv(REPO / "analysis/01_core_target_value/sparse/results/K_BASELINE_SENSITIVITIES.csv")
    ksens = pd.read_csv(REPO / "analysis/01_core_target_value/sparse/results/K_LOCAL_SENSITIVITIES.csv")
    kgrad = pd.read_csv(REPO / "analysis/01_core_target_value/sparse/results/K_TARGET_GRADIENTS.csv")
    k, _ = compute_model_surfaces("K", kbase, ksens, kgrad, "AA_RAT_MODE16D_SURFACE.csv")
    nbase = pd.read_csv(REPO / "analysis/02_robustness/variants/results/N_BASELINE_SENSITIVITIES.csv")
    nsens = pd.read_csv(REPO / "analysis/02_robustness/variants/results/N_LOCAL_SENSITIVITIES.csv")
    ngrad = pd.read_csv(REPO / "analysis/02_robustness/variants/results/N_TARGET_GRADIENTS.csv")
    n, verification = compute_model_surfaces("N", nbase, nsens, ngrad, "AA_RAT_N_VARIANT_SURFACE.csv")
    return pd.concat([k, n], ignore_index=True), verification


def main():
    aw = build_awinda()
    hu = build_human()
    rat, rat_ver = build_rat()
    surface = pd.concat([aw, hu, rat], ignore_index=True, sort=False)
    surface = attach_ranks(surface, ["system", "context", "future_target_label"])
    surface.to_csv(OUT / "AA_CROSS_SYSTEM_TARGET_SURFACE.csv", index=False)
    bands = (surface.groupby(["system", "context", "future_target_label", "future_target_mM",
                              "frequency_rank_band"], as_index=False)
             .agg(mean_raw_utility=("utility_raw", "mean"),
                  mean_rank_score=("utility_rank_score_0_to_1", "mean"),
                  mean_fraction_oracle=("fraction_of_context_oracle", "mean"),
                  min_rank_score=("utility_rank_score_0_to_1", "min"),
                  max_rank_score=("utility_rank_score_0_to_1", "max"),
                  n_frequency_candidates=("frequency_Hz", "nunique"),
                  frequency_min_Hz=("frequency_Hz", "min"), frequency_max_Hz=("frequency_Hz", "max")))
    bands.to_csv(OUT / "AA_TARGET_BAND_SUMMARY.csv", index=False)
    trends = []
    for (system, context), sub in bands.groupby(["system", "context"], sort=False):
        target_rows = []
        for (label, tm), one in sub.groupby(["future_target_label", "future_target_mM"], sort=False):
            one = one.sort_values("frequency_rank_band", key=lambda s: s.map({"low": 0, "mid": 1, "high": 2}))
            ib = int(one.mean_rank_score.to_numpy(float).argmax())
            full = surface[(surface.system == system) & (surface.context == context) &
                           (surface.future_target_label == label)].sort_values("frequency_Hz")
            ip = int(full.utility_raw.to_numpy(float).argmax())
            target_rows.append({"system": system, "context": context, "future_target_label": label,
                                "future_target_mM": float(tm), "preferred_ordinal_band": one.frequency_rank_band.iloc[ib],
                                "preferred_band_mean_rank_score": float(one.mean_rank_score.iloc[ib]),
                                "mean_rank_score_low": float(one.loc[one.frequency_rank_band.eq("low"), "mean_rank_score"].iloc[0]),
                                "mean_rank_score_mid": float(one.loc[one.frequency_rank_band.eq("mid"), "mean_rank_score"].iloc[0]),
                                "mean_rank_score_high": float(one.loc[one.frequency_rank_band.eq("high"), "mean_rank_score"].iloc[0]),
                                "peak_frequency_Hz": float(full.frequency_Hz.iloc[ip]),
                                "peak_band": str(full.frequency_rank_band.iloc[ip]),
                                "peak_utility_raw": float(full.utility_raw.iloc[ip])})
        target_rows.sort(key=lambda r: r["future_target_mM"])
        for r in target_rows:
            r["target_state_band_direction"] = ("increasing" if len(target_rows) > 1 and
                  ({"low": 0, "mid": 1, "high": 2}[target_rows[-1]["preferred_ordinal_band"]] >
                   {"low": 0, "mid": 1, "high": 2}[target_rows[0]["preferred_ordinal_band"]]) else
                  "decreasing" if len(target_rows) > 1 and
                  ({"low": 0, "mid": 1, "high": 2}[target_rows[-1]["preferred_ordinal_band"]] <
                   {"low": 0, "mid": 1, "high": 2}[target_rows[0]["preferred_ordinal_band"]]) else
                  "no_band_shift_or_single_target")
            trends.append(r)
    trenddf = pd.DataFrame(trends)
    trenddf.to_csv(OUT / "AA_TARGET_STATE_TRENDS.csv", index=False)
    report = {
        "rat_N_surface_independent_reconstruction": rat_ver,
        "Awinda_rows": int(len(aw)), "Human_rows": int(len(hu)), "Rat_rows": int(len(rat)),
        "all_surface_rows": int(len(surface)),
        "rat_frequency_grid_n": int(rat.frequency_Hz.nunique()),
        "human_frequency_grid_n": int(hu.frequency_Hz.nunique()),
        "Awinda_frequency_grid_n": int(aw.frequency_Hz.nunique()),
        "human_population": "10 non-DM + 10 T2D public preparations; analyzed separately; donor links unavailable",
        "rat_scope": "ATP0.1 and ATP1 targets only; reciprocal ATP measurement state; Model16D +0.5/+1.0 domains plus five N variants; local predictive-variance reduction only",
        "frequency_bands_frozen_before_aggregation": True,
    }
    (OUT / "AA_REBUILD_VERIFICATION.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Preferred-band state shifts:")
    print(trenddf[["system", "context", "future_target_label", "preferred_ordinal_band", "peak_frequency_Hz", "peak_band", "target_state_band_direction"]].to_string(index=False))


if __name__ == "__main__":
    main()
