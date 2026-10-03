# Executable analysis coverage

All five main claims have their frozen numerical analysis implementation in this
release. Paths were adapted to the public directory layout; candidate selection,
fitting, scoring, endpoint definitions and resampling rules were retained. The
rat repeated-dose reporting value now refers to the actual selected candidate.

| Main claim or supporting boundary | Executable script | Allowed input | Frozen/replayed output |
|---|---|---|---|
| Same measurement, different future target | `analysis/01_core_target_value/support/code/run_A_measurement_target_map.py`, `run_A_full_target_rank_stability_v2.py`, `run_A_missing_target_restart.py` | Published rat group MAT data; frozen start vectors/domain config | Support endpoint JSON and final all-target width/rank CSV |
| Matched measurement burden | `analysis/01_core_target_value/sparse/code/run_kp_design.py`, `run_bounded_local_geometry.py` | Same published rat input; frozen sensitivity and gradient intermediates | Sparse panel, bounded-width and budget tables |
| Mechanistic uncertainty | `analysis/02_robustness/variants/code/run_N_permutation_sensitivity.py` | Published model permutation fits and public rat data | Model-variant fits, sensitivity/utility tables and rank correlations |
| Robust scenario design | `analysis/02_robustness/robust/code/run_R_robust_panels.py` | Frozen model/domain sensitivity matrices | Scenario optima, selected robust panels and oracle retention |
| Independent frequency-domain prediction | `analysis/03_awinda/frequency/code/run_Q_awinda.py` | Licensed canonical exports of the fixed mouse source workbooks | Nested animal-held-out spectra and selected-pair errors |
| MgATP continuum | `analysis/03_awinda/continuum/code/run_U_mgatp_continuum.py` | Same mouse source cohort and fixed six targets | Target-frequency held-out utilities and animal-level selections |
| Utility landscape and unseen-target boundary | `analysis/03_awinda/landscape/code/run_yz_landscape.py` | Frozen all-frequency held-out mouse utilities | Additive/interaction target-holdout errors and selector regrets |
| Bounded human atrial support | `analysis/04_human_atrial/code/p2_human_stage2_bounded_nonlinear.py` | Apache-licensed public non-diabetic group data/fit | Two-radius support endpoints and exact width reductions |
| Rat drug-response prediction | `analysis/05_rlc1/code/RLC1_primary_analysis.py` | CC BY canonical seven-row repeated-dose summaries | Nested leave-one-rat-out predictions, candidates, MAEs and 720 label mappings |
| Fair human pacing comparison | `analysis/06_radbill/code/radbill_fair_comparators.py`, `radbill_within_family_perm_exact.py`, `RADBILL_primary_analysis.py` | CC0 canonical pacing table with prespecified gray exclusions | Four fair comparator errors, influence tables and 112/576 exact within-family mappings |
| Mechanical-protocol comparison | `analysis/07_tanner/code/analyze_tanner.py --outcomes primary` | CC BY canonical trabecula table and frozen protocol | Condition-held-out shared, target-specific, nearest-rate and random rules |
| Information/error divergence | `analysis/08_boundaries/preparation_support/code/run_B_human_heldout.py`, `run_I_support_error_linkage.py` | Public preparation observations; group reference used as a fitting start | Preparation-paired support widths and future stress errors |
| Dynamic and fixed-panel transfer | `analysis/08_boundaries/dynamic/code/run_T_dynamic_validation.py`, `transfer/code/evaluate_transfers.py` | Frozen source panels, canonical mouse and human spectra | Held-out dynamic predictions and transfer metrics |
| Kinetic and cross-system boundary | `analysis/08_boundaries/kinetics/code/run_V_kinetic_normalization.py`, `normalized_transfer/code/run_W_scoring.py`, `target_coordinates/code/run_X_comparisons.py` | Public kinetics and train-only frozen destination maps | Frequency/clock dispersion, rat-to-mouse transfer contrasts and target-holdout model comparisons |
| Cross-model surfaces and robust bands | `analysis/08_boundaries/state_surface/code/run_aa_surfaces.py`, `robust_bands/code/run_ab_robust.py` | Frozen model sensitivities and canonical external spectra | Ordinal frequency surfaces, robust rat bands and human transfer boundaries |

`scripts/run_analysis.py` runs each stage in the current checkout. Expensive
nonlinear endpoint optimization is documented and included but is not required
by the smoke test. `scripts/reproduce_quick.py` recomputes nested scalar results
and exact mappings, replays verified endpoint/intermediate tables, rebuilds all
figures and verifies the workbook/source hashes. The smoke report states this
scope explicitly; included code is not a claim that every optimizer was rerun.

Alternative-only model-rank reporting: `scripts/report_rank_summary.py` derives the four corrected medians from `analysis/02_robustness/variants/results/N_MODEL_RANK_STABILITY.csv`; `tests/smoke/check_reporting_patch.py` verifies exclusion of self-reference, all individual cells and the mouse aggregation contract.
