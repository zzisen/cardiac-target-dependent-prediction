# Figure source data

The tables support five main figures and three supplementary figures. They retain the numerical precision of the prespecified analysis outputs. Presentation summaries use the calculations stated below; they do not refit models or add statistical tests.

| Figure | Panel | Source tables |
|---|---|---|
| 1 | a | `Fig1_same_measurement_support.csv` |
| 1 | b | `Fig1_measurement_target_map.csv` |
| 1 | c | `Fig1_equal_budget_values.csv`, `Fig1_equal_budget_summary.csv` |
| 2 | a, b | `Fig2_robust_target_panels.csv` |
| 2 | c | `Fig2_model_rank_correlations.csv`, `Fig2_alternative_model_rank_summary.csv` |
| 3 | a | `Fig3_animal_held_out_values.csv`, `Fig3_animal_held_out_summary.csv`, `Fig3_mouse_unit_errors.csv` |
| 3 | b | `Fig3_frequency_selection_values.csv`, `Fig3_frequency_selection_summary.csv` |
| 3 | c | `Fig3_target_conditioned_landscape.csv`, `Fig3_held_out_target_models.csv` |
| 4 | a | `Fig4_rat_held_out_predictions.csv`, `Fig4_rat_unit_errors.csv`, `Fig4_rat_comparator_summary.csv` |
| 4 | b | `Fig4_human_atrial_support.csv` |
| 4 | c | `Fig4_participant_held_out_predictions.csv`, `Fig4_participant_unit_errors.csv`, `Fig4_pacing_comparator_summary.csv` |
| 5 | a | `Fig5_within_dataset_normalization.csv` |
| 5 | b | `Fig5_rat_mouse_transfer.csv` |
| 5 | c | `Fig5_support_and_accuracy_pairs.csv`, `Fig5_support_accuracy_correlations.csv` |
| Supplementary 1 | a | `Fig4_rat_comparator_summary.csv` |
| Supplementary 2 | a, b | `Fig4_pacing_comparator_summary.csv`, `Fig4_participant_unit_errors.csv`, `FigS2_within_family_mapping.csv` |
| Supplementary 3 | a | `FigS3_mechanical_comparator_summary.csv` |

For Figure 1, the domains labelled “Log-coordinate margin ±0.5” and “Log-coordinate margin ±1.0” use the coordinate box enclosing two reference/alternative parameter vectors, expanded by the stated margin and intersected with published global bounds. Coordinates are natural logs except the linear coordinate Ks/45. Figure 1c bars are medians across four targets and two domains at a specified measurement budget. Grey NA heatmap cells indicate prespecified target-matched exclusions; all negative and near-zero support-width reductions remain visible numerical values.

Figure 2c preserves all 20 individual correlations. `model_id` identifies the reference strain formulation (s16: k−2 and k3) and the four alternative strain formulations (s14: k2 and k−2; s9: k1 and k−2; s5: k−2 alone; s4: k2 alone). All use the same rapid-equilibrium ATP/Pi formulation. The grey Reference model row displays the self-comparison, correlation 1. `Fig2_alternative_model_rank_summary.csv` reports medians across the four true alternatives only, explicitly setting `reference_self_comparison_included` to false. Full-precision medians and three-decimal display values (0.609, 0.679, 0.107 and 0.175 for ATP0.1, ATP1, Pi0 and Pi5) are exported by `scripts/summarize_model_ranks.py`.

Figure 3a bars preserve the primary mean across 19 animal-condition cells per target. Individual points average the available treatment cells within each of 10 held-out mice. Gains of 0.22 and 7.45 percentage points are annotations rounded from 0.21791773469585643 and 7.4493318769334298; weighting is unchanged. Figure 3b shows medians and interquartile ranges of those ten mouse-level selections.

Figure 4a uses seven aligned rat-level repeated-measures summary rows; row indices do not assert a named row-to-rat mapping. Rat points average six future-target errors within each row. Pacing points average the common-support target errors within each participant. Overall pacing means remain weighted by the 132 participant-target cells, matching the analysis. Eighteen participants contribute to this common support from the nineteen-person source cohort; the nineteenth participant has no common-support cell.

Figure 5a records every normalization explicitly: `normalized_error = source_error / no_measurement_error`, separately within each dataset. The datasets use different original error metrics and are not pooled. `biological_units` counts units contributing to the plotted comparison; `source_cohort_units` gives the source cohort size.

Figure 5b retains the reported point estimates and 95% mouse-cluster bootstrap intervals. Its final mean first averages the six target-budget contrasts within each mouse, then averages the ten mice. Figure 5c contains two target pairs from each of twenty human preparations. Support widths were recomputed after refitting in each measurement condition using a weighted-residual Jacobian, a first-order target approximation and published parameter bounds; the resulting local sets are not nested or calibrated prediction intervals. Negative narrowing denotes widening of this local approximation. Support narrowing and error improvement are baseline minus augmented values. Descriptive Spearman correlations of −0.22 for ATP1 and −0.52 for Pi10 are reported in the legend and source table. Donor linkage is unavailable.

Supplementary Figure 3 reports peak-stress prediction MAE in mN/mm². Candidates are 10-ms perturbations at 100 or 250 s⁻¹; future responses are 100-ms perturbations at 25, 100, 250 or 1000 s⁻¹. The retrospective oracle is an optimistic reference based on held-out outcome selection.

The public plotting script reads these tables directly and produces 180-mm vector figures with editable Arial text plus RGB PNGs at 600 dpi. The analysis code and source provenance are documented elsewhere in the repository.
