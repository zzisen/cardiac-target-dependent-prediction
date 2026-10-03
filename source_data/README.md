# Figure source data

The tables support five main figures and three supplementary figures. They retain the numerical precision of the prespecified analysis outputs. Presentation summaries use the calculations stated below; they do not refit models or add statistical tests.

| Figure | Panel | Source tables |
|---|---|---|
| 1 | a | `Fig1_same_measurement_support.csv` |
| 1 | b | `Fig1_measurement_target_map.csv` |
| 1 | c | `Fig1_equal_budget_values.csv`, `Fig1_equal_budget_summary.csv` |
| 2 | a, b | `Fig2_robust_target_panels.csv` |
| 2 | c | `Fig2_model_rank_correlations.csv` |
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

For Figure 1c, each bar is the median across four targets and two domains at a specified measurement budget. Missing heatmap cells indicate prespecified target-matched exclusions; negative support-width reductions are retained.

Figure 3a bars preserve the primary mean across 19 animal-condition cells per target. Individual points average the available treatment cells within each of 10 held-out mice. Figure 3b shows medians and interquartile ranges of those ten mouse-level selections.

Figure 4a uses seven aligned rat-level repeated-measures summary rows; row indices do not assert a named row-to-rat mapping. Rat points average six future-target errors within each row. Pacing points average the common-support target errors within each participant. Overall pacing means remain weighted by the 132 participant-target cells, matching the analysis. Eighteen participants contribute to this common support from the nineteen-person source cohort; the nineteenth participant has no common-support cell.

Figure 5a records every normalization explicitly: `normalized_error = source_error / no_measurement_error`, separately within each dataset. The datasets use different original error metrics and are not pooled. `biological_units` counts units contributing to the plotted comparison; `source_cohort_units` gives the source cohort size.

Figure 5b retains the reported point estimates and 95% mouse-cluster bootstrap intervals. Its final mean first averages the six target-budget contrasts within each mouse, then averages the ten mice. Figure 5c contains two target pairs from each of twenty human preparations; it does not treat forty target-level points as independent people. Support narrowing and error improvement are baseline minus augmented values. Donor linkage is unavailable, so the displayed correlations are descriptive.

The public plotting script reads these tables directly and produces 180-mm vector figures with editable Arial text plus RGB PNGs at 600 dpi. The analysis code and source provenance are documented elsewhere in the repository.
