# Method-definition reporting patch

Date: 2026-10-03. This note states the implemented definitions underlying the existing analyses. It does not change model fits, predictions, candidate sets, thresholds or scientific conclusions. Code citations below use paths relative to the public repository and one-based line numbers in the source inspected for this patch.

## Corrected summaries

Rank-correlation medians use the four mechanistic alternatives, excluding the reference model's self-comparison of 1. The complete self-comparison row remains in the source table.

| Target | Four-alternative median | Three-decimal report |
|---|---:|---:|
| ATP0.1 | 0.6089956165427863 | 0.609 |
| ATP1 | 0.6792834000381169 | 0.679 |
| Pi0 | 0.1074899942824471 | 0.107 |
| Pi5 | 0.1749190013340956 | 0.175 |

These values were recomputed directly from `analysis/02_robustness/variants/results/N_MODEL_RANK_STABILITY.csv`, filtering `variant != published_Model16D`. The source defines each correlation against the reference model over the common single-action vocabulary: `analysis/02_robustness/variants/code/run_N_permutation_sensitivity.py:209–222`.

Awinda nested-selected mean NRMSE gains, recomputed from `analysis/03_awinda/frequency/results/Q_AWINDA_SPARSE_PANEL_RESULTS.csv`, are **0.21791773469585646** and **7.449331876933430** percentage points for ATP0.1 and ATP1. The mean uses 19 mouse-by-genotype-by-drug cells from 10 mice, giving the reported **0.22 / 7.45** points. Source code: `analysis/03_awinda/frequency/code/run_Q_awinda.py:303–340`.

## Rat model support: objective, parameters and domains

For design d, the diagonal Gaussian working likelihood is proportional to exp(−S_d(p)/2), with S_d(p) = Σ_j [(h_j(p)−y_j)/SEM_j]^2. Published group SEMs weight the real and imaginary modulus components separately at all 17 frequencies and weight steady stress. The baseline contains 34 modulus scalars plus one stress scalar. An added protocol contributes one stress scalar or a complete 34-scalar modulus spectrum. The primary SEM multiplier is 1. The working covariance treats parts, frequencies, conditions and stress blocks independently; it is a group-summary likelihood rather than individual assay-noise calibration.

Within each domain and each baseline or augmented design, p_hat_d minimizes its own S_d. The feasible support uses S_d(p) ≤ S_d(p_hat_d) + **3.841458820694124**, the nominal χ²(1) 95% increment. Target q_t(p) = 100[F_t(p)/F_baseline(p)−1] is a percentage stress response. Endpoints are the minimum and maximum feasible q_t. Width is their difference; relative narrowing = 100(width_baseline−width_augmented)/width_baseline. Measurements collected at the target condition are excluded from that target's candidate set.

Sources: `analysis/01_core_target_value/support/code/aj_core.py:95–132,194–237`; `analysis/01_core_target_value/support/configs/A_E_F_J_freeze_2026-10-01.json:23–57`; `src/common/p2/constants.py:33–42`.

The reference Model16D has 14 free parameters. Published bounds are:

| Parameter | Lower | Upper |
|---|---:|---:|
| k1 | 0.5 | 200 |
| k−1 | 0.5 | 200 |
| k2 | 0.5 | 200 |
| k−2 | 0.1 | 200 |
| k3 | 0.5 | 200 |
| φx | 0.5 | 10 |
| φv | 0.005 | 0.5 |
| φl | 1 | 5 |
| K | 2,000 | 100,000 |
| Ks | 0 | 45 |
| φs,−2 | 1 | 1,000 |
| φs3 | 1 | 1,000 |
| kd,ATP | 0.1 | 5 |
| kd,Pi | 0.05 | 5 |

Source: `src/common/p2/constants.py:4–31`. Model16D fixes rapid-equilibrium ATP/Pi mode md=4 and strain dependence on k−2 and k3. Baseline ATP/Pi is 5/1 mM; ATP0.1 and ATP1 change ATP at Pi=1 mM; Pi0 and Pi5 change Pi at ATP=5 mM. Pi0 is evaluated numerically as 10^−6 mM.

The two local boxes use **natural logarithms** z_i = ln(p_i) for all positive parameters except Ks, for which z_Ks = Ks/45. Let z^(r) and z^(a) be the two serialized parameter anchors in `A_E_F_J_start_vectors.json`. For margin m=0.5 or 1.0, coordinate limits are max(z_global_lower, min[z^(r),z^(a)]−m) and min(z_global_upper, max[z^(r),z^(a)]+m). The box encloses both anchors before expansion. These are not symmetric radii around one fit, and the Ks coordinate is linear. A concise figure label can be “Coordinate margin ±0.5” / “Coordinate margin ±1.0”, with this construction defined in the legend and Methods. A phrase such as “log-relative parameter domain” must explicitly identify the Ks exception and two-anchor envelope.

Sources: `analysis/01_core_target_value/support/code/aj_core.py:66–83`; `analysis/01_core_target_value/support/configs/A_E_F_J_start_vectors.json:6–44`.

## Seven robust scenarios and the local metric

The seven scenarios are five model configurations in their published optimizer domains and two extra domain scenarios for the reference model. They are not a model-by-domain factorial. All five mechanistic scenarios use md=4; their strain-rate indicators are indexed as (k1,k−1,k2,k−2,k3).

| Scenario | Model identifier | Strain-dependent rates | Parameter domain |
|---|---|---|---|
| Reference mechanism | published_Model16D (s=16) | k−2, k3 | Published optimizer bounds |
| Alternative mechanism 1 | md4_best_alternative (s=14) | k2, k−2 | Published optimizer bounds |
| Alternative mechanism 2 | md4_second_best_alternative (s=9) | k1, k−2 | Published optimizer bounds |
| Single-strain mechanism | md4_best_single_strain (s=5) | k−2 only | Published optimizer bounds |
| Mechanism without k−2 strain | md4_best_without_kminus2_strain (s=4) | k2 only | Published optimizer bounds |
| Reference domain 0.5 | Model16D | k−2, k3 | Two-anchor transformed-coordinate margin ±0.5, intersected with published bounds |
| Reference domain 1.0 | Model16D | k−2, k3 | Two-anchor transformed-coordinate margin ±1.0, intersected with published bounds |

The alternative configurations were selected from published source objectives plus the stated one-strain and absent-k−2 complexity/mechanism classes. Disabled strain coefficients are fixed at zero. These labels name the implemented rates rather than invented biological mechanisms. Sources: `analysis/02_robustness/variants/code/run_N_permutation_sensitivity.py:26–45,56–60`; `analysis/02_robustness/variants/code/permutation_model.py:9–21,29–34`; `analysis/02_robustness/robust/code/run_R_robust_panels.py:119–154`; `analysis/02_robustness/robust/results/R_SCENARIO_DEFINITIONS.csv`.

For a scenario, J is the baseline standardized-observation Jacobian and g is the target gradient in that scenario's coordinates. V = pinv(JᵀJ, rcond=10^−11); metric singular values below the relative cutoff are discarded, rather than inverted. No added ridge enters this robust/rank metric. For candidate rows R, v0 = max(0,gᵀVg), v_R = max[0,v0−cᵀ(I+RVRᵀ)^−1c], c=RVg. Utility is the nonnegative variance reduction u_R=v0−v_R. Oracle utility is the maximum over admissible same-budget panels within that scenario. Oracle retention is u_R/max_R u_R; a zero-utility oracle yields zero retention in this implementation. The robust target-specific panel maximizes the minimum retention across seven scenarios, breaking ties by maximum regret, then equal-weight mean retention, then panel identifier. Raw variances are not averaged across parameterizations. Near-singular calculations remain local design summaries.

Sources: `analysis/01_core_target_value/sparse/code/run_kp_design.py:118–131`; `analysis/02_robustness/robust/code/run_R_robust_panels.py:39–49,157–176,219–280`. Assay-unit budgets count one stress scalar or a real/imaginary frequency pair as one unit. Scalar budgets count individual components. Same-target-condition actions are excluded.

## Awinda animal-held-out prediction

Baseline features are measured at **5 mM MgATP**, active-state pCa **4.8**. Quality==1 is required. Fibres are averaged within animal×genotype×drug×MgATP×frequency before prediction. The four genotype×drug groups are WT-RLC or N47K, each with Control or 0.3 μM mavacamten. Complete spectra contain 72 real and 72 imaginary components.

For each spectral output, y = Gγ + Zβ + ε. G contains one indicator/intercept for each genotype×drug group, with no additional pooled intercept; Z contains the real/imaginary components of the selected baseline frequency. Group coefficients are unpenalized. Measurement coefficients use ridge penalty **α=1.0**, minimizing ||y−Gγ−Zβ||² + α||β||². Each inner and outer training fold computes feature means and population SDs (ddof=0); SD≤10^−12 is replaced by 1. Outcomes remain in kPa.

The outer split holds out all condition cells and fibres from a mouse. Within the remaining mice, frequency selection uses leave-one-mouse-out prediction. Each inner held-out mouse is scored by mean squared spectral error divided by the mean squared observed spectral amplitude, pooling that mouse's cells and real/imaginary components; the criterion is the equal mean across inner mice. The lowest score wins, with candidate ID as tie-breaker. The selected pair is refitted on the complete outer training fold.

For held-out cell c, **NRMSE_c = 100 sqrt[(1/144) Σ_j(ŷ_cj−y_cj)²] / sqrt[(1/144) Σ_j y_cj²]**. The denominator is the RMS of that cell's observed target spectrum, not a training SD, range or mean amplitude. Figure 3a bars are the arithmetic mean over **19 condition cells**; mouse points first average that mouse's condition cells, giving **10 mouse summaries**. Gains are group-only NRMSE minus selected-pair NRMSE in percentage points. The utility landscape instead averages cells within mouse and then equally over mice. These two aggregation orders must not be silently interchanged.

Sources: `analysis/03_awinda/frequency/code/run_Q_awinda.py:38–41,52–111,127–193,204–213,303–340`; `analysis/03_awinda/landscape/code/run_yz_landscape.py:179–210`.

## RLC-1 and Tanner prediction

RLC-1 uses **seven rat-level repeated-measures summary rows**, aligned across source tables; the published final workbook does not provide explicit row IDs. Candidates at **1 μM** are peak twitch tension, time to peak tension and time to 50% relaxation. The same three outcome families at **3 and 10 μM** define six targets. RT90 exactly duplicates RT50 and is excluded. Each fold standardizes both x and y using training mean and sample SD (ddof=1). It fits y_z=b0+b1x_z by ordinary least squares. The outer split leaves one rat row out; inner leave-one-rat-out selection uses standardized MAE. A global rule chooses one candidate by equal mean inner MAE over six targets; a target-specific rule selects independently for each target. Outer error is |y−ŷ|/SD_training(y), averaged over 42 rat×target cells. The reported **14.5042293082%** equals 100(MAE_global−MAE_target)/MAE_global, with MAEs 0.998561563439 and 0.853727904494; it compares these two particular rules in seven biological units. It does not imply 42 independent rats or superiority over every conventional rule.

Sources: `analysis/05_rlc1/code/RLC1_primary_analysis.py:28–35,71–107,115–140,144–175,214–238`. Peak tension is relative source units (ND=100); timing is in ms as interpreted from the source time axis.

Tanner's primary response is **peak stress response**, in **mN/mm²**. Candidate conditions are **10 ms time-to-stretch at 100 or 250 s^−1**. Future targets are **100 ms at 25, 100, 250 or 1,000 s^−1**. The biological unit is the trabecula (seven units; 28 primary target predictions). Univariate ordinary least-squares regression includes an intercept and uses native units. Outer leave-one-trabecula-out evaluation nests inner leave-one-trabecula-out candidate selection by MAE; a global candidate minimizes mean MAE over four targets, while a target-specific candidate minimizes its target's MAE. The nearest conventional rate minimizes absolute log10-rate distance, with lower-rate tie-breaking. Random-policy expectation exactly enumerates 2^4=16 assignments. The retrospective oracle chooses per target using all outer held-out errors and reuses them for scoring; it is an optimistic reference, not a deployable rule or a proven error bound.

Sources: `analysis/07_tanner/code/analyze_tanner.py:21–28,72–94,113–166,210–238`.

## Radbill participant-held-out prediction

The biological unit is the participant. **QT** means QT interval (ms); **PP** means pulse pressure, systolic minus diastolic blood pressure (mmHg). **DT5** is the fifth beat of the pacing drive train; **DTend** is its **penultimate beat**. These are exact source definitions from `HCMpacingstudydatadryadreadme.txt:97–106`, with PP difference checked against cached workbook formulae by `analysis/06_radbill/code/RADBILL_primary_analysis.py:138–148`.

The four candidate scalars at **100 bpm** are DT5 QT, DT5 PP, DTend QT and DTend PP. Future targets are DTend QT and PP at **110 bpm, 120 bpm, the first 130 bpm drive train and the last 130 bpm drive train**. Source-author gray unreliable cells are designated missing before fitting or validation; no outcome-driven deletion is used. The outer split leaves one complete participant out. An inner participant holdout chooses candidates using standardized MAE. Means and sample SDs (ddof=1) are estimated within each training fold from available measurements; OLS with intercept and slope uses paired nonmissing predictor/target rows, requiring at least three pairs. Errors are divided by the target training SD.

The family-aware global comparator chooses DT5 or DTend within the target's QT/PP family by mean inner score over that family's four rate targets. The fixed conventional comparator uses DTend QT for QT and DTend PP for PP. Target-specific C2 chooses among all four candidates separately for each target. Fair comparison uses the same **132 participant×target cells from 18 contributors in the 19-person source cohort** on which all five implemented comparators are valid. Participant dots average available target cells, whereas overall diamonds average all common-support cells.

The within-family mapping analysis enumerates **4! QT × 4! PP = 576** mappings of training-derived target-specific selection labels, preserving response families and the original all-four-candidate common-support mask. It counts mappings with mean standardized error no greater than identity (+10^−12 tolerance). The fraction **112/576** is descriptive. Its identity MAE **0.530432** and fair-comparison C2 MAE **0.520303** refer to distinct support definitions.

Sources: `analysis/06_radbill/code/RADBILL_primary_analysis.py:22–38,50–52,125–136,197–245`; `analysis/06_radbill/code/radbill_fair_comparators.py:21–49`; `analysis/06_radbill/code/radbill_within_family_perm_exact.py:19–37`.

## Utility landscape and bootstrap definitions

The six MgATP targets are 0.05, 0.10, 0.25, 0.50, 1.00 and 2.50 mM. Let x_t and x_f be log10(target) and log10(frequency), centered and divided by population SD over the full prespecified six-target and 72-frequency grids. Scaling uses design coordinates, not outcomes. Additive utility is β0+βt x_t+βtt x_t²+βf x_f+βff x_f²; interaction adds βtf x_t x_f. Ordinary least squares fits complete utility curves at the five training targets, then predicts the withheld sixth curve. Curve RMSE/MAE use frequency-wise utility errors in percentage points.

The observed/predicted best frequency is the first maximizing point on the ascending frequency grid. A best-band hit compares the low/mid/high **frequency-grid tertile**, 24 ordered frequencies each, containing those maximizing points. This differs from the top-24 utility-ranked set. Near-optimal frequencies have observed gain within **0.5 percentage points** of the observed maximum. Regret is observed maximum utility minus observed utility at the selected frequency. Oracle retention is observed utility at that frequency divided by observed maximum utility when the latter is positive.

Landscape intervals resample **whole mice**, after averaging condition cells within each mouse. The same mouse-vector resample preserves every frequency and target: **10,000** samples of ten mice with replacement, seed **20261003**, with **2.5th/97.5th percentile** intervals for mean utility. Prediction models are not refitted during this curve bootstrap. The separate MgATP low-versus-high selected-panel summary uses **50,000** paired mouse resamples; do not describe all bootstrap outputs as sharing one resample count.

Sources: `analysis/03_awinda/landscape/code/run_yz_landscape.py:36–46,70–118,130–175,179–221,236–264`; `analysis/03_awinda/continuum/code/run_U_mgatp_continuum.py:201–210`.

Figure 5b rat-to-mouse transfer first averages condition-state error within each mouse. Each target×budget contrast uses **20,000** paired mouse bootstrap samples. Its aggregate first averages the **six target×budget contrasts (two targets × budgets 1–3)** within each mouse and uses **50,000** paired whole-mouse samples; the interval is the 2.5th/97.5th percentiles. The RNG is initialized with seed **20261002**. The plotted difference is kinetic-normalized NRMSE minus absolute-frequency NRMSE. Thus positive values mean worse normalized transfer. The equal-mouse aggregate is 0.9174585907532414 points with interval [−0.0079292922378050,1.9865269362525613].

Source: `analysis/08_boundaries/normalized_transfer/code/run_W_scoring.py:98–107,120–166`; `source_data/main/Fig5_rat_mouse_transfer.csv`.

## Human group support and preparation-level local support

The human model has 11 free parameters; φx, Ks and kd,Pi remain fixed at their published reference values. Free parameter order and lower/upper limits are:

| Parameter | Lower | Upper |
|---|---:|---:|
| k1 | 0.5 | 200 |
| k−1 | 0.5 | 200 |
| k2 | 0.5 | 200 |
| k−2 | 0.1 | 500 |
| k3 | 0.5 | 200 |
| φv | 0.005 | 0.5 |
| φl | 1 | 5 |
| K | 2,000 | 10,000 |
| φs1 | 1 | 1,000 |
| φs3 | 1 | 1,000 |
| kd,ATP | 0.02 | 5 |

The group support comparison uses the public non-diabetic reference model. z_i=ln(p_i/p_reference,i) is restricted to ±0.25 or ±0.50, intersected with published parameter bounds. Baseline observations are the first 11 complex-modulus frequencies and steady stress; the added measurement is ATP0.1 stress. Group SEMs separately weight real/imaginary modulus and stress. Each design minimizes its own objective, with support cutoff equal to its own minimum plus 3.841458820694124.

Sources: `analysis/04_human_atrial/code/p2_human_local_information_pilot.py:24–28,117–129`; `analysis/04_human_atrial/code/p2_human_stage2_bounded_nonlinear.py:22–23,35–65,140–145`.

Preparation-level fits use all published human bounds for the same 11 free parameters, not the ±0.25/0.50 group-reference domains. Within each diabetic/non-diabetic group and training condition, residual scales are the **between-preparation sample SDs** of real/imaginary modulus at each frequency and of stress, floored at 10^−8. Only baseline and added ATP0.1 observations estimate these scales. Each preparation is fitted separately using baseline mechanics and again with the added ATP0.1 stress; ATP1/Pi10 outcomes are withheld.

At each fit, J is the weighted-residual Jacobian in log-relative coordinates, H=JᵀJ and q≈q_fit+gᵀδ is the first-order target response. Support endpoints minimize/maximize that linear response under δᵀHδ≤3.841458820694124 and published parameter bounds relative to **that fit**. Full-rank unconstrained solutions are used when inside the box; otherwise the ellipsoid–box problem is solved by constrained optimization. The Jacobian, target gradient, center and relative bounds are recomputed after refitting. The resulting baseline and augmented local support sets are **not nested and are not calibrated prediction intervals**. Negative narrowing therefore denotes widening of the recomputed local approximation, rather than an added constraint widening one unchanged feasible set.

Required Figure 5c legend sentence: “Support widths were recomputed after refitting in each measurement condition; negative narrowing denotes widening of this local support approximation.”

Sources: `analysis/08_boundaries/preparation_support/code/run_B_human_heldout.py:101–145,166–207,211–275`; `analysis/08_boundaries/preparation_support/code/public_human_model_evaluator.py:24–28`. Width/error pairs and their descriptive Spearman correlations are computed within target over 20 preparation rows by `analysis/08_boundaries/preparation_support/code/run_I_support_error_linkage.py:23–44,57–93`. Donor clustering is unavailable; these 20 preparations are not asserted to be 20 independent patients.

## Patch verification scope

The two corrected reporting checks were replayed from existing public numerical tables without rerunning scientific fits. All definitions above were extracted read-only from the cited code, configuration and original Radbill source README. No datasets, endpoints, model configurations, fitting algorithms or held-out predictions were added or altered by this extraction.
