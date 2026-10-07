# V3 reproduction scope

Use the exact Windows Conda binary lock `environment-v3-win64-explicit.txt` and install `requirements-v3.txt` after activation. This retains Matplotlib 3.10.0 built against FreeType 2.13.3. Pip's same-version Matplotlib wheels use FreeType 2.6.1 and produce different PNG/PDF bytes; dependency version labels alone do not identify identical renderer binaries. All commands below are run at the repository root and require a fresh output directory. They preserve supplied outputs. `requirements.txt` and older analysis entry points describe the historical implementation; the V3 commands select the new contracts explicitly.

## Quick offline replay

```sh
python scripts/reproduce_v3_quick.py --output-dir replay-v3
```

This verifies the release manifest; imports all published scientific modules; evaluates the closed-form two-action Gaussian benchmark; regenerates all 15/35 matched-partition risks and ranks from held-out cell losses; independently repeats 44-patient nested selection, fitting, and all 924 policy loss values from compact phospho summaries; independently repeats full-training channel selection and fits and all 54 Stanford transport losses; reaggregates four adaptive risks from recorded predictions; and rebuilds all 15 figure exports byte for byte in the pinned environment. It does not repeat FCS decoding or every cardiac fit. Saved outputs are verification references, not additional training input to the independent replay.

## Full prespecified cardiac fitting

```sh
python scripts/reproduce_v3_cardiac.py --analysis fixed --output-dir refit-fixed
python scripts/reproduce_v3_cardiac.py --analysis matched --output-dir refit-matched
python scripts/reproduce_v3_cardiac.py --analysis adaptive --output-dir refit-adaptive
```

The fixed command repeats RLC family and Awinda fixed-policy calculations; historical Tanner fixed estimates remain reproducible using `analysis/07_tanner/code/analyze_tanner.py`, and Radbill strict-support fixed policies are included in the V3 adaptive module (`recompute_radbill_strict_fixed_policies`). The matched command repeats all 15 RLC and 35 Radbill partitions. The adaptive command repeats the same three-layer nested action/resolution selection in all four systems; `--system RLC1_RAT` (or another registered system) restricts a deterministic rerun to one system. These are computationally heavier than the quick replay. All requisite compact cardiac inputs are bundled, including the attributed small CC-BY RLC workbook; full raw source reconstruction requires the upstream downloads listed in SOURCE_DATA.md.

Scientific function bodies in `cardiac_fixed.py`, `matched_families.py`, and `adaptive_resolution.py` were extracted verbatim from the prespecified implementations. A recorded AST comparison checks this identity. Only dependency globals for release-local file paths changed; private workflow checks and administrative packaging were omitted. See SCIENTIFIC_CODE_PROVENANCE.json. No target, action, support, estimator, loss, fold assignment, tie rule, or comparator changed.

## Gaussian finite-action numerical verification

```sh
python analysis/v3/code/finite_action_verification.py
```

Run a copied script in a fresh work directory if retaining the emitted CSV: its output defaults to the script's directory. This repeats the existing finite-action Gaussian examples, including multivariate probabilities, seeded Monte Carlo consistency checks, tied oracle actions, and singular contrast laws. It is a verification of declared theory examples, not a fit to the biological cohorts. Monte Carlo checks use explicit tolerances; the quick command only repeats the inexpensive analytic two-action benchmark.

## Raw-source parsing and training/transport

```sh
python scripts/reproduce_v3_raw_cytometry.py --ddpr-inner /path/to/ddpr_data.zip --output-dir raw-replay
```

Download the 5.86 GB upstream inner archive separately. This optional full parsing command first verifies its declared SHA-256 and then uses the original streaming FCS decoder for exactly 352 training and 27 Stanford files, preserving the source threshold (>=10) and eight-channel vocabulary. It repeats the training execution and independently verifies Stanford transport from the fixed training registry. It never refits a policy on Stanford data. A release-local outer ZIP wrapper accommodates independent downloads while preserving every inner-archive byte and member hash; the historical wrapper itself is not required. Expect substantial disk use and time.

The original `training_execution.py`, `transport_execution.py`, and `transport_training_models.py` preserve the historical implementation functions. Their administrative main methods refer to private pre-outcome provenance and are not the supported portable entry points. Use the release scripts, which verify the public manifests/model registry and invoke unchanged scientific functions. The original independent replay files are unchanged and execute in isolated directories.

## Figures and numerical interpretation

```sh
python figures/v3/figures_src/build_figures.py
```

Use the quick command to rebuild in a separate directory. Figure 1C contains arbitrary discrete schematic points with no empirical values. Figure 5D contains all 44 patient deletions, points only, retaining source order. Full-cohort dashed references do not join patients. Figures 2–4 retain the original numeric values and PNG/PDF bytes; SVG text is represented by self-contained vector glyph outlines. No new supplementary figures were added. Do not pool the native scales, interpret descriptive ranks as p-values, interpret selected-fold counts as population probabilities, or treat the strongest realized fixed estimate as a population oracle.
