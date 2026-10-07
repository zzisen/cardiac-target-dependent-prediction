# Future targets shape the predictive value of cardiac mechanical measurements

This software accompanies the cardiac Analysis by Zisen Zhou, Yaopu Zhang and
Haoran Pang. It includes the executable analyses supporting every main result,
their frozen protocols and intermediate inputs, verified outputs, all figure
source data and editable vector figures.

Measurement value depends on the future response being predicted. The analyses
compare matched budgets, mechanistic uncertainty, independent muscle experiments,
drug responses and human pacing. Local support and held-out prediction remain
separate estimands. The cross-system ladder normalizes errors within each dataset;
there is no pooled effect or meta-analysis.

## Quick reproduction

Use Python 3.10 or newer in a fresh environment:

```sh
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install -r requirements.txt
python scripts/reproduce_quick.py
```

This recomputes scalar held-out predictions, fair pacing comparators and exact
label-mapping diagnostics from canonical observations, replays expensive fitted
endpoints, rebuilds the five main and three supplementary figures, and verifies
the source-data workbook and plotting-input hashes. See REPRODUCTION_SMOKE_TEST.md.

## Analysis behind each main result

| Main result | Executable analyses |
|---|---|
| Future targets change measurement value | `analysis/01_core_target_value/`: nonlinear support and matched-budget sparse panels |
| Target dependence persists under uncertainty | `analysis/02_robustness/`: model-family fits and robust scenario enumeration |
| Frequency-domain experiments reveal a target-conditioned landscape | `analysis/03_awinda/`: nested animal-held-out prediction, MgATP continuum and utility-surface prediction |
| Pharmacological and human perturbations extend the principle | `analysis/04_human_atrial/`, `analysis/05_rlc1/`, `analysis/06_radbill/`: bounded group-model support, nested leave-one-rat-out prediction and leave-one-participant-out fair comparisons |
| Target dependence has a protocol boundary | `analysis/07_tanner/`, `analysis/08_boundaries/`: condition-holdout, paired information/error, dynamic prediction, transfer, kinetic and robust-band calculations |

`docs/CODE_COVERAGE_MAP.md` links each claim to its specific script, permitted
input and frozen output. Numerical upstream identifiers remain inside the frozen
implementation and configuration files so the code can be checked against its
verified source; reader-facing figures and documentation use scientific names.

## Full analysis

```sh
python scripts/run_analysis.py --stage awinda
python scripts/run_analysis.py --stage radbill
python scripts/run_analysis.py --stage rlc1
python scripts/run_analysis.py --stage robustness
python scripts/run_analysis.py --stage human
```

Run `python scripts/run_analysis.py --help` for the other stages. Full nonlinear
support fits can take hours and overwrite their staged result files; run them in
a copy of the release. The quick command never needs those optimizers. Full code
for every retained pipeline is included. The release verifies code/import/input
coverage and quick outputs; it does not assert a new run of every optimizer.

## Data and reuse

DATA_SOURCES.md gives source DOIs, exact filenames, versions, checksums and fetch
instructions. Licensed canonical tables are provided for offline reproduction.
Third-party raw workbooks, private data, correspondence and credentials are absent.
The Awinda analyses share one ten-mouse cohort. Rat dose predictions use seven
aligned rat-level summary rows; preparation/target cells are not biological n.

Project-authored code is MIT licensed; derived data, figures and documentation
are CC BY 4.0. Upstream inputs retain their original terms. See LICENSE,
DATA_LICENSE.md and THIRD_PARTY_LICENSES.md. The Zenodo software record uses MIT
at record level with this mixed-license policy described explicitly.

Repository: [zzisen/cardiac-target-dependent-prediction](https://github.com/zzisen/cardiac-target-dependent-prediction). Release version: v1.0.2.

Frozen input bytes are preserved by .gitattributes so recorded hashes agree across operating systems and release archives.

Archived release: [v1.0.2](https://github.com/zzisen/cardiac-target-dependent-prediction/releases/tag/v1.0.2). Zenodo DOI: [10.5281/zenodo.23114693](https://doi.org/10.5281/zenodo.23114693).
The archive records the exact v1.0.2 tagged tree, including the version DOI reserved before publication.

## Reporting patch v1.0.2

Alternative-model rank medians exclude the reference-model self-comparison.
They are 0.609, 0.679, 0.107 and 0.175 for ATP0.1, ATP1, Pi0 and Pi5, respectively.
All 20 individual rank correlations remain unchanged. Run
`python scripts/report_rank_summary.py` to reproduce the summaries from the
original mechanistic results. The four corresponding workbook medians are
explicit Excel formulae that exclude the reference rows.

Methods definitions, the seven model/domain scenarios and source-code locations
are documented in `docs/METHOD_DEFINITIONS.md`. Figure annotations and legends
have been refined; the AI-assisted-tools disclosure is finalized in
`docs/AI_ASSISTED_TOOLS.md`. Underlying model fits, held-out predictions, primary
analyses and scientific conclusions are unchanged.
