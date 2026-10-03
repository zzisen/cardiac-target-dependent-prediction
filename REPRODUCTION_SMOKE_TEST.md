# Reproduction smoke test

Result: **PASS**. Checked on 3 October 2026 in one fresh directory copied from
the final release and a newly created Python virtual environment. Requirements
were installed from the pinned requirements.txt, and pip check reported no
broken requirements. The run used Python 3.10.18, NumPy 2.2.5, pandas 2.3.1,
SciPy 1.15.3 and Matplotlib 3.10.0 on Windows.

Command: `python scripts/reproduce_quick.py`.

- Recomputed or replayed 134 headline values from permitted canonical observations
  and frozen intermediate results. Nonlinear optimizer fits were replayed; no new
  run of every optimizer is claimed.
- Recomputed all four inhibitor comparator errors, every selected input value,
  shared/target-specific/oracle prediction, and all 720 target-label mappings.
- Recomputed fair pacing MAEs: 0.819522046100299 (no measurement),
  0.5634352209353645 (family global), 0.48818911704682866 (fixed family), and
  0.5203030666697132 (target-specific), with rounding tolerance below 2e-12.
  Reproduced 112/576 = 0.19444444444444445 within-family mappings.
- Recalculated the stretch-comparator errors, all 11 within-dataset normalized
  error ratios, the two support/error correlations, and verified seven frozen
  rat-to-mouse transfer point estimates and intervals.
- Rebuilt all five main and three supplementary figures as editable PDF/SVG and
  600 dpi RGB PNG. Verified 180 mm vector artboards and absence of raster objects.
- Verified the complete source workbook: 27 scientific tables, 14,313 data cells
  and 11 explicit normalization formulae, including source CSV and workbook hashes.
- Imported all 36 analysis modules, found no missing
  statically resolvable CSV input, and checked 284 frozen
  code/input/output hashes.
- Scanned 360 release files: no private/prohibited filename,
  credential-like content, local absolute path or unlisted raw workbook was found.

Full numerical pipelines and frozen configurations remain available via
`scripts/run_analysis.py --stage ...`; the code coverage map connects all five
Results subsections to exact scripts and inputs. Running full support optimizers
can take hours. The quick test requires no external download or network access.
