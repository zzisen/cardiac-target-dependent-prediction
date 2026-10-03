# Reproduction smoke test — v1.0.2

Result: **PASS** for the numerical, reporting, figure, workbook and release
checks. The release procedure also runs this complete command in a fresh
directory extracted from the exact tagged ZIP before publication; its execution
report accompanies the final review package. Requirements were installed in a
new Python virtual environment from the pinned requirements.txt; pip check found no
broken requirements. The run used Python 3.10.18, NumPy 2.2.5, pandas 2.3.1,
SciPy 1.15.3 and Matplotlib 3.10.0 on Windows.

Command: `python scripts/reproduce_quick.py`.

- Recomputed or replayed 140 headline values from permitted canonical observations
  and frozen intermediate results. Nonlinear optimizer fits were replayed; no new
  run of every optimizer is claimed.
- Corrected the four alternative-model rank medians to 0.609, 0.679, 0.107 and
  0.175 by excluding the reference self-comparison. Verified all 20 individual
  correlations against the unchanged model output, and verified the mouse
  0.22/7.45 percentage-point gains with the original 19-cell/10-mouse aggregation.
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
- Verified the complete source workbook: 28 scientific tables, 14,357 data cells,
  11 explicit normalization formulae and four alternative-only MEDIAN formulae,
  including source CSV and workbook hashes.
- Imported all 36 analysis modules, found no missing
  statically resolvable CSV input, and checked 284 frozen
  code/input/output hashes.
- Scanned 370 release files: no private/prohibited filename,
  credential-like content, local absolute path or unlisted raw workbook was found.

Full numerical pipelines and frozen configurations remain available via
`scripts/run_analysis.py --stage ...`; the code coverage map connects all five
Results subsections to exact scripts and inputs. Running full support optimizers
can take hours. The quick test requires no external download or network access.
