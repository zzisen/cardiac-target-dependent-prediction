# Public data provenance

All external observations are retrospective secondary data. No new samples,
animals or participants were recruited. The release includes licensed canonical
exports, frozen inputs and derived results. Original third-party workbooks and
archives are not distributed; the small published model inputs retain Apache
2.0 notices. These sources are fixed to the versions below.

| Dataset | Version | Exact filename | Source | License |
|---|---|---|---|---|
| Awinda mouse mechanics | Figshare v1 | `Noise_analysis_ABC_GWN_WithFits_DQ1_ReFits_2020_08_04_Ver_2024.xlsx` | [Download/source](https://ndownloader.figshare.com/files/49733052) | CC BY 4.0 |
| Awinda mouse mechanics | Figshare v1 | `Transgenic_mice_Fiber_Tensions_with_Fits_DQ1_2019_09_06.xlsx` | [Download/source](https://ndownloader.figshare.com/files/49732494) | CC BY 4.0 |
| Awinda mouse mechanics | Figshare v1 | `WT_RLC_N47K_Control_Mavacamten_XLD_GWN_DQ1_2019_09_09.xlsx` | [Download/source](https://ndownloader.figshare.com/files/49732545) | CC BY 4.0 |
| Regulatory-light-chain inhibitor | Figshare v3 | `RLC-1_Fig6.xlsx` | [Download/source](https://ndownloader.figshare.com/files/46186848) | CC BY 4.0 |
| Tanner trabeculae | Figshare v1 | `tanner et al PLOSone_data.xlsx` | [Download/source](https://ndownloader.figshare.com/files/43612779) | CC BY 4.0 |
| Human atrial experiments | Figshare v1 | `Human atrial data.zip` | [Download/source](https://ndownloader.figshare.com/files/51622616) | CC BY 4.0 |
| Radbill pacing | Dryad v3 | `HCM_pacing_study_data_Dryad.xlsx` | [Download/source](https://datadryad.org/api/v2/files/676231/download) | CC0 1.0 |
| Radbill pacing | Dryad v3 | `HCMpacingstudydatadryadreadme.txt` | [Download/source](https://datadryad.org/api/v2/files/676230/download) | CC0 1.0 |
| Published human model | b22e5bef970adb95b7dc413979d8cd1c2ca482b3 | `ave_human_fitting_data.mat` | [Download/source](https://github.com/JuliaMusgrave/AtrialModel_2025_Human/tree/b22e5bef970adb95b7dc413979d8cd1c2ca482b3) | Apache 2.0 |
| Published human model | b22e5bef970adb95b7dc413979d8cd1c2ca482b3 | `LICENSE` | [Download/source](https://github.com/JuliaMusgrave/AtrialModel_2025_Human/tree/b22e5bef970adb95b7dc413979d8cd1c2ca482b3) | Apache 2.0 |
| Published human model | b22e5bef970adb95b7dc413979d8cd1c2ca482b3 | `ND_xb_fit.mat` | [Download/source](https://github.com/JuliaMusgrave/AtrialModel_2025_Human/tree/b22e5bef970adb95b7dc413979d8cd1c2ca482b3) | Apache 2.0 |
| Published human model | b22e5bef970adb95b7dc413979d8cd1c2ca482b3 | `XBmodel_2024_linear_perms.m` | [Download/source](https://github.com/JuliaMusgrave/AtrialModel_2025_Human/tree/b22e5bef970adb95b7dc413979d8cd1c2ca482b3) | Apache 2.0 |

The machine-readable manifest at `configs/frozen/DATA_SOURCE_MANIFEST.csv`
provides original byte counts, SHA-256, repository MD5 and the corresponding
public canonical input. Download the exact linked file, retain its filename,
and verify its hash before regenerating a canonical export. For example:

```sh
curl -L https://ndownloader.figshare.com/files/46186848 -o RLC-1_Fig6.xlsx
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('RLC-1_Fig6.xlsx').read_bytes()).hexdigest())"
```

Expected SHA-256 for this file:
`77e1b8b3a59e90138e5bc9335706b61a33aab1103804ad9a882058bad56e857e`.
The full quick reproduction works offline with the included canonical inputs.
Original workbook extraction functions are preserved where used by the analysis;
scientific field mappings and source coordinates remain in the canonical tables.

The mouse prediction, continuum and utility-landscape analyses share one
ten-mouse cohort. The inhibitor analysis uses seven aligned rat-level
repeated-measures summary rows, supported by the repeated-measures source
structure and earlier-version rat/trabecula provenance. The final summary
worksheet does not print a named row-to-rat mapping. The pacing source cohort
contains 19 participants; 18 contribute the 132 common comparison cells.
Author-marked gray cells described as unreliable were missing before fitting
and validation; no outcome-driven manual deletion was performed.

Source-data tables at `source_data/` document every plotted value, biological
unit and normalization. Upstream licenses are listed in THIRD_PARTY_LICENSES.md.
