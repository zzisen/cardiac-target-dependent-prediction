# V3 source-data provenance and fetch instructions

All quick replay inputs are included. Upstream source records and their original terms remain authoritative. The historical source URLs and checksum tables are preserved in `DATA_SOURCES.md` and `source_data/`. Resolve the listed DOI/record, select the stated version, and download its named files. Do not substitute a different version silently.

| Source | Record/version | Material and upstream terms | V3 redistribution |
|---|---|---|---|
| Awinda mouse spectra | https://figshare.com/articles/dataset/27200079/1 | Three source workbooks; CC BY 4.0; file IDs 49733052, 49732494, 49732545 | Existing attributed compact CSVs retained; no source workbook copies added |
| Regulatory light-chain rat data | https://figshare.com/articles/dataset/24354289/3 | `RLC-1_Fig6.xlsx`, file ID 46186848; CC BY 4.0 | Small attributed workbook and compact crosswalk included |
| Tanner trabecula data | https://figshare.com/articles/dataset/24796689/1 | `tanner et al PLOSone_data.xlsx`, file ID 43612779; CC BY 4.0 | Existing attributed subject-level input retained |
| Radbill human pacing | https://doi.org/10.5061/dryad.cjsxksn4k | Version 3, workbook and source README; CC0 | Existing compact, source-flagged input retained |
| Good et al. cytometry dataset, deposited by Keyes et al. | https://doi.org/10.5061/dryad.pvmcvdnxc | Version 4, 2025-07-11; `ddpr_data.zip` and README; CC0 under Dryad dataset terms | Project-derived patient/condition/channel fractions and source member/hash registries only; no raw FCS/large ZIP |

The Dryad cytometry page identifies the 2018 Good et al. source publication (doi:10.1038/nm.4505), the source-defined sample labels, and public de-identified dataset. Dataset CC0 terms are at https://datadryad.org/terms (dataset permission and copyright clauses). This differs from the CC BY license for ordinary Dryad website content. Source publication PDFs retain publisher terms and are omitted.

For raw cytometry, download `ddpr_data.zip` from the version-4 page (file-stream identifier 4162494; https://datadryad.org/downloads/file_stream/4162494). Automated direct requests may require ordinary website/session download access; use the public page if the endpoint rejects an unauthenticated fetch. Expected length **5,863,234,104 bytes**, SHA-256 **72b465bc92e6d2cb9ce94c54c3bcda46b162028ff0eb496fd29f6f441b940fa7**. It contains 525 FCS files plus source AppleDouble sidecars; the V3 training registry selects exactly 352 FCS files and the Stanford registry exactly 27. The historical outer wrapper SHA was `642d88e9f7920d1eb005c6172c2446958e52c3e341cc6efa7cec557a8fb0beff`; independent downloads use the inner hash and do not require an identical outer ZIP timestamp/container.

The RLC workbook checksum is **77e1b8b3a59e90138e5bc9335706b61a33aab1103804ad9a882058bad56e857e**. Source authors and article attribution remain in the historical DATA_SOURCES.md and original Figshare metadata. Redistributed small third-party inputs remain under their upstream CC-BY/CC0 terms rather than being rebranded as project-authored data.

Historical model code is retained with Apache-2.0 notices: JuliaMusgrave/XBModel_2024_Rat at b7c261… and JuliaMusgrave/AtrialModel_2025_Human at b22e5…. Exact commit IDs and fetch instructions remain in the historical source manifest. No new model or raw archive is added by V3.
