# P2 target-resolution analysis — V3 manuscript release

Software release **v2.0.1** accompanies the V3 manuscript, *Target resolution shapes deployable measurement design under finite biological data*, by Zisen Zhou (Auckland Bioengineering Institute, University of Auckland), Yaopu Zhang (Harbin Institute of Technology (Shenzhen)), and Haoran Pang (Harbin Institute of Technology). This is the same project as historical v1.0.2, whose tag, release, DOI, code, and data remain accessible.

V3 adds finite-action Gaussian verification, exhaustive matched-family controls (15 and 35 partitions), fully nested adaptive resolution in four cardiac systems, a 44-patient non-cardiac training benchmark, and nine-patient training-fixed Stanford transport. Numerical outputs retain their native metrics. No journal acceptance is claimed.

## Reproduction

For byte-identical artwork, use the exact Windows Conda binary environment (including FreeType 2.13.3), then install the pinned Python requirements:

```sh
conda create --name p2-v3-replay --file environment-v3-win64-explicit.txt
conda activate p2-v3-replay
python -m pip install -r requirements-v3.txt
python scripts/reproduce_v3_quick.py --output-dir replay-v3
```

The quick offline replay independently recalculates training/transport models and losses from compact published summaries, verifies theoretical benchmark roots and matched-partition ranks, checks input hashes, and rebuilds all five figures from numerical snapshots. It does not decode raw FCS files or repeat every cardiac model fit. See [complete reproduction scope and commands](docs/v3/REPRODUCTION.md), including deterministic full cardiac reruns and raw-source parsing.

Ordinary pip Matplotlib wheels use a different FreeType binary and can yield different artwork hashes even with identical source values; the explicit environment is required for the strict byte-comparison gate. Numerical analyses remain defined by their unchanged contracts.

Final 600-dpi PNG and vector PDF/SVG files are under `figures/v3/`. SVG uses self-contained vector glyph outlines; editable Python sources and numerical inputs accompany each figure. Scientific numerical tables are under `source_data/v3/`. Figure 1C is explicitly schematic; Gaussian examples are not fitted to biological data.

## Sources, attribution, and licenses

Project-authored code is MIT; project-authored derived data and documentation are CC BY 4.0. Third-party materials retain upstream licenses. The small RLC Figshare workbook redistributed in `analysis/v3/inputs/project` is CC BY 4.0 with attribution in [V3 source instructions](docs/v3/SOURCE_DATA.md). Large upstream FCS archives and source-publication PDFs are omitted; download them separately. Existing Apache-2.0 model source notices are preserved.

See [DATA_SOURCES.md](DATA_SOURCES.md), [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md), [LICENSE](LICENSE), [DATA_LICENSE.md](DATA_LICENSE.md), and [CITATION.cff](CITATION.cff). The verified shared Zenodo concept DOI is [10.5281/zenodo.23113843](https://doi.org/10.5281/zenodo.23113843); cite the specific v2.0.1 version for exact reproduction.

The historical v1.0.2 README is preserved in `docs/v3/HISTORICAL_V1_0_2_README.md`. Pre-existing top-level analysis directories retain the historical methods and inputs; the V3 entry points select the new contracts explicitly.

## Authorship and disclosure patch

Version v2.0.1 aligns final manuscript authorship and AI-assisted workflow disclosure. No scientific analysis, result, input, figure code or figure export changes. Historical v2.0.0 remains available unchanged. Cite Zisen Zhou, Yaopu Zhang and Haoran Pang (2026), P2 target-resolution analysis — V3 authorship and disclosure patch, v2.0.1, https://doi.org/10.5281/zenodo.23203165.
