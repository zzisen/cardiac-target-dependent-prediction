# Public-data audit — v1.0.2

Result: **PASS**, 3 October 2026. The fresh-directory checker inspected
370 public files and verified all licensed model binaries
against the source manifest. No private data, correspondence or credentials
were included. Original third-party workbooks and archives are absent. Only the
project-authored source-data workbook is redistributed as XLSX.

Included observations derive from fixed public Figshare and Dryad records;
published model inputs retain Apache 2.0 notices. DATA_SOURCES.md records exact
filenames, versions, original byte counts, SHA-256/MD5 and fetch links.
Project-authored code uses MIT. Derived data, documentation and figure source
tables use CC BY 4.0; canonical upstream exports retain their source licenses.

`tests/smoke/check_release.py` scans text and filenames for credentials, local
absolute paths and prohibited source identifiers, verifies the allow-listed
third-party binaries, and rejects unlisted workbooks or correspondence archives.
The provenance review also traced the published canonical tables to the permitted
public sources; this source review complements pattern scanning.
