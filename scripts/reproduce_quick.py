"""Fresh-directory quick reproduction; no optimizer or external download is required."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
for command in [['scripts/run_analysis.py','--quick'],['scripts/build_figures.py'],['scripts/build_source_data.py','--verify'],['tests/smoke/check_release.py']]:
    subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
print('PASS: all quick numerical, figure, source-data and public-release checks.')
