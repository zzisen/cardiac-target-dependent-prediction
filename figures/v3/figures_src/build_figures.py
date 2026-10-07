"""Rebuild one or all main figures using bundled frozen input snapshots."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import matplotlib
import numpy

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--figure',type=int,choices=range(1,6),help='Build only this figure; omitted means all five')
    args=p.parse_args()
    src=Path(__file__).resolve().parent
    figures=[args.figure] if args.figure else list(range(1,6))
    for n in figures:
        subprocess.run([sys.executable,'-X','utf8',str(src/f'figure{n}.py')],cwd=src,check=True)
    record={'figures':figures,'python':platform.python_version(),
        'matplotlib':matplotlib.__version__,'numpy':numpy.__version__,
        'input_mode':'bundled accepted frozen export snapshots',
        'historical_image_input':False,'scientific_analysis_rerun':False}
    (src.parent/'qa').mkdir(exist_ok=True)
    (src.parent/'qa'/'last_build_environment.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
