"""Check release integrity, input availability, imports and public-data scope."""
from __future__ import annotations
import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src/common'))

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check():
    imports=[];missing=[];failures=[]
    for i,p in enumerate(sorted((ROOT/'analysis').rglob('*.py'))):
        sys.path.insert(0,str(p.parent))
        try:
            spec=importlib.util.spec_from_file_location('smoke_analysis_'+str(i),p)
            m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
            imports.append(p.relative_to(ROOT).as_posix())
            for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='read_csv' and n.args:
                    try:
                        arg=eval(compile(ast.Expression(n.args[0]),str(p),'eval'),m.__dict__)
                        if isinstance(arg,(str,Path)) and not Path(arg).exists():missing.append((p.name,str(arg)))
                    except (NameError,KeyError,AttributeError):pass
        except Exception as e:failures.append((p.relative_to(ROOT).as_posix(),type(e).__name__,str(e)))
    assert not failures,failures
    assert not missing,missing
    assert len(imports)==36,len(imports)
    snapshots=list(csv.DictReader((ROOT/'configs/frozen/CODE_INPUT_SNAPSHOT_MANIFEST.csv').open(encoding='utf-8',newline='')))
    for row in snapshots:
        p=ROOT/row['public_file'];assert p.is_file(),row['public_file']
        assert digest(p)==row['public_sha256'],row['public_file']
    docs=['LICENSE','DATA_LICENSE.md','DATA_SOURCES.md','THIRD_PARTY_LICENSES.md','README.md','CITATION.cff','docs/CODE_COVERAGE_MAP.md']
    assert all((ROOT/p).is_file() for p in docs)
    figures=list((ROOT/'figures/main').glob('*.pdf'))+list((ROOT/'figures/supplement').glob('*.pdf'))
    assert len(figures)==8
    for p in figures:
        b=p.read_bytes()
        assert not re.search(rb'/Subtype\s*/Image',b),p.name+' contains raster objects'
        size=re.search(rb'/MediaBox\s*\[\s*[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s*\]',b)
        assert size and abs(float(size[1])-180/25.4*72)<.01,p.name+' artboard width'
        svg=p.with_suffix('.svg').read_text(encoding='utf-8')
        assert '<text' in svg and '<image' not in svg,p.name+' vector text'
        assert p.with_suffix('.png').is_file()
    # The public manifest is a provenance allow-list for third-party binaries.
    data=list(csv.DictReader((ROOT/'configs/frozen/DATA_SOURCE_MANIFEST.csv').open(encoding='utf-8',newline='')))
    for row in data:
        if row['original_file_in_release'].startswith('Yes'):
            assert digest(ROOT/row['public_input'])==row['sha256'],row['exact_filename']
    secret=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-[A-Za-z0-9]{32,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    absolute=re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/(?:Users|home|mnt|workspace)/')
    prohibited_name=re.compile(r'(?:meehan|(?:^|[/_\-])xi(?:[_\-]|$)|study[ _]?2|study[ _]?3|credential)',re.I)
    text_suffixes={'.py','.csv','.json','.md','.txt','.yml','.cff','.m','.svg','.sha256'}
    scanned=0;issues=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT).as_posix()
        if not p.is_file() or any(s in rel.split('/') for s in ['.git','__pycache__','recomputed','qa']):continue
        scanned+=1
        if prohibited_name.search(rel):issues.append((rel,'prohibited/private filename'))
        if p.suffix.lower() in text_suffixes:
            s=p.read_text(encoding='utf-8-sig',errors='replace')
            if secret.search(s):issues.append((rel,'credential-like content'))
            if absolute.search(s):issues.append((rel,'local absolute path'))
        if p.suffix.lower()=='.xlsx' and p.name!='P2_V2_NCR_SOURCE_DATA_FINAL.xlsx':issues.append((rel,'unlisted raw workbook'))
        if p.suffix.lower() in {'.docx','.zip','.log','.pem','.key'}:issues.append((rel,'unlisted binary or correspondence'))
    assert not issues,issues
    report={'status':'PASS','analysis_modules_imported':len(imports),'missing_static_csv_inputs':missing,'snapshot_hashes_verified':len(snapshots),'vector_figures_verified':len(figures),'public_files_scanned':scanned,'privacy_or_path_issues':issues}
    out=ROOT/'results/recomputed/RELEASE_CHECK.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':check()
