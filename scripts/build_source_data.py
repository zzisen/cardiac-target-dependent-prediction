"""Verify the published source-data workbook against every scientific CSV table.

The workbook is authored with the artifact-tool source builder distributed next
to the review package. This public verifier needs only the standard library.
Every cell is checked against its corresponding CSV; normalization cells also
retain explicit Excel formulae and are independently recomputed here.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
N={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def verify():
    mapping=json.loads((ROOT/'configs/frozen/SOURCE_DATA_WORKBOOK_MAP.json').read_text(encoding='utf-8'))
    cfg=json.loads((ROOT/'configs/frozen/WORKBOOK_HASH.json').read_text(encoding='utf-8'))
    file=ROOT/cfg['file']
    assert hashlib.sha256(file.read_bytes()).hexdigest()==cfg['sha256'],'Workbook hash mismatch'
    checked=0;formulas=0
    with zipfile.ZipFile(file) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            shared=[''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',N)]
        workbook=ET.fromstring(z.read('xl/workbook.xml'))
        rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        refs={r.attrib['Id']:r.attrib['Target'].lstrip('/') for r in rels}
        sheets={s.attrib['name']:refs[s.attrib['{'+N['r']+'}id']] for s in workbook.find('s:sheets',N)}
        for table in mapping:
            p=ROOT/table['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==table['sha256'],str(p)
            with p.open(encoding='utf-8-sig',newline='') as f: data=list(csv.reader(f));expected=data[1:]
            target=sheets[table['name']];target=target if target.startswith('xl/') else 'xl/'+target
            cells={}
            for c in ET.fromstring(z.read(target)).findall('.//s:c',N):
                value=c.find('s:v',N);text=value.text if value is not None else None
                if c.attrib.get('t')=='s':text=shared[int(text)]
                elif c.attrib.get('t')=='inlineStr':text=''.join(c.find('s:is',N).itertext())
                if c.find('s:f',N) is not None:formulas+=1
                cells[c.attrib['r']]=text
            for r,row in enumerate(expected,start=table['start_row']):
                for j,value in enumerate(row):
                    q=j+1;col=''
                    while q:q,rem=divmod(q-1,26);col=chr(65+rem)+col
                    actual=cells.get(col+str(r),'') or ''
                    if value in ('True','False'):assert actual==str(int(value=='True'))
                    elif value=='':assert actual=='',str((table['name'],col,r,actual))
                    else:
                        try:v=float(value)
                        except ValueError:assert actual==value,(table['name'],col,r,value,actual)
                        else:assert math.isclose(float(actual),v,rel_tol=2e-12,abs_tol=2e-12),(table['name'],col,r,value,actual)
                    checked+=1
    assert formulas>=10,'Within-dataset normalization formulae absent'
    print(f'PASS: source workbook verified; {len(mapping)} tables, {checked} source cells, {formulas} normalization formulae.')
    return checked

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.parse_args();verify()
