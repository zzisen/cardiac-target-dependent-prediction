"""Optional full parsing of the verified upstream inner DDPR archive."""
import argparse,hashlib,importlib.util,json,shutil,subprocess,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'analysis/v3/code'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def module(name):
    spec=importlib.util.spec_from_file_location(name,CODE/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--ddpr-inner',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    inner=a.ddpr_inner.resolve();out=a.output_dir.resolve()
    if out.exists():raise RuntimeError('Choose a fresh output directory')
    assert inner.stat().st_size==5863234104 and sha(inner)=='72b465bc92e6d2cb9ce94c54c3bcda46b162028ff0eb496fd29f6f441b940fa7'
    out.mkdir(parents=True);wrapper=out/'verified_source_wrapper.zip'
    with zipfile.ZipFile(wrapper,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:z.write(inner,'ddpr_data.zip')
    b1=module('training_execution');b1.EXPECTED_OUTER_SHA256=sha(wrapper)
    training=out/'training';shutil.copytree(ROOT/'source_data/v3/training',training)
    result=b1.execute(training,wrapper)
    (out/'training_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    # Only decode the fixed Stanford registry; no selection/refit uses test data.
    b2=module('transport_execution');headers=b2.readcsv(ROOT/'source_data/v3/transport/B2_SOURCE_FCS_HEADER_ROWS.csv')
    registry=b2.build_registry_rows(headers);_,channel_map=b1.read_channel_registry(training/'B13R2_FINAL_CHANNEL_REGISTRY.csv')
    outer,z,raw=b1.make_inner_zip(wrapper);summaries=[]
    try:
        for row in registry:
            _,sr=b1.decode_one_fcs(z,row,channel_map);summaries.extend(sr)
    finally:z.close();raw.close();outer.close()
    assert len(summaries)==216
    project=out/'transport_project';bp=project/'P2_V3_ABT/B1_OUTCOME_EXECUTION_LUNA14_2026-10-06';bp.mkdir(parents=True)
    shutil.copy2(training/'B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv',bp)
    test=project/'P2_V3_ABT/B2_STANFORD_TRANSPORT_LUNA15_2026-10-06';shutil.copytree(ROOT/'source_data/v3/transport',test)
    b2.writecsv(test/'B2_STANFORD_PATIENT_CONDITION_CHANNEL_SUMMARY.csv',summaries)
    shutil.copy2(CODE/'transport_independent_replay.py',test/'B2_STANFORD_INDEPENDENT_REPLAY.py')
    subprocess.run([sys.executable,'-X','utf8',str(test/'B2_STANFORD_INDEPENDENT_REPLAY.py'),'--project-root',str(project)],check=True)
    print('Verified raw training and training-fixed Stanford reproduction:',out)
if __name__=='__main__':main()
