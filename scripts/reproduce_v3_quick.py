"""Offline, isolated V3 replay from public compact summaries and figure inputs."""
import argparse,csv,hashlib,importlib.util,json,platform,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
CODE=ROOT/'analysis/v3/code'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def module(name):
    spec=importlib.util.spec_from_file_location(name,CODE/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def run(cmd,out,log):
    p=subprocess.run([sys.executable,'-X','utf8',*map(str,cmd)],cwd=out,capture_output=True,text=True,encoding='utf-8')
    (out/log).write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    if p.returncode:raise RuntimeError(f'{log}: exit {p.returncode}; inspect saved log')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir.resolve()
    if out.exists():raise RuntimeError('Choose a fresh output directory; published outputs are never overwritten')
    out.mkdir(parents=True);start=time.time();checks=[]
    for row in read(ROOT/'docs/v3/RELEASE_SOURCE_MANIFEST.csv'):
        assert sha(ROOT/row['path'])==row['sha256'],row['path']
    checks.append({'check':'release input/source/figure SHA-256 manifest','status':'PASS'})
    for name in ['cardiac_fixed','matched_families','adaptive_resolution','finite_action_verification','training_execution','transport_execution','transport_training_models']:
        module(name)
    cardiac=module('cardiac_fixed');cardiac.load_rlc_support();cardiac.load_awinda_support()
    loaded=module('adaptive_resolution').load_all_data()
    assert set(loaded)=={'RLC1_RAT','RADBILL_HUMAN_PACING','TANNER_RAT_TRABECULA','AWINDA_MOUSE'}
    checks.append({'check':'all portable cardiac loaders and prespecified grids','status':'PASS'})
    theory_copy=out/'two_action_benchmarks.py';shutil.copy2(CODE/'two_action_benchmarks.py',theory_copy)
    run([theory_copy],out,'two_action_benchmarks.log')
    theory=module('finite_action_verification');rows,root=theory.two_action_benchmark()
    assert all(r['pass']=='TRUE' for r in rows)
    checks.append({'check':'two-action analytic Gaussian benchmark','status':'PASS','root':root,'checks':len(rows)})
    matched=module('matched_families');rr,hr,_=matched.partition_registries()
    assert len(rr)==15 and len(hr)==35
    ranks={}
    for system,registry,key,targets,parts in [('RLC1',rr,'heldout_rat',6,15),('RADBILL',hr,'heldout_participant',8,35)]:
        pred=read(ROOT/f'source_data/v3/matched_families/A24_{system}_OUTER_PREDICTIONS.csv')
        summary,_=matched.rank_summaries(pred,registry,key,targets,parts)
        natural=[r for r in summary if r['is_natural_partition']][0]
        assert natural['rank_lower_is_better']==1
        expected=read(ROOT/f'source_data/v3/matched_families/A24_{system}_PARTITION_SUMMARY.csv')
        expected={r['partition_id']:r for r in expected}
        for row in summary:
            ref=expected[row['partition_id']]
            assert int(ref['rank_lower_is_better'])==row['rank_lower_is_better']
            assert abs(float(ref['primary_mean_standardized_absolute_error'])-row['primary_mean_standardized_absolute_error'])<1e-12
        ranks[system]={'partitions':parts,'natural_rank':1,'natural_risk':natural['primary_mean_standardized_absolute_error']}
    checks.append({'check':'all matched-partition risks/ranks reaggregated; no refitting','status':'PASS','systems':ranks})
    b1=out/'training';shutil.copytree(ROOT/'source_data/v3/training',b1)
    run([CODE/'training_independent_replay.py','--work-dir',b1],out,'training_replay.log')
    b1result={'report':'training/B1_INDEPENDENT_REPLAY_REPORT.md','verified_by_process_exit':True,'loss_cells':924,'patient_target_cells':308}
    checks.append({'check':'independent nested 44-patient selection and loss replay','status':'PASS','result':b1result})
    project=out/'transport_project';b1path=project/'P2_V3_ABT/B1_OUTCOME_EXECUTION_LUNA14_2026-10-06';b1path.mkdir(parents=True)
    shutil.copy2(b1/'B1_PATIENT_CONDITION_CHANNEL_SUMMARY.csv',b1path)
    b2=project/'P2_V3_ABT/B2_STANFORD_TRANSPORT_LUNA15_2026-10-06'
    shutil.copytree(ROOT/'source_data/v3/transport',b2)
    shutil.copy2(CODE/'transport_independent_replay.py',b2/'B2_STANFORD_INDEPENDENT_REPLAY.py')
    run([b2/'B2_STANFORD_INDEPENDENT_REPLAY.py','--project-root',project],out,'transport_replay.log')
    b2result=json.loads((b2/'B2_STANFORD_INDEPENDENT_REPLAY_RESULT.json').read_text(encoding='utf-8'))
    assert b2result['replay']=='PASS'
    checks.append({'check':'independent full-training model and Stanford transport replay','status':'PASS','result':b2result})
    # Adaptive fold risks and selection counts are reproduced from the recorded
    # fold-level prediction/selection outputs. Full nested fitting is separate.
    adapt=module('adaptive_resolution');adaptive={}
    expected={'RLC1_RAT':1.0024434266096864,'RADBILL_HUMAN_PACING':.5224307752998568,
        'TANNER_RAT_TRABECULA':.6325059281810032,'AWINDA_MOUSE':10.919862082400952}
    names={'RLC1_RAT':'RLC1','RADBILL_HUMAN_PACING':'RADBILL','TANNER_RAT_TRABECULA':'TANNER','AWINDA_MOUSE':'AWINDA'}
    for system,name in names.items():
        pred=read(ROOT/f'source_data/v3/adaptive/A25_{name}_OUTER_ADAPTIVE_PREDICTIONS.csv')
        units=adapt.unit_loss_from_predictions(system,pred);risk=adapt.native_risk(system,units)
        assert abs(risk-expected[system])<1e-10,(system,risk)
        adaptive[system]=risk
    checks.append({'check':'adaptive risks reaggregated from held-out predictions','status':'PASS','risks':adaptive})
    figs=out/'figures';shutil.copytree(ROOT/'figures/v3/figures_src',figs/'figures_src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    run([figs/'figures_src/build_figures.py'],out,'figure_build.log')
    comparisons=[]
    for n in range(1,6):
        for ext in ['png','pdf','svg']:
            name=f'P2_V3_Figure{n}_V3_0.{ext}';a=sha(ROOT/'figures/v3'/name);b=sha(figs/name)
            comparisons.append({'file':name,'expected_sha256':a,'rebuilt_sha256':b,'identical':a==b})
    assert all(r['identical'] for r in comparisons),'Figure hash mismatch in isolated environment'
    checks.append({'check':'all five figure source builds','status':'PASS','byte_identical_outputs':15,'files':comparisons})
    import scipy,pandas,matplotlib,PIL
    report={'status':'PASS','seconds':time.time()-start,'python':platform.python_version(),
        'environment':{'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pandas.__version__,'matplotlib':matplotlib.__version__,'Pillow':PIL.__version__},
        'scope':'Offline quick replay: training/transport independently refit, cardiac saved fold losses reaggregated, analytic two-action root verified, five figures rebuilt. Raw event parsing and full cardiac nested fits are separate commands.',
        'checks':checks}
    (out/'CLEAN_REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'status':'PASS','seconds':report['seconds'],'report':str(out/'CLEAN_REPRODUCTION_REPORT.json')},indent=2))
if __name__=='__main__':main()
