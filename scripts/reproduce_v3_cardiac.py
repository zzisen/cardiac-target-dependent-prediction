"""Repeat prespecified cardiac fits; no exploration or changes to candidate sets."""
import argparse,csv,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'analysis/v3/code'
def load(name):
    spec=importlib.util.spec_from_file_location(name,CODE/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def save(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--analysis',choices=['fixed','matched','adaptive'],required=True)
    ap.add_argument('--system',choices=['RLC1_RAT','RADBILL_HUMAN_PACING','TANNER_RAT_TRABECULA','AWINDA_MOUSE'])
    ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();out=a.output_dir.resolve()
    if out.exists():raise RuntimeError('Choose a fresh output directory')
    out.mkdir(parents=True)
    if a.analysis=='fixed':
        m=load('cardiac_fixed');x,y=m.load_rlc_support();pred,sel=m.run_rlc_family(x,y)
        pred.to_csv(out/'RLC1_family_predictions.csv',index=False);sel.to_csv(out/'RLC1_family_selections.csv',index=False)
        results=m.run_awinda(m.load_awinda_support())
        for name,frame in zip(['component_predictions','selections','policy_summary','unit_predictions','contrasts'],results):frame.to_csv(out/f'Awinda_{name}.csv',index=False)
    elif a.analysis=='matched':
        m=load('matched_families');rr,hr,parts=m.partition_registries();x,y=m.load_rlc_data()
        pred,sel,_=m.compute_rlc(x,y,parts['RLC1']);save(out/'RLC1_predictions.csv',pred);save(out/'RLC1_selections.csv',sel)
        summary,_=m.rank_summaries(pred,rr,'heldout_rat',6,15);save(out/'RLC1_partition_summary.csv',summary)
        ds,support,_=m.load_radbill();sel,maps=m.compute_radbill_selections(ds,parts['RADBILL'])
        pred=m.compute_radbill_predictions(ds,support,hr,parts['RADBILL'],maps);save(out/'Radbill_predictions.csv',pred);save(out/'Radbill_selections.csv',sel)
        summary,_=m.rank_summaries(pred,hr,'heldout_participant',8,35);save(out/'Radbill_partition_summary.csv',summary)
    else:
        m=load('adaptive_resolution');data=m.load_all_data();systems=[a.system] if a.system else list(data)
        report={}
        for system in systems:
            result=m.run_adaptive_system(system,data[system])
            for name,rows in zip(['predictions','middle_trace','outer_selections','action_scores'],result):save(out/f'{system}_{name}.csv',rows)
            units=m.unit_loss_from_predictions(system,result[0]);report[system]=m.native_risk(system,units)
        (out/'adaptive_risks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Completed prespecified cardiac reproduction:',out)
if __name__=='__main__':main()
