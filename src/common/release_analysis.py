"""Executable entry points and quick replay for the frozen cardiac analyses."""
from __future__ import annotations
import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
STAGES = {
    'core': ('01_core_target_value/support/code', ['run_A_measurement_target_map.py', 'run_A_full_target_rank_stability_v2.py', 'run_A_missing_target_restart.py', 'build_A_full_audit.py']),
    'sparse': ('01_core_target_value/sparse/code', ['run_kp_design.py', 'run_bounded_local_geometry.py']),
    'variants': ('02_robustness/variants/code', ['run_N_permutation_sensitivity.py']),
    'robustness': ('02_robustness/robust/code', ['run_R_robust_panels.py']),
    'awinda': ('03_awinda/frequency/code', ['run_Q_awinda.py']),
    'continuum': ('03_awinda/continuum/code', ['run_U_mgatp_continuum.py']),
    'landscape': ('03_awinda/landscape/code', ['run_yz_landscape.py']),
    'human': ('04_human_atrial/code', ['p2_human_stage2_bounded_nonlinear.py']),
    'rlc1': ('05_rlc1/code', ['RLC1_primary_analysis.py']),
    'radbill': ('06_radbill/code', ['radbill_fair_comparators.py', 'radbill_within_family_perm_exact.py']),
    'tanner': ('07_tanner/code', ['analyze_tanner.py']),
    'information_accuracy': ('08_boundaries/preparation_support/code', ['run_B_human_heldout.py', 'run_I_support_error_linkage.py']),
    'dynamic': ('08_boundaries/dynamic/code', ['run_T_dynamic_validation.py']),
    'transfer': ('08_boundaries/transfer/code', ['evaluate_transfers.py']),
    'kinetics': ('08_boundaries/kinetics/code', ['run_V_kinetic_normalization.py']),
    'normalized_transfer': ('08_boundaries/normalized_transfer/code', ['run_W_scoring.py']),
    'target_coordinates': ('08_boundaries/target_coordinates/code', ['run_X_comparisons.py']),
    'state_surface': ('08_boundaries/state_surface/code', ['run_aa_surfaces.py']),
    'robust_bands': ('08_boundaries/robust_bands/code', ['run_ab_robust.py']),
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run_full(stage):
    directory, files = STAGES[stage]
    for name in files:
        cmd = [sys.executable, str(REPO / 'analysis' / directory / name)]
        if stage == 'radbill':
            cmd += ['--radbill-dir', str(REPO/'analysis/06_radbill'), '--out', str(REPO/'analysis/06_radbill/results' / ('RADBILL_WITHIN_FAMILY_PERMUTATION_SUMMARY_EXACT.csv' if 'perm' in name else ''))]
        if stage == 'tanner': cmd += ['--outcomes','primary']
        subprocess.run(cmd, check=True, cwd=REPO)


def quick():
    """Recalculate headline summaries from permitted canonical/frozen upstream inputs.

    Endpoint optimization is replayed from frozen endpoint results. Rat and human
    held-out scalar predictions and target-label mappings are recomputed numerically.
    """
    rows = []
    def record(system, target, metric, value):
        rows.append(dict(system=system, target=target, metric=metric, value=float(value)))
    def read(path): return pd.read_csv(REPO / 'analysis' / path)
    a = read('01_core_target_value/support/results/A_FULL_RESTART_AUDIT.csv')
    # Independently rebuild support widths and reductions from frozen endpoints.
    np.testing.assert_allclose(a.high_q-a.low_q,a.new_width_q_pct_points,atol=1e-10,rtol=1e-10)
    for (domain,target),part in a.groupby(['domain','target']):
        base=float(part.loc[part.measurement_id=='baseline_only','new_width_q_pct_points'].iloc[0])
        ok=part.new_relative_reduction_pct.notna()
        np.testing.assert_allclose(100*(1-part.loc[ok,'new_width_q_pct_points']/base),part.loc[ok,'new_relative_reduction_pct'],atol=2e-9,rtol=1e-9)
    for r in a.itertuples():
        if np.isfinite(r.new_relative_reduction_pct):
            record('Rat model support', str(r.target), str(r.measurement_id)+' '+str(r.domain), r.new_relative_reduction_pct)
    k = read('01_core_target_value/sparse/results/K_BOUNDED_LOCAL_SUPPORT.csv')
    for (view, budget), d in k.groupby(['accounting_view', 'budget']):
        record('Matched measurement budget', str(view), 'median width reduction budget '+str(budget), d.bounded_width_reduction_pct.median())
    r = read('02_robustness/robust/results/R_ROBUST_PANEL_RESULTS.csv')
    for row in r.itertuples():
        record('Robust scenario design', row.target, 'worst oracle retention budget '+str(row.budget), row.worst_case_fraction_oracle_retained)
    q = read('03_awinda/frequency/results/Q_AWINDA_SPARSE_PANEL_RESULTS.csv')
    for target, d in q.groupby('target_ATP_mM'):
        record('Awinda', str(target), 'group-only NRMSE', d.baseline_group_only_nrmse_pct.mean())
        record('Awinda', str(target), 'selected pair NRMSE', d.outer_nrmse_pct.mean())
    u = read('03_awinda/continuum/results/U_MGATP_CONTINUUM_SELECTIONS.csv')
    # Frozen target-level selected frequency records preserve biological-unit grouping.
    for target, d in u.groupby('target_ATP_mM'):
        if 'selected_frequency_Hz' in d: record('MgATP continuum', str(target), 'median selected Hz', d.selected_frequency_Hz.median())
    y = read('03_awinda/landscape/results/Y_MODEL_COMPARISON.csv')
    for row in y.itertuples():
        record('Utility landscape', row.model, 'leave-one-target-out RMSE', row.mean_rmse_pp)
    hu = read('04_human_atrial/results_stage2/stage2_summary.csv')
    for row in hu.itertuples():
        record('Human atrial support', row.target, 'width reduction radius '+str(row.trust_radius_log), row.width_reduction_percent)
    # Exact scalar prediction algorithms from canonical observations, not frozen MAE tables.
    rl = load(REPO/'analysis/05_rlc1/code/RLC1_primary_analysis.py', 'release_rlc1')
    d = read('05_rlc1/input/RLC1_CANONICAL_DATA.csv')
    xx = {f'1uM_{family}': d[(d.family_key==family)&(d.dose=='1uM')].sort_values('source_excel_row').value.to_numpy(float) for family in rl.FAMILIES}
    yy = {t: d[(d.family_key==rl.parts(t)[1])&(d.dose==rl.parts(t)[0])].sort_values('source_excel_row').value.to_numpy(float) for t in rl.TARGETS}
    oracle={t:rl.winner(rl.inner_scores(list(range(7)),t,xx,yy)) for t in rl.TARGETS}
    ae = {c: [] for c in ('C0','C1','C2','C3')}; selections=[]; prediction_check=[]; all_candidate_errors=[]
    for held in range(7):
        train = [i for i in range(7) if i!=held]
        inner = {t: rl.inner_scores(train,t,xx,yy) for t in rl.TARGETS}
        c1 = rl.winner({c: float(np.mean([inner[t][c] for t in rl.TARGETS])) for c in rl.CANDIDATES})
        selection={t:rl.winner(inner[t]) for t in rl.TARGETS}; selections.append(selection)
        candidate_errors={}
        for t in rl.TARGETS:
            ae['C0'].append(abs((yy[t][held]-np.mean(yy[t][train]))/rl.sd(yy[t][train])))
            ae['C1'].append(rl.fit_predict(train,held,c1,t,xx,yy)['ae_std'])
            ae['C2'].append(rl.fit_predict(train,held,selection[t],t,xx,yy)['ae_std'])
            ae['C3'].append(rl.fit_predict(train,held,oracle[t],t,xx,yy)['ae_std'])
            for c in rl.CANDIDATES: candidate_errors[t,c]=rl.fit_predict(train,held,c,t,xx,yy)['ae_std']
            for comparator,candidate in [('C1',c1),('C2',selection[t]),('C3',oracle[t])]:
                fit=rl.fit_predict(train,held,candidate,t,xx,yy)
                prediction_check.append((held+1,t,comparator,candidate,fit['pred_native'],fit['ae_std'],xx[candidate][held]))
        all_candidate_errors.append(candidate_errors)
    frozen = read('05_rlc1/results/RLC1_COMPARATOR_SUMMARY.csv')
    for c in ae:
        record('RLC-1', 'six repeated-dose targets', c+' standardized MAE', np.mean(ae[c]))
        expected_rl=float(frozen[(frozen.summary_level=='overall')&(frozen.comparator==c)].mean_standardized_mae.iloc[0])
        np.testing.assert_allclose(np.mean(ae[c]),expected_rl,atol=5e-11,rtol=0)
    pred=read('05_rlc1/results/RLC1_OUTER_LOO_PREDICTIONS.csv')
    for held,t,c,candidate,pr,err,value in prediction_check:
        old=pred[(pred.heldout_row_index==held)&(pred.target==t)&(pred.comparator==c)].iloc[0]
        assert old.candidate==candidate
        np.testing.assert_allclose([pr,err,value],[old.target_predicted_native,old.absolute_error_standardized,old.candidate_1uM_value],atol=5e-9,rtol=5e-12)
    perm_values=[]
    for perm in itertools.permutations(range(6)):
        perm_values.append(np.mean([all_candidate_errors[i][t,selections[i][rl.TARGETS[perm[j]]]] for i in range(7) for j,t in enumerate(rl.TARGETS)]))
    perm_old=read('05_rlc1/results/RLC1_TARGET_LABEL_PERMUTATION.csv').mean_standardized_mae.to_numpy(float)
    np.testing.assert_allclose(perm_values,perm_old,atol=5e-11,rtol=0)
    record('RLC-1', 'target-label mappings', 'no-worse mapping fraction',sum(v<=np.mean(ae['C2'])+1e-12 for v in perm_values)/720)
    rad_dir = REPO/'analysis/06_radbill'
    dest = REPO/'results/recomputed/radbill'; dest.mkdir(parents=True,exist_ok=True)
    for script, out in [('radbill_fair_comparators.py', dest), ('radbill_within_family_perm_exact.py', dest/'within_family_mapping.csv')]:
        subprocess.run([sys.executable, str(rad_dir/'code'/script),'--radbill-dir',str(rad_dir),'--out',str(out)],check=True,stdout=subprocess.DEVNULL)
    rr = pd.read_csv(dest/'RADBILL_FAIR_COMPARATOR_SUMMARY.csv')
    expected={'C0':.819522046100299,'C1_family_global':.5634352209353645,'C1_fixed_family':.48818911704682866,'C2_target_specific':.5203030666697132}
    for c,v in expected.items():
        actual=float(rr.loc[rr.comparator==c,'mean'].iloc[0]); np.testing.assert_allclose(actual,v,rtol=0,atol=2e-12)
        record('Radbill', '132 participant-target predictions', c+' standardized MAE', actual)
    pp=pd.read_csv(dest/'within_family_mapping.csv').iloc[0]
    assert int(pp.n_permutations)==576 and int(pp.n_no_worse)==112
    record('Radbill', 'within-family label mappings', 'no-worse mapping fraction', pp.p_fraction)
    # Recalculate all Figure 5 ladder ratios from held-out errors. Each dataset
    # keeps its own metric, denominator and biological unit.
    ladder=pd.read_csv(REPO/'source_data/main/Fig5_within_dataset_normalization.csv')
    tp=read('07_tanner/results/TANNER_OUTER_LOTO_PREDICTIONS.csv')
    ts=read('07_tanner/results/TANNER_GLOBAL_COMPARATOR_RESULTS.csv')
    names={'No added measurement':'C0_TARGET_MEAN','Global/shared':'C1_GLOBAL_FIXED','Target-specific':'C2_TARGET_SPECIFIC','Nearest-rate conventional':'C1_NEAREST_RATE'}
    t_errors={}
    for label,code in names.items():
        values=tp[(tp.outcome_family=='peak_stress_response')&(tp.comparator==code)].absolute_error
        if not len(values):
            candidates=ts[(ts.outcome_family=='peak_stress_response')&(ts.target_strain_rate_s_inv=='ALL')]
            code=next(c for c in candidates.comparator if 'NEAREST' in c)
            values=tp[(tp.outcome_family=='peak_stress_response')&(tp.comparator==code)].absolute_error
        assert len(values)==28,(code,len(values))
        t_errors[label]=float(values.mean())
        record('Tanner','four future strain rates',label+' MAE',values.mean())
    for row in ladder.itertuples():
        if row.dataset=='RLC-1':
            code={'No added measurement':'C0','Global/shared':'C1','Target-specific':'C2'}[row.comparator]
            value=float(np.mean(ae[code])); baseline=float(np.mean(ae['C0']))
        elif row.dataset=='Radbill':
            code={'No added measurement':'C0','Family-aware global':'C1_family_global','Target-specific':'C2_target_specific','Fixed family-matched':'C1_fixed_family'}[row.comparator]
            value=float(rr.loc[rr.comparator==code,'mean'].iloc[0]);baseline=expected['C0']
        else:value=t_errors[row.comparator];baseline=t_errors['No added measurement']
        np.testing.assert_allclose([value,baseline,value/baseline],[row.source_error,row.no_measurement_error,row.normalized_error],atol=5e-10,rtol=1e-10)
        record(row.dataset,'within-dataset error ratio',row.comparator,value/baseline)
    # The boundary panels replay frozen fits and intervals, and recompute
    # correlations from the 20 paired preparations at each future target.
    from scipy.stats import spearmanr
    pairs=read('08_boundaries/preparation_support/paired/I_PREPARATION_PAIRED_DATA.csv')
    correlations=pd.read_csv(REPO/'source_data/main/Fig5_support_accuracy_correlations.csv')
    for row in correlations.itertuples():
        part=pairs[pairs.target==row.target]
        assert len(part)==20 and part.preparation_id.nunique()==20
        rho=float(spearmanr(part.support_width_reduction_pp,part.absolute_stress_error_improvement_pp).statistic)
        np.testing.assert_allclose(rho,row.spearman_rho,atol=1e-12,rtol=0)
        record('Human preparations',row.target,'support/error Spearman rho',rho)
    transfer=read('08_boundaries/normalized_transfer/results/W_TRANSFER_SUMMARY.csv')
    source=pd.read_csv(REPO/'source_data/main/Fig5_rat_mouse_transfer.csv')
    for row in source.itertuples():
        part=transfer[transfer.route=='rat_Model16D_to_Awinda']
        if row.future_target=='Both ATP targets':part=part[(part.target=='ATP0.1_and_ATP1')&(part.budget.astype(str)=='1_to_3_equal_weight')&(part.method=='pooled_normalized_minus_absolute_equal_mouse')]
        else:part=part[(part.target==row.future_target)&(part.budget.astype(str)==str(row.frequency_pair_budget))&(part.method=='paired_normalized_minus_absolute')]
        assert len(part)==1,(row.future_target,row.frequency_pair_budget)
        v=part.iloc[0]
        np.testing.assert_allclose([row.normalized_minus_absolute_nrmse_pp,row.bootstrap_95_lower_pp,row.bootstrap_95_upper_pp],[v.paired_delta_nrmse_normalized_minus_absolute_pp,v.paired_delta_bootstrap_95_low,v.paired_delta_bootstrap_95_high],atol=1e-11,rtol=0)
        record('Rat-to-mouse transfer',row.future_target,'normalized-minus-absolute NRMSE budget '+str(row.frequency_pair_budget),row.normalized_minus_absolute_nrmse_pp)
    for name in ['information_accuracy','dynamic','transfer','kinetics','normalized_transfer','target_coordinates','state_surface','robust_bands','tanner']:
        directory,_=STAGES[name]
        target=REPO/'analysis'/Path(directory).parent/'results'
        assert list(target.glob('*.csv')), name+' frozen results absent'
    out=REPO/'results/recomputed/HEADLINE_SUMMARIES.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False,float_format='%.17g')
    print(f'PASS: {len(rows)} recomputed/replayed headline values; Radbill fair comparators and 112/576 mapping reproduced.')
    return rows


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=STAGES); p.add_argument('--quick',action='store_true')
    args=p.parse_args()
    if args.quick or not args.stage: quick()
    else: run_full(args.stage)
