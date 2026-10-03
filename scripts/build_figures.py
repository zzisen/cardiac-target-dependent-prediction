"""Rebuild the five main and three supplementary figures from frozen source tables.

Only plotting and documented within-dataset normalization are performed here.
Run from any directory: python scripts/build_figures.py
Output is 180 mm wide; vector text remains editable; PNGs are 600 dpi.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data'
FIGURES = ROOT / 'figures'
MAIN = FIGURES / 'main'
SUPP = FIGURES / 'supplement'
QA = FIGURES / 'qa'
for d in [MAIN, SUPP, QA]:
    d.mkdir(parents=True, exist_ok=True)

NEUTRAL = ['#D9DEE3', '#A8B2BC', '#66727D']
BLUE = ['#DCE8F1', '#AFC9DD', '#78A7C8', '#477FA8', '#24577F']
TEAL = ['#D7ECE8', '#A4D3C9', '#68B4A6', '#358F7F', '#17695D']
GOLD = ['#F3E3BF', '#E8C77B', '#D7A647', '#B9822E', '#8F6122']
CORAL = '#B7655B'
PLUM = '#8A6A8F'
INK = '#26333E'
HUMAN_TARGET_COLORS = {'ATP1': '#2F7F73', 'Pi10': '#C49A32'}
DOMAINS = ['Log-coordinate margin ±0.5', 'Log-coordinate margin ±1.0']
MODEL_ORDER = ['Reference model', 'k2 + k−2 strain', 'k1 + k−2 strain',
    'k−2 strain only', 'k2 strain only']
WIDTH = 180 / 25.4
COMPARATOR_COLORS = {'No added measurement': NEUTRAL[1], 'Global/shared': BLUE[3],
    'Family-aware global': BLUE[3], 'Single global measurement': BLUE[2],
    'Target-specific': TEAL[3], 'Fixed family-matched': GOLD[2],
    'Nearest-rate conventional': GOLD[2], 'Random-policy mean': NEUTRAL[2],
    'Retrospective oracle': BLUE[4]}
COMP_SHORT = {'No added measurement': 'No added\nmeasurement', 'Global/shared': 'Global/\nshared',
    'Family-aware global': 'Family\nshared', 'Single global measurement': 'Single\nglobal',
    'Target-specific': 'Target-\nspecific', 'Fixed family-matched': 'Fixed\nfamily',
    'Nearest-rate conventional': 'Nearest\nrate', 'Random-policy mean': 'Random-policy\nmean',
    'Retrospective oracle': 'Retrospective\noracle'}

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.titlesize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': .55, 'lines.linewidth': .9, 'xtick.major.width': .5, 'ytick.major.width': .5,
    'xtick.major.size': 2, 'ytick.major.size': 2, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'savefig.facecolor': 'white',
    'figure.facecolor': 'white', 'axes.titleweight': 'normal'})
SEQ = LinearSegmentedColormap.from_list('measurement_information', ['#F5F7F8', *TEAL])
DIVERGE = LinearSegmentedColormap.from_list('coral_neutral_blue', [CORAL, '#F7F7F5', BLUE[4]])
UTILITY = LinearSegmentedColormap.from_list('coral_neutral_teal', [CORAL, '#F7F7F5', TEAL[4]])

def csv(name, area='main'):
    return pd.read_csv(DATA / area / name)

def panel(ax, label):
    ax.text(-.13, 1.075, label.lower(), transform=ax.transAxes, fontsize=8,
        fontweight='bold', color=INK, ha='left', va='bottom', clip_on=False)

def clean(ax, grid=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color(NEUTRAL[2])
    if grid:
        ax.grid(axis=grid, color='#E7EBEE', linewidth=.4, zorder=0)
        ax.set_axisbelow(True)

def heading(ax, text):
    ax.set_title(text, loc='left', pad=8)

def save(fig, stem, where):
    # No tight bounding box: the final artboard width is exactly 180 mm.
    fig.savefig(where / f'{stem}.pdf')
    fig.savefig(where / f'{stem}.svg')
    fig.savefig(where / f'{stem}.png', dpi=600)
    fig.savefig(QA / f'{stem}.png', dpi=180)
    for target, resolution in [(where / f'{stem}.png', 600), (QA / f'{stem}.png', 180)]:
        with Image.open(target) as original:
            rgb = original.convert('RGB')
        rgb.save(target, dpi=(resolution, resolution))
    plt.close(fig)

def fig1():
    anchor = csv('Fig1_same_measurement_support.csv')
    matrix = csv('Fig1_measurement_target_map.csv')
    budgets = csv('Fig1_equal_budget_summary.csv')
    fig = plt.figure(figsize=(WIDTH, 4.78), layout='constrained')
    outer = fig.add_gridspec(2, 1, height_ratios=[.65, 1.35], hspace=.20)
    top = outer[0].subgridspec(1, 2, wspace=.15)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.46, 1], wspace=.16)
    domains = DOMAINS
    for j, dom in enumerate(domains):
        ax = fig.add_subplot(top[0, j])
        sub = anchor[anchor.domain.eq(dom)].set_index('target').reindex(['ATP1', 'Pi5'])
        for iy, row in enumerate(sub.itertuples()):
            ax.plot([row.augmented_support_width_pp, row.baseline_support_width_pp], [iy, iy],
                color=NEUTRAL[0], lw=1.2, zorder=1)
            ax.scatter(row.baseline_support_width_pp, iy, s=22, color=NEUTRAL[1], zorder=3)
            ax.scatter(row.augmented_support_width_pp, iy, s=22, color=TEAL[3], zorder=3)
            ax.annotate(f'{row.relative_narrowing_percent:.1f}% narrower',
                (row.augmented_support_width_pp, iy), xytext=(0, -11),
                textcoords='offset points', ha='center', va='top', fontsize=5.8, color=TEAL[4])
        ax.set_yticks([0, 1], ['ATP1', 'Pi5'])
        ax.invert_yaxis(); ax.set_ylim(1.43, -.45)
        ax.set_xlim(0, 85)
        ax.set_xlabel('Predictive support width (percentage points)')
        heading(ax, f'Same added ATP0.1 stress: {dom.rsplit(" ", 1)[-1]} margin')
        clean(ax, 'x')
        if j == 0:
            panel(ax, 'A')
            ax.legend(handles=[Line2D([], [], marker='o', linestyle='', color=NEUTRAL[1], label='Baseline'),
                Line2D([], [], marker='o', linestyle='', color=TEAL[3], label='With added stress')],
                loc='upper left', bbox_to_anchor=(-.01, 1.01), ncol=2,
                frameon=False, fontsize=5.6, markerscale=.65, handletextpad=.4, borderpad=0)

    ax = fig.add_subplot(bottom[0, 0])
    targets = ['ATP0.1', 'ATP1', 'Pi0', 'Pi5']
    measurements = [f'{kind}, {t}' for kind in ['Stress', 'CM spectrum'] for t in targets]
    table = matrix.pivot(index='measurement', columns=['domain', 'target'], values='relative_narrowing_percent')
    z = table.reindex(index=measurements, columns=pd.MultiIndex.from_tuples([(d, t) for d in domains for t in targets])).to_numpy(float)
    cmap = UTILITY.copy(); cmap.set_bad('#E8EBEE')
    im = ax.pcolormesh(np.arange(9)-.5, np.arange(9)-.5, np.ma.masked_invalid(z),
        cmap=cmap, norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=90),
        shading='flat', rasterized=False, edgecolors='face', linewidth=.1)
    ax.set_xlim(-.5, 7.5); ax.set_ylim(7.5, -.5)
    ax.set_yticks(np.arange(8), measurements)
    ax.set_xticks(np.arange(8), targets * 2, rotation=55, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='both', length=0, pad=3, labelsize=5.7)
    ax.axvline(3.5, color='white', lw=.9)
    for iy, ix in np.argwhere(~np.isfinite(z)):
        ax.text(ix, iy, 'NA', ha='center', va='center', fontsize=5.2, color=NEUTRAL[2])
    for x, d in [(.25, domains[0]), (.75, domains[1])]:
        ax.text(x, 1.035, d.replace(' margin ', '\nmargin '), transform=ax.transAxes,
            ha='center', va='bottom', fontsize=5.8, linespacing=1.1)
    ax.set_xlabel('Future target')
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.025, ticks=[-3, 0, 30, 60, 90])
    cb.set_label('Support-width reduction (%)', fontsize=6)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5, length=2)
    # Domain labels own this header region; there is no competing floating title.
    ax.text(-.13, 1.115, 'b', transform=ax.transAxes, fontsize=8,
        fontweight='bold', color=INK, ha='left', va='bottom', clip_on=False)

    ax = fig.add_subplot(bottom[0, 1])
    subsets = [budgets[budgets.budget_type.eq(t)].sort_values('budget') for t in ['Frequency pairs', 'Scalars']]
    vals = np.r_[subsets[0].median_width_reduction_percent, subsets[1].median_width_reduction_percent]
    colors = [TEAL[1], TEAL[2], TEAL[4], BLUE[0], BLUE[1], BLUE[2], BLUE[3], BLUE[4]]
    locs = np.r_[np.arange(3), np.arange(5) + 3.75]
    labels = [f'{int(b)}' for s in subsets for b in s.budget]
    ax.bar(locs, vals, color=colors, width=.65, edgecolor='none')
    ax.set_xticks(locs, labels)
    ax.set_ylabel('Median bounded width reduction (%)')
    ax.set_ylim(0, 52)
    ax.text(1, -.13, 'Frequency pairs', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=5.8)
    ax.text(5.75, -.13, 'Scalars', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=5.8)
    heading(ax, 'Equal-budget sparse design')
    clean(ax, 'y'); panel(ax, 'C')
    save(fig, 'Figure1_Target_specific_measurement_value', MAIN)

def fig2():
    robust = csv('Fig2_robust_target_panels.csv')
    ranks = csv('Fig2_model_rank_correlations.csv')
    targets = ['ATP0.1', 'ATP1', 'Pi0', 'Pi5']
    fig = plt.figure(figsize=(WIDTH, 3.10), layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.1, 1.52], wspace=.12)
    ax = fig.add_subplot(gs[0, 0])
    one = robust[robust.budget.eq(1)].set_index('target').reindex(targets)
    colors = [TEAL[2], TEAL[4], GOLD[2], GOLD[4]]
    f = one.selected_frequencies_Hz.astype(float).to_numpy()
    ax.scatter(f, np.arange(4), color=colors, s=25, zorder=3)
    for x, y in zip(f, range(4)):
        ax.annotate(f'{x:.2f}', (x, y), xytext=(4, 0), textcoords='offset points',
            va='center', fontsize=5.8)
    ax.set_xscale('log'); ax.set_xlim(f.min()*.55, f.max()*1.6)
    ax.set_xticks([1, 5, 20], ['1', '5', '20'])
    ax.set_yticks(np.arange(4), targets); ax.set_ylim(3.6, -.6)
    ax.set_xlabel('One-pair frequency (Hz)')
    heading(ax, 'Robust frequency choices')
    clean(ax, 'x'); panel(ax, 'A')

    ax = fig.add_subplot(gs[0, 1])
    for budget, color, offset in zip([1, 2, 3], [TEAL[1], TEAL[2], TEAL[4]], [-.16, 0, .16]):
        sub = robust[robust.budget.eq(budget)].set_index('target').reindex(targets)
        ax.scatter(sub.worst_case_oracle_retained_percent, np.arange(4) + offset,
            s=18, color=color, label=f'{budget}', zorder=3)
    ax.set_yticks(np.arange(4), targets); ax.set_ylim(3.6, -.6)
    ax.set_xlim(90, 100.6); ax.set_xticks([90, 95, 100])
    ax.set_xlabel('Worst-case oracle retained (%)')
    heading(ax, 'Seven model-domain scenarios')
    ax.legend(frameon=False, title='Budget', title_fontsize=5.8, ncol=3,
        loc='lower left', fontsize=5.5, handletextpad=.1, columnspacing=.4, borderpad=0)
    clean(ax, 'x'); panel(ax, 'B')

    ax = fig.add_subplot(gs[0, 2])
    variants = MODEL_ORDER
    z = ranks.pivot(index='model', columns='target', values='spearman_rank_correlation').reindex(
        index=variants, columns=targets).to_numpy(float)
    alternative_summary = csv('Fig2_alternative_model_rank_summary.csv').set_index('target').reindex(targets)
    alternatives = ranks[ranks.model.ne('Reference model')]
    calculated = alternatives.groupby('target').spearman_rank_correlation.median().reindex(targets)
    if not np.allclose(calculated, alternative_summary.median_rank_correlation, rtol=0, atol=1e-15):
        raise AssertionError('Alternative-model medians must exclude the reference self-comparison.')
    # Preserve every cell value, while displaying self-comparisons in a neutral row.
    reference_mask = np.zeros_like(z, dtype=bool); reference_mask[0, :] = True
    cmap = DIVERGE.copy(); cmap.set_bad('#E8EBEE')
    im = ax.pcolormesh(np.arange(5)-.5, np.arange(6)-.5, np.ma.masked_array(z, mask=reference_mask), cmap=cmap,
        norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), shading='flat', rasterized=False,
        edgecolors='face', linewidth=.1)
    ax.set_xlim(-.5, 3.5); ax.set_ylim(4.5, -.5)
    ax.set_xticks(range(4), targets, rotation=45, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(5), variants)
    ax.get_yticklabels()[0].set_color(NEUTRAL[2])
    ax.tick_params(length=0, labelsize=5.5)
    for i in range(5):
        for j in range(4):
            ax.text(j, i, f'{z[i,j]:.2f}', ha='center', va='center', fontsize=5.7,
                color=NEUTRAL[2] if i == 0 else ('white' if abs(z[i,j]) > .74 else INK))
    heading(ax, 'Measurement-rank stability')
    for sp in ax.spines.values(): sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.035, ticks=[-1, 0, 1])
    cb.set_label('Spearman correlation', fontsize=6); cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.ax.tick_params(labelsize=5.5, length=2)
    panel(ax, 'C')
    save(fig, 'Figure2_Robust_target_aware_design', MAIN)

def fig3():
    summary = csv('Fig3_animal_held_out_summary.csv').sort_values('future_MgATP_mM')
    individual = csv('Fig3_mouse_unit_errors.csv')
    freq = csv('Fig3_frequency_selection_summary.csv')
    landscape = csv('Fig3_target_conditioned_landscape.csv')
    fig = plt.figure(figsize=(WIDTH, 4.85), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[.92, 1.12], width_ratios=[1, 1.13], hspace=.12, wspace=.15)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(2); w = .3
    ax.bar(x - w/2, summary.group_only_nrmse_percent, w, color=NEUTRAL[1], label='Group-only', zorder=2)
    ax.bar(x + w/2, summary.selected_pair_nrmse_percent, w, color=TEAL[3], label='Selected pair', zorder=2)
    # Mouse points average that mouse's available genotype x treatment cells.
    # Bar heights preserve the frozen primary cell-weighted mean.
    for i, target in enumerate(summary.future_MgATP_mM):
        s = individual[individual.future_MgATP_mM.eq(target)].sort_values('mouse')
        for j, row in enumerate(s.itertuples()):
            dx = (j - (len(s)-1)/2) * .008
            ax.plot([i-w/2+dx, i+w/2+dx], [row.group_only_nrmse_percent, row.selected_pair_nrmse_percent],
                color=NEUTRAL[0], lw=.5, zorder=3)
            ax.scatter([i-w/2+dx, i+w/2+dx], [row.group_only_nrmse_percent, row.selected_pair_nrmse_percent],
                s=7, facecolor='white', edgecolor=[NEUTRAL[2], TEAL[4]], lw=.45, zorder=4)
        gain = float(summary[summary.future_MgATP_mM.eq(target)].nrmse_gain_pp.iloc[0])
        point_top = float(s[['group_only_nrmse_percent', 'selected_pair_nrmse_percent']].to_numpy().max())
        ax.text(i, point_top + 1.3, f'{gain:.2f} pp gain', ha='center', va='bottom',
            fontsize=6, color=TEAL[4])
    ax.set_xticks(x, [f'{v:g}' for v in summary.future_MgATP_mM])
    ax.set_xlabel('Future MgATP (mM)'); ax.set_ylabel('Held-out NRMSE (%)')
    ax.set_ylim(0, max(individual.group_only_nrmse_percent.max(), individual.selected_pair_nrmse_percent.max()) * 1.12)
    ax.legend(frameon=False, loc='upper right', ncol=1, fontsize=5.6)
    heading(ax, 'Animal-held-out prediction')
    clean(ax, 'y'); panel(ax, 'A')

    ax = fig.add_subplot(gs[0, 1])
    xx = freq.future_MgATP_mM.to_numpy(float)
    med = freq.median_frequency_Hz.to_numpy(float)
    ax.errorbar(xx, med, yerr=np.vstack([med-freq.lower_quartile_Hz, freq.upper_quartile_Hz-med]),
        color=TEAL[4], fmt='-', lw=.9, capsize=2, elinewidth=.6, zorder=2)
    for i, (xv, yv) in enumerate(zip(xx, med)):
        color = TEAL[min(4, i)]
        ax.scatter(xv, yv, color=color, s=22, edgecolor=TEAL[4], lw=.35, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(xx, [f'{v:g}' for v in xx], minor=False)
    ax.set_yticks([1, 2, 10, 40], ['1', '2', '10', '40'])
    ax.set_ylim(1, 65); ax.set_xlabel('Future MgATP (mM)')
    ax.set_ylabel('Selected frequency (Hz)')
    heading(ax, 'Frequency shifts across targets')
    clean(ax, 'y'); panel(ax, 'B')

    ax = fig.add_subplot(gs[1, :])
    targets = np.array(sorted(landscape.future_MgATP_mM.unique()), float)
    frequencies = np.array(sorted(landscape.frequency_Hz.unique()), float)
    z = landscape.pivot(index='future_MgATP_mM', columns='frequency_Hz', values='mean_nrmse_gain_pp').reindex(
        index=targets, columns=frequencies).to_numpy(float)
    xf = np.log10(frequencies)
    edges = np.r_[xf[0] - (xf[1]-xf[0])/2, (xf[:-1]+xf[1:])/2, xf[-1] + (xf[-1]-xf[-2])/2]
    mesh = ax.pcolormesh(edges, np.arange(len(targets)+1)-.5, z, cmap=UTILITY,
        norm=TwoSlopeNorm(vmin=-3, vcenter=0, vmax=10), shading='flat', rasterized=False,
        edgecolors='face', linewidth=.1)
    ax.set_yticks(np.arange(len(targets)), [f'{t:g}' for t in targets])
    ticks = [0.5, 2, 10, 38, 100, 245]
    ax.set_xticks(np.log10(ticks), [f'{t:g}' for t in ticks])
    ax.set_xlabel('Current complex-modulus frequency (Hz; log scale)')
    ax.set_ylabel('Future MgATP (mM)')
    heading(ax, 'Target-conditioned held-out utility')
    ax.tick_params(length=2)
    cb = fig.colorbar(mesh, ax=ax, fraction=.023, pad=.022, ticks=[-3, 0, 5, 10])
    cb.set_label('NRMSE gain (percentage points)', fontsize=6)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face')
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.5, length=2)
    for sp in ax.spines.values(): sp.set_visible(False)
    panel(ax, 'C')
    save(fig, 'Figure3_Target_conditioned_landscape', MAIN)

def unit_comparison(ax, units, summary, order, unit_col, summary_col):
    table = units.pivot(index=unit_col, columns='comparator', values='mean_standardized_absolute_error').reindex(columns=order)
    jitter = np.linspace(-.095, .095, len(table))
    for (_, row), dx in zip(table.iterrows(), jitter):
        vals = row.to_numpy(float)
        ax.plot(np.arange(len(order))+dx, vals, color=NEUTRAL[0], lw=.45, zorder=1)
        for i, comp in enumerate(order):
            ax.scatter(i+dx, vals[i], s=8, facecolor='white',
                edgecolor=COMPARATOR_COLORS[comp], lw=.55, zorder=3)
    for i, comp in enumerate(order):
        value = float(summary[summary.comparator.eq(comp)][summary_col].iloc[0])
        ax.scatter(i, value, marker='D', s=24, facecolor=COMPARATOR_COLORS[comp],
            edgecolor='white', lw=.5, zorder=5)
    ax.set_xticks(range(len(order)), [COMP_SHORT[c] for c in order])
    ax.set_ylabel('Standardized absolute error')
    ax.set_ylim(0, float(np.nanmax(table.to_numpy(float))) * 1.13)
    clean(ax, 'y')

def fig4():
    rlc = csv('Fig4_rat_comparator_summary.csv')
    rlc = rlc[rlc.summary.eq('Overall')]
    rlcunits = csv('Fig4_rat_unit_errors.csv')
    human = csv('Fig4_human_atrial_support.csv')
    human = human[human.measurement_context.eq('Added ATP0.1 stress')]
    pacing = csv('Fig4_pacing_comparator_summary.csv')
    pacingunits = csv('Fig4_participant_unit_errors.csv')
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.18),
        gridspec_kw={'width_ratios': [1, .98, 1.22], 'wspace': .15}, layout='constrained')
    ax = axes[0]
    unit_comparison(ax, rlcunits, rlc, ['No added measurement', 'Global/shared', 'Target-specific'],
        'aligned_rat_summary_row', 'standardized_mae')
    heading(ax, 'Rat repeated-dose prediction')
    panel(ax, 'A')

    ax = axes[1]
    for i, (target, color) in enumerate(HUMAN_TARGET_COLORS.items()):
        sub = human[human.target.eq(target)].set_index('log_radius').reindex([.25, .5])
        ax.bar(np.arange(2)+(i-.5)*.32, sub.relative_narrowing_percent, width=.32,
            color=color, edgecolor='none', label=target)
    ax.set_xticks([0, 1], ['0.25', '0.50']); ax.set_xlabel('Log-domain radius')
    ax.set_ylabel('Support-width reduction (%)'); ax.set_ylim(0, 55)
    heading(ax, 'Human atrial target support')
    ax.legend(frameon=False, loc='upper left', fontsize=5.8)
    clean(ax, 'y'); panel(ax, 'B')

    ax = axes[2]
    unit_comparison(ax, pacingunits, pacing,
        ['No added measurement', 'Family-aware global', 'Fixed family-matched', 'Target-specific'],
        'participant', 'standardized_mae')
    heading(ax, 'Participant-held-out pacing')
    panel(ax, 'C')
    save(fig, 'Figure4_Pharmacological_and_human_evidence', MAIN)

def fig5():
    ladder = csv('Fig5_within_dataset_normalization.csv')
    transfer = csv('Fig5_rat_mouse_transfer.csv')
    paired = csv('Fig5_support_and_accuracy_pairs.csv')
    fig = plt.figure(figsize=(WIDTH, 4.78), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.04, 1.1], width_ratios=[1.13, 1], hspace=.14, wspace=.18)
    ax = fig.add_subplot(gs[0, 0])
    comps = ['No added measurement', 'Global/shared', 'Target-specific', 'Fixed family-matched']
    markers = ['o', 's', 'D', '^']
    for iy, system in enumerate(['Tanner', 'RLC-1', 'Radbill']):
        s = ladder[ladder.dataset.eq(system)]
        # All values remain independent within-system contrasts; there is no pooled estimate.
        ax.plot([s.normalized_error.min(), 1], [iy, iy], color=NEUTRAL[0], lw=.75, zorder=1)
        for _, row in s.iterrows():
            rank = 0 if row.comparator == 'No added measurement' else 1 if row.comparator in ['Global/shared', 'Family-aware global'] else 2 if row.comparator == 'Target-specific' else 3
            off = [.15, .05, -.05, -.15][rank]
            ax.scatter(row.normalized_error, iy+off, s=22, marker=markers[rank],
                color=COMPARATOR_COLORS[row.comparator], edgecolor='white', lw=.3, zorder=3)
    nrad = int(ladder[ladder.dataset.eq('Radbill')].biological_units.iloc[0])
    ax.set_yticks([0, 1, 2], ['Tanner\n7 trabeculae', 'RLC-1\n7 rats', f'Radbill\n{nrad} participants'])
    ax.set_ylim(2.6, -.6); ax.set_xlim(.15, 1.08)
    ax.set_xticks([.25, .5, .75, 1]); ax.axvline(1, color=NEUTRAL[1], lw=.65, ls=(0, (2, 2)))
    ax.set_xlabel('Within-dataset error / no-measurement error')
    heading(ax, 'Held-out predictive benefit')
    handles = [Line2D([], [], linestyle='', marker=m, markersize=3.5,
        color=COMPARATOR_COLORS[c], label=l) for c, m, l in zip(comps, markers,
            ['No added measurement', 'Global/shared', 'Target-specific', 'Conventional'])]
    ax.legend(handles=handles, frameon=False, ncol=2, fontsize=5.5, loc='upper left',
        bbox_to_anchor=(-.005, 1.07), handletextpad=.3, columnspacing=.8, borderpad=0)
    clean(ax, 'x'); panel(ax, 'A')

    ax = fig.add_subplot(gs[0, 1])
    transfer = transfer.copy()
    order = [('ATP0.1', '1'), ('ATP0.1', '2'), ('ATP0.1', '3'),
        ('ATP1', '1'), ('ATP1', '2'), ('ATP1', '3'), ('Both ATP targets', '1-3, equal weight')]
    labels = []
    for iy, (target, b) in enumerate(order):
        row = transfer[transfer.future_target.eq(target) & transfer.frequency_pair_budget.astype(str).eq(b)].iloc[0]
        value = float(row.normalized_minus_absolute_nrmse_pp)
        lower = float(row.bootstrap_95_lower_pp); upper = float(row.bootstrap_95_upper_pp)
        color = CORAL if iy < 6 else NEUTRAL[2]
        ax.errorbar(value, iy, xerr=[[value-lower], [upper-value]], fmt='o' if iy < 6 else 'D',
            markersize=3.2, color=color, capsize=1.7, lw=.65, zorder=3)
        labels.append(f'{target}, {b} {"pair" if b == "1" else "pairs"}' if iy < 6 else 'Equal-mouse mean')
    ax.set_yticks(range(7), labels); ax.set_ylim(6.6, -.7)
    ax.set_xlim(-3.2, 8.9); ax.set_xticks([-2, 0, 2, 4, 6, 8])
    ax.axvline(0, color=NEUTRAL[2], ls=(0, (2, 2)), lw=.65)
    ax.axhline(5.5, color=NEUTRAL[0], lw=.5)
    ax.set_xlabel('Normalized - absolute NRMSE (pp)')
    heading(ax, 'Rat-to-mouse transfer')
    clean(ax, 'x'); panel(ax, 'B')

    ax = fig.add_subplot(gs[1, :])
    for target, color, marker in [('ATP1', HUMAN_TARGET_COLORS['ATP1'], 'o'),
        ('Pi10', HUMAN_TARGET_COLORS['Pi10'], 's')]:
        s = paired[paired.target.eq(target)]
        ax.scatter(s.support_width_reduction_pp, s.stress_error_improvement_pp, s=19,
            marker=marker, color=color, edgecolor='white', lw=.3, label=target, zorder=3)
    ax.axvline(0, color=NEUTRAL[1], ls=(0, (2, 2)), lw=.65)
    ax.axhline(0, color=NEUTRAL[1], ls=(0, (2, 2)), lw=.65)
    ax.set_xlabel('Predictive support narrowing (percentage points)')
    ax.set_ylabel('Stress-error improvement\n(percentage points)')
    heading(ax, 'Information and realized accuracy')
    ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=5.8, handletextpad=.3, columnspacing=.8)
    clean(ax); panel(ax, 'C')
    save(fig, 'Figure5_Quantitative_boundaries', MAIN)

def supplementary_figures():
    rlc = csv('Fig4_rat_comparator_summary.csv')
    rlc = rlc[rlc.summary.eq('Future target')]
    targets = ['3 uM PeakTension', '3 uM TTP', '3 uM RT50', '10 uM PeakTension', '10 uM TTP', '10 uM RT50']
    labels = ['3 uM peak tension', '3 uM time to peak', '3 uM half-relaxation time',
        '10 uM peak tension', '10 uM time to peak', '10 uM half-relaxation time']
    fig, ax = plt.subplots(figsize=(WIDTH, 2.85), layout='constrained')
    for comp, off in zip(['No added measurement', 'Global/shared', 'Target-specific'], [-.14, 0, .14]):
        s = rlc[rlc.comparator.eq(comp)].set_index('future_target').reindex(targets)
        ax.scatter(s.standardized_mae, np.arange(6)+off, color=COMPARATOR_COLORS[comp], s=22, label=comp, zorder=3)
    ax.set_yticks(range(6), labels); ax.set_ylim(6.15, -.6)
    ax.set_xlabel('Standardized mean absolute error')
    heading(ax, 'Rat repeated-dose outcome heterogeneity')
    ax.legend(frameon=False, loc='lower right', ncol=3, fontsize=5.8)
    clean(ax, 'x'); panel(ax, 'A')
    save(fig, 'SupplementaryFigure1_Rat_outcome_heterogeneity', SUPP)

    rad = csv('Fig4_pacing_comparator_summary.csv')
    units = csv('Fig4_participant_unit_errors.csv')
    perm = csv('FigS2_within_family_mapping.csv', 'supplement').iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0),
        gridspec_kw={'width_ratios': [1.7, 1], 'wspace': .25}, layout='constrained')
    ax = axes[0]
    order = ['No added measurement', 'Single global measurement', 'Family-aware global', 'Fixed family-matched', 'Target-specific']
    unit_comparison(ax, units, rad, order, 'participant', 'standardized_mae')
    heading(ax, 'Pacing comparator definitions')
    panel(ax, 'A')
    ax = axes[1]
    ax.bar([0, 1], [perm.n_no_worse, perm.n_permutations-perm.n_no_worse], width=.58,
        color=[CORAL, NEUTRAL[1]])
    ax.set_xticks([0, 1], ['No worse\nthan identity', 'Higher\nerror'])
    ax.set_ylabel('Within-family target mappings')
    ax.set_ylim(0, float(perm.n_permutations) * 1.03)
    ax.text(.5, .98, f'{int(perm.n_no_worse)}/{int(perm.n_permutations)} = {perm.p_fraction:.3f}',
        transform=ax.transAxes, va='top', ha='center', fontsize=6)
    heading(ax, 'Fine-target mapping boundary')
    clean(ax, 'y'); panel(ax, 'B')
    save(fig, 'SupplementaryFigure2_Pacing_fair_comparators', SUPP)

    tan = csv('FigS3_mechanical_comparator_summary.csv', 'supplement')
    order = ['No added measurement', 'Global/shared', 'Target-specific', 'Random-policy mean',
        'Nearest-rate conventional', 'Retrospective oracle']
    tan = tan.set_index('comparator').reindex(order).dropna(subset=['mae']).reset_index()
    fig, ax = plt.subplots(figsize=(WIDTH, 2.6), layout='constrained')
    labels = []
    for iy, row in enumerate(tan.itertuples()):
        ax.scatter(row.mae, iy, color=COMPARATOR_COLORS[row.comparator], s=27, zorder=3)
        ax.annotate(f'{row.mae:.3f}', (row.mae, iy), xytext=(5, 0), textcoords='offset points', fontsize=5.8, va='center')
        labels.append(row.comparator)
    ax.set_yticks(range(len(tan)), labels); ax.invert_yaxis()
    ax.set_xlim(0, tan.mae.max()*1.11); ax.set_xlabel('Mean absolute error (mN/mm²)')
    heading(ax, 'Mechanical-response comparator boundary')
    clean(ax, 'x'); panel(ax, 'A')
    save(fig, 'SupplementaryFigure3_Mechanical_comparator_boundary', SUPP)

def main():
    fig1(); fig2(); fig3(); fig4(); fig5(); supplementary_figures()
    print('Built five main and three supplementary figures: PDF, editable SVG, and 600-dpi PNG.')

if __name__ == '__main__':
    main()
