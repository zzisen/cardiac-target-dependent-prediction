"""Reproduce four-alternative rank medians, excluding reference self-comparisons.

This is a reporting calculation only; individual rank-correlation cells are
unchanged. The full-precision and three-decimal summaries are both exported.
"""
from pathlib import Path
import csv
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'source_data' / 'main'
with (DATA / 'Fig2_model_rank_correlations.csv').open(encoding='utf-8', newline='') as handle:
    ranks = list(csv.DictReader(handle))

rows = []
for target in ['ATP0.1', 'ATP1', 'Pi0', 'Pi5']:
    alternatives = [r for r in ranks if r['target'] == target and r['model_id'] != 's16']
    if {r['model_id'] for r in alternatives} != {'s14', 's9', 's5', 's4'}:
        raise ValueError('Expected exactly four true mechanistic alternatives.')
    value = median(float(r['spearman_rank_correlation']) for r in alternatives)
    rows.append({'target': target, 'alternative_models': 4,
        'reference_self_comparison_included': False, 'median_rank_correlation': format(value, '.17g'),
        'median_rounded_3dp': f'{value:.3f}',
        'excluded_self_comparison': 'Reference model (s16)'})
output = DATA / 'Fig2_alternative_model_rank_summary.csv'
with output.open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print('Four-alternative medians: ' + ', '.join(r['median_rounded_3dp'] for r in rows))
