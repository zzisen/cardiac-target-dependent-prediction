"""Guard the alternative-model estimand and the mouse aggregation contract."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/common'))
from release_analysis import alternative_model_rank_summary, RANK_COLUMN


def check():
    ranks = pd.read_csv(ROOT/'analysis/02_robustness/variants/results/N_MODEL_RANK_STABILITY.csv')
    actual = alternative_model_rank_summary(ranks).set_index('target')
    expected = {'ATP0.1': .609, 'ATP1': .679, 'Pi0': .107, 'Pi5': .175}
    assert actual.alternative_models.eq(4).all()
    assert not actual.reference_self_comparison_included.any()
    assert {t: round(v, 3) for t, v in actual.median_rank_correlation.items()} == expected
    # Changing only the self-reference must have no effect on this estimand.
    changed = ranks.copy()
    changed.loc[changed.variant == 'published_Model16D', RANK_COLUMN] = -1
    np.testing.assert_array_equal(actual.median_rank_correlation,
                                  alternative_model_rank_summary(changed).set_index('target').median_rank_correlation)
    source = pd.read_csv(ROOT/'source_data/main/Fig2_alternative_model_rank_summary.csv').set_index('target')
    np.testing.assert_allclose(source.loc[actual.index, 'median_rank_correlation'],
                               actual.median_rank_correlation, rtol=0, atol=2e-15)
    assert source.alternative_models.eq(4).all() and not source.reference_self_comparison_included.any()
    # The displayed individual correlations, including self-reference, are unchanged.
    models = {'s16': 'published_Model16D', 's14': 'md4_best_alternative',
              's9': 'md4_second_best_alternative', 's5': 'md4_best_single_strain',
              's4': 'md4_best_without_kminus2_strain'}
    cells = pd.read_csv(ROOT/'source_data/main/Fig2_model_rank_correlations.csv')
    assert len(cells) == len(ranks) == 20
    for row in cells.itertuples():
        value = ranks[(ranks.target == row.target) & (ranks.variant == models[row.model_id])][RANK_COLUMN]
        assert len(value) == 1
        np.testing.assert_allclose(row.spearman_rank_correlation, value.iloc[0], rtol=0, atol=2e-15)
    q = pd.read_csv(ROOT/'analysis/03_awinda/frequency/results/Q_AWINDA_SPARSE_PANEL_RESULTS.csv')
    summary = pd.read_csv(ROOT/'source_data/main/Fig3_animal_held_out_summary.csv')
    for row in summary.itertuples():
        part = q[q.target_ATP_mM == row.future_MgATP_mM]
        assert len(part) == row.animal_condition_cells == 19
        assert part.heldout_animal.nunique() == row.mice == 10
        gain = float((part.baseline_group_only_nrmse_pct - part.outer_nrmse_pct).mean())
        np.testing.assert_allclose(gain, row.nrmse_gain_pp, rtol=0, atol=2e-13)
        assert round(gain, 2) == {.1: .22, 1.: 7.45}[float(row.future_MgATP_mM)]
    print('PASS: four alternative-only medians; all20 correlation cells unchanged; mouse gains0.22/7.45pp retain19cell/10mouse weighting.')


if __name__ == '__main__':
    check()
