"""Report alternative-only rank medians from the unchanged mechanistic results."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/common'))
from release_analysis import alternative_model_rank_summary

if __name__ == '__main__':
    output = ROOT/'results/recomputed/ALTERNATIVE_MODEL_RANK_SUMMARY.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = alternative_model_rank_summary()
    summary.to_csv(output, index=False, float_format='%.17g')
    print(summary.to_string(index=False))
