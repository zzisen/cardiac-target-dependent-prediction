"""Copy accepted frozen numeric inputs with byte hashes; never read old artwork.

This is provenance preparation, not a scientific rerun. The resulting snapshots
permit independent figure builds without access to large raw-data archives.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

PACKAGE=Path(__file__).resolve().parents[1]
WORKSPACE=PACKAGE.parent
LEDGER='P2_V3_CANONICAL_EVIDENCE_LEDGER_2026-10-06'
ABT='P2_V3_ABT'
A23=f'{ABT}/A23_EXECUTION_LUNA7_RESUMED_2026-10-06'
A24=f'{ABT}/A24_MATCHED_COMPLEXITY_FAMILY_CONTROL_CODEX_2026-10-06'
A25=f'{ABT}/A25_ADAPTIVE_TARGET_RESOLUTION_CODEX_2026-10-06'
B1=f'{ABT}/B1_OUTCOME_EXECUTION_LUNA14_2026-10-06'
B2=f'{ABT}/B2_STANFORD_TRANSPORT_LUNA15_2026-10-06'
SOURCES={
 'canonical_master.csv':(f'{LEDGER}/P2_V3_MASTER_EVIDENCE_LEDGER.csv','1–5','Evidence rows, accepted metrics and claim boundaries'),
 'canonical_adaptive.csv':(f'{LEDGER}/P2_V3_A25_ADAPTIVE_LEDGER.csv','2;4','System rows; exact fixed and adaptive point risks'),
 'canonical_natural_family.csv':(f'{LEDGER}/P2_V3_A24_NATURAL_FAMILY_LEDGER.csv','3','Natural partition identity/risk and rank denominators'),
 'canonical_training.csv':(f'{LEDGER}/P2_V3_B1_LEDGER.csv','5B;5C','Policy risks; paired contrasts and intervals; selected-channel counts'),
 'canonical_transport.csv':(f'{LEDGER}/P2_V3_B2_LEDGER.csv','5A;5E;5F','Overall point risks and provenance boundaries'),
 'canonical_theory.md':(f'{LEDGER}/P2_V3_TSTAR_M_CANONICAL_LEDGER.md','1','Finite-action proposition and schematic boundaries'),
 'accepted_frontier.csv':(f'{A23}/A23_CROSS_SYSTEM_FRONTIER_REGISTRY.csv','2','Overall and separate comparator rows'),
 'rlc_natural_ranks.csv':(f'{A24}/A24_RLC1_LEAVE_ONE_UNIT_RANKS.csv','3B','is_natural_partition=1; source-order rat deletions'),
 'radbill_natural_ranks.csv':(f'{A24}/A24_RADBILL_LEAVE_ONE_UNIT_RANKS.csv','3D','is_natural_partition=1; 17 scored participant deletions'),
 'resolution_frequencies.csv':(f'{A25}/A25_RESOLUTION_SELECTION_FREQUENCIES.csv','4','Counts of scored outer biological units'),
 'training_risks.csv':(f'{B1}/B1_PRIMARY_RISK_SUMMARY.csv','5B','Three accepted policy risks'),
 'training_contrasts.csv':(f'{B1}/B1_PRIMARY_CONTRASTS.csv','5B','Paired point contrasts'),
 'training_bootstrap.csv':(f'{B1}/B1_BOOTSTRAP_STABILITY.csv','5B','Descriptive patient bootstrap percentiles'),
 'training_target_summary.csv':(f'{B1}/B1_TARGET_LEVEL_SUMMARY.csv','5C','Target-specific frequency and risk summaries'),
 'training_exact_actions.csv':(f'{B1}/B1_EXACT_SELECTION_MAP.csv','5C','One selected channel per target per outer patient fold'),
 'training_shared_actions.csv':(f'{B1}/B1_SHARED_SELECTION_MAP.csv','5C','One channel per outer patient fold'),
 'training_deletion_contrasts.csv':(f'{B1}/B1_LEAVE_ONE_PATIENT_SENSITIVITY.csv','5D','Three contrasts after each one-patient deletion'),
 'transport_risks.csv':(f'{B2}/B2_STANFORD_PRIMARY_RISK_SUMMARY.csv','5E','Three overall point risks'),
 'transport_target_summary.csv':(f'{B2}/B2_STANFORD_TARGET_LEVEL_SUMMARY.csv','5E','Dasatinib and IL-7 three-policy native risks'),
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,default=WORKSPACE)
    args=parser.parse_args()
    dest=PACKAGE/'figures_src'/'inputs'
    dest.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,(rel,panels,scope) in SOURCES.items():
        original=args.workspace/rel
        if not original.is_file():
            raise FileNotFoundError(original)
        snapshot=dest/name
        shutil.copyfile(original,snapshot)
        original_hash=sha(original)
        assert original_hash==sha(snapshot)
        rows.append(dict(panels=panels,input_file=str(snapshot.relative_to(PACKAGE)).replace('\\','/'),
            authoritative_source=rel,source_sha256=original_hash,snapshot_sha256=sha(snapshot),
            selected_rows_or_fields=scope,transformation='byte-identical input snapshot',
            artifact_type='accepted frozen numerical/formal input'))
    manifest=PACKAGE/'P2_V3_FIGURE_SOURCE_MANIFEST.csv'
    with manifest.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({'copied_inputs':len(rows),'hash_mismatches':0,
        'textual_authority_status':'owner-confirmed P2 V3 directory; exact source filenames retained in authority map'},indent=2))

if __name__=='__main__':main()
