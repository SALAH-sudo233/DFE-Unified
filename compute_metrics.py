#!/usr/bin/env python3
"""Compute metrics from docking results and append to results JSON."""
import json, sys, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

dock_output = sys.argv[1]
results_file = sys.argv[2]
pocket = sys.argv[3]

with open(dock_output) as f:
    results = json.load(f)

total = len(results)
valid = [r for r in results if r.get('valid')]
valid_count = len(valid)
smiles_list = [r.get('smiles') for r in valid if r.get('smiles')]
unique_smiles = set(smiles_list)
qeds = [r.get('qed', 0) for r in valid if r.get('qed') is not None]
mws = [r.get('mw', 0) for r in valid if r.get('mw') is not None]
docked = [r for r in valid if r.get('docking_score') is not None]
dock_scores = [r['docking_score'] for r in docked]

diversity = 0.0
try:
    mols_fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2)
            mols_fps.append(fp)
    if len(mols_fps) > 1:
        sims = DataStructs.BulkTanimotoSimilarity(mols_fps[0], mols_fps[1:])
        diversity = 1.0 - np.mean(sims)
except:
    pass

metrics = {
    'pocket': pocket,
    'total_molecules': total,
    'valid_molecules': valid_count,
    'unique_molecules': len(unique_smiles),
    'success_rate': valid_count / max(total, 1),
    'unique_rate': len(unique_smiles) / max(valid_count, 1),
    'diversity': float(diversity),
    'docking_score_mean': float(np.mean(dock_scores)) if dock_scores else None,
    'docking_score_median': float(np.median(dock_scores)) if dock_scores else None,
    'docking_score_min': float(np.min(dock_scores)) if dock_scores else None,
    'docking_score_max': float(np.max(dock_scores)) if dock_scores else None,
    'qed_mean': float(np.mean(qeds)) if qeds else None,
    'mw_mean': float(np.mean(mws)) if mws else None,
    'docked_count': len(dock_scores),
}

print(json.dumps(metrics, indent=2))

with open(results_file, 'r') as f:
    all_results = json.load(f)
all_results[pocket] = metrics
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)
